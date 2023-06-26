use indextree::Arena;
use indextree::NodeId;
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray_linalg::Inverse;
use pyo3::prelude::*;
use std::ops::Add;

use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use pyo3::wrap_pyfunction;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Deterministic position vector
pub struct DetPos {
  pos: Array1<f64>,
}

/// Probabilitstic position vector. It is represented via a Gaussian mixture
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct ProbPos {
  mus: Array2<f64>,
  covs: Array3<f64>,
  weights: Array1<f64>,
  #[pyo3(get, set)]
  n_components: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pos {
  Det(DetPos),
  Prob(ProbPos),
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Module {
  /// Position vec of module centroid
  centroid: Pos,

  /// labels for points of inter
  labels: Vec<i64>,

  /// position of next module
  p_vector: Pos,

  /// reference frame of next module
  next_ref_frame: Array2<f64>,

  /// tracked points:
  tracked_points: Vec<ProbPos>,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct GeneralModule {
  // Assumes that the first of the p_vectors is aligned on the x axis
  /// Position vec of module centroid
  centroid: Pos,

  /// labels for points of inter
  labels: Vec<i64>,

  /// position of next module
  p_vectors: Vec<Pos>,

  /// a vector of reference frame of next modules
  next_ref_frames: Vec<Array2<f64>>,

  /// reference frame
  ref_frame: Array2<f64>,

  /// tracked points:
  tracked_points: Vec<ProbPos>,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Construct {
  // Raw modules that are used in the construct.
  raw_modules: Vec<GeneralModule>,

  // Edges are stored as a tuple of ((node, connection_point), (node, connection_point))
  edges: Vec<((usize, usize), (usize, usize))>,

  // Modules assembled in the given order and dynamics propagated.
  assembled_modules: Vec<GeneralModule>,

  // node_ids
  node_ids: Vec<NodeId>,

  // Tree constructed using the given edges.
  tree: Arena<usize>,

  // Adjacency matrix constructed internally using given edges
  adjacency_matrix: Array2<i64>,
}

pub trait Vector {
  fn get_mean_pos(&self) -> Array1<f64>;
  fn rotate_mut(&mut self, rot_mat: &Array2<f64>);
  fn translate_mut(&mut self, trans: &Array1<f64>);
}

impl Vector for DetPos {
  fn get_mean_pos(&self) -> Array1<f64> {
    self.pos.clone()
  }

  fn rotate_mut(&mut self, rot_mat: &Array2<f64>) {
    self.pos = rot_mat.dot(&self.pos);
  }

  fn translate_mut(&mut self, trans: &Array1<f64>) {
    self.pos += trans;
  }
}

/// Generic methods that both DetPos and DetVec must implement
impl Vector for ProbPos {
  fn get_mean_pos(&self) -> Array1<f64> {
    let mut pos: Array1<f64> = self.mus.slice(s![0, ..,]).into_owned();
    for i in 0..self.mus.nrows() {
      for j in 0..self.mus.ncols() {
        pos[j] += self.weights[i] * self.mus[[i, j]];
      }
    }
    pos
  }

  fn rotate_mut(&mut self, rot_mat: &Array2<f64>) {
    // rotate means
    for n in 0..self.n_components {
      let mu = self.mus.slice(s![n, ..]).into_owned();
      self.mus.slice_mut(s![n, ..]).assign(&rot_mat.dot(&mu));
      let mut cov = self.covs.slice(s![n, .., ..]).into_owned();
      let mut cov_rot_t = self.covs.slice(s![n, .., ..]).into_owned();
      // let cov_l = &cov.dot(&rot_mat.t());
      cov_rot_t
        .slice_mut(s![.., ..])
        .assign(&cov.dot(&rot_mat.t()));
      self
        .covs
        .slice_mut(s![n, .., ..])
        .assign(&rot_mat.dot(&cov_rot_t))
    }
  }

  fn translate_mut(&mut self, trans: &Array1<f64>) {
    for n in 0..self.n_components {
      let mu = self.mus.slice(s![n, ..]).into_owned();
      self.mus.slice_mut(s![n, ..]).assign(&(trans + &mu));
    }
  }
}

impl Vector for Pos {
  fn get_mean_pos(&self) -> Array1<f64> {
    match self {
      Self::Det(pos) => pos.get_mean_pos(),
      Self::Prob(pos) => pos.get_mean_pos(),
    }
  }

  fn rotate_mut(&mut self, rot_mat: &Array2<f64>) {
    match self {
      Self::Det(pos) => pos.rotate_mut(rot_mat),
      Self::Prob(pos) => pos.rotate_mut(rot_mat),
    }
  }

  fn translate_mut(&mut self, trans: &Array1<f64>) {
    match self {
      Self::Det(pos) => pos.translate_mut(trans),
      Self::Prob(pos) => pos.translate_mut(trans),
    }
  }
}

impl Add for DetPos {
  type Output = DetPos;
  fn add(self, other: DetPos) -> DetPos {
    DetPos {
      pos: self.pos + other.pos,
    }
  }
}

impl From<DetPos> for ProbPos {
  fn from(det_pos: DetPos) -> Self {
    let mut pos = Self::new_zero(0);
    pos.mus.slice_mut(s![0, ..]).assign(&det_pos.pos);
    pos
  }
}

/// Allows one to use the '+'  operator on two ProbPos
impl Add for ProbPos {
  type Output = Self;
  fn add(self, other: Self) -> Self {
    let self_n_comps = self.mus.len_of(Axis(0));
    let other_n_comps = other.mus.len_of(Axis(0));
    let n_components = self_n_comps * other_n_comps;

    let mut mus = Array::zeros((self_n_comps * other_n_comps, self.mus.ncols()));
    let mut covs = Array::zeros((mus.nrows(), self.mus.ncols(), self.mus.ncols()));
    let mut weights = Array::from_vec(vec![0.0; mus.nrows()]);

    for i in 0..self_n_comps {
      for j in 0..other_n_comps {
        let idx = i * other.mus.nrows() + j;
        mus.slice_mut(s![idx, ..]).assign(
          &(self.mus.slice(s![i, ..]).into_owned() + other.mus.slice(s![j, ..]).into_owned()),
        );
        covs.slice_mut(s![idx, .., ..]).assign(
          &(self.covs.slice(s![i, .., ..]).into_owned()
            + other.covs.slice(s![j, .., ..]).into_owned()),
        );
        weights[idx] = self.weights[i] * other.weights[j];
      }
    }

    Self {
      mus,
      covs,
      weights,
      n_components,
    }
  }
}

impl Add for Pos {
  type Output = Pos;
  fn add(self, other: Pos) -> Pos {
    match (self, other) {
      (Pos::Det(a), Pos::Det(b)) => Pos::Det(a + b),
      (Pos::Prob(a), Pos::Prob(b)) => Pos::Prob(a + b),
      (Pos::Det(a), Pos::Prob(b)) => Pos::Prob(b.det_add(a)),
      (Pos::Prob(a), Pos::Det(b)) => Pos::Prob(a.det_add(b)),
    }
  }
}

impl ProbPos {
  /// create a new zero vector
  fn new_zero(n_components: usize) -> Self {
    Self {
      mus: Array::zeros((n_components, 3)),
      covs: Array::zeros((n_components, 3, 3)),
      weights: Array::from_vec(vec![1.0 / n_components as f64; n_components]),
      n_components,
    }
  }

  fn det_add_mut(&mut self, shift: DetPos) {
    for n in 0..self.n_components {
      let mu = self.mus.slice(s![n, ..]).into_owned();
      self
        .mus
        .slice_mut(s![n, ..])
        .assign(&(mu + shift.pos.clone()));
    }
  }

  fn det_add(&self, shift: DetPos) -> ProbPos {
    let mut mus = self.mus.clone();
    let covs = self.covs.clone();
    let weights = self.weights.clone();
    for j in 0..mus.len_of(Axis(0)) {
      mus
        .slice_mut(s![j, ..])
        .assign(&(self.mus.slice(s![j, ..]).into_owned() + shift.pos.clone().into_owned()));
    }
    Self {
      mus,
      covs,
      weights,
      n_components: self.n_components,
    }
  }

  /// add two probabilitstic vectors together
  fn prob_add(&self, other: Self) -> Self {
    self.clone() + other
  }

  fn total_mean(&self) -> Array1<f64> {
    self.weights.dot(&self.mus)
  }

  fn total_cov(&self) -> Array2<f64> {
    // total sqaured mean
    let mut total_s_mu = Array2::zeros((3, 3));

    let total_mean = self.total_mean();

    for i in 0..self.weights.len() {
      let cov = self.covs.slice(s![i, .., ..]).into_owned();
      let mu = self.mus.slice(s![i, ..]).into_owned();
      let mu_s = mu.slice(s![.., NewAxis]).dot(&mu.slice(s![NewAxis, ..]));

      let cum_term = &total_s_mu + self.weights[i] * (cov + mu_s);
      total_s_mu.assign(&cum_term);
    }

    total_s_mu
      - total_mean
        .slice(s![.., NewAxis])
        .dot(&total_mean.slice(s![NewAxis, ..]))
  }
}

#[pymethods]
impl DetPos {
  #[new]
  fn new(array_pos: &PyArray1<f64>) -> Self {
    Self {
      pos: array_pos.readonly().as_array().into_owned(),
    }
  }

  #[getter(pos)]
  fn get_pos(&self, py: Python) -> Py<PyArray1<f64>> {
    self.pos.clone().into_pyarray(py).to_owned()
  }

  #[setter(pos)]
  fn set_pos(&mut self, array_pos: &PyArray1<f64>) {
    self.pos = array_pos.readonly().as_array().into_owned();
  }
}

#[pymethods]
impl ProbPos {
  #[new]
  fn new(mus: &PyArray2<f64>, covs: &PyArray3<f64>, weights: &PyArray1<f64>) -> Self {
    Self {
      mus: mus.readonly().as_array().into_owned(),
      covs: covs.readonly().as_array().into_owned(),
      weights: weights
        .readonly()
        .readonly()
        .as_array()
        .iter()
        .map(|a| *a as f64)
        .collect(),
      n_components: mus.readonly().as_array().nrows(),
    }
  }

  #[getter(weights)]
  fn get_weights(&self, py: Python) -> Py<PyArray1<f64>> {
    self.weights.clone().into_pyarray(py).to_owned()
  }

  #[getter(mus)]
  fn get_mus(&self, py: Python) -> Py<PyArray2<f64>> {
    self.mus.clone().into_pyarray(py).to_owned()
  }

  #[getter(covs)]
  fn get_covs(&self, py: Python) -> Py<PyArray3<f64>> {
    self.covs.clone().into_pyarray(py).to_owned()
  }

  #[setter(mus)]
  /// Set the meansof the Gaussian mixture.
  fn set_mus(&mut self, mus: &PyArray2<f64>) -> PyResult<()> {
    self.mus = mus.readonly().as_array().into_owned();
    Ok(())
  }

  #[setter(covs)]
  /// Set the covariance matrices of the Gaussian mixture.
  fn set_covs(&mut self, covs: &PyArray3<f64>) -> PyResult<()> {
    self.covs = covs.readonly().as_array().into_owned();
    Ok(())
  }

  #[setter(weights)]
  /// Set the covariance matrices of the Gaussian mixture.
  fn set_weights(&mut self, weights: &PyArray1<f64>) -> PyResult<()> {
    self.weights = weights.readonly().as_array().into_owned();
    Ok(())
  }

  /// add into memory
  /// TODO inefficient
  fn add_mut(&mut self, other: &Self) {
    let c = self.clone() + other.clone();
    self.mus = c.mus;
    self.weights = c.weights;
    self.covs = c.covs;
  }

  fn __iadd__(&mut self, other: &Self) {
    self.add_mut(other);
  }

  /// Return the weighted mean of the mixture.
  fn mean(&self, py: Python) -> Py<PyArray1<f64>> {
    self.total_mean().into_pyarray(py).to_owned()
  }

  /// Return the weighted cov of the mixture.
  fn cov(&self, py: Python) -> Py<PyArray2<f64>> {
    self.total_cov().into_pyarray(py).to_owned()
  }
}

#[pymethods]
impl Module {
  #[new]
  fn new(p_vector: ProbPos, next_ref_frame: &PyArray2<f64>) -> Self {
    Self {
      centroid: Pos::Det(DetPos {
        pos: array![0.0, 0.0, 0.0],
      }),
      labels: vec![0; 0],
      p_vector: Pos::Prob(p_vector),
      next_ref_frame: next_ref_frame.readonly().as_array().into_owned(),
      tracked_points: vec![ProbPos::new_zero(1); 0],
    }
  }

  fn add_prob_point_of_interest(&mut self, point: ProbPos) {
    self.tracked_points.push(point);
  }

  fn add_det_point_of_interest(&mut self, point: DetPos) {
    self.tracked_points.push(point.into());
  }

  fn attachment_transform_mut(&mut self, other_module: &Module) {
    self.next_ref_frame = other_module.next_ref_frame.view().dot(&self.next_ref_frame);
    self.p_vector.rotate_mut(&other_module.next_ref_frame);
    // self.p_vector = self.p_vector.clone() + other_module.p_vector.clone();
    self.centroid = other_module.centroid.clone() + other_module.p_vector.clone();

    for pos in self.tracked_points.iter_mut() {
      let mut pos_c = pos.clone();
      pos_c.rotate_mut(&other_module.next_ref_frame);
      match &self.centroid {
        Pos::Prob(cent) => *pos = cent.clone() + pos_c,
        Pos::Det(cent) => *pos = ProbPos::from(cent.clone()) + pos_c,
      }
    }
  }

  #[getter(p_vector)]
  fn get_p_vector(&self) -> ProbPos {
    match &self.p_vector {
      Pos::Prob(x) => x.clone(),
      Pos::Det(x) => ProbPos::new_zero(1),
    }
  }

  #[getter(next_ref_frame)]
  fn get_next_ref_frame(&self, py: Python) -> Py<PyArray2<f64>> {
    self.next_ref_frame.clone().into_pyarray(py).to_owned()
  }

  #[setter(next_ref_frame)]
  fn set_next_ref_frame(&mut self, next_ref_frame: &PyArray2<f64>) {
    self.next_ref_frame = next_ref_frame.readonly().as_array().into_owned();
  }

  #[setter(p_vector)]
  fn set_p_vector(&mut self, p_vector: ProbPos) {
    self.p_vector = Pos::Prob(p_vector);
  }

  // TODO: getter for DetPos
  #[getter(centroid)]
  fn get_centroid(&self) -> ProbPos {
    match &self.centroid {
      Pos::Prob(x) => x.clone(),
      Pos::Det(x) => ProbPos::new_zero(1),
    }
  }

  // TODO: getter for DetPos
  #[setter(centroid)]
  fn set_centroid(&mut self, centroid: ProbPos) {
    self.centroid = Pos::Prob(centroid);
  }

  #[getter(tracked_points)]
  fn get_tracked_points(&self) -> Vec<ProbPos> {
    self.tracked_points.clone()
  }
}

impl GeneralModule {
  fn new(p_vectors: Vec<ProbPos>, next_ref_frames: Vec<Array2<f64>>) -> Self {
    Self {
      centroid: Pos::Det(DetPos {
        pos: array![0.0, 0.0, 0.0],
      }),
      labels: vec![0; 0],
      p_vectors: p_vectors.iter().map(|x| Pos::Prob(x.clone())).collect(),
      next_ref_frames: next_ref_frames,
      tracked_points: vec![ProbPos::new_zero(1); 0],
      align_p_idx: 0,
    }
  }

    }
  }

  fn from_module(module: Module) -> Self {
    Self {
      centroid: module.centroid.clone(),
      p_vectors: vec![module.p_vector.clone()],
      next_ref_frames: vec![module.next_ref_frame.clone()],
      tracked_points: module.tracked_points.clone(),
      align_p_idx: 0,
      labels: module.labels.clone(),
    }
  }

  fn rotate_mut(&mut self, rot_matrix: Array2<f64>) {
    self.ref_frame = rot_matrix.dot(&self.ref_frame);
    self.centroid.rotate_mut(&rot_matrix); 
    for pos in self.p_vectors.iter_mut() {
      pos.rotate_mut(&rot_matrix);
    }
    for pos in self.tracked_points.iter_mut() {
      pos.rotate_mut(&rot_matrix);
    }
  }

  /// Translate this module by the given translation vector.
  fn translate_mut(&mut self, trans: Array1<f64>) {
    self.centroid.translate_mut(&trans);
    for pos in self.p_vectors.iter_mut() {
      pos.translate_mut(&trans);
    }
    for pos in self.tracked_points.iter_mut() {
      pos.translate_mut(&trans);
    }
  }


  fn align_mut(&mut self, align_idx: usize) {

  }

  /// Transform this module such that it is attached to the other.
  fn attachment_transform_mut(
    &mut self,
    c_att_pnt: usize,
    o_att_pnt: usize,
    other_module: &GeneralModule,
  ) {
    self.realign_module(c_att_pnt);

    self.next_ref_frames[c_att_pnt] = other_module.next_ref_frames[o_att_pnt]
      .view()
      .dot(&self.next_ref_frames[c_att_pnt]);
    self.p_vectors[c_att_pnt].rotate_mut(&other_module.next_ref_frames[o_att_pnt]);
    self.centroid = other_module.centroid.clone() + other_module.p_vectors[o_att_pnt].clone();

    for pos in self.tracked_points.iter_mut() {
      let mut pos_c = pos.clone();
      pos_c.rotate_mut(&other_module.next_ref_frames[o_att_pnt]);
      match &self.centroid {
        Pos::Prob(cent) => *pos = cent.clone() + pos_c,
        Pos::Det(cent) => *pos = ProbPos::from(cent.clone()) + pos_c,
      }
    }
  }
}

#[pymethods]
impl GeneralModule {
  #[new]
  /// GeneralModule(p_vectors, next_ref_frames)
  /// Creates a GeneralModule
  /// Parameters
  /// ----------
  /// p_vectors: list of ProbPos
  /// next_ref_frames: list of 2d np.arrays corresponding to the p_vectors. Each column vector in
  ///                  the array corresponds with x,y,z. The corresponding p_vector is aligned with
  ///                  the first column vector.
  fn new_py(p_vectors: Vec<ProbPos>, next_ref_frames: Vec<&PyArray2<f64>>) -> Self {
    Self {
      centroid: Pos::Det(DetPos {
        pos: array![0.0, 0.0, 0.0],
      }),
      labels: vec![0; 0],
      p_vectors: p_vectors.iter().map(|x| Pos::Prob(x.clone())).collect(),
      next_ref_frames: next_ref_frames
        .iter()
        .map(|x| x.readonly().as_array().into_owned())
        .collect(),
      tracked_points: vec![ProbPos::new_zero(1); 0],
      align_p_idx: 0,
    }
  }

  fn get_p_vector(&self, idx: usize) -> ProbPos {
    match &self.p_vectors[idx] {
      Pos::Prob(x) => x.clone(),
      Pos::Det(x) => ProbPos::new_zero(1),
    }
  }

  fn get_next_ref_frame(&self, py: Python, idx: usize) -> Py<PyArray2<f64>> {
    self.next_ref_frames[idx]
      .clone()
      .into_pyarray(py)
      .to_owned()
  }

  fn set_next_ref_frame(&mut self, next_ref_frame: &PyArray2<f64>, idx: usize) {
    self.next_ref_frames[idx] = next_ref_frame.readonly().as_array().into_owned();
  }

  fn set_p_vector(&mut self, p_vector: ProbPos, idx: usize) {
    self.p_vectors[idx] = Pos::Prob(p_vector);
  }

  // TODO: getter for DetPos
  #[getter(centroid)]
  fn get_centroid(&self) -> ProbPos {
    match &self.centroid {
      Pos::Prob(x) => x.clone(),
      Pos::Det(x) => ProbPos::new_zero(1),
    }
  }

  // TODO: getter for DetPos
  #[setter(centroid)]
  fn set_centroid(&mut self, centroid: ProbPos) {
    self.centroid = Pos::Prob(centroid);
  }

  #[getter(tracked_points)]
  fn get_tracked_points(&self) -> Vec<ProbPos> {
    self.tracked_points.clone()
  }

  #[setter(tracked_points)]
  fn set_tracked_points(&mut self, tracked_points: Vec<ProbPos>) {
    self.tracked_points = tracked_points;
  }
}

impl Construct {
  fn edges_to_adjacency_matrix(
    num_nodes: usize,
    edges: &Vec<((usize, usize), (usize, usize))>,
  ) -> Array2<i64> {
    let mut adjacency_matrix = Array2::zeros((num_nodes, num_nodes));
    // Use edge list to fill adjacency matrix
    for edge in edges.iter() {
      adjacency_matrix[[edge.0 .0, edge.1 .0]] = 1;
      adjacency_matrix[[edge.1 .0, edge.0 .0]] = 1;
    }

    adjacency_matrix
  }

  fn build_tree_inner(
    stack: &mut Vec<usize>,
    node_ids: &mut Vec<NodeId>,
    adj: &mut Array2<i64>,
    tree: &mut Arena<usize>,
  ) {
    if let Some(c_node) = stack.pop() {
      if adj.slice(s![c_node, ..]).sum() == 0 {
        return;
      }

      for i in 0..node_ids.len() {
        if adj[[c_node, i]] == 1 {
          node_ids[c_node].append(node_ids[i], tree);
          adj[[i, c_node]] = 0;
          stack.push(i);
        } else {
        }
      }

      Construct::build_tree_inner(stack, node_ids, adj, tree);
    } else {
      return;
    }
  }

  // TODO: It may be possible to remove the edges without re-initiallizing the tree
  fn clear_tree(&mut self) {
    self.tree = Arena::new();
    self.node_ids = Vec::<NodeId>::new();
    for i in 0..self.raw_modules.len() {
      let node = self.tree.new_node(i);
      self.node_ids.push(node);
    }
  }

  /// Searches the edge list to get the module attachment p idxs
  fn get_attachment_idxs_from_edge_list(
    &self,
    parent_id: usize,
    current_id: usize,
  ) -> (usize, usize) {
    for edge in self.edges.iter() {
      if edge.0 .0 == parent_id && edge.1 .0 == current_id {
        return (edge.0 .1, edge.1 .1);
      } else if edge.1 .0 == parent_id && edge.0 .0 == current_id {
        return (edge.1 .1, edge.0 .1);
      }
    }
    panic!("parent and current do not have an edge!");
    (0, 0)
  }
}

#[pymethods]
impl Construct {
  #[new]
  fn new(raw_modules: Vec<GeneralModule>, edges: Vec<((usize, usize), (usize, usize))>) -> Self {
    let adjacency_matrix = Construct::edges_to_adjacency_matrix(raw_modules.len(), &edges);
    let mut tree = Arena::new();
    let mut node_ids = Vec::<NodeId>::new();
    for i in 0..raw_modules.len() {
      let node = tree.new_node(i);
      node_ids.push(node);
    }

    Self {
      raw_modules,
      edges,
      tree,
      node_ids,
      assembled_modules: Vec::<GeneralModule>::new(),
      adjacency_matrix,
    }
  }

  /// Builds a tree for the construct with starting_node as the root.
  fn build_tree(&mut self, starting_node: usize) {
    self.clear_tree();
    let mut stack = vec![starting_node];
    Construct::build_tree_inner(
      &mut stack,
      &mut self.node_ids,
      &mut self.adjacency_matrix.clone(),
      &mut self.tree,
    );
  }

  /// Add a module to the construct.
  fn add_module(&mut self, module: GeneralModule) {
    self.raw_modules.push(module);
  }

  /// Add an edge to the construct.
  fn add_edges(&mut self, edge: ((usize, usize), (usize, usize))) {
    self.edges.push(edge);
  }

  // TODO: Clean this up
  /// Propagate the dynamics from the starting node
  fn propagate(&mut self, starting_node: usize) {
    self.build_tree(starting_node);
    self.assembled_modules = self.raw_modules.clone();
    let mut tree_iter = self.node_ids[starting_node].descendants(&self.tree);
    for c_node in tree_iter {
      let c_node_id = *self.tree[c_node].get();
      let mut tree_jiter = self.node_ids[c_node_id].descendants(&self.tree);
      tree_jiter.next();

      let mut through_node_id = c_node_id.clone();
      for n_node in tree_jiter {
        let n_node_id = *self.tree[n_node].get();

        if let Some(other) = self.tree[n_node].parent() {
          if other == c_node {
            through_node_id = n_node_id.clone();
          }
        }

        let (c_node_p_id, n_node_p_id) =
          self.get_attachment_idxs_from_edge_list(c_node_id, through_node_id);

        let c_pos = self.assembled_modules[c_node_id].p_vectors[c_node_p_id].clone();
        let c_pos_total_cov = match c_pos {
          Pos::Prob(pos) => pos.total_cov(),
          Pos::Det(pos) => Array2::zeros((3, 3)),
        };
        let mut raw_pos = ProbPos::new_zero(1);
        raw_pos
          .covs
          .slice_mut(s![0, .., ..])
          .assign(&c_pos_total_cov);

        for p_vec in self.assembled_modules[n_node_id].p_vectors.iter_mut() {
          let n_pos = p_vec.clone();
          *p_vec = n_pos + Pos::Prob(raw_pos.clone());
        }
      }
    }

    let mut tree_iter = self.node_ids[starting_node].descendants(&self.tree);
    // skip the root
    tree_iter.next();
    for node_id in tree_iter {
      let (parent_id, current_id) = match self.tree[node_id].parent() {
        Some(parent_id) => (*self.tree[parent_id].get(), *self.tree[node_id].get()),
        None => {
          panic!("Current node has no parent!");
          (0, *self.tree[node_id].get())
        }
      };

      let (parent_p_id, current_p_id) =
        self.get_attachment_idxs_from_edge_list(parent_id, current_id);

      let mut current_module = self.assembled_modules[current_id].clone();
      current_module.attachment_transform_mut(
        current_p_id,
        parent_p_id,
        &self.assembled_modules[parent_id],
      );

      self.assembled_modules[current_id] = current_module;
    }
  }

  fn get_assembled_modules(&self) -> Vec<GeneralModule> {
    self.assembled_modules.clone()
  }
}

/// A Python module implemented in Rust.
#[pymodule]
fn dynamo(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<ProbPos>()?;
  m.add_class::<Construct>()?;
  m.add_class::<DetPos>()?;
  m.add_class::<Module>()?;
  m.add_class::<GeneralModule>()?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::abs_diff_eq;
  use ndarray::linalg::Dot;
  use ndarray::prelude::*;
  #[test]
  fn add_two_det_pos() {
    let a = DetPos {
      pos: array![0.4, 20.5, 0.5],
    };

    let b = DetPos {
      pos: array![234.8, 0.2, 0.8],
    };

    let c = a + b;

    assert_eq!(true, abs_diff_eq!(c.pos[[0]], 235.2, epsilon = 0.0000001));
    assert_eq!(true, abs_diff_eq!(c.pos[[1]], 20.7, epsilon = 0.0000001));
    assert_eq!(true, abs_diff_eq!(c.pos[[2]], 1.3, epsilon = 0.0000001));
  }

  #[test]
  fn add_det_pos_to_prob_pos() {
    let a = DetPos {
      pos: array![0.4, 20.5, 0.5],
    };

    let mut b = ProbPos::new_zero(2);
    b.mus[[1, 0]] = 3.8;
    b.mus[[1, 1]] = 7.8;

    let c = b.det_add(a);

    assert_eq!(true, abs_diff_eq!(c.mus[[0, 0]], 0.4, epsilon = 0.0000001));
    assert_eq!(
      true,
      abs_diff_eq!(c.mus[[1, 0]], 3.8 + 0.4, epsilon = 0.0000001)
    );
    assert_eq!(
      true,
      abs_diff_eq!(c.mus[[1, 1]], 7.8 + 20.5, epsilon = 0.0000001)
    );
  }

  #[test]
  fn prob_pos_weigted_mean() {
    let mut a = ProbPos::new_zero(3);
    a.mus[[0, 0]] = 3.8;
    a.mus[[2, 1]] = 7.8;
    a.mus[[1, 0]] = 1.8;
    a.mus[[1, 2]] = 9.8;

    // mus = [
    //   [3.8, 0.0, 0.0],
    //   [1.8, 0.0, 9.8],
    //   [0.0, 7.8, 0.0],
    // ]
    let weighted_mu = array![(3.8 + 1.8) / 3.0, (7.8) / 3.0, (9.8) / 3.0,];

    assert_eq!(
      true,
      abs_diff_eq!(
        (weighted_mu - a.total_mean()).sum(),
        0.0,
        epsilon = 0.0000001
      )
    );
  }

  #[test]
  fn prob_pos_weigted_cov() {
    let mut a = ProbPos::new_zero(2);

    let mus = array![
      [19.11174950, -3.80991797, -2.11702118],
      [19.02044437, -4.25629735, -2.17426382],
    ];

    let covs = array![
      [
        [0.00740698, 0.01750531, 0.00219118],
        [0.01750531, 0.06673698, 0.00875887],
        [0.00219118, 0.00875887, 0.01309119],
      ],
      [
        [0.00692701, 0.01411599, 0.00555987],
        [0.01411599, 0.04624028, 0.01463649],
        [0.00555987, 0.01463649, 0.01702779],
      ]
    ];

    let weights = array![0.56388056, 0.43611944];

    a.mus = mus.clone();
    a.covs = covs.clone();
    a.weights = weights.clone();

    let total_cov = array![
      [0.00924779, 0.02605003, 0.00494564],
      [0.02605003, 0.10679851, 0.01760593],
      [0.00494564, 0.01760593, 0.01561383],
    ];

    assert_eq!(
      true,
      abs_diff_eq!((total_cov - a.total_cov()).sum(), 0.0, epsilon = 0.0000001)
    );
  }

  #[test]
  fn add_prob_pos_to_prob_pos_mus() {
    let mut a = ProbPos::new_zero(3);
    a.mus[[0, 0]] = 3.8;
    a.mus[[2, 1]] = 7.8;

    let mut b = ProbPos::new_zero(2);
    b.mus[[1, 0]] = 234.123;
    b.mus[[1, 1]] = 7.8;

    let c = a.clone() + b.clone();

    assert_eq!(c.mus.len_of(Axis(0)), 6);
    assert_eq!(
      true,
      abs_diff_eq!(
        c.mus[[0, 0]],
        a.mus[[0, 0]] + b.mus[[0, 0]],
        epsilon = 0.0000001
      )
    );
    assert_eq!(
      true,
      abs_diff_eq!(
        c.mus[[1, 0]],
        a.mus[[0, 0]] + b.mus[[1, 0]],
        epsilon = 0.0000001
      )
    );
    assert_eq!(
      true,
      abs_diff_eq!(
        c.mus[[4, 1]],
        a.mus[[2, 1]] + b.mus[[0, 1]],
        epsilon = 0.0000001
      )
    );
    assert_eq!(
      true,
      abs_diff_eq!(
        c.mus[[5, 1]],
        a.mus[[2, 1]] + b.mus[[1, 1]],
        epsilon = 0.0000001
      )
    );
  }

  #[test]
  fn rotate_prob_pos() {
    let cov = array![[1.0, 2.0, 3.0], [2.0, 2.4, 0.8], [3.0, 0.8, 1.3],];

    let mu = array![3.8, 2.7, 9.2];

    let mut a = ProbPos::new_zero(1);
    a.mus[[0, 0]] = 3.8;
    a.mus[[0, 1]] = 2.7;
    a.mus[[0, 2]] = 9.2;

    for i in 0..3 {
      a.mus[[0, i]] = mu[[i]];
      for j in 0..3 {
        a.covs[[0, i, j]] = cov[[i, j]];
      }
    }

    let rot_mat = array![[0.5, 0., 0.866], [0.866, 0., -0.5], [-0., 1., 0.]];

    let mu_r = (&rot_mat.dot(&mu));
    let cov_r = (&cov.dot(&rot_mat.t())).to_owned();
    let cov_r = (&rot_mat.dot(&cov_r));

    a.rotate_mut(&rot_mat);

    assert_eq!(a.mus.slice(s![0, ..]), mu_r);
    assert_eq!(a.covs.slice(s![0, .., ..]), cov_r);
  }

  #[test]
  fn translate_prob_pos() {
    let cov = array![[1.0, 2.0, 3.0], [2.0, 2.4, 0.8], [3.0, 0.8, 1.3],];

    let mu = array![3.8, 2.7, 9.2];

    let mut a = ProbPos::new_zero(1);
    a.mus[[0, 0]] = 3.8;
    a.mus[[0, 1]] = 2.7;
    a.mus[[0, 2]] = 9.2;

    for i in 0..3 {
      a.mus[[0, i]] = mu[[i]];
      for j in 0..3 {
        a.covs[[0, i, j]] = cov[[i, j]];
      }
    }

    let rot_mat = array![[0.5, 0., 0.866], [0.866, 0., -0.5], [-0., 1., 0.]];

    let trans = array![1.0, 2.0, 3.0];
    let mu_r = mu + trans.clone();

    a.translate_mut(&trans);
    assert_eq!(a.mus.slice(s![0, ..]), mu_r);
  }

  #[test]
  fn build_adjacency_matrix_from_edges() {
    let edges = vec![
      ((0, 0), (1, 0)),
      ((1, 1), (2, 0)),
      ((1, 2), (3, 0)),
      ((3, 1), (4, 0)),
    ];

    let adj: Array2<i64> = array![
      [0, 1, 0, 0, 0],
      [1, 0, 1, 1, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 1, 0],
    ];

    assert_eq!(adj, Construct::edges_to_adjacency_matrix(5, &edges));
  }

  #[test]
  fn build_tree() {
    let edges = vec![
      ((0, 0), (1, 0)),
      ((1, 1), (2, 0)),
      ((1, 2), (3, 0)),
      ((3, 1), (4, 0)),
    ];

    let adj: Array2<i64> = array![
      [0, 1, 0, 0, 0],
      [1, 0, 1, 1, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 1, 0],
    ];

    let mut tree = Arena::<usize>::new();
    let mut nodes: Vec<NodeId> = vec![];
    for i in 0..5 {
      let node = tree.new_node(i);
      nodes.push(node);
    }

    nodes[1].append(nodes[0], &mut tree);
    nodes[1].append(nodes[2], &mut tree);
    nodes[3].append(nodes[1], &mut tree);
    nodes[4].append(nodes[3], &mut tree);

    let prob_poses = vec![ProbPos::new_zero(3)];
    let next_ref_frames = vec![Array::eye(3)];
    let module = GeneralModule::new(prob_poses.clone(), next_ref_frames.clone());
    let modules = vec![module.clone(); 5];

    let mut construct = Construct::new(modules, edges.clone());
    construct.build_tree(1);
    construct.build_tree(4);
    assert_eq!(tree, construct.tree.clone());
  }
  #[test]
  fn build_tree_string() {
    let edges = vec![
      ((0, 1), (1, 0)),
      ((1, 1), (2, 0)),
      ((2, 1), (3, 0)),
      ((3, 1), (4, 0)),
      ((4, 1), (5, 0)),
      ((5, 1), (6, 0)),
      ((6, 1), (7, 0)),
    ];

    let adj: Array2<i64> = array![
      [0, 1, 0, 0, 0, 0, 0, 0],
      [1, 0, 1, 0, 0, 0, 0, 0],
      [0, 1, 0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0, 1, 0, 1],
      [0, 0, 0, 0, 0, 0, 1, 0],
    ];

    let mut tree = Arena::<usize>::new();
    let mut flipped_tree = Arena::<usize>::new();
    let mut branch_tree = Arena::<usize>::new();
    let mut nodes: Vec<NodeId> = vec![];
    let mut flipped_nodes: Vec<NodeId> = vec![];
    let mut branch_nodes: Vec<NodeId> = vec![];
    for i in 0..8 {
      let node = tree.new_node(i);
      let f_node = flipped_tree.new_node(i);
      let b_node = branch_tree.new_node(i);
      nodes.push(node);
      flipped_nodes.push(f_node);
      branch_nodes.push(f_node);
    }

    for i in 0..7 {
      nodes[i].append(nodes[i + 1], &mut tree);
      flipped_nodes[7 - i].append(flipped_nodes[6 - i], &mut flipped_tree);
    }

    branch_nodes[4].append(branch_nodes[3], &mut branch_tree);
    branch_nodes[3].append(branch_nodes[2], &mut branch_tree);
    branch_nodes[2].append(branch_nodes[1], &mut branch_tree);
    branch_nodes[1].append(branch_nodes[0], &mut branch_tree);

    branch_nodes[4].append(branch_nodes[5], &mut branch_tree);
    branch_nodes[5].append(branch_nodes[6], &mut branch_tree);
    branch_nodes[6].append(branch_nodes[7], &mut branch_tree);

    let prob_poses = vec![ProbPos::new_zero(3)];
    let next_ref_frames = vec![Array::eye(3)];
    let module = GeneralModule::new(prob_poses.clone(), next_ref_frames.clone());
    let modules = vec![module.clone(); 8];

    let mut construct = Construct::new(modules, edges.clone());
    construct.build_tree(0);
    println!("zero {}", NodeId::debug_pretty_print(&nodes[0], &tree));
    println!(
      "zero c {}",
      NodeId::debug_pretty_print(&construct.node_ids[0], &construct.tree)
    );
    assert_eq!(tree, construct.tree.clone());

    construct.build_tree(7);
    println!(
      "flip {}",
      NodeId::debug_pretty_print(&flipped_nodes[7], &flipped_tree)
    );
    println!(
      "flip c {}",
      NodeId::debug_pretty_print(&construct.node_ids[7], &construct.tree)
    );
    assert_eq!(flipped_tree, construct.tree.clone());

    construct.build_tree(4);
    println!(
      "branch {}",
      NodeId::debug_pretty_print(&branch_nodes[4], &branch_tree)
    );
    println!(
      "branch c {}",
      NodeId::debug_pretty_print(&construct.node_ids[4], &construct.tree)
    );

    construct.build_tree(4);
    assert_eq!(branch_tree, construct.tree.clone());
  }
}
