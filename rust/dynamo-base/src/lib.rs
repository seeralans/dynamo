use ndarray::linalg::Dot;
use ndarray::prelude::*;
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
  weights: Vec<f64>,
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
  tracked_points: Vec<Pos>,
}

pub trait Vector {
  fn get_mean_pos(&self) -> Array1<f64>;
  fn rotate_mut(&mut self, rot_mat: &Array2<f64>);
}

impl Vector for DetPos {
  fn get_mean_pos(&self) -> Array1<f64> {
    self.pos.clone()
  }

  fn rotate_mut(&mut self, rot_mat: &Array2<f64>) {
    self.pos = rot_mat.dot(&self.pos);
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
}

#[derive(Debug, Clone, PartialEq)]
enum Pos {
  Det(DetPos),
  Prob(ProbPos),
}

#[pyclass]
struct Module {
  /// Position vec of module centroid
  centroid: Pos,
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
}

  /// position of next module
  p_vector: Pos,
#[pymethods]
impl Module {
  #[new]
  fn new(p_vector: ProbPos, next_ref_frame: &PyArray2<f64>) -> Self {
    Self {
      centroid: Pos::Det(DetPos{pos: array![0.0, 0.0, 0.0] }),
      labels: vec![0; 0],
      p_vector: Pos::Prob(p_vector),
      next_ref_frame: next_ref_frame.readonly().as_array().into_owned(),
      tracked_points: vec![Pos::Det(DetPos { pos: array![0.0, 0.0, 0.0] })],
    }
  }

  fn attachement_transform_mut(&mut self, other_module: &Module) {
    self.next_ref_frame = other_module.next_ref_frame.view().dot(&self.next_ref_frame);
    self.p_vector.rotate_mut(&other_module.next_ref_frame);
    self.centroid = other_module.centroid.clone() + other_module.p_vector.clone();

    for pos in self.tracked_points.iter_mut() {
      *pos = self.centroid.clone() + pos.clone();
    }
  }

  #[getter(p_vector)] 
  fn get_p_vector(&self) -> ProbPos {
    match &self.p_vector {
      Pos::Prob(x) => x.clone(),
      Pos::Det(x) => ProbPos::new_zero(1),
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

/// Allows one to use the '+'  operator on two ProbPos
impl Add for ProbPos {
  type Output = Self;
  fn add(self, other: Self) -> Self {
    let self_n_comps = self.mus.len_of(Axis(0));
    let other_n_comps = other.mus.len_of(Axis(0));

    let mut mus = Array::zeros((self_n_comps * other_n_comps, self.mus.ncols()));
    let mut covs = Array::zeros((mus.nrows(), self.mus.ncols(), self.mus.ncols()));
    let mut weights = vec![0.0; mus.nrows()];

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

    Self { mus, covs, weights }
  }
}

impl ProbPos {
  fn zero(n_components: usize) -> Self {
    Self {
      mus: Array::zeros((n_components, 3)),
      covs: Array::zeros((n_components, 3, 3)),
      weights: vec![1.0 / n_components as f64; n_components],
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
    Self { mus, covs, weights }
  }

  /// add two probabilitstic vectors together
  fn prob_add(&self, other: Self) -> Self {
    self.clone() + other
  }

#[pymethods]
impl ProbPos {
  #[new]
  fn new( mus: &PyArray2<f64>, covs: &PyArray3<f64>, weights: &PyArray1<f64>) -> Self {
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
  fn set_mus(&mut self, mus: &PyArray2<f64>)  -> PyResult<()> {
    self.mus =  mus.readonly().as_array().into_owned();
    Ok(())
  }

  #[setter(covs)]
  fn set_covs(&mut self, covs: &PyArray3<f64>) -> PyResult<()> {
    self.covs =  covs.readonly().as_array().into_owned();
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

#[pyfunction]
/// Formats the sum of two numbers as string.
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
  Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn protein_dynamics(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::abs_diff_eq;

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

    let mut b = ProbPos::zero(2);
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
  fn add_prob_pos_to_prob_pos_mus() {
    let mut a = ProbPos::zero(3);
    a.mus[[0, 0]] = 3.8;
    a.mus[[2, 1]] = 7.8;

    let mut b = ProbPos::zero(2);
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
}
