import pyvista as pv
import numpy as np

def create_plotter(**kwargs):
    """
    Create and return a new PyVista Plotter.
    Parameters:
        **kwargs: keyword arguments passed to pv.Plotter (e.g. window_size, off_screen)
    Returns:
        plotter (pv.Plotter)
    """
    return pv.Plotter(**kwargs)


def show(plotter):
    """
    Display the plotter window (blocking).
    Parameters:
        plotter (pv.Plotter): the plotter to display
    """
    plotter.show()


def save_screenshot(plotter, filename, **kwargs):
    """
    Save the current plotter view to an image file.
    Parameters:
        plotter (pv.Plotter): the plotter to capture
        filename (str): output file path (e.g. 'output.png')
        **kwargs: keyword arguments passed to plotter.screenshot
    """
    plotter.screenshot(filename, **kwargs)


def _ensure_plotter(plotter):
    """Return the given plotter, or create a new one if None."""
    if plotter is None:
        plotter = create_plotter()
    return plotter


def sorted_eig(mat):
    """
    Returns eigenvalues and eigenvectors sorted by eigenvalues in descending order.
    """
    vals, vecs = np.linalg.eig(mat)
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs


def smooth_points(points, num_interps=50):
    """
    Returns interpolated values between the given points.
    Parameters:
        points (np.array): points to be interpolated, shape (N, 3)
        num_interps (int): number of points to be interpolated between each point
    Returns:
        smooth_points (np.array): interpolated points, shape (num_interps, 3)
    """
    from scipy.interpolate import splprep, splev

    tck, u = splprep(points.T, s=0)
    smooth = splev(np.linspace(0, 1, num_interps), tck)
    return np.array(smooth).T


def reorder_band_points(upper, lower):
    """
    Reorder band points to cope with twists.
    Parameters:
        upper (np.array): upper points of the band
        lower (np.array): lower points of the band
    Returns:
        upper_a (np.array): reordered upper points
        lower_a (np.array): reordered lower points
    """
    upper_a = [upper[0]]
    lower_a = [lower[0]]
    for i in range(1, len(upper)):
        current_u = upper_a[i - 1]
        current_l = lower_a[i - 1]

        duu = np.linalg.norm(current_u - upper[i])
        dul = np.linalg.norm(current_u - lower[i])
        dlu = np.linalg.norm(current_l - upper[i])
        dll = np.linalg.norm(current_l - lower[i])

        if duu + dll < dul + dlu:
            upper_a.append(upper[i])
            lower_a.append(lower[i])
        else:
            upper_a.append(lower[i])
            lower_a.append(upper[i])

    return np.array(upper_a), np.array(lower_a)


def _spline_tube(points, plotter, radius=0.1, color=(1.0, 1.0, 1.0), opacity=1.0, **kwargs):
    """
    Draw a tube along a polyline of points.
    Parameters:
        points (np.array): (N, 3) array of points
        plotter (pv.Plotter): the plotter to add to
        radius (float): tube radius
        color (tuple): RGB colour
        opacity (float): opacity
    """
    spline = pv.Spline(points, n_points=max(len(points), 2))
    tube = spline.tube(radius=radius)
    plotter.add_mesh(tube, color=color, opacity=opacity, **kwargs)


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_fins_between_two_curves(upper, lower, scalars, plotter=None, colormap='viridis', **kwargs):
    """
    Draws fins (triangulated surface) between the upper and lower curves.
    Parameters:
        upper (np.array): upper points of the band, shape (N, 3)
        lower (np.array): lower points of the band, shape (N, 3)
        scalars (np.array): scalars for colouring the surface, shape (N,)
        plotter (pv.Plotter or None): plotter to add to (created if None)
        colormap (str): colourmap name
        **kwargs: keyword arguments passed to plotter.add_mesh
    Returns:
        plotter (pv.Plotter)
    """
    plotter = _ensure_plotter(plotter)
    num_centroids = upper.shape[0]

    # draw the two boundary curves as tubes
    _spline_tube(upper, plotter, radius=0.05, **kwargs)
    _spline_tube(lower, plotter, radius=0.05, **kwargs)

    # build triangulated surface between upper and lower
    road = np.vstack((upper, lower))
    tris = []
    for i in range(num_centroids - 1):
        tris.append([3, i, i + 1, i + num_centroids])
        tris.append([3, i + num_centroids + 1, i + 1, i + num_centroids])
    faces = np.array([v for tri in tris for v in tri])

    mesh = pv.PolyData(road, faces=faces)
    mesh.point_data['scalars'] = np.hstack((scalars, scalars))

    # filter out 'color' from kwargs since we're using scalar colouring
    mesh_kwargs = {k: v for k, v in kwargs.items() if k not in ('color',)}
    plotter.add_mesh(mesh, scalars='scalars', cmap=colormap, show_scalar_bar=False,
                     **mesh_kwargs)

    return plotter


def arrow_from_a_to_b(a, b, plotter=None, color=(1.0, 1.0, 1.0), **kwargs):
    """
    Draws an arrow from point a to point b.
    Parameters:
        a (np.array): start point
        b (np.array): end point
        plotter (pv.Plotter or None): plotter to add to (created if None)
        color (tuple): RGB colour
        **kwargs: keyword arguments passed to plotter.add_mesh
    Returns:
        plotter (pv.Plotter)
    """
    plotter = _ensure_plotter(plotter)
    direction = np.array(b) - np.array(a)
    arrow = pv.Arrow(start=a, direction=direction, scale='auto')
    plotter.add_mesh(arrow, color=color, **kwargs)
    return plotter


def draw_helices(hels, plotter=None, hel_radius=1, hel_opacity=1.0,
                 hel_colour=(1.0, 1.0, 1.0), **kwargs):
    """
    Draws helices as tubes (cylinders).
    Parameters:
        hels (np.array): helices to be drawn, shape (N_helices, 2, 3) — each
                         helix is defined by two endpoints
        plotter (pv.Plotter or None): plotter to add to (created if None)
        hel_radius (float): radius of the helix tubes
        hel_opacity (float): opacity of the helices
        hel_colour (tuple): RGB colour
        **kwargs: keyword arguments passed to plotter.add_mesh
    Returns:
        plotter (pv.Plotter)
    """
    plotter = _ensure_plotter(plotter)
    for h in hels:
        _spline_tube(h, plotter, radius=hel_radius, color=hel_colour,
                     opacity=hel_opacity, **kwargs)
    return plotter


def draw_fins_from_means_and_variances(
    means,
    variances,
    plotter=None,
    interp_points=None,
    fin_dims=[0, 1],
    arrow_colours=[(1.0, 0.2, 0.2),
                   (0.2, 1.0, 0.2),
                   (0.2, 0.2, 1.0)],
    centroid_colour=(1.0, 1.0, 1.0),
    centroid_tube_colour=(1.0, 1.0, 1.0),
    centroid_tube_radius=0.1,
    fin_colourmap='viridis',
    **kwargs,
):
    """
    Uses fins to visualise the dynamics of centroids.
    Parameters:
        means (np.array): means of centroids, shape (N, 3)
        variances (np.array): covariance matrices, shape (N, 3, 3)
        plotter (pv.Plotter or None): plotter to add to (created if None)
        interp_points (int or None): number of interpolation points (None = no interpolation)
        fin_dims (list): indices of eigenvector dimensions to draw fins for
        arrow_colours (list): RGB colours for the three eigenvector directions
        centroid_colour (tuple): RGB colour for centroid points
        centroid_tube_colour (tuple): RGB colour for the tube connecting centroids
        centroid_tube_radius (float): radius of the centroid tube
        fin_colourmap (str): colourmap name for the fins
        **kwargs: keyword arguments passed to draw_fins_between_two_curves
    Returns:
        plotter (pv.Plotter)
    """
    plotter = _ensure_plotter(plotter)

    if interp_points is None:
        smooth_it = lambda points: points
    else:
        smooth_it = lambda points: smooth_points(points, interp_points)

    # plot centroids as spheres
    cloud = pv.PolyData(means)
    plotter.add_mesh(cloud, color=centroid_colour, point_size=10,
                     render_points_as_spheres=True)

    # plot centroid tube
    _spline_tube(smooth_it(means), plotter, radius=centroid_tube_radius,
                 color=centroid_tube_colour)

    # compute eigendecompositions
    eigs = [sorted_eig(var) for var in variances]
    eigvals = np.array([vals for vals, _ in eigs])

    num_centroids = len(means)
    vecs = np.zeros((num_centroids, 3, 3, 3))
    for i, (vals, evecs) in enumerate(eigs):
        for xyz in range(3):
            p0 = means[i] - evecs[:, xyz] * vals[xyz] ** 0.5 * 2
            p1 = means[i]
            p2 = means[i] + evecs[:, xyz] * vals[xyz] ** 0.5 * 2
            vec = np.stack((p0, p1, p2))
            vecs[i][xyz] = vec
            _spline_tube(vec, plotter, radius=0.1, color=arrow_colours[xyz])

    # draw fins for the requested dimensions
    for dim in fin_dims:
        upper, lower = reorder_band_points(vecs[:, dim, 0], vecs[:, dim, -1])
        draw_fins_between_two_curves(
            smooth_it(upper),
            smooth_it(lower),
            smooth_it(eigvals[:, dim][:, None])[:, 0],
            plotter=plotter,
            colormap=fin_colourmap,
            **kwargs,
        )

    return plotter


def visualise_centroids_of_modules(modules, plotter=None, **kwargs):
    """
    Visualises the centroids of modules using the fins representation.
    Parameters:
        modules (list): list of modules to be visualised (each must have .centroid
                        with .mean() and .cov() methods)
        plotter (pv.Plotter or None): plotter to add to (created if None)
        **kwargs: keyword arguments passed to draw_fins_from_means_and_variances
    Returns:
        plotter (pv.Plotter)
    """
    plotter = _ensure_plotter(plotter)
    centroids = [mod.centroid for mod in modules]
    centroid_mus = np.array([centroid.mean() for centroid in centroids])
    centroid_vas = np.array([centroid.cov() for centroid in centroids])
    return draw_fins_from_means_and_variances(centroid_mus, centroid_vas, plotter, **kwargs)
