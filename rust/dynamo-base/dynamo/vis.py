from mayavi import mlab
import numpy as np


def sorted_eig(mat):
    """
    Returns eigenvalues and eigenvectors sorted by eigenvalues in ascending order
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
      points (np.array): points to be interpolated
      num_interps (int): number of points to be interpolated between each point
    returns:
      smooth_points (np.array): interpolated points
    """
    import scipy
    from scipy.interpolate import splprep

    tck, u = splprep(points.T, s=0)
    smooth_points = scipy.interpolate.splev(np.linspace(0, 1, num_interps), tck)
    return np.array(smooth_points).T


def reorder_band_points(upper, lower):
    """
    Reorder band points to cope with twists.
    Parameters:
      upper (np.array): upper points of the band
      lower (np.array): lower points of the band
    Returns;
      upper_a (np.array): reordered upper points
      upper_a (np.array): reordered lower points
    """
    upper_a, lower_a = upper.copy(), lower.copy()
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
    upper_a, lower_a = np.array(upper_a), np.array(lower_a)
    return upper_a, lower_a


def draw_fins_between_two_curves(upper, lower, scalars, fig, **kwargs):
    """
    Draws fins between the upper and lower curves
    Parameters:
      upper (np.array): upper points of the band
      lower (np.array): lower points of the band
      scalars (np.array): scalars to be used for colouring the fins
      fig (mlab.figure): figure to plot on
      **kwargs: keyword arguments to be passed to mlab.mesh
    """
    from scipy.spatial import Delaunay

    num_centroids = upper.shape[0]
    road = np.vstack((upper, lower))
    mlab.plot3d(*upper.T, figure=fig, **kwargs)
    mlab.plot3d(*lower.T, figure=fig, **kwargs)
    tri = Delaunay(road)

    tris_u = [[i, i + num_centroids, 0] for i in range(num_centroids - 1)]

    for i in range(len(tris_u)):
        a, b, c = tuple(tris_u[i])
        p1 = a + 1
        p2 = b + 1
        if np.linalg.norm(road[a] - road[p1]) < np.linalg.norm(road[a] - road[p2]):
            tris_u[i][-1] = p1
        else:
            tris_u[i][-1] = p2

    tris = tris_u

    tris = [(i, i + 1, i + num_centroids) for i in range(num_centroids - 1)]
    tris.extend(
        [
            (i + num_centroids + 1, i + 1, i + num_centroids)
            for i in range(num_centroids - 1)
        ]
    )
    mlab.triangular_mesh(
        *road.T, tris, scalars=np.hstack((scalars, scalars)), figure=fig, **kwargs
    )


def arrow_from_a_to_b(a, b):
    """
    Draws and arrow_from_a_to_b
    Parameters:
      a (np.array): start point
      b (np.array): end point
    Returns:
      ar1 (mlab.arrow): arrow object
    """
    from tvtk.tools import visual

    x1, y1, z1 = tuple(a)
    x2, y2, z2 = tuple(a)
    ar1 = visual.arrow(x=x1, y=y1, z=z1)
    ar1.length_cone = 0.4

    arrow_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos / arrow_length
    ar1.axis = [x2 - x1, y2 - y1, z2 - z1]
    return ar1


def draw_helices(
    hels, fig, hel_radius=1, hel_opacity=1.0, hel_colour=(1.0, 1.0, 1.0), **kwargs
):
    """
    Draws helices as cylinders.
    Parameters:
      hels (np.array): helices to be drawn
      fig (mlab.figure): figure to plot on
      Optional:
        hel_radius (float): radius of the helices
        hel_opacity (float): opacity of the helices
      **kwargs: keyword arguments to be passed to mlab.plot3d
    Returns:
      None
    """
    for i, h in enumerate(hels):
        mlab.plot3d(
            *h.T,
            tube_radius=hel_radius,
            opacity=hel_opacity,
            color=hel_colour,
            figure=fig,
            **kwargs
        )


def visualise_centroids(
    means,
    variances,
    fig,
    centroid_tube_colour=(1.0, 1.0, 1.0),
    interp_points=None,
    fin_dims=[0, 1],
):
    """
    Uses fins to visualise the dynamics of centroids.
    Parameters: 
      means (np.array): means of centroids to be visualised
      variances (np.array): variances of centroids to be visualised
      fig (mlab.figure): figure to plot on
      Optional:
        centroid_tube_colour (tuple): colour of the tube around the centroids
        interp_points (int): number of points to interpolate between each point
        fin_dims (list): dimensions to draw fins for
    Returns:
      None
    """

    # if interp_points is None, don't interpolate
    if interp_points is None:
        smooth_it = lambda points: points
    else:
        smooth_it = lambda points: smooth_points(points, interp_points)

    # plot centroids
    mlab.points3d(
        *means.T, scale_factor=1, scale_mode="none", figure=fig, color=(1.0, 1.0, 1.0)
    )

    # plot centroids tube
    mlab.plot3d(
        *smooth_it(means).T, tube_radius=0.1, color=centroid_tube_colour, figure=fig
    )

    # compute eig
    eigs = [sorted_eig(var) for var in variances]
    eigvals = np.array([vals for vals, _ in eigs])

    # colours for the free directions
    arrow_colours = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]

    num_centroids = len(means)
    vecs = np.zeros((num_centroids, 3, 3, 3))
    for i, eigvvals in enumerate(eigs):
        for xyz in range(3):
            p0 = means[i] - eigvvals[1][:, xyz] * eigvvals[0][xyz] ** (1 / 2) * 2
            p1 = means[i]
            p2 = means[i] + eigvvals[1][:, xyz] * eigvvals[0][xyz] ** (1 / 2) * 2
            vec = np.stack((p0, p1, p2))
            vecs[i][xyz] = vec
            mlab.plot3d(*vec.T, tube_radius=0.1, color=arrow_colours[xyz], figure=fig)

    # draw fins for given dimensions
    for dim in fin_dims:
        upper, lower = reorder_band_points(vecs[:, dim, 0], vecs[:, dim, -1])
        draw_fins_between_two_curves(
            smooth_it(upper),
            smooth_it(lower),
            smooth_it(eigvals[:, 0][:, None])[:, 0],
            fig,
            colormap="viridis",
        )
    return
