#For interpolate function
#Autor: Metpy Developers
#Modified: Albenis Pérez Alarcón
from __future__ import division
import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.spatial import cKDTree, ConvexHull, Delaunay, qhull
from scipy.spatial.distance import cdist
import logging
import math
from scipy.spatial import cKDTree



def get_points_within_r(center_points, target_points, r):
    r"""Get all target_points within a specified radius of a center point.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        location from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (X, Y) ndarray
        A list of points within r distance of, and in the same
        order as, center_points

    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return tree.data[indices].T


def get_point_count_within_r(center_points, target_points, r):
    r"""Get count of target points within a specified radius from center points.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        locations from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (N, ) ndarray
        A list of point counts within r distance of, and in the same
        order as, center_points

    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return np.array([len(x) for x in indices])


def triangle_area(pt1, pt2, pt3):
    r"""Return the area of a triangle.

    Parameters
    ----------
    pt1: (X,Y) ndarray
        Starting vertex of a triangle
    pt2: (X,Y) ndarray
        Second vertex of a triangle
    pt3: (X,Y) ndarray
        Ending vertex of a triangle

    Returns
    -------
    area: float
        Area of the given triangle.

    """
    a = 0.0

    a += pt1[0] * pt2[1] - pt2[0] * pt1[1]
    a += pt2[0] * pt3[1] - pt3[0] * pt2[1]
    a += pt3[0] * pt1[1] - pt1[0] * pt3[1]

    return abs(a) / 2


def dist_2(x0, y0, x1, y1):
    r"""Return the squared distance between two points.

    This is faster than calculating distance but should
    only be used with comparable ratios.

    Parameters
    ----------
    x0: float
        Starting x coordinate
    y0: float
        Starting y coordinate
    x1: float
        Ending x coordinate
    y1: float
        Ending y coordinate

    Returns
    -------
    d2: float
        squared distance

    See Also
    --------
    distance

    """
    d0 = x1 - x0
    d1 = y1 - y0
    return d0 * d0 + d1 * d1


def distance(p0, p1):
    r"""Return the distance between two points.

    Parameters
    ----------
    p0: (X,Y) ndarray
        Starting coordinate
    p1: (X,Y) ndarray
        Ending coordinate

    Returns
    -------
    d: float
        distance

    See Also
    --------
    dist_2

    """
    return math.sqrt(dist_2(p0[0], p0[1], p1[0], p1[1]))


def circumcircle_radius_2(pt0, pt1, pt2):
    r"""Calculate and return the squared radius of a given triangle's circumcircle.

    This is faster than calculating radius but should only be used with comparable ratios.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    -------
    r: float
        circumcircle radius

    See Also
    --------
    circumcenter

    """
    a = distance(pt0, pt1)
    b = distance(pt1, pt2)
    c = distance(pt2, pt0)

    t_area = triangle_area(pt0, pt1, pt2)
    prod2 = a * b * c

    if t_area > 0:
        radius = prod2 * prod2 / (16 * t_area * t_area)
    else:
        radius = np.nan

    return radius


def circumcircle_radius(pt0, pt1, pt2):
    r"""Calculate and return the radius of a given triangle's circumcircle.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    -------
    r: float
        circumcircle radius

    See Also
    --------
    circumcenter

    """
    a = distance(pt0, pt1)
    b = distance(pt1, pt2)
    c = distance(pt2, pt0)

    t_area = triangle_area(pt0, pt1, pt2)

    if t_area > 0:
        radius = (a * b * c) / (4 * t_area)
    else:
        radius = np.nan

    return radius


def circumcenter(pt0, pt1, pt2):
    r"""Calculate and return the circumcenter of a circumcircle generated by a given triangle.

    All three points must be unique or a division by zero error will be raised.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    -------
    cc: (x, y)
        circumcenter coordinates

    See Also
    --------
    circumcenter

    """
    a_x = pt0[0]
    a_y = pt0[1]
    b_x = pt1[0]
    b_y = pt1[1]
    c_x = pt2[0]
    c_y = pt2[1]

    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    cb_x_diff = c_x - b_x
    ac_x_diff = a_x - c_x
    ba_x_diff = b_x - a_x

    d_div = (a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff)

    if d_div == 0:
        raise ZeroDivisionError

    d_inv = 0.5 / d_div

    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y

    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv

    return cx, cy


def find_natural_neighbors(tri, grid_points):
    r"""Return the natural neighbor triangles for each given grid cell.

    These are determined by the properties of the given delaunay triangulation.
    A triangle is a natural neighbor of a grid cell if that triangles circumcenter
    is within the circumradius of the grid cell center.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation.
    grid_points: (X, Y) ndarray
        Locations of grids.

    Returns
    -------
    members: dictionary
        List of simplex codes for natural neighbor
        triangles in 'tri' for each grid cell.
    triangle_info: dictionary
        Circumcenter and radius information for each
        triangle in 'tri'.

    """
    tree = cKDTree(grid_points)

    in_triangulation = tri.find_simplex(tree.data) >= 0

    triangle_info = {}

    members = {key: [] for key in range(len(tree.data))}

    for i, simplices in enumerate(tri.simplices):

        ps = tri.points[simplices]

        cc = circumcenter(*ps)
        r = circumcircle_radius(*ps)

        triangle_info[i] = {'cc': cc, 'r': r}

        qualifiers = tree.query_ball_point(cc, r)

        for qualifier in qualifiers:
            if in_triangulation[qualifier]:
                members[qualifier].append(i)

    return members, triangle_info


def find_nn_triangles_point(tri, cur_tri, point):
    r"""Return the natural neighbors of a triangle containing a point.

    This is based on the provided Delaunay Triangulation.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation
    cur_tri: int
        Simplex code for Delaunay Triangulation lookup of
        a given triangle that contains 'position'.
    point: (x, y)
        Coordinates used to calculate distances to
        simplexes in 'tri'.

    Returns
    -------
    nn: (N, ) array
        List of simplex codes for natural neighbor
        triangles in 'tri'.

    """
    nn = []

    candidates = set(tri.neighbors[cur_tri])

    # find the union of the two sets
    candidates |= set(tri.neighbors[tri.neighbors[cur_tri]].flat)

    # remove instances of the "no neighbors" code
    candidates.discard(-1)

    for neighbor in candidates:

        triangle = tri.points[tri.simplices[neighbor]]
        cur_x, cur_y = circumcenter(triangle[0], triangle[1], triangle[2])
        r = circumcircle_radius_2(triangle[0], triangle[1], triangle[2])

        if dist_2(point[0], point[1], cur_x, cur_y) < r:

            nn.append(neighbor)

    return nn


def find_local_boundary(tri, triangles):
    r"""Find and return the outside edges of a collection of natural neighbor triangles.

    There is no guarantee that this boundary is convex, so ConvexHull is not
    sufficient in some situations.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation
    triangles: (N, ) array
        List of natural neighbor triangles.

    Returns
    -------
    edges: (2, N) ndarray
        List of vertex codes that form outer edges of
        a group of natural neighbor triangles.

    """
    edges = []

    for triangle in triangles:

        for i in range(3):

            pt1 = tri.simplices[triangle][i]
            pt2 = tri.simplices[triangle][(i + 1) % 3]

            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))

            elif (pt2, pt1) in edges:
                edges.remove((pt2, pt1))

            else:
                edges.append((pt1, pt2))

    return edges


def area(poly):
    r"""Find the area of a given polygon using the shoelace algorithm.

    Parameters
    ----------
    poly: (2, N) ndarray
        2-dimensional coordinates representing an ordered
        traversal around the edge a polygon.

    Returns
    -------
    area: float

    """
    a = 0.0
    n = len(poly)

    for i in range(n):
        a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]

    return abs(a) / 2.0


def order_edges(edges):
    r"""Return an ordered traversal of the edges of a two-dimensional polygon.

    Parameters
    ----------
    edges: (2, N) ndarray
        List of unordered line segments, where each
        line segment is represented by two unique
        vertex codes.

    Returns
    -------
    ordered_edges: (2, N) ndarray

    """
    edge = edges[0]
    edges = edges[1:]

    ordered_edges = [edge]

    num_max = len(edges)
    while len(edges) > 0 and num_max > 0:

        match = edge[1]

        for search_edge in edges:
            vertex = search_edge[0]
            if match == vertex:
                edge = search_edge
                edges.remove(edge)
                ordered_edges.append(search_edge)
                break
        num_max -= 1

    return ordered_edges


def cressman_point(sq_dist, values, radius):
    r"""Generate a Cressman interpolation value for a point.

    The calculated value is based on the given distances and search radius.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    radius: float
        Maximum distance to search for observations to use for
        interpolation.

    Returns
    -------
    value: float
        Interpolation value for grid point.

    """
    weights = tools.cressman_weights(sq_dist, radius)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))


def barnes_point(sq_dist, values, kappa, gamma=None):
    r"""Generate a single pass barnes interpolation value for a point.

    The calculated value is based on the given distances, kappa and gamma values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    kappa: float
        Response parameter for barnes interpolation.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 1.

    Returns
    -------
    value: float
        Interpolation value for grid point.

    """
    if gamma is None:
        gamma = 1
    weights = tools.barnes_weights(sq_dist, kappa, gamma)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))


def natural_neighbor_point(xp, yp, variable, grid_loc, tri, neighbors, triangle_info):
    r"""Generate a natural neighbor interpolation of the observations to the given point.

    This uses the Liang and Hale approach [Liang2010]_. The interpolation will fail if
    the grid point has no natural neighbors.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_loc: (float, float)
        Coordinates of the grid point at which to calculate the
        interpolation.
    tri: object
        Delaunay triangulation of the observations.
    neighbors: (N, ) ndarray
        Simplex codes of the grid point's natural neighbors. The codes
        will correspond to codes in the triangulation.
    triangle_info: dictionary
        Pre-calculated triangle attributes for quick look ups. Requires
        items 'cc' (circumcenters) and 'r' (radii) to be associated with
        each simplex code key from the delaunay triangulation.

    Returns
    -------
    value: float
       Interpolated value for the grid location

    """
    edges =  find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in  order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    c1 =  circumcenter(grid_loc, tri.points[p1], tri.points[p2])
    polygon = [c1]

    area_list = []
    total_area = 0.0

    for i in range(num_vertices):

        p3 = edge_vertices[(i + 2) % num_vertices]

        try:

            c2 =  circumcenter(grid_loc, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(triangle_info[check_tri]['cc'])

            pts = [polygon[i] for i in ConvexHull(polygon).vertices]
            value = variable[(tri.points[p2][0] == xp) & (tri.points[p2][1] == yp)]

            cur_area =  area(pts)

            total_area += cur_area

            area_list.append(cur_area * value[0])

        except (ZeroDivisionError, qhull.QhullError) as e:
            message = ('Error during processing of a grid. '
                       'Interpolation will continue but be mindful '
                       'of errors in output. ') + str(e)

            log.warning(message)
            return np.nan

        polygon = [c2]

        p2 = p3

    return sum(x / total_area for x in area_list)


def natural_neighbor_to_points(points, values, xi):
    r"""Generate a natural neighbor interpolation to the given points.

    This assigns values to the given interpolation points using the Liang and Hale
    [Liang2010]_. approach.

    Parameters
    ----------
    points: array_like, shape (n, 2)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, 2)
        Points to interpolate the data onto.

    Returns
    -------
    img: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`

    See Also
    --------
    natural_neighbor_to_grid

    """
    tri = Delaunay(points)

    members, triangle_info =  find_natural_neighbors(tri, xi)

    img = np.empty(shape=(xi.shape[0]), dtype=values.dtype)
    img.fill(np.nan)

    for ind, (grid, neighbors) in enumerate(members.items()):

        if len(neighbors) > 0:

            points_transposed = np.array(points).transpose()
            img[ind] = natural_neighbor_point(points_transposed[0], points_transposed[1],
                                              values, xi[grid], tri, neighbors, triangle_info)

    return img


def inverse_distance_to_points(points, values, xi, r, gamma=None, kappa=None, min_neighbors=3,
                               kind='cressman'):
    r"""Generate an inverse distance weighting interpolation to the given points.

    Values are assigned to the given interpolation points based on either [Cressman1959]_ or
    [Barnes1964]_. The Barnes implementation used here based on [Koch1983]_.

    Parameters
    ----------
    points: array_like, shape (n, 2)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, 2)
        Points to interpolate the data onto.
    r: float
        Radius from grid center, within which observations
        are considered and weighted.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    min_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation
        for a point. Default is 3.
    kind: str
        Specify what inverse distance weighting interpolation to use.
        Options: 'cressman' or 'barnes'. Default 'cressman'

    Returns
    -------
    img: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`

    See Also
    --------
    inverse_distance_to_grid

    """
    obs_tree = cKDTree(points)

    indices = obs_tree.query_ball_point(xi, r=r)

    img = np.empty(shape=(xi.shape[0]), dtype=values.dtype)
    img.fill(np.nan)

    for idx, (matches, grid) in enumerate(zip(indices, xi)):
        if len(matches) >= min_neighbors:

            x1, y1 = obs_tree.data[matches].T
            values_subset = values[matches]
            dists =  dist_2(grid[0], grid[1], x1, y1)

            if kind == 'cressman':
                img[idx] = cressman_point(dists, values_subset, r)
            elif kind == 'barnes':
                img[idx] = barnes_point(dists, values_subset, kappa, gamma)

            else:
                raise ValueError(str(kind) + ' interpolation not supported.')

    return img

def calc_kappa(spacing, kappa_star=5.052):
    r"""Calculate the kappa parameter for barnes interpolation.

    Parameters
    ----------
    spacing: float
        Average spacing between observations
    kappa_star: float
        Non-dimensional response parameter. Default 5.052.

    Returns
    -------
        kappa: float

    """
    return kappa_star * (2.0 * spacing / np.pi)**2

def points_interpolation(points, values, xi, interp_type='linear', minimum_neighbors=3,
                          gamma=0.25, kappa_star=5.052, search_radius=None, rbf_func='linear',
                          rbf_smooth=0):
    r"""Interpolate unstructured point data to the given points.

    This function interpolates the given `values` valid at `points` to the points `xi`. This is
    modeled after `scipy.interpolate.griddata`, but acts as a generalization of it by including
    the following types of interpolation:

    - Linear
    - Nearest Neighbor
    - Cubic
    - Radial Basis Function
    - Natural Neighbor (2D Only)
    - Barnes (2D Only)
    - Cressman (2D Only)

    Parameters
    ----------
    points: array_like, shape (n, D)
        Coordinates of the data points.
    values: array_like, shape (n,)
        Values of the data points.
    xi: array_like, shape (M, D)
        Points to interpolate the data onto.
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from `scipy.interpolate`.
        2) "natural_neighbor", "barnes", or "cressman" from `metpy.interpolate`.
        Default "linear".
    minimum_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See `scipy.interpolate.Rbf` for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    values_interpolated: (M,) ndarray
        Array representing the interpolated values for each input point in `xi`.

    Notes
    -----
    This function primarily acts as a wrapper for the individual interpolation routines. The
    individual functions are also available for direct use.

    See Also
    --------
    interpolate_to_grid

    """
    # If this is a type that `griddata` handles, hand it along to `griddata`
    if interp_type in ['linear', 'nearest', 'cubic']:
        return griddata(points, values, xi, method=interp_type)

    # If this is natural neighbor, hand it along to `natural_neighbor`
    elif interp_type == 'natural_neighbor':
        return natural_neighbor_to_points(points, values, xi)

    # If this is Barnes/Cressman, determine search_radios and hand it along to
    # `inverse_distance`
    elif interp_type in ['cressman', 'barnes']:
        ave_spacing = cdist(points, points).mean()

        if search_radius is None:
            search_radius = ave_spacing

        if interp_type == 'cressman':
            return inverse_distance_to_points(points, values, xi, search_radius,
                                              min_neighbors=minimum_neighbors,
                                              kind=interp_type)
        else:
            kappa = calc_kappa(ave_spacing, kappa_star)
            return inverse_distance_to_points(points, values, xi, search_radius, gamma, kappa,
                                              min_neighbors=minimum_neighbors,
                                              kind=interp_type)

    # If this is radial basis function, make the interpolator and apply it
    elif interp_type == 'rbf':

        rbfi = Rbf(points, values,  function=rbf_func,
                   smooth=rbf_smooth)
        return rbfi(xi)

    else:
        raise ValueError('Interpolation option not available. '
                         'Try: linear, nearest, cubic, natural_neighbor, '
                         'barnes, cressman, rbf')
