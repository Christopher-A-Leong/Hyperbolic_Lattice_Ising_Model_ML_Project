from hypertiling import HyperbolicTiling
from hypertiling.graphics.plot import convert_polygons_to_patches
import numpy as np
import matplotlib.pyplot as plt
from Fundamental.Hamiltonian_PeierlsSubstitution import Number_Plaquets
from Fundamental.Number_Points import points
# from Fundamental.Honeycomb_Lattice import honeycomb_lattice
# from Fundamental.Honeycomb_Lattice import honeycomb_points

# def honeycomb_flake_plot(num_levels):
#     a1vec = np.array([np.sqrt(3) / 2, -1 / 2])
#     a2vec = np.array([0, 1])
#     b1vec = np.array([1 / (2 * np.sqrt(3)), 1 / 2])
#     b2vec = np.array([1 / (2 * np.sqrt(3)), -1 / 2])
#     b3vec = np.array([-1 / np.sqrt(3), 0])
#
#     point_one = np.array([0, 1])
#     point_two = point_one + b3vec
#     point_three = point_two - b1vec
#     point_four = point_three + b2vec
#     point_five = point_four - b3vec
#     point_six = point_five + b1vec
#
#     points = np.array([point_one, point_two, point_three, point_four, point_five, point_six])
#
#     points_per_level = honeycomb_points(num_levels)[0]
#
#     def motif_even_one(points, level):
#         for i in range(level - 1):
#             if i % 2 == 0:
#                 points = np.vstack((points, points[-1] - b2vec))
#             else:
#                 points = np.vstack((points, points[-1] + b3vec))
#         return (points)
#
#     def motif_odd_one(points, level):
#         for i in range(level - 1):
#             if i % 2 == 0:
#                 points = np.vstack((points, points[-1] + b3vec))
#             else:
#                 points = np.vstack((points, points[-1] - b2vec))
#         return (points)
#
#     def motif_two(points, level):
#         motif_length = level - 1
#         for i in range(motif_length):
#             points = np.vstack((points, points[-1] + b3vec))
#             points = np.vstack((points, points[-1] - b1vec))
#         points = np.vstack((points, points[-1] + b3vec))
#         return (points)
#
#     def motif_three(points, level):
#         for i in range(level):
#             points = np.vstack((points, points[-1] - b1vec))
#             points = np.vstack((points, points[-1] + b2vec))
#         return (points)
#
#     def motif_four(points, level):
#         for i in range(level - 1):
#             points = np.vstack((points, points[-1] - b3vec))
#             points = np.vstack((points, points[-1] + b2vec))
#         return (points)
#
#     def motif_five(points, level):
#         for i in range(level):
#             points = np.vstack((points, points[-1] - b3vec))
#             points = np.vstack((points, points[-1] + b1vec))
#         return (points)
#
#     def motif_six(points, level):
#         for i in range(level - 1):
#             points = np.vstack((points, points[-1] - b2vec))
#             points = np.vstack((points, points[-1] + b1vec))
#         return (points)
#
#     def motif_seven_odd(points, level):
#         for i in range(int((level - 1) / 2)):
#             points = np.vstack((points, points[-1] - b2vec))
#             points = np.vstack((points, points[-1] + b3vec))
#         return (points)
#
#     def motif_seven_even(points, level):
#         for i in range(int(level / 2 - 1)):
#             points = np.vstack((points, points[-1] - b2vec))
#             points = np.vstack((points, points[-1] + b3vec))
#         points = np.vstack((points, points[-1] - b2vec))
#         return (points)
#
#     first_to_first_dist = b1vec
#     last_first_point = point_one
#     for n in range(2, num_levels + 1):
#         points = np.vstack((points, last_first_point + first_to_first_dist))
#         last_first_point = np.copy(points[-1])
#         if n % 2 == 0:
#             first_to_first_dist = -b3vec + b1vec - b2vec
#             points = motif_even_one(points, n)
#             points = motif_two(points, n)
#             points = motif_three(points, n)
#             points = motif_four(points, n)
#             points = motif_five(points, n)
#             points = motif_six(points, n)
#             points = motif_seven_even(points, n)
#         else:
#             first_to_first_dist = b1vec
#             points = motif_odd_one(points, n)
#             points = motif_two(points, n)
#             points = motif_three(points, n)
#             points = motif_four(points, n)
#             points = motif_five(points, n)
#             points = motif_six(points, n)
#             points = motif_seven_odd(points, n)
#
#     tbham = honeycomb_lattice(num_levels)
#     conns = np.where(tbham != 0)
#
#     plt.plot([points[conns[0], 0], points[conns[1], 0]], [points[conns[0], 1], points[conns[1], 1]], color='black')
#     plt.scatter(points[:, 0], points[:, 1], color='black')
#     # plt.show()
#
#     return(points, conns)

def get_hyperbolicq3_site_coordinates(p, num_levels):

    num_plaqs_per_level = Number_Plaquets(p, 3, num_levels)[0]

    ht_object = HyperbolicTiling(p, 3, num_levels)
    mpl_patch_collection = convert_polygons_to_patches(ht_object)
    mpl_paths = mpl_patch_collection.get_paths()
    coords = []
    for i in range(len(mpl_paths)):
        coords.append(((mpl_paths[i]).vertices)[:-1,:])

    angles = np.array([0])
    xavglocs = np.array([])
    yavglocs = np.array([])
    for c in coords:
        angles_local = np.array([])
        xavglocs = np.append(xavglocs, np.average(c[:,0]))
        yavglocs = np.append(yavglocs, np.average(c[:,1]))
        for v in c:
            angles_local = np.append(angles_local, np.arctan(v[1]/v[0]))
        if (len(c[c[:,0]>0,0]) == len(c[c[:,0]<0,0]) or np.all(np.abs(c[:,0]) < 10**(-3))) and yavglocs[-1] > 0:
            print('pika')
            angles = np.append(angles, np.pi/2)
        elif (len(c[c[:,0]>0,0]) == len(c[c[:,0]<0,0]) or np.all(np.abs(c[:,0]) < 10**(-3))) and yavglocs[-1] < 0:
            print('chu')
            angles = np.append(angles, 3*np.pi/2)
        elif xavglocs[-1] > 0 and yavglocs[-1] > 0: #Quadrant I
            angles = np.append(angles, np.average(angles_local))
        elif xavglocs[-1] < 0 and yavglocs[-1] > 0: #Quadrant II
            angles = np.append(angles, np.pi + np.average(angles_local))
        elif xavglocs[-1] < 0 and yavglocs[-1] < 0: #Quadrant III
            angles = np.append(angles, np.pi + np.average(angles_local))
        elif xavglocs[-1] > 0 and yavglocs[-1] < 0:  # Quadrant IV
            angles = np.append(angles, 2*np.pi + np.average(angles_local))

    poincare_dist = np.array([])
    for i in range(len(xavglocs)):
        poincare_dist = np.append(poincare_dist, np.sqrt(xavglocs[i]**2 + yavglocs[i]**2)/(1-(xavglocs[i]**2 + yavglocs[i]**2)))

    coords = np.array(coords)[np.argsort(poincare_dist)]
    angles = angles[np.argsort(poincare_dist)]

    gen_indices = []
    for n in range(1, num_levels+1):
        gen_indices.append(np.arange(np.sum(num_plaqs_per_level[:n-1]), np.sum(num_plaqs_per_level[:n]), dtype=int))

    ordering = np.array([])
    for gi in gen_indices:
        first_index = gi[0]
        rel_angles = angles[gi]
        ordering = np.append(ordering, first_index + np.argsort(rel_angles))
    ordering = [int(o) for o in ordering]

    coords = (coords[ordering]).reshape(-1, 2)
    unique_coords = np.unique(np.around(coords, 5), axis=0, return_index=True)[1]

    coords = coords[np.sort(unique_coords), :]

    points_per_level = points(p, 3, num_levels)[0]
    sites_on_level = []
    for n in range(1, num_levels + 1):
        sites_on_level.append(np.arange(np.sum(points_per_level[:n - 1]), np.sum(points_per_level[:n]), dtype=int))

    final_coords = np.zeros((1,2))
    for soli in range(len(sites_on_level)):
        rel_coords = coords[sites_on_level[soli][0]:sites_on_level[soli][-1]+1]
        local_angles = []
        for rc in rel_coords:
            local_angles.append(np.arctan(rc[1]/rc[0]))
        for i in range(len(local_angles)):
            if np.abs(rel_coords[i][0]) < 10**(-3) and rel_coords[i][1] > 0:
                local_angles[i] = np.pi/2
            elif np.abs(rel_coords[i][0]) < 10**(-3) and rel_coords[i][1] < 0:
                local_angles[i] = 3*np.pi/2
            elif rel_coords[i][0] < 0 and rel_coords[i][1] > 0:
                local_angles[i] = local_angles[i] + np.pi
            elif rel_coords[i][0] < 0 and rel_coords[i][1] < 0:
                local_angles[i] = local_angles[i] + np.pi
            elif rel_coords[i][0] > 0 and rel_coords[i][1] < 0:
                local_angles[i] = local_angles[i] + 2*np.pi
        if soli <= 1:
            final_coords = np.vstack((final_coords, rel_coords[np.argsort(local_angles)]))
        else:
            final_coords = np.vstack((final_coords, np.roll(rel_coords[np.argsort(local_angles)], -int((p-4)/2), axis=0)))


    return(final_coords[1:, :])

# def get_hyperbolicq3_site_coordinates(p, num_levels):
#
#     num_plaqs_per_level = Number_Plaquets(p, 3, num_levels)[0]
#
#     ht_object = HyperbolicTiling(p, 3, num_levels)
#     mpl_patch_collection = convert_polygons_to_patches(ht_object)
#     mpl_paths = mpl_patch_collection.get_paths()
#     coords = np.zeros((1,2))
#     used_path_indices = []
#     #First gen plaquet sites
#     coords = np.vstack((coords, ((mpl_paths[0]).vertices)[:-1,:]))
#     used_path_indices.append(0)
#
#     #Higher gen plaquet sites other than last gen of plaquets
#     for n in range(1, num_levels):
#         first_plaq_index = np.sum(num_plaqs_per_level[:n])
#         num_plaqs_on_levels = num_plaqs_per_level[n]
#         skipfactor = (np.sum(num_plaqs_per_level[n+1:]) / p) + 1
#         for i in np.arange(first_plaq_index, num_plaqs_on_levels*skipfactor + 1, skipfactor):
#             coords = np.vstack((coords, ((mpl_paths[int(i)]).vertices)[:-1,:]))
#             used_path_indices.append(i)
#
#     #Finally handle last gen of plaquets
#     for i in range(len(mpl_paths)):
#         if i not in used_path_indices:
#             coords = np.vstack((coords, ((mpl_paths[i]).vertices)[:-1, :]))
#
#     coords = coords[1:, :]
#     #Remove overlapping points
#     unique_coord_indices = np.unique(np.around(coords, 5), axis=0, return_index=True)[1]
#     coords = coords[np.sort(unique_coord_indices), :]
#
#     return(coords)



### plot_tiling_custom is copied from the source code of the hypertiling python package
### the custom part is the code I added to the function to add ploting of points on sites and color of said points
### BE SURE TO GIVE CREDIT TO hypertiling PACKAGE:
### Manuel Schrauth, Yanick Thurn, Florian Goth, Jefferson S.E. Portela, Dietmar Herdt and Felix Dusel. (2023). The hypertiling project. Zenodo. https://doi.org/10.5281/zenodo.7559393
def plot_tiling_custom(tiling, colors=None, unitcircle=False, symmetric_colors=False, plot_colorbar=False, cutoff=None,
                xcrange=(-1, 1), ycrange=(-1, 1), dpi=120, **kwargs):
    """
    Plots a hyperbolic tiling

    Parameters
    ----------

    tiling: HyperbolicTiling
        A hyperbolic tiling object, requires proper "get"-interfaces and iterator functionality

    colors: None or colorvalue or array-like
        Used for colormapping the PatchCollection. If None, all polygons are mapped with transparent faces;
        Other valid options are matplotlib color strings (e.g. "k", "white" or RGBA (0,0,1,1))
        or an array with the same length as the tiling, containing floats

    unitcircle: Bool, default: False
        If True, the unit circle (boundary of the Poincare disk) is added to the plot

    symmetric_colors: Bool, default: False
        If True, sets the colormap so that the center of the colormap corresponds to the center of colors.

    plot_colorbar: Bool, default: False
        If True, plots a colorbar.

    cutoff: float, default: None
        Actives lazy plotting; If set, only polygons which are located entirely inside the cutoff radius are being drawn

    xcrange: (2,) array-like, default: (-1,1)
        Sets the x limits of the plot.

    ycrange: (2,) array-like, default: (-1,1)
        Sets the y limits of the plot.


    Returns
    -------

    out: Axes
        Axes object containing the hyperbolic tiling plot.

    Other Parameters:
    -----------------

    **kwargs
        Further properties (such as linewidth, alpha, facecolor, edgecolor, ...)

    """

    # create figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    # draw unit circle
    if unitcircle:
        circle = plt.Circle((0, 0), 1, lw=0.7, fc=(0, 0, 0, 0), ec="k")
        ax.add_patch(circle)

    # convert to matplotlib format
    pgons = convert_polygons_to_patches(tiling, colors, cutoff, **kwargs)

    # draw patches
    ax.add_collection(pgons)

    # symmetric colorbar
    if symmetric_colors:
        cmin = np.min(colors)
        cmax = np.max(colors)
        clim = np.maximum(-cmin, cmax)
        pgons.set_clim([-clim, clim])

    if plot_colorbar:
        plt.colorbar(pgons)



    plt.xlim(xcrange)
    plt.ylim(ycrange)
    plt.axis("off")

    return ax