
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon

def finite_polygons(voronoi, diameter):

    '''
    helper function for bounded_polygons().

    Parameter:  
    ----------
    voronoi: Voronoi object
    diameter: float
    
    Return:
    ----------
    Polygon object, generator of finite voronoi regions in the same order of the input points. 
    The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    Reference code: 
    https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region

    '''

    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def bounded_polygons(G, vor):
    '''
    Parameter:  
    ----------
    G: nx graph object
    vor: Voronoi object
    
    Return:
    ----------
    bounded_regions: list of faces (verties)
    '''
    bounded_regions = []

    diameter = np.linalg.norm(np.array(G.graph['boundary']).ptp(axis=0))
    
    boundary_polygon = Polygon(np.array(G.graph['boundary']))
    
    for p in finite_polygons(vor, diameter):
        # big periphery polygon intersecting with the bounded area in the blade:
        bounded_regions.append(list(p.intersection(boundary_polygon).exterior.coords))
    
    return bounded_regions

def overlap_test(G, seeds, bounded_regions):
    '''
    Parameter:  
    ----------
    G: nx graph object
    seeds: list of coordinates of dots
    bounded_regions: list of faces (list of vertix coordinates)

    Return:
    ----------
    bounded_regions: J_list, list of J_index (intersected area/union area)
    '''

    # find the indices of the passed points in the list seed:
    passed_index = []
    for passed_point in G.graph['dots_passed']:
        passed_index.append(seeds.index(passed_point))

    L = len(G.graph['dots_passed'])
    shared_area_list = np.zeros(L)
    union_area_list = np.zeros(L)

    for i in range(L):
        shared_area_list[i] = Polygon(G.graph['faces_passed'][i]).intersection(Polygon(bounded_regions[passed_index[i]])).area
        union_area_list[i] = Polygon(G.graph['faces_passed'][i]).union(Polygon(bounded_regions[passed_index[i]])).area

    J_list = shared_area_list/union_area_list
    return J_list

def hybrid_seeds(G):

    '''
    Parameter:  
    ----------
    G: nx graph object

    Return:
    ----------
    centroid_seeds/random_seeds: list of point coordinates, same length as seeds
    '''


    centroid_seeds = []
    random_seeds = []

    for n in G.nodes:
        # if it is not the single dot:
        if  G.nodes[n]['type'] == 'dot':
            # copy locaion of of the dot to two seeding list:
            centroid_seeds.append(n)
            random_seeds.append(n)
        # if is the single dot:
        elif G.nodes[n]['type'] == 'single_dot':
            index = G.graph['dots_passed'].index(n)
            # apend other dots associated with the face the dot is in:
            centroid_seeds.append(list(Polygon(G.graph['faces_passed'][index]).centroid.coords)[0])
            random_seeds.append(list(Polygon(G.graph['faces_passed'][index]).representative_point().coords)[0])
        
    return centroid_seeds, random_seeds
