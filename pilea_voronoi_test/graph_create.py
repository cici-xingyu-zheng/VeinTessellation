import numpy as np
import pandas as pd
import os
import cv2
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def create_dot_graph(dot_file):
    '''
    Parameter:  
    ----------
    dot_file: string, path to dot img file
    
    Returns:
    ----------
    G_dots: nx graph, 
        with node attribute 'type' == 'dot'
    dot_list: list, contains tuple of xy coordinate for each dot
    '''
    # load in grayscale mode
    img = cv2.imread(dot_file, 0)
    # threshold optimization:
    th, threshed = cv2.threshold(img, 100, 255,  cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # output contour list for dots:
    contour = cv2.findContours(threshed, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    dot_list = []
    for n in range(len(contour)):
        x_cor = [contour[n][i][0][0] for i in range(contour[n].shape[0])]
        y_cor = [contour[n][i][0][1] for i in range(contour[n].shape[0])]
        # note that we flip x and y, to match the x, y in the vein graph:
        dot_list.append((np.around(np.array(y_cor).mean(), 2), np.around(np.array(x_cor).mean(), 2)) )
    
    G_dots = nx.Graph()
    G_dots.add_nodes_from(dot_list) 
    nx.set_node_attributes(G_dots, 'dot', name= 'type')
    
    return dot_list, G_dots

def read_nefi_graph(vein_file):
    '''
    Parameter:  
    ----------
    vein_file: string, path to vein txt file, graph extracted from nefi
    
    Returns:
    ----------
    G_vein: nx graph,
        with node attribute 'type' == 'dot'
    '''
    # read in output from nefi:
    G_vein = nx.read_multiline_adjlist(vein_file, delimiter='|')
    
    # remove unwanted attr:
    for e in G_vein.edges:
        del G_vein.edges[e]['pixels']
        del G_vein.edges[e]['width']
        del G_vein.edges[e]['width_var']
        
    # node attr:
    name_map = {}
    for n in G_vein:
        name_map[n] = tuple(map(int, n[1:-1].split(', ')))
        G_vein.nodes[n]['type'] = 'vein' # assign type to vein 
    # rename the node to coordinate tuples to replace the strings:
    G_vein = nx.relabel_nodes(G_vein, name_map)
    
    # rm helper edges:
    G_vein = trim_fake_edge(G_vein)
    
    return G_vein

def trim_fake_edge(G):
   
    'helper func for read_nefi_graph(), recursively remove nodes when node deg == 1'
    
    deg_node_list = G.degree()
    degs = [d for n, d in G.degree()]
    
    if min(degs) !=1:
        return G
    else:
        kept_node = [n[0] for n in deg_node_list if n[1]!= 1]
        G = G.subgraph(kept_node)
        return trim_fake_edge(G)

def merge_graphs(G_vein, G_dot):
    '''
    Parameter:  
    ----------
    G_vein: nx graph; trimmed graph for veins
    G_dot: nx graph; dots as nodes
    
    Returns:
    ----------
    G: nx graph, combined vein and dot graphs, with
        graph attribute for dimensions (xmin, xmax, y_min, y_max, xy ratio)
    
    '''
    
    G = nx.compose(G_vein, G_dot)
    
    mins = np.min(np.array(list(G.nodes())), 0)
    maxes = np.max(np.array(list(G.nodes())), 0)
    
    G.graph['x_min'] = mins[0]
    G.graph['y_min'] = mins[1]
    
    G.graph['x_max'] = maxes[0]
    G.graph['y_max'] = maxes[1]
    
    G.graph['ratio'] = (maxes[0] - mins[0]) / (maxes[1] - mins[1])
    
    return G

def get_faces(G, G_eb, bound = 30):
    '''
    Parameter:  
    ----------
    G: nx graph
    G_eb: G's planar embedding
    bound: int, upper bound for # of edge
    
    Returns:
    ----------
    faces: list of list, each inner list contains nodes for a face in G
    '''
    # create embedding: 
    _, G_eb = nx.check_planarity(G)
    
    faces = []
    faces_sorted = []
    
    G.graph['boundary'] = []
    blade_area = 0
    # traverse clockwise and counter clockwise for every edge: 
    for edge in G.edges():
        # one direction: 
        new_face = G_eb.traverse_face(edge[0], edge[1])
        new_face_sorted = sorted(new_face)

        # if len(new_face) > bound and len(new_face) > len(G.graph['boundary']):
        # ATTENTION NEEDED: the boundary face sometimes can  be confused for cases when the interior faces does not connect to the exterior.
        # Solution for now is to raise the threshold for boundary: 
        if len(new_face) > bound and Polygon(new_face).area > blade_area:
            G.graph['boundary'] = new_face
            blade_area = Polygon(new_face).area
            
        # duplicate check: 
        if (new_face_sorted not in faces_sorted) and (len(new_face) < bound):
            faces.append(new_face)
            faces_sorted.append(new_face_sorted)


        # the other direction:
        new_face_2 = G_eb.traverse_face(edge[1], edge[0])
        new_face_2_sorted = sorted(new_face_2)
            
        if (new_face_2_sorted not in faces_sorted) and (len(new_face_2) < bound):
            faces.append(new_face_2)
            faces_sorted.append(new_face_2_sorted)
            
    return faces

def one_per_loop(G, faces, dot_list):
    '''
    Parameter:  
    ----------
    G: nx graph
    faces: list of nodes in each face
    dot_list: list of xy coordinates of dots
    
    Returns:
    ----------
    dot_bool: list of boolean for if there is only one dot in the loop
    dots_passed: list of dots that are the single ones in the face
    faces_passed: list of faces that have only one dot
    '''
    dot_count = np.zeros((len(faces)))
    dot_in = [ [] for _ in range(len(faces)) ]

    for i in range(len(faces)): 
        polygon = Polygon(faces[i])
        for j in range(len(dot_list)):
            point = Point(dot_list[j][0], dot_list[j][1])
            if polygon.contains(point):
                dot_count[i] += 1
                dot_in[i].append(dot_list[j])

    dot_bool = [dot_count[n] == 1 for n in range(dot_count.shape[0])]

    faces_passed = []
    dots_passed = []

    for i in range(len(faces)):
        if dot_bool[i]:
            G.nodes[dot_in[i][0]]['type'] = 'single_dot'
            faces_passed.append(faces[i])
            dots_passed.append(dot_in[i][0])

    return dot_bool, dots_passed, faces_passed

# ------ main ----------
def graph_creation(sample, dot_folder = 'dot_images', vein_folder = 'vein_file'):

    '''
    Parameter:  
    ----------
    sample: string, sample name of the original image taken 
    
    Return:
    ----------
    G: nx graph, 
        Graph attributes: 
        'x_min', 
        'y_min', 
        'x_max', 
        'y_max',
        'ratio',
        'faces', # list of face nodes
        'dot_bool', # boolean for passed dots/faces
        'dots_passed', # list of passed dot nodes  
        'faces_passed' # list of passed faces
    '''
    
    print('Creating graph from vein and dot tracing images.')
    print()

    print('- Step1: reading files...')
    print()
    
    dot_file = f'{dot_folder}/{sample}_dots.jpg'
    vein_file = f'{vein_folder}/{sample}.txt'
    
    assert (os.path.exists(dot_file)), 'dot image file does not exist!'
    assert (os.path.exists(vein_file)), 'vein graph txt does not exist!'
    
    print('- Step2: create dot graph...')
    print()
    
    dot_list, G_dot = create_dot_graph(dot_file)
    
    print('- Step3: read vein graph...')
    print()
    
    G_vein = read_nefi_graph(vein_file)
   
    print('- Step4: merge graphs...')
    print()
    
    G = merge_graphs(G_vein, G_dot)
    
    print('- Step5: find testable faces...')
    print()
    
    planar_tf, G_eb = nx.check_planarity(G)
    
    assert (planar_tf), 'we only test for planar graph!'
    
    faces = get_faces(G, G_eb)
    dot_bool, dots_passed, faces_passed = one_per_loop(G, faces, dot_list)
    
    G.graph['faces'] = faces
    G.graph['dot_bool'] = dot_bool
    G.graph['dots_passed'] = dots_passed
    G.graph['faces_passed'] = faces_passed 

    print('Graph creation completed.')
    print()

    return G


# Okay the dual can only be created after running the tests in voronoi_local.py!!!
# perhaps we can do a warning here using assert (05/03/22)

def make_dual(G, cent_in_faces, mid_in_faces, rand_in_faces, result_mat):
    '''
    This is used to create the dual graph by itself and add the referene point test results in,
    thus needs the local test results.
    '''
    G_dual = nx.Graph()
    for i in range(len(cent_in_faces)):
        node1 = G.graph['dots_passed'][i]
        G_dual.add_node(node1)
        G_dual.nodes[node1]['label'] = i
        G_dual.nodes[node1]['centroid'] = cent_in_faces[i]
        G_dual.nodes[node1]['midpoint'] = mid_in_faces[i]
        G_dual.nodes[node1]['random'] = rand_in_faces[i]

        for j in range(i+1, (len(cent_in_faces))):
            if result_mat[i][j]:
                node2 = G.graph['dots_passed'][j]
                G_dual.add_edge(node1, node2)
                G_dual.edges[node1, node2]['angle'] = result_mat[i][j][0] 
                G_dual.edges[node1, node2]['dist'] = result_mat[i][j][1] 
                
    return G_dual


def create_dual_subgraph(G_dual, rst_df, cent_df, dual_edge_list, attr = 'angle'):
    '''
    Select the worst 10 performing pairs.
    Return the dual subgraph of those 10 edges.
    '''
    comp_df = pd.concat([rst_df[f'{attr}_diff'], cent_df[f'{attr}_diff']], axis=1)
    comp_df.columns = ['dot', 'centroid']
    comp_df['worse_by'] = comp_df['dot'] - comp_df['centroid']
    worse_10 = comp_df.sort_values(by=['worse_by'], ascending = False).index.tolist()[0:10]
    bad_apples = [dual_edge_list[i] for i in worse_10]
    G_dual_bad_apples =  G_dual.edge_subgraph(bad_apples)

    for (i, e) in enumerate(bad_apples):
        G_dual_bad_apples.edges[e][f'centroid_{attr}'] = comp_df['centroid'][worse_10[i]]
    
    return G_dual_bad_apples
