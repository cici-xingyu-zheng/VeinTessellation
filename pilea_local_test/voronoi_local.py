import numpy as np
import pandas as pd
import networkx as nx
import math

from matplotlib.path import Path


# ---- angle and distance error calculate suite ----

def shared_edges(G, threshold = 50):
    
    '''
    Parameters:  
    ----------
    G: nx graph
    threshold: float/int, pixel threshold to be considered as an valid shared edge

    Returns:
    ----------
    adj_faces_mat (A): N by N boolean ndarray; A_ij = 1 if face i and j are adjacent 
    shared_edges_mat (S): N by N ndarray of lists;
                          S_ij contains a list of shared edge between face i and j
    '''
    
    M = len(G.graph['faces_passed'])
    adj_faces_mat = np.zeros((M, M))
    shared_edges_mat = np.ndarray((M, M), dtype=np.object)
    nx.set_edge_attributes(G, 'not_shared', name = 'shared')
    
    for i in range(M):
        for j in range(i+1, M):
            # intersection of node list of two faces:
            shared_nodes = set(G.graph['faces_passed'][i]) & set(G.graph['faces_passed'][j])
            # to enable use of len function:
            if bool(shared_nodes): 
                # when we have a shared edge rather than just one single node:
                if len(shared_nodes) > 1: 
                    
                    shared_nodes_list = list(shared_nodes)
                    
                    shared_edges_mat[i][j] = []
                    shared_edges_mat[j][i] = []
                    
                    # when there is more than one edge:
                    if len(shared_nodes_list) > 2:
                        for k in range(len(shared_nodes_list)):
                            for l in range(k+1, len(shared_nodes_list)):
                                # if the two nodes makes an edge: 
                                if G.has_edge(shared_nodes_list[k], shared_nodes_list[l]):  
                                    # to make sure the edge is not artifect resulted from NEFI placing two dots
                                    # very close to each other:  
                                    edge_vec =  (np.array(shared_nodes_list[k]) - np.array(shared_nodes_list[l])).tolist()                                 
                                    if np.linalg.norm(edge_vec) > threshold: 
                                        shared_edges_mat[i][j].append((shared_nodes_list[k], shared_nodes_list[l]))
                                        shared_edges_mat[j][i].append((shared_nodes_list[k], shared_nodes_list[l]))
                                        G.edges[(shared_nodes_list[k], shared_nodes_list[l])]['shared'] = 'shared'
                                        adj_faces_mat[i][j] = 1
                                        adj_faces_mat[j][i] = 1
                                        
                    else:    
                        edge_vec =  (np.array(shared_nodes_list[0]) - np.array(shared_nodes_list[1])).tolist()
                        if np.linalg.norm(edge_vec) > threshold:

                            shared_edges_mat[i][j].append((shared_nodes_list[0], shared_nodes_list[1]))
                            shared_edges_mat[j][i].append((shared_nodes_list[0], shared_nodes_list[1]))       
                            G.edges[(shared_nodes_list[0], shared_nodes_list[1])]['shared'] = 'shared'
                            adj_faces_mat[i][j] = 1
                            adj_faces_mat[j][i] = 1
                    
    return adj_faces_mat, shared_edges_mat

def slope_intercept(p1, p2):
    '''
    helper_func A for error_calculate()
    
    Returns the slope and intercept for the line 
    determined by the two points provided.
    '''
    # slope:
    try:
        s = (p1[1] - p2[1])/(p1[0] - p2[0])
        if math.isinf(s) and s > 0:
            s = 10000
        elif math.isinf(s) and s < 0:  
            s = -10000
    except ZeroDivisionError:
        s = 10000
   
        
    # intercept: 
    d = - s*p1[0] + p1[1]
    
    return s, d

def intersect(s1, d1, s2, d2):
    '''
    helper_func B for error_calculate()

    returns the intersection point of two lines defined by: 
    y1 = s1*x + d1;
    y2 = s2*x + d2.
    '''
    A = np.zeros((2,2))
    A[0][0] = s1
    A[0][1] = -1
    A[1][0] = s2
    A[1][1] = -1
    b = np.array([-d1, -d2])
    intersection = np.linalg.lstsq(A, b, rcond= None)[0]
    
    return intersection

def error_calculate(dot1, dot2, vein_node1, vein_node2):
    
    '''
    helper function for local_test().

    given dots and nodes of an edge,
    return angle of intersection and dist
    '''
    s1, d1 = slope_intercept(dot1, dot2)
    s2, d2 = slope_intercept(vein_node1, vein_node2)
    
    intersection = intersect(s1, d1, s2, d2)
    
    tan_theta = (s2 - s1)/(1 + s1*s2)
    theta = abs(np.degrees(np.arctan(tan_theta)))
    
    theta_diff = 90 - theta
    
    dist1 = np.linalg.norm(dot1 - intersection)
    dist2 = np.linalg.norm(dot2 - intersection)
    
    dist_diff = abs(dist1 - dist2)/ (.5* (dist1 + dist2))
    
    return (theta_diff, dist_diff)


# ----  baseline points generation -----------------
def scatter_everywhere(x_min, x_max, y_min, y_max, tot = 5000000):  
    '''
    helper function for random dots and centroid genenration functions. 
    return an 2d array of random points coordinates.
    '''
    rand_x = np.random.uniform(x_min, x_max, tot)
    rand_y = np.random.uniform(y_min, y_max, tot)
    points = np.transpose(np.stack((rand_x, rand_y)))
    
    return points

def randlist_n_centroid(G):
    '''
    Parameters:  
    ----------
    G: nx graph

    Returns:
    ----------
    points_in_faces: 3d array, 1000 random dots for each face
    centroid_in_faces: 2d array, centroid for each face

    '''
    tot = 5000000
    print(f'Scattering {tot} random points and binning by face, takes about 30 s...')
    print()
    points = scatter_everywhere(G.graph['x_min'], G.graph['x_max'], 
                                G.graph['y_min'], G.graph['y_max'], tot)
    
    L = len(G.graph['faces_passed'])

    points_in_faces = np.ndarray((L,1000,2))
    centroid_in_faces = np.ndarray((L,), dtype = 'object')

    for i in range(L):
        # create path for polygon: 
        face_pos = G.graph['faces_passed'][i]
        path = Path(face_pos)
        # find points inside:
        inside = path.contains_points(points)

        inside_pos = points[inside]
        try:
            points_in_faces[i] = inside_pos[0:1000,:] 
        except IndexError:
            print(f'Not enough random points are in face {i}!')
        
        centroid_in_faces[i] = (np.mean(inside_pos[:,0]), np.mean(inside_pos[:,1]))
    
    print('Binning completed.')
    print()
    return points_in_faces, centroid_in_faces


# ---- local test main -----------------------------
def local_test(adj_faces_mat, shared_edges_mat, dot_list, *Graph, dot_bool = False):
                
    '''
    Parameters:  
    ----------
    adj_faces_mat: M by M boolean np array 
    shared_edges_mat:  M by M np array, dtype = list
    dot_list: M by 2 array
    dot_bool: boolean, default to Flase for baseline comparison; if set to true, 
                will set the tested edge attr 'shared' to 'tested_shared'.
    Returns:
    ----------
    result_mat: M by M array, dtype = tuple, 
            if passed face i and j are adjacent, 
            entry ij stores the angle and distance error.
    result_df: list, append all local test results in result_mat to a list
    result_summary_df: dataframe, of summary statistics

    '''
    M = len(shared_edges_mat)
    result_mat = np.ndarray((M, M), dtype=np.object)
    result_list = []
    
    # for each pair of faces:
    for i in range(M):
        for j in range(i+1, M):
            # if they are adjacent:
            if adj_faces_mat[i][j]:
                # Case 1: if more than one edge is shared:
                if len(shared_edges_mat[i][j]) > 1:   
                    rst_list = []
     
                    for n in range(len(shared_edges_mat[i][j])):
                        
                        vein_node1 = shared_edges_mat[i][j][n][0]
                        vein_node2 = shared_edges_mat[i][j][n][1]
                        dot1 = dot_list[i]
                        dot2 = dot_list[j]

                        rst = error_calculate(dot1, dot2, vein_node1, vein_node2)
                        rst_list.append(rst)

                    # pick the edge that has the smallest error: 
                    error_list = [rst_list[k][0] for k in range(len(rst_list))]
                    min_index = error_list.index(min(error_list))

                    result_mat[i][j] = rst_list[min_index]
                    result_mat[j][i] = rst_list[min_index]

                    if dot_bool:
                        # Graph returns a tuple with the first one being the graph object:
                        Graph[0].edges[shared_edges_mat[i][j][min_index]]['shared'] = 'tested_shared'

                # Case 2: only one edge is shared:
                else:
                    vein_node1 = np.array(shared_edges_mat[i][j][0][0])        
                    vein_node2 = np.array(shared_edges_mat[i][j][0][1])
        
                    dot1 = dot_list[i]
                    dot2 = dot_list[j]
                    
                    rst = error_calculate(dot1, dot2, vein_node1, vein_node2)
                    
                    result_mat[i][j] = rst
                    result_mat[j][i] = rst

                    if dot_bool:
                   
                        Graph[0].edges[shared_edges_mat[i][j][0]]['shared'] = 'tested_shared'

                result_list.append(result_mat[i][j])

    # create result list df and summary: 
    result_df = pd.DataFrame(np.array(result_list),columns = ['angle_diff', 'dist_diff'])
   
    result_summary = [result_df['angle_diff'].mean(), 
                      result_df['angle_diff'].std(), 
                      result_df['dist_diff'].mean(), 
                      result_df['dist_diff'].std()]

    result_summary_df = pd.DataFrame(columns = ['mean angle error', 'std angle error', 'mean distance error', 'std distance error'])  
    result_summary_df.loc[0] = result_summary
             
    return (result_mat, result_df, result_summary_df)

def random_rounds(adj_faces_mat, shared_edges_mat, points_in_faces):

    '''
    convenient function for 1000 round of random point local test.
    returns list of mean angle and dsitance error, and an example result for plotting.
    '''
    mean_angle_error = np.zeros(1000)
    mean_dist_error = np.zeros(1000)

    for i in range(1000):
        result_mat, result_df, result_summary_df = local_test(adj_faces_mat, shared_edges_mat, points_in_faces[:,i,:])
        mean_angle_error[i] = result_summary_df.loc[0][0]
        mean_dist_error[i] = result_summary_df.loc[0][2]
    
    # save last result as an example:
    rst = (result_mat, result_df, result_summary_df)

    return mean_angle_error, mean_dist_error, rst

