import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
import shapely 
from scipy.spatial import voronoi_plot_2d


def quick_plot(G):
    '''
    quick plotting for spatial graphs with attr "pos" as a length 2 coordinate vector  
    '''
    node_positions = {}
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1'}
    node_color = []
    
    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])
    
    fig, ax = plt.subplots(figsize=(9, 9/G.graph['ratio']))
   
    nx.draw(G, pos=node_positions, node_size= 20, node_color= node_color, ax = ax) 
    
    plt.tight_layout()
    plt.show()   
        
    return



def plot_baseline(G, G_dual, pt_type = 'centroid'):
    '''
    quick plotting for graph and dual  
    '''
    node_position_G = {}
    
    compared_position_dual = {}
    node_position_dual = {}
                                                                                                        
    for node in G.nodes:
        node_position_G[node] = node
    
    for node in G_dual.nodes:
        compared_position_dual[node] = (G_dual.nodes[node][pt_type][0], G_dual.nodes[node][pt_type][1])
        node_position_dual[node] = node
     
    color_dict = {'centroid': 'red', 'midpoint': 'deeppink','random':'purple'}  
    
    fig, ax = plt.subplots(figsize=(9,9/G.graph['ratio']))
    
    # later might need two colors for two ind of nodes, dk if networkx do it automatically 
    nx.draw_networkx_edges(G, 
            pos=node_position_G, 
            edge_color = 'C7', ax = ax) 
    nx.draw_networkx_nodes(G_dual, 
                           pos = node_position_dual,  
                           node_size = 20,
                           node_color = 'C1')
    nx.draw_networkx_nodes(G_dual, 
                           pos = compared_position_dual,
                           node_size = 20,
                           node_color = color_dict[pt_type])
    ax.set_title(f"{pt_type} overlay", fontsize = 16)
    plt.tight_layout()
    plt.show()
           
    return 

def plot_dual(G, G_dual, attr = "angle"):

    '''
    assert for attr 
    '''

    node_position_G = {}
    
    node_position_dual = {}
    # node_label_dual = {}
    
    edge_color_dual = []
    
    for node in G.nodes:
        node_position_G[node] = node
    
    edge_style = ['solid' if G.edges[e]['shared'] =='tested_shared' else 'dashed' for e in G.edges]
    edge_col = ['black' if G.edges[e]['shared'] =='tested_shared' else 'C7' for e in G.edges]
    for node in G_dual.nodes:
        node_position_dual[node] = node
        # node_label_dual[node] = G_dual.nodes[node]['label']
     
    for edge in G_dual.edges:
        edge_color_dual.append(G_dual.edges[edge][attr])
    
    selected_nodes = [n for n,v in G.nodes(data=True) if v['type'] == 'vein']  
    
    cmap_dict = {"angle":plt.cm.viridis, "dist":plt.cm.magma}
    
    fig, ax = plt.subplots(figsize=(10,10/G.graph['ratio']))
    
    nx.draw_networkx_edges(G, pos=node_position_G, edge_color = edge_col, style = edge_style, ax = ax) 
    nx.draw_networkx_nodes(G, pos=node_position_G, 
                           nodelist = selected_nodes, node_size= 5, node_color = 'C0', ax = ax) 
    nx.draw(G_dual, pos=node_position_dual, node_size= 20,  node_color= 'C1', 
            edge_color = edge_color_dual ,  edge_cmap =  cmap_dict[attr], width = 2.5, alpha = .5,
             ax = ax)
    
    # add colorbar:
    cbar_ax = fig.add_axes([0.2, .1, .6, 0.02])

    cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', 
                                   cmap= cmap_dict[attr],
                                   norm=mpl.colors.Normalize(np.array(edge_color_dual).min(), 
                                                             np.array(edge_color_dual).max()),
                                   label=f'{attr} difference to ideal')
    
    
    plt.show()   

    return
    
def plot_random_rounds(mean_angle_error, mean_dist_error, rst_summary):
    
    fig, ax = plt.subplots(nrows = 2, figsize = (8,12))

    sns.histplot(mean_angle_error, kde=True, ax = ax[0], color = "C5")
    ax[0].set_title('intersection angle', fontsize = 14)
    ax[0].axvline(x =  rst_summary.iloc[0][0], c = "C1")
    ax[0].axvline(x = np.array(mean_angle_error).mean(), c = "C5")

    ax[0].set_xlim([0, 50])

    sns.histplot(mean_dist_error,  kde=True, ax = ax[1])
    ax[1].set_title('percent distance difference', fontsize = 14)
    ax[1].axvline(x = rst_summary.iloc[0][2], c = 'C1')
    ax[1].axvline(x = np.array(mean_dist_error).mean(), c = 'C0')

    ax[1].set_xlim([0, 1])
    fig.suptitle("mean error of dots v.s. 1000 rounds of random points", fontsize = 16)
    
    plt.show()
    
    return

def plot_dist(df, test = 'angle'):

    fig, ax = plt.subplots(nrows =2, figsize = (8, 12))
    
    sns.histplot(df, x = f'{test}_diff', hue = 'type', kde = True, ax = ax[0])

    ax[1] = sns.violinplot(x = f'{test}_diff', y = 'type' , 
                            data = df, inner = 'quartile')
    for l in ax[1].lines:
        l.set_linestyle('--')
        l.set_linewidth(1)
        l.set_color('brown')
        l.set_alpha(0.8)
    for l in ax[1].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    
    fig.suptitle(f'{test} difference distribution', fontsize = 16)

    #fig.tight_layout() # no they overlaps

    plt.show()
    return

def plot_voronoi(G, vor):
    '''
    '''
    node_positions = {}
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1'}
    node_color = []
    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])

    fig , ax = plt.subplots(figsize=(9,9/G.graph['ratio']))
    nx.draw(G, pos = node_positions, node_size= 20, node_color= node_color, ax = ax) 
    voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.6, point_size=2, ax = ax)
    plt.show()
    return 

def plot_vor_regions(G, seeds, single_dot, bounded_regions, dot_type = 'dots'):
    
    dot_color = {'dots':'C1', 'centroid':'red','midpoint': 'hotpink', 'random':'purple'}

    fig, ax = plt.subplots(figsize = (8, 8/G.graph['ratio']))

    ax.scatter(np.array(seeds)[:,0], np.array(seeds)[:,1], s = 10, c = 'C7')

    for i in range(len(bounded_regions)):
        ax.plot(np.array(bounded_regions[i])[:,0], np.array(bounded_regions[i])[:,1], alpha = .7)
        if single_dot[i]:
            p = mpl.patches.Polygon(bounded_regions[i], facecolor = dot_color[dot_type], alpha = .2)
            ax.add_patch(p)
            ax.scatter(seeds[i][0], seeds[i][1], s = 10, c = dot_color[dot_type])

    node_positions = {}
    
    for node in G.nodes:
        node_positions[node] = node
   
    nx.draw_networkx_edges(G, pos=node_positions, edge_color = 'C7', ax = ax, width = 1.5, alpha = .8) 

    ax.set_title(f'Voronoi regions by {dot_type}')      
    
    plt.show()

    return


def plot_subdual(G, G_dual,  attr = "angle"):

    node_position_G = {}
        
    node_position_dual = {}
    compared_position_dual = {}

    edge_color_dual = []
    edge_color_comp_dual = []

    for node in G.nodes:
        node_position_G[node] = node

    edge_style = ['solid' if G.edges[e]['shared'] =='tested_shared' else 'dashed' for e in G.edges]
    edge_col = ['black' if G.edges[e]['shared'] =='tested_shared' else 'C7' for e in G.edges]

    for node in G_dual.nodes:
        node_position_dual[node] = node
        compared_position_dual[node] = (G_dual.nodes[node]['centroid'][0], G_dual.nodes[node]['centroid'][1])

        
    for edge in G_dual.edges:
        edge_color_dual.append(G_dual.edges[edge][attr])
        edge_color_comp_dual.append(G_dual.edges[edge][f'centroid_{attr}'])
    selected_nodes = [n for n,v in G.nodes(data=True) if v['type'] == 'vein']  

    cmap_dict = {"angle":plt.cm.viridis, "dist":plt.cm.magma}

    fig, ax = plt.subplots(figsize=(10,10/G.graph['ratio']))

    for i in range(len(G.graph['faces_passed'])):
        p = mpl.patches.Polygon(G.graph['faces_passed'][i], facecolor = 'C7', alpha = .1)
        ax.add_patch(p)

    nx.draw_networkx_edges(G, pos=node_position_G, edge_color = edge_col, style = edge_style, ax = ax) 
    nx.draw_networkx_nodes(G, pos=node_position_G, 
                            nodelist = selected_nodes, node_size= 5, node_color = 'C7', ax = ax) 
    nx.draw(G_dual, pos=node_position_dual, node_size= 20,  node_color= 'C1', 
            edge_color = edge_color_dual ,  edge_cmap =  cmap_dict[attr], width = 2, alpha = .9,
                ax = ax)
    
    nx.draw(G_dual, pos=compared_position_dual, node_size= 20,  node_color= 'purple', 
            edge_color = edge_color_comp_dual ,  edge_cmap =  cmap_dict[attr], width = 2, alpha = .9,
                ax = ax)


    # add colorbar:
    cbar_ax = fig.add_axes([0.2, .1, .6, 0.02])

    cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', 
                                    cmap= cmap_dict[attr],
                                    norm=mpl.colors.Normalize(np.array(edge_color_dual).min(), 
                                                                np.array(edge_color_dual).max()),
                                    label=f'{attr} difference to ideal')

    ax.set_title(f'{attr} 10 bad apples', fontsize = 16)
    
    plt.show()   


def plot_testable(G, G_dual):

    'shade the testable polygons, can use after we perform the local test.'

    node_position_G = {}
        
    node_position_dual = {}


    for node in G.nodes:
        node_position_G[node] = node

    edge_style = ['solid' if G.edges[e]['shared'] =='tested_shared' else 'dashed' for e in G.edges]
    edge_col = ['black' if G.edges[e]['shared'] =='tested_shared' else 'C7' for e in G.edges]

    for node in G_dual.nodes:
        node_position_dual[node] = node

    fig, ax = plt.subplots(figsize=(10,10/G.graph['ratio']))

    for i in range(len(G.graph['faces_passed'])):
        p = mpl.patches.Polygon(G.graph['faces_passed'][i], facecolor = 'C7', alpha = .1)
        ax.add_patch(p)

    nx.draw_networkx_edges(G, pos=node_position_G, edge_color = edge_col, style = edge_style, ax = ax) 

    nx.draw(G_dual, pos=node_position_dual, node_size= 20,  node_color= 'seagreen', 
            edge_color ='seagreen',  width = 1, ax = ax)


    plt.show()   


def plot_overlap_diff(G, bounded_regions, seeds, single_dot, type = 'dot'):

    'this is the new function for visualizing similarity test results'

    fig, ax = plt.subplots(figsize = (8, 8/G.graph['ratio']))

    node_positions = {}

    for node in G.nodes:
        node_positions[node] = node

    dot_color = {'dot':'C1', 'centroid':'red','midpoint': 'hotpink', 'random':'purple'}
    color = dot_color[type]

    # plot veins:     
    nx.draw_networkx_edges(G, pos=node_positions, edge_color = 'C7', ax = ax, width = 1.5, alpha = .8) 

    # plot voronoi polygons:
    for i in range(len(bounded_regions)):
        ax.plot(np.array(bounded_regions[i])[:,0], np.array(bounded_regions[i])[:,1], alpha = .7, color = 'C7') # change color to grey

    # plot diff patches:
    for geom in G.graph[f'diff_geom_{type}']:  

        if isinstance(geom, shapely.geometry.polygon.Polygon):
            contour = list(geom.exterior.coords)
           
            if len(contour):
                p = mpl.patches.Polygon(contour, facecolor = color, alpha = .3)
                ax.add_patch(p)
                
            if len(list(geom.interiors)):
                interior = list(geom.interiors[0].coords)
                p_in = mpl.patches.Polygon(contour, facecolor = 'white', alpha = 1)
                ax.add_patch(p_in)
                
        else: # multipolygon.MultiPolygon
            for ploy in geom:
                contour = list(ploy.exterior.coords)
                if len(contour):
                    p = mpl.patches.Polygon(contour, facecolor = color, alpha = .3)
                    ax.add_patch(p)

                    
                if len(list(ploy.interiors)):
                    interior = list(ploy.interiors[0].coords)
                    p_in = mpl.patches.Polygon(interior, facecolor = 'white', alpha = 1)
                    ax.add_patch(p_in)
    
    # plot seeds:
    ax.scatter(np.array(seeds)[:,0], np.array(seeds)[:,1], s = 10, c = 'C7')

    for i in range(len(bounded_regions)):
        if single_dot[i]:
            ax.scatter(seeds[i][0], seeds[i][1], s = 10, c = dot_color[type])


    plt.show()