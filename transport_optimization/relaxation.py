import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

K = 1
gamma = 0.5 

def parameterize_leaf(N):
    
    'add transport netowrk variables'

    G = nx.generators.triangular_lattice_graph(N,2*N)
    
    for i in range(0,int(N/2)-1):
        G.remove_nodes_from([(i,N/2-2*i-1),(i,N/2-2*i-2)])
        G.remove_nodes_from([(i,N/2+2*i+1),(i,N/2+2*i+2)])
    for j in range(0,int(N/2)-2):
        G.remove_nodes_from([(N-j,N/2-2*j-2),(N-j,N/2-2*j-3)])
        G.remove_nodes_from([(N-j,N/2+2*j+2),(N-j,N/2+2*j+3)])
    # keep the largest component:
    main_comp = np.max([len(comp) for comp in list(nx.connected_components(G))])
    for comp in list(nx.connected_components(G)):
        if len(comp)!=main_comp:
            G.remove_nodes_from(comp)

    # add edge attr:
    nx.set_node_attributes(G, -1, 'P') # set node strength
    G.nodes[(0,3)]['P'] = G.number_of_nodes() - 1 # set supply node strength
    nx.set_node_attributes(G, 0, 'theta') # initialize potentials, not important, will be the first thing to calculate
    nx.set_node_attributes(G, 'C0', 'color')

    # add node attr:
    nx.set_edge_attributes(G, 1/(G.number_of_nodes()**(1/gamma)), 'weight') # set initial capacity dist. to be uniform, use 'weight' instead of 'k' for easy access of laplacian matrix
    nx.set_edge_attributes(G, 0, 'F') # let's assume that flow is always a positive number here


    return G


def parameterize_leaf_with_hydathode(N, strength = 2, hydathode = (2,1)):

    'add one node as hydathode that is a stronger sink, we keep the supply consistant'
    
    G = nx.generators.triangular_lattice_graph(N,2*N)
    
    for i in range(0,int(N/2)-1):
        G.remove_nodes_from([(i,N/2-2*i-1),(i,N/2-2*i-2)])
        G.remove_nodes_from([(i,N/2+2*i+1),(i,N/2+2*i+2)])
    for j in range(0,int(N/2)-2):
        G.remove_nodes_from([(N-j,N/2-2*j-2),(N-j,N/2-2*j-3)])
        G.remove_nodes_from([(N-j,N/2+2*j+2),(N-j,N/2+2*j+3)])
    # keep the largest component:
    main_comp = np.max([len(comp) for comp in list(nx.connected_components(G))])
    for comp in list(nx.connected_components(G)):
        if len(comp)!=main_comp:
            G.remove_nodes_from(comp)

    # add edge attr, renormalize sinks: 
    re_norm = (G.number_of_nodes() - 1)/(G.number_of_nodes() - 2 + strength) 
    nx.set_node_attributes(G, -1*re_norm, 'P') # set node strength
    G.nodes[(0,3)]['P'] = G.number_of_nodes() - 1 # keep supply consistent
    G.nodes[hydathode]['P'] = -strength*re_norm
    
    nx.set_node_attributes(G, 0, 'theta')

    nx.set_node_attributes(G, 'C0', 'color')
    G.nodes[hydathode]['color'] = 'C1'

    # add node attr:
    nx.set_edge_attributes(G, 1/(G.number_of_nodes()**(1/gamma)), 'weight') # uniform capacity 
    nx.set_edge_attributes(G, 0, 'F') # let's assume that flow is always a positive number here


    return G

def plot_network(mesh):
    '''
    plot network with edge width proportional to sqrt of capacity, which can be seen as proportional to the diameter
    highlight hydathode color
    '''

    node_positions = {}
        
    for node in mesh.nodes:
        node_positions[node] = mesh.nodes[node]['pos']

    diam_list = [np.sqrt(mesh[u][v]['weight'])*100 for u,v in mesh.edges()]
    
    color_list =[ mesh.nodes[node]['color'] for node in mesh.nodes()]

    fig, ax = plt.subplots(figsize=(8,7))

    nx.draw(mesh, 
            with_labels = True,
            pos=node_positions, 
            node_size= 30,  
            node_color = color_list,
            width=diam_list,
            ax = ax) 

    plt.tight_layout()
    plt.show()   


def plot_pressure(mesh):
    '''
    plot network with edge width proportional to sqrt of capacity, which can be seen as proportional to the diameter
    highlight hydathode color
    '''
    plt.rcParams['text.usetex'] = True

    node_positions = {}
    node_color = []
    for node in mesh.nodes:
        node_positions[node] = mesh.nodes[node]['pos']
        node_color.append(mesh.nodes[node]['theta'])

    edge_color = []
    for edge in mesh.edges:
        edge_color.append(abs(mesh.nodes[edge[0]]['theta']- mesh.nodes[edge[1]]['theta']))

    fig, ax = plt.subplots(figsize=(8,7))

    nx.draw(mesh, 
            pos=node_positions,       
            node_color = node_color,
            cmap = plt.cm.viridis,
            edge_color =  edge_color,  
            edge_cmap  = plt.cm.Blues,
            node_size= 100,  
            edge_width = 10,
            ax = ax) 

    cbar_ax = fig.add_axes([.2, 0, .6, 0.02])

    cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', 
                                   cmap=plt.cm.viridis,
                                   norm=mpl.colors.Normalize(np.array(node_color).min(), 
                                                             np.array(node_color).max()),
                                   label = 'theta (potential) at each node')
                                   
    plt.show()   


def randomized_k(G):
    
    'start with a different capacity distribution'

    rand_list = np.random.randint(0,10000, G.number_of_edges())
    norm_list = rand_list/rand_list.sum()
    k_list = norm_list**(1/gamma)
    nx.set_edge_attributes(G, dict(zip(list(G.edges),k_list)), 'weight')


def solve_P_theta(G):
    
    L = nx.laplacian_matrix(G, nodelist=None, weight='weight')
    P_vec = [G.nodes[node]['P'] for node in G.nodes]
    theta_vec = np.linalg.solve(L.todense(), P_vec)
    nx.set_node_attributes(G, dict(zip(list(G.nodes),theta_vec)), 'theta')

    # positive flow: the direction is always toward the node with lower potential: 
    for e in G.edges:
        G[e[0]][e[1]]['F'] =  abs(G[e[0]][e[1]]['weight']*(G.nodes[e[0]]['theta'] - G.nodes[e[1]]['theta']))


# set a threshold to remove the edge: 
weight_lower = 1e-20 

def new_k(G):
    
    F_vec = np.array([G[e[0]][e[1]]['F']  for e in G.edges])
    F_scaling_sum = (((F_vec**2)**(gamma/(1+gamma))).sum())**(1/gamma)

    for e in G.edges:
        weight = (G[e[0]][e[1]]['F']**2)**(1/(1+gamma)) / F_scaling_sum
        if weight < weight_lower:
            G.remove_edge(*e)
        else:
            G[e[0]][e[1]]['weight'] = weight 

    # calculate dissipation:
    weight_vec = np.array([G[e[0]][e[1]]['weight']  for e in G.edges])
    F_vec = np.array([G[e[0]][e[1]]['F']  for e in G.edges]) # need to redefine here, since length might change due to edge drop

    G.graph['D'] = np.dot(F_vec**2, 1/weight_vec)
    

