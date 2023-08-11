import numpy as np
import math

# general utils
def distance(nodes, edges):
    e0, e1 = edges
    p0 = nodes[e0]
    p1 = nodes[e1]
    return math.dist(p0, p1)

# graph utils
def create_adjacency_matrix(edges, nodes):
    # Initialize an empty adjacency matrix
    num_vertices = len(nodes)
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=float)

    # Populate the adjacency matrix based on the edges
    for edge in edges:
        src, dest = edge
        d = np.linalg.norm(nodes[src]-nodes[dest])               
        adj_matrix[src][dest] = d
        adj_matrix[dest][src] = d  # If the graph is undirected, add this line as well

    return adj_matrix

def edges_from_matrix(adj_matrix):
    new_edges = []
    for row in range(0,len(adj_matrix)):
        for col in range(row+1,len(adj_matrix)):
            if adj_matrix[row,col] != 0:
                new_edges.append([row,col])
    return new_edges



# functions

# -- create dense
def split_edge(n0,n1,edges,s, unique_edges):   
    w = n1-n0
    length = np.linalg.norm(n0 - n1)   
      
    # Calculate the number of new nodes needed
    n_steps = math.ceil(length / s) - 1     
    
    # Calculate the increment to add to each coordinate for each new node
    increment_x = w[0] / (n_steps + 1)
    increment_y = w[1] / (n_steps + 1)
    increment_z = w[2] / (n_steps + 1)
    
    # Calculate the new nodes
    new_nodes = []
    for i in range(1, n_steps+1):
        new_node = (
            n0[0] + i * increment_x,
            n0[1] + i * increment_y,
            n0[2] + i * increment_z
        )       
        new_nodes.append(new_node)
    
    new_edges = np.array([[unique_edges[-1]+1*i,unique_edges[-1]+1*i+1] for i in range(0,n_steps+1)]) 
    new_edges[0][0] = edges[0]
    new_edges[-1][1] = edges[1]

    new_nodes = np.array(new_nodes)  
  
    return new_nodes, new_edges

def update_adjacency_matrix(old_matrix, s, new_edges, old_edges):
    num_vertices = len(old_matrix) - 1 + len(new_edges) 
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=float) 
    adj_matrix[0:len(old_matrix),0:len(old_matrix)] = old_matrix
    
    for edge in new_edges:                                                                                                                                                                                                                                                                
        src, dest = edge          
        adj_matrix[src][dest] = s
        adj_matrix[dest][src] = s  # If the graph is undirected, add this line as well

    # set irrelevant old edge to zero
    src, dest = old_edges         
    adj_matrix[src][dest] = 0
    adj_matrix[dest][src] = 0

  
    return adj_matrix

def create_dense(nodes, edges, s = 0.5, return_adj=False):

    lengths = np.array([distance(nodes, e) for e in edges])
    
    dense_nodes = nodes.copy()
    dense_edges = None    
    
    current_edges = edges.copy()
    adj_matrix = create_adjacency_matrix(edges, nodes)  
   
    for i,l in enumerate(lengths):          
        if l > s:    
            # print(edges[i])
            new_nodes, new_edges = split_edge(nodes[edges[i][0]], nodes[edges[i][1]], edges[i], s, np.unique(current_edges))                   
                  
            adj_matrix = update_adjacency_matrix(adj_matrix, s, new_edges, edges[i])   
            current_edges = edges_from_matrix(adj_matrix)

            dense_nodes = np.concatenate((dense_nodes,new_nodes))     

    dense_edges = edges_from_matrix(adj_matrix) 

    if return_adj:
        return np.array(dense_nodes), np.array(dense_edges), adj_matrix

    return np.array(dense_nodes), np.array(dense_edges)

# ---