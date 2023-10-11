import numpy as np
import math
from scipy.optimize import linear_sum_assignment

from .utils import *
from .graph_utils import *

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
    """Create a dense graph by breaking up any segments that are longer than l."""

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

# --- match polyline

def point_near_line_segment(P,A,B,t):
    x, y, z = P
    x1, y1, z1 = A
    x2, y2, z2 = B

    # Calculate the vectors
    vector_AB = (x2 - x1, y2 - y1, z2 - z1)
    vector_AC = (x - x1, y - y1, z - z1)

    # Calculate the projection P onto vector AB
    dot_AB_AB = np.dot(vector_AB, vector_AB)
    dot_AC_AB = np.dot(vector_AC, vector_AB)
    scalar = dot_AC_AB / dot_AB_AB
    projection_P = (vector_AB[0] * scalar, vector_AB[1] * scalar, vector_AB[2] * scalar)

    # Calculate the vector from C to P
    vector_CP = (vector_AC[0] - projection_P[0], vector_AC[1] - projection_P[1], vector_AC[2] - projection_P[2])

    # Calculate the distance from C to P
    distance_CP = magnitude(vector_CP)

    return distance_CP <= t

def match_polyline_graphs(graph1, graph2, nodes1, nodes2, thresh, line_dist_thresh = 0.25):
    """
    Matches graph2 against graph 1 (GT)
    Returns a dictionary of the 1-1 matched indices between the two graphs
    """
    match_dict = {}
    keys = graph1.keys()   

    # PART 1: find lines between the nodes in the gt
    line_segments = [] 
     
    # identify contiguous sections and check if there are points on g2 that match g1 near these lines
    # these are tp but not tp that we count (so we can compare again the same number of tp across 
    # the diff algs)
    contiguous = []
    g_frag = split_into_fragments(graph1)   

    for frag in g_frag: 
        b =  split_into_branches(frag)
        contiguous.extend(b)
        
    for i in range(0,len(contiguous)):  
        contig = contiguous[i]
       
        for p in range(0,len(contig)-1):
            index_i = contig[p]
            index_i1 = contig[p+1]            

            A = nodes1[index_i]
            B = nodes1[index_i1]

            line_segments.append((A,B))           
    
    g2_tp = []
    for i in list(graph2.keys()):
        for A,B in line_segments:
            if point_near_line_segment(nodes2[i], A, B, line_dist_thresh): # distance from polyline
                g2_tp.append(i)

    
    # PART 2: compute the cost matrix and find the best 1-1 fit 
    cost_matrix = np.ones((len(graph1.keys()), len(graph2.keys()))) * 1000
    for k1 in graph1.keys():        
        for k2 in graph2.keys():         
            d = np.linalg.norm(nodes1[k1] - nodes2[k2])            
            if d <= thresh:
                cost_matrix[k1][k2] = d 

          
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False) # what if there are unmatched ones, they need to come out as -1

    for i in zip(row_ind,col_ind):
        k,v = i
        match_dict[k] = v 
        
    for k1 in graph1.keys():        
        if len(np.unique(cost_matrix[k1,:])) == 1 or k1 not in match_dict.keys():
            # unmatched
            match_dict[k1] = -1          
                   
    return match_dict, list(set(g2_tp))

# --- confusion matrix

def corresponding_tp(match_dict):  
    g2_matches = np.array([match_dict[k] for k in match_dict.keys()])    
    g2_matches = g2_matches[g2_matches != -1]

    return g2_matches

def find_unmatched(match_dict, graph2):
    
    g2_matches = corresponding_tp(match_dict) 
    g2_unmatched = graph2.keys()- g2_matches           
        
    return list(g2_unmatched)

def confusion_matrix(match_dict, g2): 
    """NB: If you are using match_polyline_graph then the fp value here
    might contain some tp so should recalcuate as fp = list(set(g2.keys()) - set(tp_e))
    where tp_e are returned by match_polyline_graph"""  
    tp = []
    fn = []
 
    for k in match_dict.keys():
        if match_dict[k] != -1:
            tp.append(k)
        else:
            fn.append(k)

    fp = find_unmatched(match_dict, g2)
   
    return tp,fn, fp

#  --- find length matched path

def find_length_matched_path(graph, adj_matrix, gt_dist):
    """Given a graph, find the leaf-leaf length that is closest to a reference"""
   
    all_paths = find_all_paths_between_leaf_nodes(
        graph
    )  # (edgepair) : [[edge0, edge1...]]

    # print(all_paths.keys()) # normalise keys
    # exit()

    
    end_pairs = all_paths.keys()
    # print(end_pairs)
  
    unique_edges = list(set(map(normalize_edge, end_pairs)))
    # print(unique_edges)   

    est_dists = [sum_path(graph, adj_matrix, e0, e1) for e0, e1 in unique_edges]    
    dist_err = np.array([np.abs(gt_dist - d) for d in est_dists])    

    i_min = np.argmin(dist_err) 
    # print(i_min) 
    
    # print(unique_edges[i_min])
    # print(all_paths.keys())
    # exit()

   
    try:
        matching_segment_path = all_paths[unique_edges[i_min]]   
    except:
        inverted = unique_edges[i_min][1],unique_edges[i_min][0]       
        matching_segment_path = all_paths[inverted]


    return est_dists[i_min], matching_segment_path