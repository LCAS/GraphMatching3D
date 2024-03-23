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

def point_between_points(P, A, B):
    # Convert points to numpy arrays
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    # Calculate vectors AB and AP
    vector_AB = B - A
    vector_AP = P - A

    # Calculate dot product
    dot_product = np.dot(vector_AB, vector_AP)

    # Check if point lies between A and B
    if 0 <= dot_product <= np.dot(vector_AB, vector_AB):
        return True
    else:
        return False

def point_near_line_segment(P,A,B,t):  
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    # Calculate the vectors
    vector_AB = B - A
    vector_AP = P - A
   
    # Calculate the projection P onto vector A   
    scalar_projection = np.dot(vector_AB, vector_AP) / np.dot(vector_AB, vector_AB)

    # Calculate projection of P onto AB
    projection_P = A + scalar_projection * vector_AB
            
    # check if the projected point is between A and B
    if not point_between_points(projection_P, A, B):
        return False      
    
    # Calculate the vector from C to P
    vector_CP = P-projection_P   

    # Calculate the distance from C to P
    distance_CP = magnitude(vector_CP)     

    return distance_CP <= t

def match_polyline_graphs(graph1, graph2, nodes1, nodes2, thresh, line_dist_thresh = 0.25):
    """
    Matches graph2 against graph 1 (GT)
    Returns a dictionary of the 1-1 matched indices between the two graphs
    """
    match_dict = {}
    # keys = graph1.keys()         
     
    # PART 1: compute the cost matrix and find the best 1-1 fit 
    cost_matrix = np.ones((len(graph1.keys()), len(graph2.keys()))) * 1000
    for k1 in graph1.keys():        
        for k2 in graph2.keys():         
            d = np.linalg.norm(nodes1[k1] - nodes2[k2])            
            if d <= thresh:
                cost_matrix[k1][k2] = d 

    # Hungarian            
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False) # what if there are unmatched ones, they need to come out as -1

    for i in zip(row_ind,col_ind):
        k,v = i
        match_dict[k] = v 
        
    for k1 in graph1.keys():        
        if len(np.unique(cost_matrix[k1,:])) == 1 or k1 not in match_dict.keys():
            # unmatched
            match_dict[k1] = -1     
    
    # PART 2
    def filter_dict_by_value(d):
        return {key: value for key, value in d.items() if value != -1} 
    
    tp_dict = filter_dict_by_value(match_dict)
    tp = list(tp_dict.keys())
            
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

            if index_i in tp and index_i1 in tp: # if it is between positive matches
                line_segments.append((A,B))           
    
    g2_tp = []
    for i in list(graph2.keys()):
        for A,B in line_segments:
            if point_near_line_segment(nodes2[i], A, B, line_dist_thresh): # distance from polyline ####
                g2_tp.append(i)

    for k1 in graph1.keys(): 
        if len(np.unique(cost_matrix[k1,:])) != 1 and match_dict[k1] not in g2_tp and match_dict[k1] != -1: #it is matched             
            g2_tp.append(match_dict[k1])
            # print(k1,match_dict[k1])
    
    # check if a G node is within t_line of a line segment of E ****true positves****
    # -- get contiguous segments of E
    contiguous_e = []
    line_segments_e = []
    
    e_frag = split_into_fragments(graph2)  # 8 fragments of graph 2 (estimate)
    
    for frag in e_frag: 
        b =  split_into_branches(frag)
        contiguous_e.extend(b)
            
    for i in range(0,len(contiguous_e)):  # 8 contiguous sections
        contig = contiguous_e[i]
       
        for p in range(0,len(contig)-1):
            index_i = contig[p]
            index_i1 = contig[p+1]            

            A = nodes2[index_i]
            B = nodes2[index_i1]

            if index_i in g2_tp and index_i1 in g2_tp: # ***should be TP segments*** 
                line_segments_e.append((A,B))  

    # -- find elements that fall within t_line
    g1_tp_line = [] # these are not matched with hungarian but    
    for i in list(graph1.keys()):
        for A,B in line_segments_e:
            criteria = point_near_line_segment(nodes1[i], A, B, line_dist_thresh)             
            if criteria and i not in g1_tp_line: # distance from polyline
                g1_tp_line.append(i)           
    
    for k1 in graph1.keys():        
        if match_dict[k1] == -1:
            if k1 in g1_tp_line:
                match_dict[k1] = -2 # matched but not to a node 
                   
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
       
    end_pairs = all_paths.keys()
  
    unique_edges = list(end_pairs)
    # print(list(set(end_pairs)))
    # unique_edges = list(set(map(normalize_edge, end_pairs)))
    # print(list(set(unique_edges)))  

    est_dists = [sum_path(graph, adj_matrix, e0, e1) for e0, e1 in unique_edges]    
    dist_err = np.array([np.abs(gt_dist - d) for d in est_dists])    

    i_min = np.argmin(dist_err)     

   
    # try:
    matching_segment_path = all_paths[unique_edges[i_min]]   
    # except:
    #     inverted = unique_edges[i_min][1],unique_edges[i_min][0]       
    #     matching_segment_path = all_paths[inverted]


    return est_dists[i_min], matching_segment_path
