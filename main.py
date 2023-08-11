import numpy as np
import math
from scipy.optimize import linear_sum_assignment

# general utils
def distance(nodes, edges):
    e0, e1 = edges
    p0 = nodes[e0]
    p1 = nodes[e1]
    return math.dist(p0, p1)

def magnitude(vector):
    # return math.sqrt(sum(pow(element, 2) for element in vector))
    return np.linalg.norm(vector)

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

def find_branch_nodes(graph):
    branch_nodes = []
    for node in graph:
        if len(graph[node]) > 2:
            branch_nodes.append(node)
    return branch_nodes

def create_graph_from_adjacency_matrix(adj_matrix):
    graph = {}
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        neighbors = []
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                neighbors.append(j)
        graph[i] = neighbors

    return graph

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
    
def find_contiguous(graph, start_node, true_positive_nodes, visited):
    section = set()  # To store the current contiguous section
    stack = [start_node]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            section.add(node)
            stack.extend(neighbor for neighbor in graph.get(node, [])
                         if neighbor in true_positive_nodes)
    
    return list(section)

def find_contiguous_sections(graph, nodes):
    visited = set()  # To keep track of visited nodes
    contiguous_sections = []

    for node in nodes:
        if node not in visited:
            section = find_contiguous(graph, node, nodes, visited)            
            contiguous_sections.append(section)    
    
    return contiguous_sections

def split_into_branches(graph):
    branch_nodes = find_branch_nodes(graph)
    branches = None
    for this_branch in branch_nodes:  
        connections = graph[this_branch] #  follow each connection until it is a leaf or a branch node
        
        branches = []
        visited = set([])

        def branch_dfs(node, branch):
            visited.add(node)
            branch.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited and neighbor not in branch_nodes:
                    branch_dfs(neighbor, branch)
                if neighbor in branch_nodes:
                    branch.append(neighbor)

        for node in connections:
            branch = []
            branch_dfs(node, branch)              
            branches.append(branch)

    # if no branches
    if branches:
        # order branches
        sorted_branches = []
        for branch in branches:
            c = find_contiguous_sections(graph, branch)
            sorted_branches.append(c[0])
    
        return sorted_branches
    else:
        section = find_contiguous_sections(graph, list(graph.keys()))      
        return section

def match_polyline_graphs(graph1, graph2, nodes1, nodes2, thresh, line_dist_thresh = 0.25):
    """
    Matches graph2 against graph 1 (GT)
    Returns a dictionary of the matched indices between the two graphs
    """
    match_dict = {}
    keys = graph1.keys()   

    # find lines between the nodes in the gt
    line_segments = [] 
     
    contiguous = split_into_branches(graph1)
    
    # print(contiguous)
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

    cost_matrix = np.ones((len(graph1.keys()), len(graph2.keys()))) * 1000
    for k1 in graph1.keys():        
        for k2 in graph2.keys():         
            d = np.linalg.norm(nodes1[k1] - nodes2[k2])            
            if d < thresh:
                cost_matrix[k1][k2] = d         
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False) # what if there are unmatched ones, they need to come out as -1
          
    for i in zip(row_ind,col_ind):
        k,v = i
        match_dict[k] = v 
        
    for k1 in graph1.keys():        
        if len(np.unique(cost_matrix[k1,:])) == 1 or k1 not in match_dict.keys():
            # unmatched
            match_dict[k1] = -1 
            
    for k in keys:
        if k not in match_dict.keys():
            print(k)
    
              
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