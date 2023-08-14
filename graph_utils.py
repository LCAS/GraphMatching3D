import numpy as np

# graph utils

def normalize_edge(edge):
    n1, n2,  = edge
    if n1 > n2: # use a custom compare function if desired
        n1, n2 = n2, n1
    return (n1, n2)

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

def find_leaf_nodes(graph):
    leaf_nodes = []
    for node in graph:
        if len(graph[node]) == 1:
            leaf_nodes.append(node)
    return leaf_nodes

def dfs_paths(graph, start_node, end_node):
        stack = [(start_node, [start_node])]
        paths = []

        while stack:
            node, path = stack.pop()

            if node == end_node:
                paths.append(path)

            for neighbor in graph[node]:
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

        return paths

def find_all_paths_between_leaf_nodes(graph):
    """ Returns a dictionary with keys as tuples of the leaf nodes and values
    as a list of the path to traverse from the one to the other"""

    all_paths = {}
    # split into connected sections
    contig_sections = find_contiguous_sections(graph, list(graph.keys()))   
    
    for j in range(0,len(contig_sections)):
        subgraph = {}
        for i in contig_sections[j]:      
            subgraph[i] = graph[i]   
        
        # print(subgraph)
        
        leaf_nodes = find_leaf_nodes(subgraph)       
      
        for start_node in leaf_nodes:        
            for end_node in leaf_nodes:
                if start_node != end_node:         
                    path = list(dfs_paths(subgraph, start_node, end_node))                     
                    # print(start_node,end_node,path)                              
                    all_paths[(start_node, end_node)] = path

    # remove entries that are the same in reverse
    pairs_list = list(all_paths.keys())

    # Create a set to store unique pairs
    unique_pairs = set()

    # Iterate through the list of tuples and add them to the set (order matters)
    for pair in pairs_list:
        if pair not in unique_pairs and (pair[1], pair[0]) not in unique_pairs:
            unique_pairs.add(pair)

    # Convert the set back to a list of tuples
    unique_pairs_list = list(unique_pairs)
    
    filtered_graph = {}
    for k in unique_pairs_list:
        filtered_graph[k] = all_paths[k]  
    
  
    return {k: v[0] for k, v in filtered_graph.items() if len(v) != 0}

def sum_path(graph, adj_matrix, start, end):
    dist = 0

    path = list(dfs_paths(graph,start,end))[0]
    path.append(end)    

    for i in range(0,len(path)-1):
        e0 =path[i]
        e1 =path[i+1] 
        dist += adj_matrix[e0][e1]

    return dist

    
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

def split_into_fragments(graph):
    num_nodes = len(graph)
    visited = [False] * num_nodes
    components = []

    def dfs(graph, node, visited, component):
        visited[node] = True
        component.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(graph, neighbor, visited, component)

    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(graph, node, visited, component)
            components.append(component)

    sub_graphs = []
    for fragment in components:
        g = {}
        
        for i in fragment:
            g[i] = graph[i]
        
        sub_graphs.append(g)

    return sub_graphs

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
