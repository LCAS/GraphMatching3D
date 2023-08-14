import numpy as np

from main import create_graph_from_adjacency_matrix, create_adjacency_matrix

# tested functions
from main import (confusion_matrix, corresponding_tp, create_dense,
                  find_all_paths_between_leaf_nodes, find_contiguous_sections,
                  find_length_matched_path, match_polyline_graphs,
                  split_into_branches, split_into_fragments)


# test dense 
def test_create_dense():
    a = [0,0,0]
    b = [2,0,0]
    c = [4,0,0]

    nodes = np.array([a,b,c])
    edges = np.array([[0,1],[1,2]])

    l = 1

    dense_nodes, dense_edges = create_dense(nodes,edges,l)

    expected = [[0, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [1, 0, 0],
                [3, 0, 0]]

    assert np.array_equal(dense_nodes,expected)  
    assert np.array_equal(dense_edges, [[0, 3],[1, 3],[1, 4],[2, 4]])

def test_create_dense2():
    a = [0,0,0]
    b = [1.5,0,0]
    c = [4,0,0]
    d = [8,0,0]

    nodes = np.array([a,b,c,d])
    edges = np.array([[0,1],[1,2],[2,3]])

    l = 1

    dense_nodes, dense_edges = create_dense(nodes,edges,l)

    expected = [[0, 0, 0],
                [1.5, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [0.75, 0, 0],   # from 0:1.5
                [2.3333, 0, 0],  # from 1.5:4 -> d = 4-1.5 = (d) 2.5 / l = 2.5 => 2 steps, d/(steps+1) = increment => 2.5/3 => 0.833 + previous step
                [3.1666, 0, 0],  # from 1.5:4. 2.333+0.833
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0,]]

    
    for i,sub in enumerate(dense_nodes):
        # print(sub, expected[i])
        assert np.allclose(sub,expected[i],atol=1e-4)  
    assert np.array_equal(dense_edges, [[0, 4],
                                        [1, 4],
                                        [1, 5],
                                        [2, 6],
                                        [2, 7],
                                        [3, 9],
                                        [5, 6],
                                        [7, 8],
                                        [8, 9],])

# test match

def test_match_graphs():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes = np.array([node0, node1, node2, node3, node4])
    edges = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )
    g = create_graph_from_adjacency_matrix(adj_matrix)
    match_dict,_ = match_polyline_graphs(g, g, nodes, nodes, 1)

    assert match_dict[0] == 0
    assert match_dict[1] == 1
    assert match_dict[2] == 2
    assert match_dict[3] == 3
    assert match_dict[4] == 4

def test_match_graphs2():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    nodes2 = np.array([node0, node1, node2, node3, node4])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)

    assert match_dict[0] == 0
    assert match_dict[1] == 1
    assert match_dict[2] == 2
    assert match_dict[3] == 3
    assert match_dict[4] == -1

def test_match_reversed():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    node5 = [20, 25, 20]
    nodes2 = np.array([node5, node1, node2, node3, node4, node0])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 3, 0, 5, 1],
            [0, 3, 0, 6, 0, 0],
            [0, 0, 6, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)

    assert match_dict[0] == 5, match_dict[0]
    assert match_dict[1] == 1, match_dict[1]
    assert match_dict[2] == 2, match_dict[2]
    assert match_dict[3] == 3, match_dict[3]
    assert match_dict[4] == -1, match_dict[4]

def test_match_graphs_diff_density():
    node0 = [0,0,1]
    node1 = [0,0,2]
    node2 = [0,0,3] # branch
    node3 = [0,0,4]
    node4 = [0,1,3]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 2]])

    adj_matrix = create_adjacency_matrix(edges1,nodes1)

    g1 = create_graph_from_adjacency_matrix(adj_matrix)
    
    node0 = [0,0,1]   
    node00 = [0,0,1.5] 
    node1 = [0,0,2]
    node10 = [0,0,2.5]
    node2 = [0,0,3] # 4- branch
    node3 = [0,0,4]
    node4 = [0,1,3] 
    node5 = [0,0.3,1.5]  
    node6 = [0,0.5,3]
    nodes2 = np.array([node0,node00,node1,node10,node2,node3,node4,node5,node6])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [3,4], [4,5], [4,7], [7,8], [8,6]])

    adj_matrix = create_adjacency_matrix(edges2, nodes2)

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict, tp_e = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)   

    assert np.array_equal(tp_e, [0, 1, 2, 3, 4, 5, 6, 8]) 

def test_match_graphs_fractured():
    a = [0,0,0]
    b = [1,0,0]
    c = [2,0,0]
    
    # fracture here
    d = [3,0,0]
    e = [4,0,0]
    nodes = np.array([a,b,c,d,e])
    edges = np.array([[0,1],[1,2],[3,4]])

    adj_matrix = create_adjacency_matrix(edges,nodes)
    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict, tp_e = match_polyline_graphs(g1, g1, nodes, nodes, 1) 

    expected = [0,1,2,3,4]
    assert np.array_equal(expected,tp_e)

def test_match_graphs_fracture_and_branch():
    a = [0,0,0] 
    b = [1,0,0] # branch-d-e
    c = [2,0,0] 

    # brach here
    d = [1,1,0] 
    e = [1,2,0]  

    # fracture here - these are NOT connected
    f = [3,0,0]
    g = [4,0,0]
    nodes = np.array([a,b,c,d,e,f,g])
    edges = np.array([[0,1],[1,2],[1,3],[3,4],[5,6]])

    adj_matrix = create_adjacency_matrix(edges,nodes)
    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    # nodes2 = np.array([a,b,c,d,e,d,[4.4,0,0]])
    # edges2 = np.array([[0,1],[1,2],[3,4],[1,5],[5,6]])

    # adj_matrix = create_adjacency_matrix(edges2,nodes2)
    # g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict, tp_e = match_polyline_graphs(g1, g1, nodes, nodes, 1) 

    expected = [0,1,2,3,4,5,6]
    assert np.array_equal(expected,tp_e)

def test_meta_split_into_branches():
    a = [0,0,0] 
    b = [1,0,0] # branch-d-e
    c = [2,0,0] 

    # brach here
    d = [1,1,0] 
    e = [1,2,0]  

    # fracture here - these are NOT connected
    f = [3,0,0]
    g = [4,0,0]
    nodes = np.array([a,b,c,d,e,f,g])
    edges = np.array([[0,1],[1,2],[1,3],[3,4],[5,6]])

    adj_matrix = create_adjacency_matrix(edges,nodes)
    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    branches = []
    g_frag = split_into_fragments(g1)   

    for frag in g_frag: 
        b =  split_into_branches(frag)
        branches.extend(b)
    expected = [[0,1],[1,2],[1,3,4],[5,6]]
    # print(branches)
    # assert np.array_equal(branches,expected)
    assert branches == expected

def test_match_graphs3():
    node0 = [1,0,0] # matches with 1,0,0.1
    node1 = [2,0,0]
    node2 = [3,0,0]
    node3 = [4,0,0]
    node4 = [5,0,0]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [3,4]])

    adj_matrix = create_adjacency_matrix(edges1,nodes1)
    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [0.5,0,0]
    node1 = [1,0,0.1]
    node2 = [6, 7, 10]
    node3 = [20,20,20]   
    nodes2 = np.array([node0, node1, node2, node3])
    edges2 = np.array([[0, 1], [1, 2], [2, 3]])

    
    adj_matrix = create_adjacency_matrix(edges2,nodes2)
    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)
    print(match_dict)

    assert match_dict[0] == 1
    assert match_dict[1] == -1
    assert match_dict[2] == -1
    assert match_dict[3] == -1
    assert match_dict[4] == -1


#  --- confusion matrix
def test_match_graphs_diff_density():
    node0 = [0,0,1]
    node1 = [0,0,2]
    node2 = [0,0,3] # branch
    node3 = [0,0,4]
    node4 = [0,1,3]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 2]])

    adj_matrix = create_adjacency_matrix(edges1,nodes1)

    g1 = create_graph_from_adjacency_matrix(adj_matrix)
    
    node0 = [0,0,1]   
    node00 = [0,0,1.5] 
    node1 = [0,0,2]
    node10 = [0,0,2.5]
    node2 = [0,0,3] # 4- branch
    node3 = [0,0,4]
    node4 = [0,1,3] 
    node5 = [0,0.3,1.5]  
    node6 = [0,0.5,3]
    nodes2 = np.array([node0,node00,node1,node10,node2,node3,node4,node5,node6])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [3,4], [4,5], [4,7], [7,8], [8,6]])

    adj_matrix = create_adjacency_matrix(edges2, nodes2)

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict, tp_e = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)   

    assert np.array_equal(tp_e, [0, 1, 2, 3, 4, 5, 6, 8]) 

    tp, fn, _ = confusion_matrix(match_dict, g2) #
    fp = list(set(g2.keys()) - set(tp_e))
    tp_e = corresponding_tp(match_dict) 

    false_positives = nodes2[fp]
    true_positives_e = nodes2[tp_e] 
    true_positives = nodes1[tp]
    false_negatives = nodes1[fn]    

    assert len(false_positives) == 1, false_positives
    assert np.array_equal(false_positives[0],[0,0.3,1.5]), false_positives

def test_confusion_matrix_reversed():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    node5 = [20, 25, 20]
    nodes2 = np.array([node5, node1, node2, node3, node4, node0])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 3, 0, 5, 1],
            [0, 3, 0, 6, 0, 0],
            [0, 0, 6, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)

    tp, fn, fp = confusion_matrix(match_dict, g2)

    assert np.array_equal(tp, [0, 1, 2, 3]), tp  # indicies in g1
    assert np.array_equal(fn, [4]), fn  # indicies in g1
    assert np.array_equal(fp, [0, 4]), fp  # indicies in g2

def test_confusion_matrix():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    nodes2 = np.array([node0, node1, node2, node3, node4])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)
    tp, fn, fp = confusion_matrix(match_dict, g2)

    assert np.array_equal(tp, [0, 1, 2, 3])
    assert np.array_equal(fn, [4])
    assert np.array_equal(fp, [4])

    assert len(tp) + len(fn) == len(g1)
    assert len(tp) + len(fp) == len(g2)

def test_confusion_matrix2():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [6, 5, 4]
    nodes1 = np.array([node0, node1, node2, node3, node4])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 3, 0, 5],
            [0, 3, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 5, 0, 0, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    node5 = [20, 25, 20]
    nodes2 = np.array([node0, node1, node2, node3, node4, node5])
    edges2 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 3, 0, 5, 0],
            [0, 3, 0, 6, 0, 0],
            [0, 0, 6, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)
    tp, fn, fp = confusion_matrix(match_dict, g2)

    assert np.array_equal(tp, [0, 1, 2, 3])
    assert np.array_equal(fn, [4])
    assert np.array_equal(fp, [4, 5])

    assert len(tp) + len(fn) == len(g1)
    assert len(tp) + len(fp) == len(g2)

def test_confusion_matrix3():
    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    nodes1 = np.array([node0, node1])
    edges1 = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])

    adj_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )

    g1 = create_graph_from_adjacency_matrix(adj_matrix)

    node0 = [1, 2, 3]
    node1 = [4, 5, 6]
    node2 = [6, 7, 8]
    node3 = [8, 10, 11]
    node4 = [20, 20, 20]
    node5 = [20, 25, 20]
    nodes2 = np.array([node0, node1, node2, node3, node4, node5])

    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 3, 0, 5, 0],
            [0, 3, 0, 6, 0, 0],
            [0, 0, 6, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ]
    )

    g2 = create_graph_from_adjacency_matrix(adj_matrix)

    match_dict,_ = match_polyline_graphs(g1, g2, nodes1, nodes2, 1)
    tp, fn, fp = confusion_matrix(match_dict, g2)

    assert np.array_equal(tp, [0, 1])
    assert np.array_equal(fn, [])
    assert np.array_equal(fp, [2, 3, 4, 5])

    assert len(tp) + len(fn) == len(g1)
    assert len(tp) + len(fp) == len(g2)

# --- find contiguous sections

def test_find_contiguous_on_simple():
    graph = {0: [1], 1: [0,2], 2: [1,3], 3:[2]}
    contig = find_contiguous_sections(graph, [0,1,2,3])
    assert np.array_equal(contig, [[0,1,2,3]]), contig

    contig = find_contiguous_sections(graph, [0,2,3])     
    assert contig == [[0], [2,3]]    

def test_find_contiguous_on_branched():
    graph = {0: [1], 1: [0, 2, 4], 2: [1,3], 3:[2], 4: [1,5], 5: [4]}
    contig = find_contiguous_sections(graph, [0,1,2,3,4,5])
    assert contig == [[0,1,2,3,4,5]], contig

    contig = find_contiguous_sections(graph, [5,4,3,2,1,0])
    assert contig == [[0,1,2,3,4,5]], contig

    contig = find_contiguous_sections(graph, [0,1,3,4,5])    
    assert contig == [[0,1,4,5],[3]], contig

def test_find_contiguous_on_fractured():
    graph = {0: [1], 1: [0,2], 2: [1], 3: [4], 4: [3,5], 5: [4]}
    
    # 0-1-2  X  3-4-5
    contig = find_contiguous_sections(graph, list(graph.keys()))
    assert contig == [[0,1,2],[3,4,5]], contig

# --- find length matched path

def test_find_all_paths_between_leaf_nodes():
    graph = {0: [1], 1: [0, 2, 4], 2: [1,3], 3:[2], 4: [1,5], 5: [4]}
    all_paths = find_all_paths_between_leaf_nodes(graph)  
    assert all_paths == {(0,3): [0,1,2,3], (0, 5): [0,1,4,5], (3,5): [3,2,1,4,5]}, all_paths 

def test_matched_length_path():
    a = [0,0,0]
    b = [1,0,0]
    c = [2,0,0]
    
    # fracture here
    d = [3,0,0]
    e = [4,0,0]

    nodes = np.array([a,b,c,d,e])
    edges = np.array([[0,1],[1,2],[3,4]])

    adj = create_adjacency_matrix(edges,nodes)
    graph = create_graph_from_adjacency_matrix(adj)

    dist_to_match = 5
    d, path = find_length_matched_path(graph, adj, dist_to_match)
    assert d == 2, d
    assert path == [0,1,2], path
    
    
