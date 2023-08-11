import numpy as np
from main import create_dense

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



