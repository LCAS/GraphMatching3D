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



