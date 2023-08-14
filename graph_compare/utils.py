import math
import numpy as np

# general utils

def distance(nodes, edges):
    e0, e1 = edges
    p0 = nodes[e0]
    p1 = nodes[e1]
    return math.dist(p0, p1)

def magnitude(vector):
    # return math.sqrt(sum(pow(element, 2) for element in vector))
    return np.linalg.norm(vector)
