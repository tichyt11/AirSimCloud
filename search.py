import astar
import numpy as np


def astar_search(start, goal, occupancy_grid):

    def in_bounds(n):
        row, col = n
        h, w = occupancy_grid.shape
        return 0 <= row < h and 0 <= col < w

    def distance(n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def neighbors(node):
        drs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        c = np.array(node)
        return [n for i in drs if in_bounds(n:=tuple(c+i)) and occupancy_grid[n] == 0]

    path = list(astar.find_path(start, goal, neighbors, distance, distance))
    return path
