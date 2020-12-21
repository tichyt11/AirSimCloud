import astar
import numpy as np


def astar_search(start, goal, occupancy_grid, vals):

    def los(n1, n2): # TODO implement
        return False

    def in_bounds(n):
        row, col = n
        h, w = occupancy_grid.shape
        return 0 <= row < h and 0 <= col < w

    def heuristic(n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def cost(n1, n2):
        p1 = np.array(n1)
        p2 = np.array(n2)
        height_diff = abs(vals[tuple(p1)] - vals[tuple(p2)])
        return np.linalg.norm(p1 - p2) + height_diff

    def neighbors(node):
        drs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        c = np.array(node)
        return [n for i in drs if in_bounds(n:=tuple(c+i)) and occupancy_grid[n] == 0]

    path = list(astar.find_path(start, goal, neighbors_fnct=neighbors,
                                heuristic_cost_estimate_fnct=heuristic, distance_between_fnct=cost))
    return path
