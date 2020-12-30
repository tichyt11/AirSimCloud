import astar
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw.draw import line

def astar_search(start, goal, occupancy_grid, vals, bias=1):

    def lineOfsight(n1, n2):
        if not n1:
            print('no came_from Node')
            return False
        r0, c0 = n1
        r1, c1 = n2
        z0 = vals[r0, c0]
        z1 = vals[r1, c1]
        difz = z1 - z0
        rs, cs = line(r0, c0, r1, c1)
        num_tiles = len(rs)
        tiles = [(rs[i], cs[i]) for i in range(num_tiles)]

        halfinc = difz / (2 * (len(rs) - 1))
        ze = z0 + bias + halfinc  # at y/x = 1

        if ze < vals[tiles[0]]:  # check end of first one
            return False

        for tile in tiles[1:-1]:
            zb = ze
            ze = ze + 2 * halfinc
            z = vals[tile]
            if occupancy_grid[tile] == 1:  # obstacle in the way
                return False
            if zb < z:
                return False
            if ze < z:
                return False

        if ze < vals[tiles[-1]]:  # check beginning of last one
            return False

        return True

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
        if height_diff > 10:  # penalize high altitude difference
            height_diff *= 100
        return np.linalg.norm(p1 - p2) + height_diff

    def neighbors(node):
        drs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        c = np.array(node)
        return [n for i in drs if in_bounds(n:=tuple(c+i)) and occupancy_grid[n] == 0]

    # path = list(astar.find_path(start, goal, neighbors_fnct=neighbors,
    #                             heuristic_cost_estimate_fnct=heuristic, distance_between_fnct=cost))

    path = list(theta_find_path(start, goal, lineOfsight=lineOfsight, neighbors_fnct=neighbors,
                                heuristic_cost_estimate_fnct=heuristic, distance_between_fnct=cost))

    return path


from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop, heapify

Infinite = float('inf')

class ThetaStar:
    __metaclass__ = ABCMeta
    __slots__ = ()

    class SearchNode:
        __slots__ = ('data', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = ThetaStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    @abstractmethod
    def lineOfsight(self, n1, n2):
        raise NotImplementedError

    @abstractmethod
    def heuristic_cost_estimate(self, current, goal):
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1, n2):
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node):
        raise NotImplementedError

    def is_goal_reached(self, current, goal):
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def thetastar(self, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = ThetaStar.SearchNodeDict()
        startNode = searchNodes[start] = ThetaStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                if current.came_from is not None and self.lineOfsight(current.came_from.data, neighbor.data):  # there is line of sight from parent to neighbor
                    tentative_gscore = current.came_from.gscore + self.distance_between(current.came_from.data, neighbor.data)
                    if tentative_gscore >= neighbor.gscore:
                        continue
                    neighbor.came_from = current.came_from
                else:   # no line of sight
                    tentative_gscore = current.gscore + self.distance_between(current.data, neighbor.data)
                    if tentative_gscore >= neighbor.gscore:
                        continue
                    neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

def theta_find_path(start, goal, lineOfsight, neighbors_fnct, reversePath=False, heuristic_cost_estimate_fnct=lambda a, b: Infinite, distance_between_fnct=lambda a, b: 1.0, is_goal_reached_fnct=lambda a, b: a == b):
    """A non-class version of the path finding algorithm"""
    class FindPath(ThetaStar):

        def lineOfsight(self, n1, n2):
            return lineOfsight(n1, n2)

        def heuristic_cost_estimate(self, current, goal):
            return heuristic_cost_estimate_fnct(current, goal)

        def distance_between(self, n1, n2):
            return distance_between_fnct(n1, n2)

        def neighbors(self, node):
            return neighbors_fnct(node)

        def is_goal_reached(self, current, goal):
            return is_goal_reached_fnct(current, goal)
    return FindPath().thetastar(start, goal, reversePath)