import numpy as np
cimport numpy as np
from skimage.draw.draw import line
from libc.math cimport sqrt, fabs


ctypedef np.uint8_t uint8
DOUBLE_TYPE = np.double
BOOL_TYPE = np.bool
Infinite = float('inf')
ctypedef (Py_ssize_t, Py_ssize_t) tile

ctypedef struct node:
    tile data
    double gscore, fscore
    bint closed, out_openset
    node *came_from

cdef init_node(node n, tile data, double gscore, double fscore):
    n.data = data
    n.gscore = gscore
    n.fscore = fscore
    n.came_from = NULL

cdef class SearchNode:
    cdef public tile data
    cdef public double gscore, fscore
    cdef public bint closed, out_openset
    cdef public SearchNode came_from
    def __cinit__(self, tile data, double gscore=Infinite, double fscore=Infinite):
        self.data = data
        self.gscore = gscore
        self.fscore = fscore
        self.closed = False
        self.out_openset = True
        self.came_from = None

def heappush(heap, SearchNode item):
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    cdef SearchNode lastelt, returnitem
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if len(heap) > 0:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def _siftdown(heap, Py_ssize_t startpos, Py_ssize_t pos):
    cdef SearchNode newitem, parent
    cdef Py_ssize_t parentpos
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem.fscore < parent.fscore:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup(heap, Py_ssize_t pos):
    cdef SearchNode newitem, child, right
    cdef Py_ssize_t endpos, startpos, childpos, rightpos
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos:
            child = heap[childpos]
            right = heap[rightpos]
            if not child.fscore < right.fscore:
                childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

class SearchNodeDict(dict):

    def __missing__(self, k):
        v = SearchNode(k)
        self.__setitem__(k, v)
        return v

cdef bint is_goal_reached(tile current, tile goal):
    return current[0] == goal[0] and current[1] == goal[1]

cdef double dist(Py_ssize_t x0, Py_ssize_t y0, Py_ssize_t x1, Py_ssize_t y1):
    return sqrt(<double>((x1 - x0)**2 + (y1 - y0)**2))

cdef bint in_bounds(Py_ssize_t row, Py_ssize_t col, Py_ssize_t h, Py_ssize_t w):
    return 0 <= row < h and 0 <= col < w

cpdef double heuristic(n0, n1):
    r0 = n0[0]
    c0 = n0[1]
    r1 = n1[0]
    c1 = n1[1]
    return dist(r0, c0, r1, c1)

cdef class ThetaStar:
    cdef public Py_ssize_t h, w
    cdef public double[:,:] vals_view
    cdef public uint8[:,:] occ_view
    cdef public double alt

    def __init__(self, occupancy_grid, vals, bint reversePath=0, double alt=1):
        self.h = occupancy_grid.shape[0]
        self.w = occupancy_grid.shape[1]

        assert vals.dtype == DOUBLE_TYPE
        assert occupancy_grid.dtype == BOOL_TYPE
        self.vals_view = vals
        self.occ_view = np.ubyte(occupancy_grid)
        self.alt = alt

    def lineOfsight(self, n0, n1):
        if not n0:
            print('no came_from Node')
            return 0
        cdef Py_ssize_t[::1] rs
        cdef Py_ssize_t[::1] cs
        cdef Py_ssize_t  num_tiles, i, r0, r1, c0, c1, r, c
        cdef double z0, z1, defz, zb, ze, halfinc, z

        r0 = n0[0]
        c0 = n0[1]
        r1 = n1[0]
        c1 = n1[1]

        z0 = self.vals_view[r0, c0]
        z1 = self.vals_view[r1, c1]

        rs, cs = line(r0, c0, r1, c1)
        num_tiles = len(rs)

        difz = z1 - z0
        halfinc = difz / ((2 * (num_tiles - 1)))
        ze = z0 + self.alt + halfinc  # at y/x = 1

        if ze < z0:  # check end of first one
            return 0

        for i in range(1, num_tiles-1):
            zb = ze
            ze = ze + 2 * halfinc
            r = rs[i]
            c = cs[i]
            z = self.vals_view[r, c]
            if self.occ_view[r, c] == 1 or zb < z or ze < z:  # obstacle in the way
                return 0

        if ze < z1:  # check beginning of last one
            return 0

        return 1

    def heuristic_cost_estimate(self, tile current, tile goal):
        cdef Py_ssize_t r0, c0, r1, c1
        r0 = current[0]
        c0 = current[1]
        r1 = goal[0]
        c1 = goal[1]
        return dist(r0, c0, r1, c1)

    def distance_between(self, n0, n1):
        cdef Py_ssize_t r0, c0, r1, c1
        cdef double height_diff

        r0 = n0[0]
        c0 = n0[1]
        r1 = n1[0]
        c1 = n1[1]
        height_diff = fabs(self.vals_view[r0, c0] - self.vals_view[r1, c1])
        if height_diff > 10:  # penalize high altitude difference
            height_diff *= 100
        return dist(r0, c0, r1, c1) + height_diff

    def neighbors(self, node):
        cdef Py_ssize_t r, c
        r = node[0]
        c = node[1]
        cdef tile[8] drs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        return [(r + i[0], c + i[1]) for i in drs if in_bounds(r + i[0], c + i[1], self.h, self.w) and self.occ_view[r + i[0], c + i[1]] == 0]

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return list(_gen())
        else:
            return list(reversed(list(_gen())))

    def thetastar(self, start, goal, reversePath=False):
        cdef SearchNode startNode, current, neighbor
        if is_goal_reached(start, goal):
            return [start]
        searchNodes = SearchNodeDict()
        startNode = searchNodes[start] = SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if is_goal_reached(current.data, goal):
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
