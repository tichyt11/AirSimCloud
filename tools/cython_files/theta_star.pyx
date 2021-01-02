import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libc.stdlib cimport abs as cabs
from fast_heap cimport FastUpdateBinaryHeap, REFERENCE_T, VALUE_T
import cython
from libc.stdlib cimport malloc, free

DOUBLE_TYPE = np.double
BOOL_TYPE = np.bool
Infinite = float('inf')

ctypedef np.uint8_t uint8

ctypedef (Py_ssize_t, Py_ssize_t) tile

ctypedef struct line_data:
    Py_ssize_t *rr
    Py_ssize_t *cc
    Py_ssize_t num_tiles

ctypedef struct node:
    tile data
    double gscore, fscore
    bint closed, out_openset
    node *came_from

cdef init_node(node *n, tile data, double gscore, double fscore):
    n.data = data
    n.gscore = gscore
    n.fscore = fscore
    n.came_from = NULL

cdef Py_ssize_t cmax(Py_ssize_t a, Py_ssize_t b):
    if a > b:
        return a
    else:
        return b

cdef class SearchNode:
    cdef public tile data
    cdef public double gscore, fscore
    cdef public bint closed
    cdef public SearchNode came_from
    def __cinit__(self, tile data, double gscore=Infinite, double fscore=Infinite):
        self.data = data
        self.gscore = gscore
        self.fscore = fscore
        self.closed = False
        self.came_from = None

class SearchNodeDict(dict):

    def __missing__(self, tile k):
        cdef SearchNode v = SearchNode(k)
        self.__setitem__(k, v)
        return v

cdef bint is_goal_reached(tile current, tile goal):
    return current[0] == goal[0] and current[1] == goal[1]

cdef bint in_bounds(Py_ssize_t row, Py_ssize_t col, Py_ssize_t h, Py_ssize_t w):
    return 0 <= row < h and 0 <= col < w

cdef double dist(tile n0, tile n1):
    return sqrt(<double>((n1[0] - n0[0])**2 + (n1[1] - n0[1])**2))

cdef line_data _line(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1):  # from skimage/draw
    cdef char steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = cabs(r1 - r0)
    cdef Py_ssize_t dc = cabs(c1 - c0)
    cdef Py_ssize_t sr, sc, d, i

    cdef Py_ssize_t num_tiles = cmax(dc, dr) + 1
    cdef Py_ssize_t *rr = <Py_ssize_t*> malloc(num_tiles * sizeof(Py_ssize_t))
    cdef Py_ssize_t *cc = <Py_ssize_t*> malloc(num_tiles * sizeof(Py_ssize_t))
    # cdef Py_ssize_t[::1] rr = np.zeros(num_tiles, dtype=np.intp)
    # cdef Py_ssize_t[::1] cc = np.zeros(num_tiles, dtype=np.intp)

    if rr is NULL or cc is NULL:
        raise MemoryError()

    with nogil:
        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc

        for i in range(dc):
            if steep:
                rr[i] = c
                cc[i] = r
            else:
                rr[i] = r
                cc[i] = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)

        rr[dc] = r1
        cc[dc] = c1

    cdef line_data ret
    ret.cc = cc
    ret.rr = rr
    ret.num_tiles = dc +1
    return ret

cdef class ThetaStar:
    cdef public Py_ssize_t h, w
    cdef public double[:,::1] vals_view
    cdef public unsigned char[:,::1] occ_view
    cdef public double alt

    def __init__(self, np.ndarray[np.uint8_t, ndim=2] occupancy_grid, np.ndarray[np.float64_t, ndim=2] vals, bint reversePath=0, double alt=1):
        self.h = occupancy_grid.shape[0]
        self.w = occupancy_grid.shape[1]

        self.vals_view = vals
        self.occ_view = occupancy_grid
        self.alt = alt

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Activate C division
    @cython.initializedcheck(False) # deactivate init check
    cdef bint lineOfsight(self, tile n0, tile n1):
        cdef Py_ssize_t *rs
        cdef Py_ssize_t *cs
        cdef Py_ssize_t  num_tiles, i, r0, r1, c0, c1, r, c
        cdef double z0, z1, defz, zb, ze, halfinc, z

        r0 = n0[0]
        c0 = n0[1]
        r1 = n1[0]
        c1 = n1[1]

        z0 = self.vals_view[r0, c0]
        z1 = self.vals_view[r1, c1]

        cdef line_data ret = _line(r0, c0, r1, c1)
        rs = ret.rr
        cs = ret.cc
        num_tiles = ret.num_tiles


        difz = z1 - z0
        halfinc = difz / ((2 * (num_tiles - 1)))
        ze = z0 + self.alt + halfinc  # at y/x = 1

        if ze < z0:  # check end of first one
            free(rs)
            free(cs)
            return 0

        for i in range(1, num_tiles-1):
            zb = ze
            ze = ze + 2 * halfinc
            r = rs[i]
            c = cs[i]
            z = self.vals_view[r, c]
            if self.occ_view[r, c] == 1 or zb < z or ze < z:  # obstacle in the way
                free(rs)
                free(cs)
                return 0
        free(rs)
        free(cs)
        if ze < z1:  # check beginning of last one
            return 0

        return 1

    @cython.initializedcheck(False) # deactivate init check
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef double distance_between(self, tile n0, tile n1):
        cdef Py_ssize_t r0, c0, r1, c1
        cdef double height_diff
        r0 = n0[0]
        c0 = n0[1]
        r1 = n1[0]
        c1 = n1[1]
        height_diff = fabs(self.vals_view[r0, c0] - self.vals_view[r1, c1])
        if height_diff > 10:  # penalize high altitude difference
            height_diff *= 100
        return dist(n0, n1) + height_diff

    @cython.initializedcheck(False) # deactivate init check
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef neighbors(self, tile n):
        cdef Py_ssize_t r, c, i, rr, cc
        ret = []
        r = n[0]
        c = n[1]
        cdef tile[8] drs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for i in range(8):
            rr = r + drs[i][0]
            cc = c + drs[i][1]
            if in_bounds(rr, cc, self.h, self.w) and self.occ_view[rr, cc] == 0:
                ret.append((rr, cc))
        return ret
        # return [(r + i[0], c + i[1]) for i in drs if in_bounds(r + i[0], c + i[1], self.h, self.w) and self.occ_view[r + i[0], c + i[1]] == 0]

    def reconstruct_path(self, SearchNode last):
        cdef SearchNode current
        path = []
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        return reversed(list(_gen()))

    cdef REFERENCE_T tile2ref(self, tile node):
        cdef Py_ssize_t r, c
        r = node[0]
        c = node[1]
        return r*self.w + c

    @cython.cdivision(True)  # Activate C division
    cdef tile ref2tile(self, REFERENCE_T ref):
        cdef tile node
        node[0] = ref // self.w
        node[1] = ref % self.w
        return node

    def thetastar(self, tile start, tile goal, reversePath=False):
        fastHeap = FastUpdateBinaryHeap(max_reference=self.w*self.h-1)  # push(val, ref), value_of(ref), val, ref = pop()
        cdef REFERENCE_T ref  # PY_ssize_t
        cdef VALUE_T val  # double
        cdef tile current_tile
        cdef SearchNode current, neighbor
        if is_goal_reached(start, goal):
            return [start]
        searchNodes = SearchNodeDict()
        cdef double fscore = dist(start, goal)
        startNode = searchNodes[start] = SearchNode(
            start, .0, fscore)
        fastHeap.push_fast(fscore, self.tile2ref(start))
        while True:
            fastHeap.pop_fast()
            ref = fastHeap._popped_ref
            current_tile = self.ref2tile(ref)
            current = searchNodes[current_tile]  # TODO get rid of dict
            if is_goal_reached(current.data, goal):
                return self.reconstruct_path(current)
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
                neighbor.fscore = tentative_gscore + dist(neighbor.data, goal)
                fastHeap.push_fast(neighbor.fscore, self.tile2ref(neighbor.data))
            if fastHeap.count == 0:  # no more nodes to explore
                break
        return None
