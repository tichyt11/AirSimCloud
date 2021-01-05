import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
from heap cimport FastUpdateBinaryHeap, REFERENCE_T, VALUE_T
import cython

cdef extern from "pyport.h":
  double Py_HUGE_VAL
cdef VALUE_T Infinite = Py_HUGE_VAL

DOUBLE_TYPE = np.double
BOOL_TYPE = np.bool

ctypedef np.uint8_t uint8

ctypedef (Py_ssize_t, Py_ssize_t) tile

cdef tile[8] directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

ctypedef struct line_data:
    Py_ssize_t *rr
    Py_ssize_t *cc
    Py_ssize_t num_tiles

ctypedef struct node:
    tile data
    double gscore
    bint closed, valid
    node *came_from

cdef void init_node(node *n, tile data, bint valid, double gscore) nogil:
    n.data = data
    n.gscore = gscore
    n.closed = 0
    n.valid = valid
    n.came_from = NULL

cdef Py_ssize_t cmax(Py_ssize_t a, Py_ssize_t b) nogil:
    return a if a > b else b

cdef bint is_goal_reached(tile current, tile goal) nogil:
    return current[0] == goal[0] and current[1] == goal[1]

cdef bint in_bounds(Py_ssize_t row, Py_ssize_t col, Py_ssize_t h, Py_ssize_t w) nogil:
    return 0 <= row < h and 0 <= col < w

cdef double dist(tile n0, tile n1) nogil:
    return sqrt(<double>((n1[0] - n0[0])**2 + (n1[1] - n0[1])**2))

cdef Py_ssize_t cabs(Py_ssize_t x) nogil:
    return x if x >= 0 else -x

cdef line_data _line(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1) nogil:  # from skimage/draw
    cdef bint    steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = cabs(r1 - r0)
    cdef Py_ssize_t dc = cabs(c1 - c0)
    cdef Py_ssize_t sr, sc, d, i

    cdef Py_ssize_t num_tiles = cmax(dc, dr) + 1
    cdef Py_ssize_t *rr = <Py_ssize_t*> malloc(num_tiles * sizeof(Py_ssize_t))
    cdef Py_ssize_t *cc = <Py_ssize_t*> malloc(num_tiles * sizeof(Py_ssize_t))

    if rr is NULL or cc is NULL:
        raise MemoryError()

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

cdef class PathFinder:
    cdef public Py_ssize_t h, w
    cdef public double[:,::1] vals_view
    cdef public unsigned char[:,::1] occ_view
    cdef double bias

    def __init__(self, np.ndarray[np.uint8_t, ndim=2] occupancy_grid, np.ndarray[np.float64_t, ndim=2] vals, double bias=0.5):
        self.h = occupancy_grid.shape[0]
        self.w = occupancy_grid.shape[1]

        self.vals_view = vals
        self.occ_view = occupancy_grid
        self.bias = bias

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Activate C division
    @cython.initializedcheck(False) # deactivate init check
    cdef bint lineOfsight(self, tile n0, tile n1, double alt) nogil:
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
        ze = z0 + alt + halfinc - self.bias

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
            if zb < z or ze < z or self.occ_view[r, c] == 1:  # obstacle in the way
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
    cdef double distance_between(self, tile n0, tile n1, double diff_threshold) nogil:
        cdef Py_ssize_t r0, c0, r1, c1
        cdef double height_diff
        r0 = n0[0]
        c0 = n0[1]
        r1 = n1[0]
        c1 = n1[1]
        height_diff = fabs(self.vals_view[r0, c0] - self.vals_view[r1, c1])
        if height_diff > diff_threshold:  # penalize high altitude difference
            height_diff *= 100
        return dist(n0, n1) + height_diff

    @cython.initializedcheck(False) # deactivate init check
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef Py_ssize_t neighbors(self, tile n, tile *adjacent) nogil:
        cdef Py_ssize_t r, c, i, rr, cc, total = 0
        cdef tile drs[8]
        cdef char valid[8]
        cdef tile temp

        r = n[0]
        c = n[1]

        for i in range(8):
            rr = drs[i][0] = r + directions[i][0]
            cc = drs[i][1] = c + directions[i][1]
            if in_bounds(rr, cc, self.h, self.w) and self.occ_view[rr, cc] == 0:
                valid[i] = 1
                total += 1
            else:
                valid[i] = 0

        cdef Py_ssize_t j = 0
        i = 0
        while j < total:
            if valid[i]:
                adjacent[j] = drs[i]
                j += 1
            i += 1

        return total

    cdef reconstruct_path(self, node* last, node* AllTheNodes):
        cdef node* current = last
        path = []
        while current is not NULL:
            path.append(current.data)
            current = current.came_from
        free(AllTheNodes)
        return list(reversed(path))

    cdef REFERENCE_T tile2ref(self, tile n) nogil:
        return n[0]*self.w + n[1]

    @cython.cdivision(True)  # Activate C division
    cdef tile ref2tile(self, REFERENCE_T ref) nogil:
        return ref // self.w, ref % self.w

    cpdef thetastar(self, tile start, tile goal, double alt=1, double threshold=10):  # from https://github.com/jrialland/python-astar
        cdef Py_ssize_t num_neighbors, i, ref
        cdef double fscore, tentative_gscore, val

        fastHeap = FastUpdateBinaryHeap(128, self.w*self.h-1)
        cdef tile adjacent_tiles[8]
        cdef tile neighbor_tile
        cdef node *current = NULL
        cdef node *neighbor = NULL
        cdef node *SearchNodes

        if is_goal_reached(start, goal):
            return [start]

        SearchNodes = <node*> malloc(self.w * self.h * sizeof(node))  # create grid of nodes ready
        if SearchNodes is NULL:
            raise MemoryError()
        for i in range(self.w * self.h):
            init_node(& SearchNodes[i], (0,0), 0, Infinite)

        fscore = dist(start, goal)
        init_node(& SearchNodes[self.tile2ref(start)], start, 1, .0)
        fastHeap.push_fast(fscore, self.tile2ref(start))
        while True:
            fastHeap.pop_fast()
            ref = fastHeap._popped_ref
            current = & SearchNodes[ref]
            if is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, SearchNodes)  # pass SearchNodes for freeing memory
            current.closed = 1
            num_neighbors = self.neighbors(current.data, adjacent_tiles)
            for i in range(num_neighbors):
                neighbor_tile = adjacent_tiles[i]
                ref = self.tile2ref(neighbor_tile)
                neighbor = & SearchNodes[ref]  # node
                if not neighbor.valid:  # node not initialized yet
                    init_node(neighbor, neighbor_tile, 1, Infinite)
                if neighbor.closed:
                    continue
                if current.came_from is not NULL and self.lineOfsight(current.came_from.data, neighbor_tile, alt):  # there is line of sight from parent to neighbor
                    tentative_gscore = current.came_from.gscore + self.distance_between(current.came_from.data, neighbor_tile, threshold)
                    if tentative_gscore >= neighbor.gscore:
                        continue
                    neighbor.came_from = current.came_from
                else:   # no line of sight
                    tentative_gscore = current.gscore + self.distance_between(current.data, neighbor_tile, threshold)
                    if tentative_gscore >= neighbor.gscore:
                        continue
                    neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                fscore = tentative_gscore + dist(neighbor_tile, goal)
                fastHeap.push_fast(fscore, ref)
            if fastHeap.count == 0:  # no more nodes to explore
                break
        free(SearchNodes)
        return None
