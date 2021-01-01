import cython
from libc.stdlib cimport malloc, free

cdef extern from "pyport.h":
  double Py_HUGE_VAL

ctypedef double VALUE_T
ctypedef Py_ssize_t REFERENCE_T
ctypedef REFERENCE_T INDEX_T
ctypedef unsigned char BOOL_T
ctypedef unsigned char LEVELS_T
cdef VALUE_T inf = Py_HUGE_VAL

cdef inline INDEX_T index_min(INDEX_T a, INDEX_T b) nogil:
    return a if a <= b else b


cdef class BinaryHeap:
    cdef readonly INDEX_T count
    cdef readonly LEVELS_T levels, min_levels
    cdef VALUE_T *_values
    cdef REFERENCE_T *_references
    cdef REFERENCE_T _popped_ref

    cdef void _add_or_remove_level(self, LEVELS_T add_or_remove) nogil
    cdef void _update(self) nogil
    cdef void _update_one(self, INDEX_T i) nogil
    cdef void _remove(self, INDEX_T i) nogil

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil
    cdef VALUE_T pop_fast(self) nogil
    def __cinit__(self, INDEX_T initial_capacity=128, *args, **kws):
        cdef LEVELS_T levels = 0
        while 2**levels < initial_capacity:
            levels += 1
        self.min_levels = self.levels = levels

        self.count = 0

        cdef INDEX_T number = 2**self.levels
        self._values = <VALUE_T *>malloc(2 * number * sizeof(VALUE_T))
        self._references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))

    def __init__(self, INDEX_T initial_capacity=128):
        if self._values is NULL or self._references is NULL:
            raise MemoryError()
        self.reset()

    def reset(self):
        cdef INDEX_T number = 2**self.levels
        cdef INDEX_T i
        cdef VALUE_T *values = self._values
        for i in range(number * 2):
            values[i] = inf

    def __dealloc__(self):
        free(self._values)
        free(self._references)

    def __str__(self):
        s = ''
        for level in range(1, self.levels + 1):
            i0 = 2**level - 1  # LevelStart
            s += 'level %i: ' % level
            for i in range(i0, i0 + 2**level):
                s += '%g, ' % self._values[i]
            s = s[:-1] + '\n'
        return s

    cdef void _add_or_remove_level(self, LEVELS_T add_or_remove) nogil:
        cdef INDEX_T i, i1, i2, n
        cdef LEVELS_T new_levels = self.levels + add_or_remove

        cdef INDEX_T number = 2**new_levels
        cdef VALUE_T *values
        cdef REFERENCE_T *references
        values = <VALUE_T *>malloc(number * 2 * sizeof(VALUE_T))
        references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))
        if values is NULL or references is NULL:
            free(values)
            free(references)
            with gil:
                raise MemoryError()

        for i in range(number * 2):
            values[i] = inf
        for i in range(number):
            references[i] = -1

        cdef VALUE_T *old_values = self._values
        cdef REFERENCE_T *old_references = self._references
        if self.count:
            i1 = 2**new_levels - 1  # LevelStart
            i2 = 2**self.levels - 1  # LevelStart
            n = index_min(2**new_levels, 2**self.levels)
            for i in range(n):
                values[i1+i] = old_values[i2+i]
            for i in range(n):
                references[i] = old_references[i]

        free(self._values)
        free(self._references)
        self._values = values
        self._references = references

        self.levels = new_levels
        self._update()

    cdef void _update(self) nogil:
        cdef VALUE_T *values = self._values

        cdef INDEX_T i0, i, ii, n
        cdef LEVELS_T level

        for level in range(self.levels, 1, -1):
            i0 = (1 << level) - 1  # 2**level-1 = LevelStart
            n = i0 + 1  # 2**level
            for i in range(i0, i0+n, 2):
                ii = (i-1) // 2  # CalcPrevAbs
                if values[i] < values[i+1]:
                    values[ii] = values[i]
                else:
                    values[ii] = values[i+1]

    cdef void _update_one(self, INDEX_T i) nogil:
        cdef VALUE_T *values = self._values

        if i % 2 == 0:
            i -= 1

        cdef INDEX_T ii
        cdef LEVELS_T level
        for level in range(self.levels, 1, -1):
            ii = (i-1) // 2  # CalcPrevAbs

            # test
            if values[i] < values[i+1]:
                values[ii] = values[i]
            else:
                values[ii] = values[i+1]
            # next
            if ii % 2:
                i = ii
            else:
                i = ii - 1

    cdef void _remove(self, INDEX_T i1) nogil:
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        cdef INDEX_T i0 = (1 << levels) - 1  # 2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references

        values[i1] = values[i2]
        references[r1] = references[r2]

        values[i2] = inf

        self.count -= 1
        count -= 1
        if (levels > self.min_levels) and (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil:
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        if count >= (1 << levels):  # 2**self.levels:
            self._add_or_remove_level(1)
            levels += 1

        cdef INDEX_T i = ((1 << levels) - 1) + count  # LevelStart + n
        self._values[i] = value
        self._references[count] = reference

        self.count += 1
        self._update_one(i)

        return count

    cdef VALUE_T pop_fast(self) nogil:
        cdef VALUE_T *values = self._values

        cdef LEVELS_T level
        cdef INDEX_T i = 1
        cdef LEVELS_T levels = self.levels
        # search tree (using absolute indices)
        for level in range(1, levels):
            if values[i] <= values[i+1]:
                i = i * 2 + 1  # CalcNextAbs
            else:
                i = (i+1) * 2 + 1  # CalcNextAbs

        if values[i] <= values[i+1]:
            i = i
        else:
            i += 1

        cdef INDEX_T ir = i - ((1 << levels) - 1) # (2**self.levels-1)
                                                  # LevelStart
        cdef VALUE_T value = values[i]
        self._popped_ref = self._references[ir]

        if self.count:
            self._remove(i)
        return value

    def push(self, VALUE_T value, REFERENCE_T reference=-1):
        self.push_fast(value, reference)

    def min_val(self):
        cdef VALUE_T *values = self._values

        if values[1] < values[2]:
            return values[1]
        else:
            return values[2]

    def values(self):
        cdef INDEX_T i0 = 2**self.levels - 1  # LevelStart
        return [self._values[i] for i in range(i0, i0+self.count)]

    def references(self):
        return [self._references[i] for i in range(self.count)]

    def pop(self):
        if self.count == 0:
            raise IndexError('pop from an empty heap')
        value = self.pop_fast()
        ref = self._popped_ref
        return value, ref


cdef class FastUpdateBinaryHeap(BinaryHeap):

    cdef readonly REFERENCE_T max_reference
    cdef INDEX_T *_crossref
    cdef BOOL_T _invalid_ref
    cdef BOOL_T _pushed

    cdef VALUE_T value_of_fast(self, REFERENCE_T reference)
    cdef INDEX_T push_if_lower_fast(self, VALUE_T value,
                                    REFERENCE_T reference) nogil
    def __cinit__(self, INDEX_T initial_capacity=128, max_reference=None):
        if max_reference is None:
            max_reference = initial_capacity - 1
        self.max_reference = max_reference
        self._crossref = <INDEX_T *>malloc((self.max_reference + 1) *
                                           sizeof(INDEX_T))

    def __init__(self, INDEX_T initial_capacity=128, max_reference=None):
        if self._crossref is NULL:
            raise MemoryError()
        BinaryHeap.__init__(self, initial_capacity)

    def __dealloc__(self):
        free(self._crossref)

    def reset(self):
        BinaryHeap.reset(self)
        cdef INDEX_T i
        for i in range(self.max_reference + 1):
            self._crossref[i] = -1

    cdef void _remove(self, INDEX_T i1) nogil:
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count

        cdef INDEX_T i0 = (1 << levels) - 1  # 2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references
        cdef INDEX_T *crossref = self._crossref

        crossref[references[r2]] = r1
        crossref[references[r1]] = -1  # disable removed item

        values[i1] = values[i2]
        references[r1] = references[r2]

        values[i2] = inf

        self.count -= 1
        count -= 1
        if (levels > self.min_levels) & (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil:
        if not (0 <= reference <= self.max_reference):
            return -1

        cdef INDEX_T i

        cdef INDEX_T ir = self._crossref[reference]

        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            self._values[i] = value
            self._update_one(i)
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir

    cdef INDEX_T push_if_lower_fast(self, VALUE_T value,
                                    REFERENCE_T reference) nogil:

        if not (0 <= reference <= self.max_reference):
            return -1

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        cdef VALUE_T *values = self._values
        self._pushed = 1
        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            if values[i] > value:
                values[i] = value
                self._update_one(i)
            else:
                self._pushed = 0
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir

    cdef VALUE_T value_of_fast(self, REFERENCE_T reference):
        if not (0 <= reference <= self.max_reference):
            self._invalid_ref = 1
            return inf

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        self._invalid_ref = 0
        if ir == -1:
            self._invalid_ref = 1
            return inf
        i = (1 << self.levels) - 1 + ir
        return self._values[i]

    def push(self, double value, int reference):

        if self.push_fast(value, reference) == -1:
            raise ValueError('reference outside of range [0, max_reference]')

    def push_if_lower(self, double value, int reference):
        if self.push_if_lower_fast(value, reference) == -1:
            raise ValueError('reference outside of range [0, max_reference]')
        return self._pushed == 1

    def value_of(self, int reference):
        value = self.value_of_fast(reference)
        if self._invalid_ref:
            raise ValueError('invalid reference')
        return value

    def cross_references(self):
        return [self._crossref[i] for i in range(self.max_reference + 1)]