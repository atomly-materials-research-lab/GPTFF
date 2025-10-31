# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from libc.stdlib cimport free
from cpython.mem cimport PyMem_Malloc, PyMem_Free

np.import_array()

cdef extern from 'source/utils.cc':
    
    ctypedef struct Results_ptr2:
        vector[int] center_indices, neigh_indices;
        vector[double] dist_list;
        vector[int] offset_list;

    Results_ptr2 *find_neighbors_cc(const double *coords, const double *lattice, const int *cell_shift, const int*pbc, const int *num_cell_xyz, const int *range_xyz, const int *cell_index_1d, const int *cell_index, const int cell_index_1d_max, const int n_atoms, const double cutoff);
    void free_results2(Results_ptr2* p)

cpdef find_neighbors(np.ndarray[double, ndim=2] coords, np.ndarray[double, ndim=2] lattice, cutoff, np.ndarray[int, ndim=1] pbc):
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)

    if not lattice.flags['C_CONTIGUOUS']:
        lattice = np.ascontiguousarray(lattice, dtype=np.float64)

    
    lattice_inv = np.linalg.pinv(lattice).T
    # face distance
    f_dist = np.linalg.norm(lattice_inv, axis=1)
    f_dist = np.array([1/x if x > 0 else 1 for x in f_dist])

    cell_size = cutoff

    cdef np.ndarray[int, ndim=1] num_cell_xyz = np.maximum((f_dist / cell_size).astype(int), [1, 1, 1]).astype(np.int32)

    num_cells = np.prod(num_cell_xyz)

    # for situation that cell_size(cutoff) > face distance
    range_x, range_y, range_z = np.ceil(cell_size * num_cell_xyz / f_dist).astype(np.int32)

    # cell_size(cutoff) < face distance and unperiodic
    range_x = 0 if num_cell_xyz[0] == 1 and not pbc[0] else range_x
    range_y = 0 if num_cell_xyz[1] == 1 and not pbc[1] else range_y
    range_z = 0 if num_cell_xyz[2] == 1 and not pbc[2] else range_z
    cdef np.ndarray[int, ndim=1] range_xyz = np.array([range_x, range_y, range_z]).astype(np.int32)

    frac_coords = np.linalg.solve(lattice.T, coords.T).T

    cdef np.ndarray[int, ndim=2] cell_index = np.floor(frac_coords * num_cell_xyz).astype(np.int32)

    if not cell_index.flags['C_CONTIGUOUS']:
        cell_index = np.ascontiguousarray(cell_index, dtype=np.int32)
    
    cdef np.ndarray[int, ndim=2] cell_shift = np.zeros_like(cell_index)

    if not cell_shift.flags['C_CONTIGUOUS']:
        cell_shift = np.ascontiguousarray(cell_shift, dtype=np.int32)

    for c in range(3):
        if pbc[c]:
            cell_shift[:, c], cell_index[:, c] = \
                divmod(cell_index[:, c], num_cell_xyz[c])
        else:
            cell_index[:, c] = np.clip(cell_index[:, c], 0, num_cell_xyz[c] - 1)
    
    cdef np.ndarray[int, ndim=1] cell_index_1d = (cell_index[:, 0] + num_cell_xyz[0] * (cell_index[:, 1] + num_cell_xyz[1] * cell_index[:, 2]))
    cell_index_1d_max = np.max(cell_index_1d)

    cdef int *cell_index_1d_ptr = &cell_index_1d[0]
    cdef int *cell_index_ptr = &cell_index[0, 0] # Nx3
    cdef int *range_xyz_ptr = &range_xyz[0]
    cdef int *num_cell_xyz_ptr = &num_cell_xyz[0]
    cdef int *pbc_ptr = &pbc[0]
    cdef int *cell_shift_ptr = &cell_shift[0, 0]
    cdef double *coords_ptr = &coords[0, 0]
    cdef double *lattice_ptr = &lattice[0, 0]

    res = find_neighbors_cc(coords_ptr, lattice_ptr, cell_shift_ptr, pbc_ptr, num_cell_xyz_ptr, range_xyz_ptr, cell_index_1d_ptr, cell_index_ptr, <int> cell_index_1d_max, <int> len(coords), cutoff)
    cdef vector[int] i_idx = res.center_indices
    cdef vector[int] j_idx = res.neigh_indices
    cdef vector[int] o = res.offset_list
    cdef vector[double] d = res.dist_list 

    # Convert to numpy (copies from vectors), then free C++ struct to avoid leaks
    i_np = np.asarray(i_idx, dtype=np.int32)
    j_np = np.asarray(j_idx, dtype=np.int32)
    o_np = np.asarray(o, dtype=np.int32).reshape(-1, 3)
    d_np = np.asarray(d, dtype=np.float64)

    free_results2(res)

    return i_np, j_np, o_np, d_np