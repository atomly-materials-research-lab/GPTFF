# distutils: language = c++

import numpy as np
cimport numpy as np

from libc.stdlib cimport free
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from 'source/utils.cc':

    ctypedef struct Results_ptr:
        int num_tps
        int *bond_pairs

    Results_ptr *cal_tp(int *n_bond_pairs_atom, int *n_bond_pairs_bond, const int *nbr_atoms, const int *idx_mapping, const int n_atoms, const size_t rows, const size_t cols)


np.import_array()

cpdef compute_tp_cc(np.ndarray[int, ndim=2] nbr_atoms, d_ij, tp_cut, n_atoms):
    if not nbr_atoms.flags['C_CONTIGUOUS']:
        nbr_atoms = np.ascontiguousarray(nbr_atoms, dtype=np.int32)

    cdef int num_nbrs = nbr_atoms.shape[0]

    mask = d_ij <= tp_cut

    cdef np.ndarray[int, ndim=1] idx_mapping = np.nonzero(mask)[0].astype(np.int32)
    cdef np.ndarray[int, ndim=2] filtered_nbr_atoms = nbr_atoms[mask]
    cdef size_t rows = filtered_nbr_atoms.shape[0]
    cdef size_t cols = filtered_nbr_atoms.shape[1]

    cdef np.ndarray[int, ndim=1] n_bond_pairs_atom_arr = np.zeros(n_atoms).astype(np.int32)
    cdef np.ndarray[int, ndim=1] n_bond_pairs_bond_arr
    cdef np.ndarray[int, ndim=2] bond_pairs_indices 
    cdef np.ndarray[int, ndim=1] bond_pairs 

    cdef int* n_bond_pairs_atom 
    cdef int* n_bond_pairs_bond 
    cdef int* filtered_nbr_atoms_ptr 
    cdef int* idx_mapping_ptr

    cdef Results_ptr *c
    cdef int num_tps 
    cdef np.npy_intp* dims

    if rows:
        n_bond_pairs_bond_arr = np.zeros(num_nbrs).astype(np.int32)
        n_bond_pairs_atom = &n_bond_pairs_atom_arr[0]
        n_bond_pairs_bond = &n_bond_pairs_bond_arr[0]
        filtered_nbr_atoms_ptr = &filtered_nbr_atoms[0,0]
        idx_mapping_ptr = &idx_mapping[0]
        c = cal_tp(n_bond_pairs_atom, n_bond_pairs_bond, filtered_nbr_atoms_ptr, idx_mapping_ptr, n_atoms, rows, cols)
        num_tps = c.num_tps 

        dims = <np.npy_intp*>PyMem_Malloc(1 * sizeof(np.npy_intp))
        dims[0] = num_tps * 2 # Cast length to npy_intp
        bond_pairs = np.PyArray_SimpleNewFromData(
            1, dims, np.NPY_INT32, <void*>c.bond_pairs)

        bond_pairs_indices =  bond_pairs.reshape(num_tps, 2)

        free(c)
        PyMem_Free(dims)

    else:
        n_bond_pairs_bond_arr = np.array([0] * num_nbrs, dtype=np.int32)

        bond_pairs_indices = np.array([], dtype=np.int32).reshape(-1, 2)

    return n_bond_pairs_atom_arr, n_bond_pairs_bond_arr, bond_pairs_indices
