#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_map>


typedef struct Results {
    int num_tps;
    int *bond_pairs;
} Results_ptr;

typedef struct Results2 {
        std::vector<int> center_indices, neigh_indices;
        std::vector<double> dist_list;
        std::vector<int> offset_list;
    } Results_ptr2;

int mmod(int a, int b){
        int res = a % b;

        if ((a < 0) != (b < 0) && a%b != 0) {
            res += b;
        }
        return res;
    }

int mdiv(int a, int b){
    int res = a / b;
    if ((a < 0) != (b < 0) && a%b != 0) {
        res -= 1;
    }
    return res;
}

Results_ptr *cal_tp(int *n_bond_pairs_atom, int *n_bond_pairs_bond, const int *nbr_atoms, const int *idx_mapping, const int n_atoms, const size_t rows, const size_t cols){

    Results_ptr *results = new Results_ptr;
    // std::unique_ptr<Results_ptr> results = std::make_unique<Results_ptr>()

    int *num_bonds_atom = new int[n_atoms]();
    for (int i=0; i<rows; i++){
        int idx = nbr_atoms[i * cols];
        num_bonds_atom[idx] ++;
    }

    int num_tps = 0;
    for (int i=0; i<n_atoms; i++){
        n_bond_pairs_atom[i] = num_bonds_atom[i] * (num_bonds_atom[i]-1);
        num_tps += n_bond_pairs_atom[i];
    }

    if (idx_mapping != NULL){
        for (int i=0; i<rows; i++){
            n_bond_pairs_bond[idx_mapping[i]] = num_bonds_atom[nbr_atoms[i * cols]] - 1; 
            }
    }
    else {
        for (int i=0; i<rows; i++){
            n_bond_pairs_bond[i] = num_bonds_atom[nbr_atoms[i * cols]] - 1; 
            }
    }

    int *bond_pairs = new int[num_tps * 2];

    int index = 0;
    int tmp = 0;

    for (int i = 0; i < n_atoms; i++) {
        int n = num_bonds_atom[i];

        if (n > 0){

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (j != k) { 
                    if (idx_mapping != NULL){
                        bond_pairs[index * 2] = idx_mapping[tmp + j];
                        bond_pairs[index * 2 + 1] = idx_mapping[tmp + k];
                    }
                    else {
                        bond_pairs[index * 2] = tmp + j;
                        bond_pairs[index * 2 + 1] = tmp + k;
                    }
                    index++;
                    }
                }
            }
        }
        tmp += n;
    }

    results->bond_pairs = new int[num_tps * 2];
    
    std::copy(bond_pairs, bond_pairs + num_tps * 2, results->bond_pairs);

    delete[] bond_pairs;
    delete[] num_bonds_atom;
    results->num_tps = num_tps;
    return results;
}   

Results_ptr2 *find_neighbors_cc(const double *coords, const double *lattice, const int *cell_shift, const int *pbc, const int *num_cell_xyz, const int *range_xyz, const int *cell_index_1d, const int *cell_index, const int cell_index_1d_max, const int n_atoms, const double cutoff){

    Results_ptr2 *results = new Results_ptr2;
    std::unordered_map<int, std::vector<int>> cell_rmap;
        for (int ii=0; ii<n_atoms; ii++){
            int idx = cell_index_1d[ii];
            cell_rmap[idx].push_back(ii);
         }
    
    std::vector<int> center_indices, neigh_indices, offset_list;
    std::vector<double> dist_list;
    for (int center_idx=0; center_idx<n_atoms; center_idx++){
        int cx_atom_cell_idx = cell_index[center_idx * 3 + 0];
        int cy_atom_cell_idx = cell_index[center_idx * 3 + 1];
        int cz_atom_cell_idx = cell_index[center_idx * 3 + 2];

        for (int dx=-range_xyz[0]; dx<=range_xyz[0]; dx++){
            for (int dy=-range_xyz[1]; dy<=range_xyz[1]; dy++){
                for (int dz=-range_xyz[2]; dz<=range_xyz[2]; dz++){
                    int neigh_x = mmod(cx_atom_cell_idx + dx, num_cell_xyz[0]);
                    int neigh_y = mmod(cy_atom_cell_idx + dy, num_cell_xyz[1]);
                    int neigh_z = mmod(cz_atom_cell_idx + dz, num_cell_xyz[2]);

                    int shift_x = mdiv(cx_atom_cell_idx + dx, num_cell_xyz[0]);
                    int shift_y = mdiv(cy_atom_cell_idx + dy, num_cell_xyz[1]);
                    int shift_z = mdiv(cz_atom_cell_idx + dz, num_cell_xyz[2]);

                    if (pbc[0] == 0){
                        shift_x = 0;
                        }
                    if (pbc[1] == 0){
                        shift_y = 0;
                        }
                    if (pbc[2] == 0){
                        shift_z = 0;
                        }

                    int search_cell_idx = neigh_x + num_cell_xyz[0] * (neigh_y + num_cell_xyz[1] * neigh_z);
                    if (cell_rmap.find(search_cell_idx) == cell_rmap.end()){
                            continue;
                        }
                        else {
                            for (int p=0; p < cell_rmap[search_cell_idx].size(); p++){
                                int j_atom_idx = cell_rmap[search_cell_idx][p];
                                int offset_xyz[3];

                                offset_xyz[0] = shift_x + cell_shift[center_idx * 3 + 0] - cell_shift[j_atom_idx * 3 + 0];
                                offset_xyz[1] = shift_y + cell_shift[center_idx * 3 + 1] - cell_shift[j_atom_idx * 3 + 1];
                                offset_xyz[2] = shift_z + cell_shift[center_idx * 3 + 2] - cell_shift[j_atom_idx * 3 + 2];

                                double dist_vec[3];

                                for (int kk=0; kk<3; kk++){
                                    double tmp_val = 0;
                                    for (int kkk=0; kkk<3; kkk++){
                                        tmp_val += offset_xyz[kkk] * lattice[kkk*3 + kk];
                                    }
                                    dist_vec[kk] = coords[j_atom_idx * 3 + kk] - coords[center_idx * 3 + kk] + tmp_val;
                                }

                                double dist_r = std::sqrt(dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] + dist_vec[2] * dist_vec[2]);
                            
                                if ((dist_r <= cutoff) && (center_idx != j_atom_idx)){
                                    center_indices.push_back(center_idx);
                                    neigh_indices.push_back(j_atom_idx);
                                    dist_list.push_back(dist_r);
                                    for (int kk=0; kk<3; kk++){
                                        offset_list.push_back(offset_xyz[kk]);
                                    }
                                }
                                
                            }
                        }
                }
            }
        }
        

    }

    results->center_indices = center_indices;
    results->neigh_indices = neigh_indices;
    results->dist_list = dist_list;
    results->offset_list = offset_list;

    return results;
}   