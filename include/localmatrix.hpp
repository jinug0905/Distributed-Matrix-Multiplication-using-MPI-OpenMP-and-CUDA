#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include "common.hpp"

class LocalMatrix {
public:
    int g_rows, g_cols;     // Global shape
    int l_rows, l_cols;     // Local shape
    int row_off, col_off;   // offsets of this tile

    Dist2D grid;                // Process grid info (P x P)

    std::vector<double> vals;   // Local value for calculation : [i * l_cols + j]

    LocalMatrix(int g_m, int g_n, const Dist2D& d_) : g_rows(g_m), g_cols(g_n), grid(d_){
        // sizes and offsets
        std::vector<int> row_sizes, row_offs, col_sizes, col_offs;
        split_sizes(g_rows, grid.P, row_sizes, row_offs);
        split_sizes(g_cols, grid.P, col_sizes, col_offs);

        // Local shape + starting offsets
        l_rows = row_sizes[grid.myr];
        l_cols = col_sizes[grid.myc];
        row_off = row_offs[grid.myr];
        col_off = col_offs[grid.myc];

        // initialize with zero for local tiles
        vals.assign(static_cast<size_t>(l_rows) * static_cast<size_t>(l_cols), 0.0);
    }

    // For C matrix later, initialize to zero
    void zero() {
        std::fill(vals.begin(), vals.end(), 0.0);
    }

    // Initialize A(i,j) = i + j
    void initialize_A() {
        for (int li = 0; li < l_rows; ++li) {
            const int gi = row_off + li;
            for (int lj = 0; lj < l_cols; ++lj) {
                const int gj = col_off + lj;
                vals[static_cast<size_t>(li) * l_cols + lj] = static_cast<double>(gi + gj);
            }
        }
    }

    // Initialize B(i,j) = i - j
    void initialize_B() {
        for (int li = 0; li < l_rows; ++li) {
            const int gi = row_off + li;
            for (int lj = 0; lj < l_cols; ++lj) {
                const int gj = col_off + lj;
                vals[static_cast<size_t>(li) * l_cols + lj] = static_cast<double>(gi - gj);
            }
        }
    }
};