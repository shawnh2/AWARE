#include "Matrix.h"

using namespace wrf;

Matrix::Matrix(int n, int m, double a) {
    this->n = n;
    this->m = m;
    // Init data.
    this->rows = new Vector[n];
    for (int i = 0; i < n; ++i) this->rows[i] = Vector(a, m);
}

void Matrix::col(int i, Vector &out) const {
    int k = 0, at = i < 0 ? i + this->m : i;
    while (k < this->n) {
        out[k] = this->rows[k][at];
        ++k;
    }
}

void Matrix::col(int i, const Indexes &idx, Vector &out) const {
    int k = 0, at = i < 0 ? i + this->m : i, N = idx.size();
    while (k < N) {
        out[k] = this->rows[idx[k]][at];
        ++k;
    }
}

Vector& Matrix::operator[](unsigned int i) const {
    return this->rows[i];
}
