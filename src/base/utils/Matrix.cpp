#include "Matrix.h"

using namespace wrf;

Matrix::Matrix(int n, int m, double a) {
    this->n = n;
    this->m = m;
    this->rows = new Vector[n];
    for (int i = 0; i < n; ++i) this->rows[i] = Vector(a, m);
}

Matrix::Matrix(int n, double eye) {
    this->n = n;
    this->m = n;
    this->rows = new Vector[n];
    for (int i = 0; i < n; ++i) {
        this->rows[i] = Vector(0.0, n);
        this->rows[i][i] = eye;
    }
}

Vector Matrix::col(int i) const {
    Vector out(this->n);
    int k = 0, at = i < 0 ? i + this->m : i;
    while (k < this->n) {
        out[k] = this->rows[k][at];
        ++k;
    }
    return out;
}

void Matrix::col(int i, const Indexes &idx, Vector &out) const {
    int k = 0, at = i < 0 ? i + this->m : i;
    for(int pos : idx) {
        out[k] = this->rows[pos][at];
        ++k;
    }
}

Matrix Matrix::T() const {
    Matrix res(this->m, this->n, 0.0);
    for (int i = 0; i < this->m; ++i) {
        for (int j = 0; j < this->n; ++j) {
            res[i][j] = this->rows[j][i];
        }
    }
    return res;
}

Vector& Matrix::operator[](unsigned int i) const {
    return this->rows[i];
}

Matrix Matrix::operator-(const Matrix &mat) const {
    assert(this->n == mat.n && this->m == mat.m);

    Matrix res(this->n, this->m, 0.0);
    for (int i = 0; i < this->n; ++i) {
        for (int j = 0; j < this->m; ++j) {
            res.rows[i][j] = this->rows[i][j] - mat.rows[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &mat) const {
    assert(this->m == mat.n);

    Matrix res(this->n, mat.m, 0.0);
    for (int i = 0; i < this->n; ++i) {
        for (int j = 0; j < mat.m; ++j) {
            for (int k = 0; k < mat.m; ++k) {
                res.rows[i][j] += this->rows[i][k] * mat.rows[k][j];
            }
        }
    }
    return res;
}

Vector Matrix::operator*(const Vector &vec) const {
    assert(this->m == vec.size());

    Vector res(0.0, this->n);
    for (int i = 0; i < this->n; ++i) {
        for (int j = 0; j < this->m; ++j) {
            res[i] += this->rows[i][j] * vec[j];
        }
    }
    return res;
}
