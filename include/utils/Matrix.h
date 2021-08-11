#ifndef WRF_MATRIX_H
#define WRF_MATRIX_H

#include <valarray>
#include <vector>

namespace wrf {

    typedef std::valarray<double> Vector;
    typedef std::vector<int> Indexes;

    class Matrix {
    public:
        // The number of rows and columns.
        int n, m;

        // Constructor
        Matrix(int n, int m, double a);
        Matrix(int n, double eye);
        Matrix(int n, int m);

        // Destructor
        ~Matrix();

        // Get one column of data in matrix.
        Vector col(int i) const;
        // Get one column of data with selected rows in matrix.
        void col(int i, const Indexes &idx, Vector &out) const;

        // Transpose
        Matrix T() const;

        // Matrix[i]: access row
        Vector& operator[](unsigned int i) const;
        // Matrix1(i*j) - Matrix2(i*j) = Matrix(i*j)
        Matrix operator-(const Matrix &mat) const;
        // Matrix1(i*j) * Matrix2(j*k) = Matrix(i*k)
        Matrix operator*(const Matrix &mat) const;
        // Matrix1(i*j) * Vector(j*1) = Vector(i*1)
        Vector operator*(const Vector &vec) const;

    private:
        // Values
        Vector *rows;
    };
}

#endif //WRF_MATRIX_H
