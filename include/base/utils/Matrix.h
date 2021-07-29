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

        // Get one column of data in matrix.
        void col(int i, Vector &out) const;
        void col(int i, const Indexes &idx, Vector &out) const;

        // Matrix[i]: access row
        Vector& operator[](unsigned int i) const;

    private:
        // Values
        Vector *rows;
    };
}

#endif //WRF_MATRIX_H
