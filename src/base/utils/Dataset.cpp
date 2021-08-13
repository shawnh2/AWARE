#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <random>
#include <numeric>

void wrf::loadDataset(const DatasetInfo &dataset, Matrix &out, int randomState) {
    const int H = dataset.height, W = dataset.width;
    std::ifstream fin("../dataset/" + dataset.name + ".csv");
    std::string line;

    // Get the indexes of where to put this line of data.
    Indexes idxes(H);
    std::iota(idxes.begin(), idxes.end(), 0);
    if (randomState >= 0) {
        std::shuffle(idxes.begin(), idxes.end(), std::default_random_engine(randomState));
    }

    // Clip if i >= height.
    int i = 0;
    while (getline(fin, line) && i < H) {
        std::istringstream sin(line);
        std::string value;
        // Clip if j >= width.
        int j = 0;
        Vector &row = out[idxes[i]];
        while (getline(sin, value, ',') && j < W) {
            row[j] = stod(value);
            ++j;
        }
        ++i;
    }
}

void wrf::trainTestSplit(const Matrix &src, Matrix &train, Matrix &test, unsigned int randomState) {
    const int H = src.n, N = train.n, M = test.n;

    // Get a random indexes map.
    Indexes idx(H);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(randomState));
    // Split train.
    int i = 0;
    while (i < N) {
        train[i] = src[idx[i]];
        ++i;
    }
    // Split test.
    int j = 0;
    while (j < M) {
        test[j] = src[idx[i]];
        ++j;
        ++i;
    }
}

void wrf::trainTestSplit(const Matrix &src, const int *ridx, Matrix &train, Matrix &test, int testA, int testZ) {
    const int N = src.n;
    int i = 0, x = 0, y = 0;
    while (i < testA) {
        train[x] = src[ridx[i]];
        ++x;
        ++i;
    }
    while (i <= testZ) {
        test[y] = src[ridx[i]];
        ++y;
        ++i;
    }
    while (i < N) {
        train[x] = src[ridx[i]];
        ++x;
        ++i;
    }
}
