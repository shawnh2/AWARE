#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <random>
#include <numeric>

void wrf::loadDataset(const DatasetInfo &dataset, Matrix &out) {
    const int H = dataset.height, W = dataset.width;

    // Load dataset from csv file.
    std::ifstream fin("../dataset/" + dataset.name + ".csv");
    std::string line;

    // Clip if i >= height.
    int i = 0;
    while (getline(fin, line) && i < H) {
        std::istringstream sin(line);
        std::string value;
        // Clip if j >= width.
        int j = 0;
        Vector &row = out[i];
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
