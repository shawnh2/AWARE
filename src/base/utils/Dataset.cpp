#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <random>

void wrf::loadDataset(const DatasetInfo &dataset, Matrix &trainSet, Matrix &testSet, unsigned randomState) {
    const int H = dataset.height, W = dataset.width, N = trainSet.n;

    // Load dataset from csv file.
    std::ifstream fin("../dataset/" + dataset.name + ".csv");
    std::string line;

    // Get a random indexes map.
    int i;
    Indexes idx(H);
    for (i = 0; i < H; ++i) idx[i] = i;
    std::shuffle(std::begin(idx), std::end(idx), std::default_random_engine(randomState));

    // Clip if i >= height.
    i = 0;
    while (getline(fin, line) && i < H) {
        std::istringstream sin(line);
        std::string value;
        // Clip if j >= width.
        int j = 0, to = idx[i++];
        Vector &row = to < N ? trainSet[to] : testSet[to - N];
        while (getline(sin, value, ',') && j < W) {
            row[j++] = stod(value);
        }
    }
}
