#ifndef WRF_DATASET_H
#define WRF_DATASET_H

#include "Matrix.h"

#include <string>
#include <array>

namespace wrf {

    // Define some information about dataset.
    typedef struct datasetInfo {
        int width;
        int height;
        int categories;
        std::string name;
    } DatasetInfo;

    // + Add more dataset here.
    static std::array<DatasetInfo, 20> DatasetList = {
        DatasetInfo {31, 569, 2, "breast"},
        DatasetInfo {7, 1728, 4, "car"},
        DatasetInfo {25, 1000, 2, "credit"},
        DatasetInfo {8, 336, 8, "ecoli"},
        DatasetInfo {28, 523, 4, "forest"},
        DatasetInfo {11, 214, 7, "glass"},
        DatasetInfo {13, 615, 4, "hcv"},
        DatasetInfo {20, 2310, 7, "image"},
        DatasetInfo {8, 90, 2, "immuno"},
        DatasetInfo {17, 20000, 26, "letter"},
        DatasetInfo {7, 345, 2, "liver"},
        DatasetInfo {9, 12960, 5, "nursery"},
        DatasetInfo {23, 195, 2, "parkinsons"},
        DatasetInfo {10, 58000, 7, "shuttle"},
        DatasetInfo {61, 208, 2, "sonar"},
        DatasetInfo {17, 470, 2, "thoraric"},
        DatasetInfo {10, 958, 2, "tic-tac-toe"},
        DatasetInfo {5, 748, 2, "transfusion"},
        DatasetInfo {22, 5000, 3, "waveform"},
        DatasetInfo {6, 4839, 2, "wilt"}
    };

    // Load dataset from disk and shuffle it if random state >= 0.
    void loadDataset(const DatasetInfo &dataset, Matrix &out, int randomState = -1);

    // Split dataset into train and test set randomly.
    void trainTestSplit(const Matrix &src, Matrix &train, Matrix &test, unsigned randomState = 0);

    // Split dataset into train and test set by the begin and end index of test set.
    void trainTestSplit(const Matrix &src, const int *ridx, Matrix &train, Matrix &test, int testA, int testZ);
}

#endif //WRF_DATASET_H
