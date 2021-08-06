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
    static std::array<DatasetInfo, 18> DatasetList = {
        DatasetInfo {9, 4177, 28, "abalone"},
        DatasetInfo {31, 569, 2, "breast"},
        DatasetInfo {25, 1000, 2, "credit"},
        DatasetInfo {8, 336, 8, "ecoli"},
        DatasetInfo {11, 214, 7, "glass"},
        DatasetInfo  {13, 615, 4, "hcv"},
        DatasetInfo {20, 2310, 7, "image"},
        DatasetInfo {8, 90, 2, "immuno"},
        DatasetInfo {17, 20000, 26, "letter"},
        DatasetInfo {7, 345, 2, "liver"},
        DatasetInfo {9, 12960, 5, "nursery"},
        DatasetInfo {23, 195, 2, "parkinsons"},
        DatasetInfo {10, 58000, 7, "shuttle"},
        DatasetInfo {61, 208, 2, "sonar"},
        DatasetInfo {5, 748, 2, "transfusion"},
        DatasetInfo {22, 5000, 3, "waveform"},
        DatasetInfo {6, 4839, 2, "wilt"},
        DatasetInfo {9, 1484, 10, "yeast"}
    };

    // Load dataset and split it into train and test set.
    void loadDataset(const DatasetInfo &dataset, Matrix &trainSet, Matrix &testSet, unsigned randomState = 0);
}

#endif //WRF_DATASET_H
