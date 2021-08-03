#ifndef WRF_DATASET_H
#define WRF_DATASET_H

#include "Matrix.h"

#include <string>

namespace wrf {

    // Define some information about dataset.
    typedef struct datasetInfo {
        int width;
        int height;
        int categories;
        std::string name;
    } DatasetInfo;

    // + Add more dataset here.
    static DatasetInfo ds_Abalone     = { 9,  4177, 28, "abalone"};
    static DatasetInfo ds_Breast      = {31,   569,  2, "breast"};
    static DatasetInfo ds_Credit      = {25,  1000,  2, "credit"};
    static DatasetInfo ds_Ecoli       = { 8,   336,  8, "ecoli"};
    static DatasetInfo ds_Glass       = {11,   214,  7, "glass"};
    static DatasetInfo ds_Hcv         = {13,   615,  4, "hcv"};
    static DatasetInfo ds_Image       = {20,  2310,  7, "image"};
    static DatasetInfo ds_Immuno      = { 8,    90,  2, "immuno"};
    static DatasetInfo ds_Letter      = {17, 20000, 26, "letter"};
    static DatasetInfo ds_Liver       = { 7,   345,  2, "liver"};
    static DatasetInfo ds_Nursery     = { 9, 12960,  5, "nursery"};
    static DatasetInfo ds_Parkinsons  = {23,   195,  2, "parkinsons"};
    static DatasetInfo ds_Shuttle     = {10, 58000,  7, "shuttle"};
    static DatasetInfo ds_Sonar       = {61,   208,  2, "sonar"};
    static DatasetInfo ds_Transfusion = { 5,   748,  2, "transfusion"};
    static DatasetInfo ds_Waveform    = {22,  5000,  3, "waveform"};
    static DatasetInfo ds_Wilt        = { 6,  4839,  2, "wilt"};
    static DatasetInfo ds_Yeast       = { 9,  1484, 10, "yeast"};
    // +

    // Load dataset and split it into train and test set.
    void loadDataset(const DatasetInfo &dataset, Matrix &trainSet, Matrix &testSet, unsigned randomState = 0);
}

#endif //WRF_DATASET_H
