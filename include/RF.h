#ifndef WRF_RF_H
#define WRF_RF_H

#include "CART.h"

#include <random>

namespace wrf {

    class RandomForestClassifier {
    public:
        // The number of base estimators in forest.
        int nEstimators;

        // The number of samples to draw from train-set to train each base estimator.
        float maxSamplesRatio;

        /* The number of features to consider when looking for the best split.
        Should be "sqrt" or "log2". If not, then it will consider all features. */
        std::string maxFeatures;

        /* The maximum depth of tree. If is -1, then nodes are expanded until all leaves are pure
        or until all leaves contain less than minSamplesSplit samples. */
        int maxDepth;

        // The minimum number of samples required to split an interal node.
        int minSamplesSplit;

        // The minimum number of samples required to be at a leaf node.
        int minSamplesLeaf;

        // The minimum information gain of one split.
        double minSplitGain;

        explicit RandomForestClassifier(
            int nEstimators = 100,
            int maxDepth = 10,
            int randomState = -1,
            float maxSamplesRatio = 0.8,
            const std::string &maxFeatures = "sqrt",
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            double minSplitGain = 0.0
        );

        void fit(const Matrix &train, int categories);

        void predict(const Matrix &test, Vector &preds);

    protected:
        // The number of features and samples while Bagging.
        int nFeatures = 0, nSamples = 0;

        // The categories number of current training dataset.
        int nCategories = 0;

        std::default_random_engine randomEngine;

        // Store all the base estimators.
        std::vector<CART*> estimators;

        // Store the indexes of out-of-bag.
        std::vector<int*> oobIndexes;

        // Sample some data from train set to sub set.
        void bootstrap(
            const Matrix &train,
            Matrix &subTrain,
            Indexes &featuresIdx,
            int epoch
        );
    };

}

#endif //WRF_RF_H
