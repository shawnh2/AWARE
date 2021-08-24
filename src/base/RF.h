#ifndef WRF_RF_H
#define WRF_RF_H

#include "CART.h"

#include <random>

namespace wrf {

    enum class MaxFeature {SQRT, LOG2, ALL};

    class RandomForestClassifier {
    public:
        // The number of base estimators in forest.
        int nEstimators;

        // The number of samples to draw from train-set to train each base estimator.
        float maxSamplesRatio;

        // The number of features to consider when looking for the best split.
        MaxFeature maxFeatures;

        // maxDepth, minSamplesSplit, minSamplesLeaf, minSplitGain are used for CART construction.
        int maxDepth;
        int minSamplesSplit;
        int minSamplesLeaf;
        double minSplitGain;

        explicit RandomForestClassifier(
            int nEstimators = 100,
            int maxDepth = 10,
            float maxSamplesRatio = 0.8,
            MaxFeature maxFeatures = MaxFeature::SQRT,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            double minSplitGain = 0.0
        );

        ~RandomForestClassifier();

        void fit(const Matrix &train, int categories, int randomState = -1);

        Vector predict(const Matrix &test);

    protected:
        // The number of features and samples while Bagging.
        int nFeatures{0}, nSamples{0};

        // The categories number of current training dataset.
        int nCategories{0};

        // Store all the base estimators.
        std::vector<CART*> estimators;

        // Store the indexes of out-of-bag indicated by 1.
        std::vector<std::valarray<int>> oobIndexes;

        // Random engine for bootstrap.
        std::default_random_engine randomEngine;

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
