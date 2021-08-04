#ifndef WRF_WAVE_H
#define WRF_WAVE_H

#include "RF.h"

namespace wrf {
    /*
        This weighted voting classification ensemble method uses two weight
    vectors: a weight vector of classifiers and a weight vector of instances.
    The instance weight vector assigns higher weights to observations that are
    hard to classify. The weight vector of classifiers puts larger weights on
    classifiers that perform better on hard-to-classify instances.

    Reference:
    [1] Kim, Moon, Ahn. A Weight-Adjusted Voting Algorithm for Ensembles of Classifiers.
        Journal of the Korean Statistical Society, 2011.
     */
    class WAVE: public RandomForestClassifier {
    public:
        explicit WAVE(
            int nEstimators = 100,
            int maxDepth = 10,
            int randomState = -1,
            float maxSamplesRatio = 0.8,
            const std::string &maxFeatures = "sqrt",
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            double minSplitGain = 0.0
        );

        Vector predict(const Matrix &test, const Matrix &train);

    private:
        Vector estimatorsW;

        void getWeights(const Matrix &train);
    };
}

#endif //WRF_WAVE_H
