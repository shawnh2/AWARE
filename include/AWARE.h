#ifndef WRF_AWARE_H
#define WRF_AWARE_H

#include "RF.h"

namespace wrf {

    /* Adaptive Weighted voting Aggregation for Ensemble.

    Reference:
    [1] Friedman, Hastie, Tibshirani. Additive logistic regression: A statistical view of boosting.
        Annals of Statistics, 2000.
    [2] Zhu, Zou, Hastie. Multi-class AdaBoost. Statistics and its Interface, 2006.
    */
    class AWARE: public RandomForestClassifier {
    public:
        explicit AWARE(
            int nEstimators = 100,
            int maxDepth = 10,
            float maxSamplesRatio = 0.8,
            MaxFeature maxFeatures = MaxFeature::SQRT,
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

#endif //WRF_AWARE_H
