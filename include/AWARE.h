#ifndef WRF_AWARE_H
#define WRF_AWARE_H

#include "RF.h"

namespace wrf {
    /*
    Reference:
    [1] Zhu, Zou, Hastie. Multi-class AdaBoost. Statistics and its Interface, 2006.
     */
    class AWARE: public RandomForestClassifier {
    public:
        explicit AWARE(
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

#endif //WRF_AWARE_H
