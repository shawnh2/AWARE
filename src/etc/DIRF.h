#ifndef WRF_DIRF_H
#define WRF_DIRF_H

#include "RF.h"

namespace wrf {

    /* Dynamic Integration with Random Forests.

    This algorithm, Dynamic Integration, which is based on local performance
    estimates of base predictors, can be used instead of majority voting.
    Dynamic integration always increases the margin. A bias/variance
    decomposition demonstrates that dynamic integration decreases the error
    by significantly decreasing the bias component while leaving the same
    or insignificantly increasing the variance.

    Reference:
    [1] Tsymbal, Pechenizkiy, Cunningham. Dynamic Integration with Random Forests. Machine Learning: ECML, 2006.
    [2] Tripoliti, Fotiadis, Argyropoulou, Manis. A six stage approach for the diagnosis of the Alzheimer's disease
        based on fMRI data. Journal of Biomedical Informatics, 2010.
    [3] Robnik-Sikonja. Improving Random Forests. Machine Learning: ECML, 2004.
    */
    class DIRF: public RandomForestClassifier {
    public:
        explicit DIRF(
            int nEstimators = 100,
            int maxDepth = 10,
            float maxSamplesRatio = 0.8,
            MaxFeature maxFeatures = MaxFeature::SQRT,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            double minSplitGain = 0.0
        );

        Vector predict(const Matrix &test, const Matrix &train);
    };

}

#endif //WRF_DIRF_H
