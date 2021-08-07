#ifndef WRF_BTA_H
#define WRF_BTA_H

#include "RF.h"

namespace wrf {

    /* Bayesian Tree Aggregation.

    The BTA algorithm is inspired by the naive Bayes classifier. It computes the
    conditional probability of the label, y, of sample x given the predictions of
    individual trees. The final classification label y* of sample x is given by
    the maximal conditional probability.

    Reference:
    [1] Brabec, Machlica. Decision-forest Voting Scheme for Classification of Rare Classes
        in Network Intrusion Detection. IEEE International Conference on Systems, Man, and Cybernetics, 2018.
    [2] Kuncheva, Rodriguez. A Weighted Voting Framework for Classifiers Ensembles.
        Knowledge and Information Systems, 2014.
    [3] Zhang, Wang. Weighted Random Forest Algorithm Based on Bayesian Algorithm.
        Journal of Physics: Conference Series, 2021.
    */
    class BTA: public RandomForestClassifier {
    public:
        explicit BTA(
            int nEstimators = 100,
            int maxDepth = 10,
            int randomState = -1,
            float maxSamplesRatio = 0.8,
            MaxFeature maxFeatures = MaxFeature::SQRT,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            double minSplitGain = 0.0
        );

        Vector predict(const Matrix &test, const Matrix &train);
    };

}

#endif //WRF_BTA_H
