#include "BTA.h"
#include "Metric.h"

using namespace wrf;

BTA::BTA(
    int nEstimators,
    int maxDepth,
    float maxSamplesRatio,
    MaxFeature maxFeatures,
    int minSamplesSplit,
    int minSamplesLeaf,
    double minSplitGain
): RandomForestClassifier(
    nEstimators, maxDepth, maxSamplesRatio, maxFeatures, minSamplesSplit, minSamplesLeaf, minSplitGain
) {}

Vector BTA::predict(const Matrix &test, const Matrix &train) {
    const int C = this->nCategories, N = this->nEstimators;

    std::vector<Matrix> CMs;
    CMs.reserve(N);
    for (int i = 0; i < N; ++i) {
        // Initialize OOB data.
        int *oobIdx = this->oobIndexes[i];
        const int n = oobIdx[0];
        Matrix oob(n, train.m, 0.0);
        for (int j = 1; j < n + 1; ++j) oob[j - 1] = train[oobIdx[j]];

        // Initialize confusion matrix.
        CMs.emplace_back(C, 0.0);
        Matrix &cm = CMs[i];  // i: true, j: pred
        Vector Y = oob.col(-1);
        Vector L = this->estimators[i]->predict(oob);
        for (int j = 0; j < n; ++j) cm[Y[j]][L[j]] += 1.0;
    }

    Vector res(test.n);
    // Probability P(y) can be estimated from the prevalence of y in train-set.
    Vector P_y = distribution(train.col(-1), C);
    P_y = log(P_y / train.n);

    for (int i = 0; i < test.n; ++i) {
        Vector probs(0.0, C);
        for (int c = 0; c < C; ++c) {
            // Conditional probability can be estimated from the confusion matrix.
            double CP = 0.0;
            for (int k = 0; k < this->nEstimators; ++k) {
                const Matrix &cm = CMs[k];
                double y = this->estimators[k]->predict(test[i]);
                double Ny = cm[c].sum();
                // Conditional probabilities smoothing.
                double p = sqrt((cm[c][y] + 1.0 / C) / (Ny + 1.0));
                CP += log(p);
            }
            probs[c] = P_y[c] + CP;
        }
        res[i] = argmax(probs);
    }
    return res;
}
