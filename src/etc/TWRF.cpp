#include "TWRF.h"
#include "Metric.h"

using namespace wrf;

TWRF::TWRF(
    int nEstimators,
    int maxDepth,
    int randomState,
    float maxSamplesRatio,
    const std::string &maxFeatures,
    int minSamplesSplit,
    int minSamplesLeaf,
    double minSplitGain
): RandomForestClassifier(
    nEstimators, maxDepth, randomState, maxSamplesRatio,
    maxFeatures, minSamplesSplit, minSamplesLeaf, minSplitGain
) {
    this->estimatorsW = Vector(0.0, this->nEstimators);
}

Vector TWRF::predict(const Matrix &test, const Matrix &train) {
    const int N = test.n;

    // Collect all predictions.
    Matrix labels(this->nEstimators, N, 0.0);
    for (int i = 0; i < this->nEstimators; ++i) {
        labels[i] = this->estimators[i]->predict(test);
    }

    // Aggregate predictions with weights.
    Vector preds(0.0, N);
    this->getWeights(train);
    for (int i = 0; i < N; ++i) {
        Vector votes = labels.col(i), dist(0.0, this->nCategories);
        for (int j = 0; j < this->nEstimators; ++j) {
            dist[votes[j]] += this->estimatorsW[j];
        }
        preds[i] = argmax(dist);
    }
    return preds;
}

void TWRF::getWeights(const Matrix &train) {
    for (int i = 0; i < this->nEstimators; ++i) {
        // Initialize OOB data.
        int *oobIdx = this->oobIndexes[i];
        const int N = oobIdx[0];
        Matrix oob(N, train.m, 0.0);
        for (int j = 1; j < N + 1; ++j) oob[j - 1] = train[oobIdx[j]];

        // Compute accuracy.
        Vector right(0.0, N), labels = oob.col(-1);
        Vector preds = this->estimators[i]->predict(oob);
        right[preds == labels] = 1.0;

        this->estimatorsW[i] = right.sum() / N;
    }
}
