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
    this->learnerWeights = Vector(0.0, this->nEstimators);
}

void TWRF::predict(const Matrix &test, const Matrix &train, Vector &preds) {
    const int N = test.n;
    // Collect all predictions.
    Matrix labels(this->nEstimators, N, 0.0);
    for (int i = 0; i < this->nEstimators; ++i) {
        this->estimators[i]->predict(test, labels[i]);
    }
    // Aggregate predictions with weights.
    this->getWeights(train);
    for (int i = 0; i < N; ++i) {
        Vector votes(this->nEstimators), dist(0.0, this->nCategories);
        labels.col(i, votes);
        for (int j = 0; j < this->nEstimators; ++j) {
            dist[votes[j]] += this->learnerWeights[j];
        }
        preds[i] = argmax(dist);
    }
}

void TWRF::getWeights(const Matrix &train) {
    for (int i = 0; i < this->nEstimators; ++i) {
        // Initialize OOB data.
        int *oobIdx = this->oobIndexes[i];
        const int N = oobIdx[0];
        Matrix oob(N, train.m, 0.0);
        for (int j = 1; j < N + 1; ++j) oob[j - 1] = train[oobIdx[j]];

        // Compute accuracy.
        Vector right(0.0, N), labels(0.0, N), preds(0.0, N);
        oob.col(-1, labels);
        this->estimators[i]->predict(oob, preds);
        right[preds == labels] = 1.0;

        this->learnerWeights[i] = right.sum() / N;
    }
}
