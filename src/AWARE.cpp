#include "AWARE.h"
#include "Metric.h"

#include <numeric>

using namespace wrf;

AWARE::AWARE(
    int nEstimators,
    int maxDepth,
    float maxSamplesRatio,
    MaxFeature maxFeatures,
    int minSamplesSplit,
    int minSamplesLeaf,
    double minSplitGain
): RandomForestClassifier(
    nEstimators, maxDepth, maxSamplesRatio, maxFeatures, minSamplesSplit, minSamplesLeaf, minSplitGain
) {
    this->estimatorsW = Vector(0.0, this->nEstimators);
}

Vector AWARE::predict(const Matrix &test, const Matrix &train) {
    const int N = test.n;

    // Collect all predictions.
    Matrix labels(this->nEstimators, N);
    for (int i = 0; i < this->nEstimators; ++i) {
        labels[i] = this->estimators[i]->predict(test);
    }

    // Aggregate predictions with weights.
    Vector preds(0.0, N);
    this->getWeights(train);
    for (int i = 0; i < N; ++i) {
        Vector y = labels.col(i), dist(0.0, this->nCategories);
        for (int j = 0; j < this->nEstimators; ++j) {
            dist[y[j]] += this->estimatorsW[j];
        }
        preds[i] = argmax(dist);
    }
    return preds;
}

void AWARE::getWeights(const Matrix &train) {
    const int N = train.n, K = this->nEstimators;
    const Vector Y = train.col(-1);

    // Initialize prediction error.
    Vector predErr(0.0, K);
    for (int i = 0; i < K; ++i) {
        // Initialize OOB data.
        const std::valarray<int> &oobIdx = this->oobIndexes[i];
        const int n = oobIdx.sum(), size = oobIdx.size();
        Matrix oob(n, train.m);
        for (int j = 0, x = 0; j < size; ++j) {
            if (oobIdx[j] == 1) oob[x++] = train[j];
        }
        // Evaluate OOB prediction error.
        Vector y = oob.col(-1), pred = this->estimators[i]->predict(oob), wrong(0.0, n);
        wrong[y != pred] = 1.0;
        predErr[i] = wrong.sum() / n;
    }

    // Sort (argsort) the estimators by its performance.
    Indexes sorted(K);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) noexcept -> bool {
        return predErr[a] < predErr[b];
    });

    // Initialize the observation and estimators weights.
    Vector OW(1.0 / N, N), EW(0.0, K);
    for (int si : sorted) {
        // Compute error using weight.
        Vector pred = this->estimators[si]->predict(train), wrong(0.0, N);
        wrong[pred != Y] = 1.0;
        double err = (wrong * OW).sum() / OW.sum();
        // compute alpha.
        double alpha = log(1 / (err + 1e-6) - 1) + log(this->nCategories - 1);
        EW[si] = alpha;
        // Update and re-normalize weights.
        OW *= exp(alpha * wrong);
        OW /= OW.sum();
    }

    this->estimatorsW = EW / EW.sum();
}
