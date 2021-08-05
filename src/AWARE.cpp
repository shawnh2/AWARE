#include "AWARE.h"
#include "Metric.h"

#include <numeric>

using namespace wrf;

AWARE::AWARE(
    int nEstimators,
    int maxDepth,
    int randomState,
    float maxSamplesRatio,
    MaxFeature maxFeatures,
    int minSamplesSplit,
    int minSamplesLeaf,
    double minSplitGain
): RandomForestClassifier(
    nEstimators, maxDepth, randomState, maxSamplesRatio,
    maxFeatures, minSamplesSplit, minSamplesLeaf, minSplitGain
) {
    this->estimatorsW = Vector(0.0, this->nEstimators);
}

Vector AWARE::predict(const Matrix &test, const Matrix &train) {
    const int N = test.n;

    // Collect all predictions.
    Matrix labels(this->nEstimators, N, 0.0);
    for (int i = 0; i < this->nEstimators; ++i) {
        labels[i] = this->estimators[i].predict(test);
    }

    // Aggregate predictions with weights.
    Vector preds(0.0, N);
    this->getWeights(train);
    for (int i = 0; i < N; ++i) {
        Vector y = labels.col(i), dist(0.0, this->nCategories);
        for (int j = 0; j < this->nEstimators; ++j) {
            Vector factor(-1.0 / (this->nCategories - 1), this->nCategories);
            factor[y[j]] = 1.0;
            dist += factor * this->estimatorsW[j];
        }
        preds[i] = argmax(dist);
    }
    return preds;
}

void AWARE::getWeights(const Matrix &train) {
    const int N = train.n, K = this->nEstimators, C = this->nCategories;

    // Initialize the performance matrix.
    Vector label = train.col(-1), count(0.0, K);
    Matrix wrong(K, N, 0.0);
    for (int i = 0; i < K; ++i) {
        Vector pred = this->estimators[i].predict(train);
        wrong[i][pred != label] = 1.0;
        count[i] = wrong[i].sum();
    }

    // Sort (argsort) the estimators by its performance.
    Indexes sorted(K);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) noexcept -> bool {
        return count[a] < count[b];
    });

    // Initialize the observation and estimators weights.
    Vector OW(1.0 / N, N), EW(0.0, K);
    for (int si : sorted) {
        // Compute error using weight.
        double err = (wrong[si] * OW).sum() / OW.sum();
        // compute alpha.
        double alpha = log(1 / (err + 1e-5) - 1) + log(C - 1);
        // Update and re-normalize observation weights.
        OW *= exp(alpha * wrong[si]);
        OW /= OW.sum();
        EW[si] = alpha;
    }
    // Re-normalize estimator weights.
    EW /= EW.sum();
    this->estimatorsW = EW;
}
