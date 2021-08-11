#include "WAVE.h"
#include "Metric.h"

using namespace wrf;

WAVE::WAVE(
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

Vector WAVE::predict(const Matrix &test, const Matrix &train) {
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
        Vector votes = labels.col(i), dist(0.0, this->nCategories);
        for (int j = 0; j < this->nEstimators; ++j) {
            dist[votes[j]] += this->estimatorsW[j];
        }
        preds[i] = argmax(dist);
    }
    return preds;
}

void WAVE::getWeights(const Matrix &train) {
    const int N = train.n, K = this->nEstimators;
    Matrix X_t(K, N, 0.0);
    Vector y = train.col(-1);
    for (int i = 0; i < K; ++i) {
        Vector pred = this->estimators[i]->predict(train);
        X_t[i][pred == y] = 1.0;
    }

    // X is Performance Matrix: indicating whether the classification is right (1) or wrong (0).
    Matrix X = X_t.T(), J_nk(N, K, 1.0), J_kk(K, K, 1.0), I_k(K, 1.0);
    Vector i_k(1.0, K);

    // Set an initial instance weight vector and a classifiers weight vector.
    Matrix cmn = (J_nk - X) * (J_kk - I_k);
    Vector Q = cmn * i_k, P;
    Q /= Q.sum();

    // Stop the iteration when the weight vectors become stable.
    int epoch = std::max(this->nFeatures, 3);
    for (int m = 0; m < epoch; ++m) {
        // Calculate a classifier weight vector.
        P = X_t * Q;
        P /= P.sum();
        // Update the instance weight vector.
        Q = cmn * P;
        Q /= Q.sum();
    }
    this->estimatorsW = P;
}
