#include "WAVE.h"
#include "Metric.h"

using namespace wrf;

WAVE::WAVE(
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

void WAVE::predict(const Matrix &test, const Matrix &train, Vector &preds) {
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
            dist[votes[j]] += this->estimatorsW[j];
        }
        preds[i] = argmax(dist);
    }
}

void WAVE::getWeights(const Matrix &train) {
    const int N = train.n, K = this->nEstimators;
    Matrix X_t(K, N, 0.0);
    Vector y(N);
    train.col(-1, y);
    for (int i = 0; i < K; ++i) {
        Vector pred(N);
        this->estimators[i]->predict(train, pred);
        X_t[i][pred == y] = 1.0;
    }

    // X is Performance Matrix: indicating whether the classification is right (1) or wrong (0).
    Matrix X = X_t.T();
    Matrix J_nk(N, K, 1.0), J_kk(K, K, 1.0), I_k(K, 1.0);
    Vector i_k(1.0, K);

    // Set an initial instance weight vector.
    Matrix cmn = (J_nk - X) * (J_kk - I_k);  // n*k
    Vector num1 = cmn * i_k;  // n*1;
    Vector Q = num1 / num1.sum();
    // Set an initial classifiers weight vector.
    Vector P(0.0, K);

    // Stop the iteration when the weight vectors become stable.
    for (int m = 0; m < this->nFeatures; ++m) {
        // Calculate a classifier weight vector.
        Vector num2 = X_t * Q;  // k*1
        P = num2 / num2.sum();
        // Update the instance weight vector.
        Vector num3 = cmn * P;  // n*1
        Q = num3 / num3.sum();
    }
    this->estimatorsW = P;
}
