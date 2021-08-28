#include "DIRF.h"
#include "Metric.h"

using namespace wrf;

DIRF::DIRF(
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

Vector DIRF::predict(const Matrix &test, const Matrix &train) {
    const int N1 = test.n, N2 = train.n;
    const int M = train.m - 1; // Except labels.
    const int K = N2 / 3;      // Select K nearest neighbours of the new instance.
    const Vector Y = train.col(-1);

    // Compute the width of the range of values of the features on training set.
    Vector range(0.0, M);
    for (int i = 0; i < M; ++i) {
        const Vector &y = train.col(i);
        range[i] = y.max() - y.min() + 1e-5;
    }

    Vector res(0.0, N1);
    // Compute the distance between the new instance and all training instances.
    for (int i = 0; i < N1; ++i) {
        const Vector &x_new = test[i];
        // The distance-based weight coefficients reflect similarity between the two distances.
        Vector coff(0.0, N2);
        for (int j = 0; j < N2; ++j) {
            const Vector &x = train[j];
            // Compute the heterogeneous euclidean/overlap metric in the instance spaces.
            double heom = 0.0;
            for (int k = 0; k < M; ++k) {
                double dis = std::abs(x_new[k] - x[k]) / range[k];
                heom += pow(dis, 2);
            }
            coff[j] = 1.0 / sqrt(heom);
        }

        // Compute the margin of each tree on the k-th nearest neighbours of new instance.
        Indexes nearest(N2);
        std::iota(nearest.begin(), nearest.end(), 0);
        std::sort(nearest.begin(), nearest.end(), [&](int a, int b) noexcept -> bool {
            return coff[a] < coff[b];
        });

        Vector dist(0.0, this->nCategories);
        for (int j = 0; j < this->nEstimators; ++j) {
            CART *estimator = this->estimators[j];
            double pred = estimator->predict(x_new);
            // Compute the weight of each tree.
            double w1 = 0.0, w2 = 0.0;
            for (int k = 0; k < K; ++k) {
                const int at = nearest[k];
                // Margin is defined as usual for a classifier with crisp outputs.
                // 1 for a correct prediction, and -1 for a wrong one.
                double mrg = estimator->predict(train[at]) == Y[at] ? 1.0 : -1.0;
                // Indicator indicates whether this training instance belongs to OOB.
                double ind = this->oobIndexes[j][at];
                w1 += ind * coff[at] * mrg;
                w2 += ind * coff[at] + 1e-5;
            }
            dist[pred] += w1 / w2;
        }

        res[i] = argmax(dist);
    }
    return res;
}
