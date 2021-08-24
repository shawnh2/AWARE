#include "RF.h"
#include "Metric.h"

#include <numeric>

using namespace wrf;

RandomForestClassifier::RandomForestClassifier(
    int nEstimators,
    int maxDepth,
    float maxSamplesRatio,
    MaxFeature maxFeatures,
    int minSamplesSplit,
    int minSamplesLeaf,
    double minSplitGain
) {
    this->nEstimators = nEstimators;
    this->maxSamplesRatio = maxSamplesRatio;
    this->maxFeatures = maxFeatures;
    this->maxDepth = maxDepth;
    this->minSamplesSplit = minSamplesSplit;
    this->minSamplesLeaf = minSamplesLeaf;
    this->minSplitGain = minSplitGain;

    this->oobIndexes = std::vector<std::valarray<int>>(nEstimators);
    // Initialize estimators.
    this->estimators.reserve(nEstimators);
}

RandomForestClassifier::~RandomForestClassifier() {
    for (int i = 0; i < this->nEstimators; ++i) {
        delete this->estimators[i];
    }
    this->estimators.clear();
    this->oobIndexes.clear();
}

void RandomForestClassifier::fit(const Matrix &train, int categories, int randomState) {
    const int N = train.n, k = train.m - 1;  // except labels
    switch (this->maxFeatures) {
        case MaxFeature::SQRT:
            this->nFeatures = sqrt(k);
            break;
        case MaxFeature::LOG2:
            this->nFeatures = log2(k);
            break;
        case MaxFeature::ALL:
            this->nFeatures = k;
            break;
    }
    this->nSamples = int(N * this->maxSamplesRatio);
    this->nCategories = categories;

    int i = 0;
    this->randomEngine = std::default_random_engine(randomState < 0 ? time(nullptr) : randomState);
    // Fitting each base estimator.
    while (i < this->nEstimators) {
        // Get training data from bootstrap.
        Matrix subTrain(this->nSamples, this->nFeatures + 1, 0.0);
        Indexes featuresIdx(k);
        this->oobIndexes[i] = std::valarray<int>(1, N);
        this->bootstrap(train, subTrain, featuresIdx, i);

        // Feed training data to train CART.
        CART *cart = new CART(
            this->maxDepth,
            this->minSamplesSplit,
            this->minSamplesLeaf,
            this->minSplitGain
        );
        cart->fit(subTrain, featuresIdx, categories);
        this->estimators[i++] = cart;
    }
}

Vector RandomForestClassifier::predict(const Matrix &test) {
    const int N = test.n;

    // Collect all predictions.
    Matrix labels(this->nEstimators, N);
    for (int i = 0; i < this->nEstimators; ++i) {
        labels[i] = this->estimators[i]->predict(test);
    }

    // Aggregate predictions with majority votes.
    Vector preds(0.0, N);
    for (int i = 0; i < N; ++i) {
        Vector votes = labels.col(i);
        Vector dist = distribution(votes, this->nCategories);
        preds[i] = argmax(dist);
    }
    return preds;
}

void RandomForestClassifier::bootstrap(
    const Matrix &train,
    Matrix &subTrain,
    Indexes &featuresIdx,
    int epoch
) {
    int i;
    const int H = train.n;

    // Draw samples from train set with replacement.
    int samplesIdx[this->nSamples];
    // Collect the out-of-bag samples.
    std::valarray<int> &oobIdx = this->oobIndexes[epoch];
    std::uniform_int_distribution<int> rdis(0, H - 1);
    for (i = 0; i < this->nSamples; ++i) {
        int pos = rdis(this->randomEngine);
        samplesIdx[i] = pos;
        oobIdx[pos] = 0;
    }

    // Draw features from train set without replacement.
    std::iota(featuresIdx.begin(), featuresIdx.end(), 0);
    std::shuffle(featuresIdx.begin(), featuresIdx.end(), this->randomEngine);
    featuresIdx.resize(this->nFeatures);

    // Generating a sub set.
    i = 0;
    while (i < this->nSamples) {
        int j = 0;
        Vector &src = train[samplesIdx[i]];
        Vector &dst = subTrain[i];
        while (j < this->nFeatures) {
            dst[j] = src[featuresIdx[j]];
            ++j;
        }
        dst[j] = src[train.m - 1]; // The last column will be labels.
        ++i;
    }
}
