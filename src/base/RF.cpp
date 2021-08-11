#include "RF.h"
#include "Metric.h"

#include <numeric>
#include <random>

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

    this->oobIndexes = std::vector<int*>(nEstimators);
    // Initialize estimators.
    this->estimators.reserve(nEstimators);
}

RandomForestClassifier::~RandomForestClassifier() {
    this->estimators.clear();
    this->oobIndexes.clear();
}

void RandomForestClassifier::fit(const Matrix &train, int categories, int randomState) {
    const int k = train.m - 1;  // except labels
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
    this->nSamples = int(train.n * this->maxSamplesRatio);
    this->nCategories = categories;

    int i = 0;
    unsigned seed = randomState < 0 ? time(nullptr) : randomState;
    // Fitting each base estimator.
    while (i < this->nEstimators) {
        // Get training data from bootstrap.
        Matrix subTrain(this->nSamples, this->nFeatures + 1, 0.0);
        Indexes featuresIdx(k);
        this->bootstrap(train, subTrain, featuresIdx, seed, i);

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
    unsigned randomState,
    int epoch
) {
    int i;
    const int H = train.n;
    auto randomEngine = std::default_random_engine(randomState);

    // Draw samples from train set with replacement.
    int samplesIdx[this->nSamples];
    std::valarray<int> unsampled(1, H);
    std::uniform_int_distribution<int> rdis(0, H - 1);
    for (i = 0; i < this->nSamples; ++i) {
        int pos = rdis(randomEngine);
        samplesIdx[i] = pos;
        unsampled[pos] = 0;
    }

    // Collect the out-of-bag samples.
    int oobN = unsampled.sum(), l = 1;
    int *oob = new int[oobN + 1]; // Extra one for element: length (at [0]).
    for (i = 0; i < H; ++i) if (unsampled[i]) oob[l++] = i;
    oob[0] = oobN;
    this->oobIndexes[epoch] = oob;

    // Draw features from train set without replacement.
    std::iota(featuresIdx.begin(), featuresIdx.end(), 0);
    std::shuffle(featuresIdx.begin(), featuresIdx.end(), randomEngine);
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
