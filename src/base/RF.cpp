#include "RF.h"
#include "Metric.h"

using namespace wrf;

RandomForestClassifier::RandomForestClassifier(
    int nEstimators,
    int maxDepth,
    int randomState,
    float maxSamplesRatio,
    const std::string &maxFeatures,
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

    unsigned seed = randomState == -1 ? time(nullptr) : randomState;
    this->randomEngine = std::default_random_engine(seed);

    this->estimators = std::vector<CART*>(nEstimators);
    this->oobIndexes = std::vector<int*>(nEstimators);
}

void RandomForestClassifier::fit(const Matrix &train, int categories) {
    const int k = train.m - 1;
    if (this->maxFeatures == "sqrt") this->nFeatures = sqrt(k);
    else if (this->maxFeatures == "log2") this->nFeatures = log2(k);
    else this->nFeatures = k;

    this->nSamples = int(train.n * this->maxSamplesRatio);
    this->nCategories = categories;

    int i = 0;
    // Fitting each base estimator.
    while (i < this->nEstimators) {
        // Get training data from bootstrap.
        Matrix subTrain(this->nSamples, this->nFeatures + 1, 0.0);
        Indexes featuresIdx(k);
        this->bootstrap(train, subTrain, featuresIdx, i);

        // Feed training data to train CART.
        CART *estimator = new CART(
            this->maxDepth,
            this->minSamplesSplit,
            this->minSamplesLeaf,
            this->minSplitGain
        );
        estimator->fit(subTrain, featuresIdx, categories);

        this->estimators[i++] = estimator;
    }
}

void RandomForestClassifier::predict(const Matrix &test, Vector &preds) {
    int i;
    const int H = test.n;
    // Collect all predictions.
    Matrix labels(this->nEstimators, H, 0.0);
    i = 0;
    while (i < this->nEstimators) {
        this->estimators[i]->predict(test, labels[i]);
        ++i;
    }
    // Aggregate predictions with weights.
    i = 0;
    while (i < H) {
        Vector votes(this->nEstimators), dist(0.0, this->nCategories);
        labels.col(i, votes);
        distribution(votes, dist);
        preds[i++] = argmax(dist);
    }
}

void RandomForestClassifier::bootstrap(
    const Matrix &train,
    Matrix &subTrain,
    Indexes &featuresIdx,
    int epoch
) {
    int i;
    const int N = featuresIdx.size(), H = train.n;

    // Draw samples from train set with replacement.
    int samplesIdx[this->nSamples];
    std::valarray<int> unsampled(1, H);
    std::uniform_int_distribution<int> rdis(0, H - 1);
    for (i = 0; i < this->nSamples; ++i) {
        int pos = rdis(this->randomEngine);
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
    for (i = 0; i < N; ++i) featuresIdx[i] = i;
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
