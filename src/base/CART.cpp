#include "CART.h"
#include "Metric.h"

#include <algorithm>

using namespace wrf;

CART::CART(int maxDepth, int minSamplesSplit, int minSamplesLeaf, double minSplitGain) {
    this->maxDepth = std::max(1, maxDepth);
    this->minSamplesSplit = std::max(0, minSamplesSplit);
    this->minSamplesLeaf = std::max(0, minSamplesLeaf);
    this->minSplitGain = std::max(0.0, minSplitGain);

    this->nodes.reserve(pow(2, maxDepth));
}

void CART::fit(const Matrix &train, const Indexes &featuresIdx, int categories) {
    // Select the training samples.
    const int N = train.n;
    Indexes trainIdx(N);
    for (int i = 0; i < N; ++i) trainIdx[i] = i;

    this->nCategories = categories;
    this->buildTree(train, trainIdx, featuresIdx, 1, 1);
}

void CART::predict(const Vector &vec, double &pred) {
    pred = this->predict(vec, this->nodes[1]);
}

void CART::predict(const Matrix &test, Vector &preds) {
    for (int i = 0; i < test.n; ++i) {
        const Vector &row = test[i];
        preds[i] = this->predict(row, this->nodes[1]);
    }
}

void CART::buildTree(
    const Matrix &train,
    const Indexes &trainIdx,
    const Indexes &featuresIdx,
    int depth,
    int nodeIdx
) {
    const int H = trainIdx.size();
    // Init node.
    Node *node = new Node;
    node->splitFeature = -1;
    node->splitValue = 0.0;
    node->leaf = 0.0;
    node->leftChild = 0;
    node->rightChild = 0;

    Vector labels(H), distLabels(0.0, this->nCategories);
    train.col(-1, trainIdx, labels);
    distribution(labels, distLabels);

    // If labels are all the same or samples are less than minimum samples number.
    if (H <= this->minSamplesSplit || H == distLabels.max()) {
        node->leaf = argmax(distLabels);
        this->nodes[nodeIdx] = node;
    }

    // Check depth
    if (depth < this->maxDepth) {
        // Get one best split.
        bestSplit best = {1.0, 0.0, -1, -1};
        this->getBestSplit(train, trainIdx, featuresIdx, labels, best);

        // Split train set into left and right set.
        Indexes lti(H), rti(H);
        int l = 0, r = 0;
        for (int i = 0; i < H; ++i) {
            if (train[trainIdx[i]][best.curFeature] <= best.splitValue) lti[l++] = trainIdx[i];
            else rti[r++] = trainIdx[i];
        }
        lti.resize(l);
        rti.resize(r);

        // Check leaf and gain threshold.
        if (l <= this->minSamplesLeaf ||
            r <= this->minSamplesLeaf ||
            best.splitGain <= this->minSplitGain) {
            node->leaf = argmax(distLabels);
            this->nodes[nodeIdx] = node;
        }

        node->splitFeature = best.splitFeature;
        node->splitValue = best.splitValue;
        // Grow the left and right child.
        node->leftChild = nodeIdx * 2;
        node->rightChild = nodeIdx * 2 + 1;

        this->buildTree(train, lti, featuresIdx, depth + 1, node->leftChild);
        this->buildTree(train, rti, featuresIdx, depth + 1, node->rightChild);
        this->nodes[nodeIdx] = node;
    }
    // Exceed the maximum depth.
    else {
        node->leaf = argmax(distLabels);
        this->nodes[nodeIdx] = node;
    }
}

void CART::getBestSplit(
    const Matrix &train,
    const Indexes &trainIdx,
    const Indexes &featuresIdx,
    const Vector &labels,
    bestSplit &best
) const {
    const int H = trainIdx.size(), FN = featuresIdx.size();
    int i = 0;
    double splitGain;
    Vector features(H), ufeatures(H);
    Vector ldist(0.0, this->nCategories), rdist(0.0, this->nCategories);

    while (i < FN) {
        train.col(i, trainIdx, features);
        train.col(i, trainIdx, ufeatures);
        // Get unique features.
        std::sort(std::begin(ufeatures), std::end(ufeatures));
        double *pos = std::unique(std::begin(ufeatures), end(ufeatures));
        double *j = std::begin(ufeatures);

        while (j < pos) {
            // Get the distribution of left and right labels.
            int k = 0;
            while (k < H) {
                if (*j >= features[k]) ldist[labels[k]] += 1.0;
                else rdist[labels[k]] += 1.0;
                ++k;
            }
            int l = ldist.sum();
            int r = rdist.sum();

            // Calculate split gain.
            // splitGain += (double)l / H * (1.0 - pow(ldist / l, 2).sum());
            // splitGain += (double)r / H * (1.0 - pow(rdist / r, 2).sum());
            splitGain = 1.0 - (pow(ldist, 2).sum() / l + pow(rdist, 2).sum() / r) / H;

            if (splitGain < best.splitGain) {
                best.splitGain = splitGain;
                best.splitValue = *j;
                best.splitFeature = featuresIdx[i];
                best.curFeature = i;
            }
            ++j;
            ldist = 0.0;
            rdist = 0.0;
        }
        ++i;
        features = 0.0;
    }
}

double CART::predict(const Vector &vec, Node *node) {
    double y;

    if (!node->leftChild && !node->rightChild) return node->leaf;

    if (vec[node->splitFeature] <= node->splitValue)
         y = this->predict(vec, this->nodes[node->leftChild]);
    else y = this->predict(vec, this->nodes[node->rightChild]);

    return y;
}
