#include "CART.h"
#include "Metric.h"

#include <numeric>

using namespace wrf;

CART::CART(int maxDepth, int minSamplesSplit, int minSamplesLeaf, double minSplitGain) {
    this->maxDepth = std::max(1, maxDepth);
    this->minSamplesSplit = std::max(0, minSamplesSplit);
    this->minSamplesLeaf = std::max(0, minSamplesLeaf);
    this->minSplitGain = std::max(0.0, minSplitGain);

    this->nodes.reserve(pow(2, maxDepth));
}

CART::~CART() {
    this->nodes.clear();
}

void CART::fit(const Matrix &train, const Indexes &featuresIdx, int categories) {
    // Select the training samples.
    const int N = train.n;
    Indexes trainIdx(N);
    std::iota(trainIdx.begin(), trainIdx.end(), 0);

    this->nCategories = categories;
    this->buildTree(train, trainIdx, featuresIdx, 1, 1);
}

double CART::predict(const Vector &vec) {
    return this->_predict(vec, this->nodes[1]);
}

Vector CART::predict(const Matrix &test) {
    Vector preds(0.0, test.n);
    for (int i = 0; i < test.n; ++i) {
        preds[i] = this->_predict(test[i], this->nodes[1]);
    }
    return preds;
}

Matrix CART::predictProb(const Matrix &test) {
    Matrix probs(test.n, this->nCategories, 0.0);
    for (int i = 0; i < test.n; ++i) {
        probs[i] = this->_predictProb(test[i], this->nodes[1]);
    }
    return probs;
}

void CART::buildTree(
    const Matrix &train,
    const Indexes &trainIdx,
    const Indexes &featuresIdx,
    int depth,
    int iNode
) {
    const int H = trainIdx.size();

    Vector labels(H);
    train.col(-1, trainIdx, labels);
    Vector distLabels = distribution(labels, this->nCategories);

    // Initialize node.
    Node *node = new Node;
    node->splitFeature = -1;
    node->splitValue = 0.0;
    node->leaf = 0.0;
    node->leftChild = 0;
    node->rightChild = 0;
    node->prob = distLabels / H;

    // Check depth
    if (depth < this->maxDepth) {
        // If labels are all the same or samples are less than minimum samples number.
        if (H <= this->minSamplesSplit || H == distLabels.max()) {
            node->leaf = argmax(distLabels);
            this->nodes[iNode] = node;
            return;
        }

        // Get one best split.
        bestSplit best = {1.0, 0.0, -1, -1};
        this->getBestSplit(train, trainIdx, featuresIdx, labels, best);

        // Split train set into left and right set.
        Indexes lti(H), rti(H);
        int l = 0, r = 0;
        for (int ti : trainIdx) {
            if (train[ti][best.curFeature] <= best.splitValue) lti[l++] = ti;
            else rti[r++] = ti;
        }
        lti.resize(l);
        rti.resize(r);

        // Check leaf and gain threshold.
        if (l <= this->minSamplesLeaf ||
            r <= this->minSamplesLeaf ||
            best.splitGain <= this->minSplitGain) {
            node->leaf = argmax(distLabels);
            this->nodes[iNode] = node;
            return;
        }

        node->splitFeature = best.splitFeature;
        node->splitValue = best.splitValue;
        node->leftChild = iNode * 2;
        node->rightChild = iNode * 2 + 1;
        // Grow the left and right child.
        this->buildTree(train, lti, featuresIdx, depth + 1, node->leftChild);
        this->buildTree(train, rti, featuresIdx, depth + 1, node->rightChild);

        this->nodes[iNode] = node;
    }
    // Exceed the maximum depth.
    else {
        node->leaf = argmax(distLabels);
        this->nodes[iNode] = node;
        return;
    }
}

void CART::getBestSplit(
    const Matrix &train,
    const Indexes &trainIdx,
    const Indexes &featuresIdx,
    const Vector &labels,
    bestSplit &best
) const {
    const int H = trainIdx.size(), FN = featuresIdx.size(), C = this->nCategories;
    int i = 0, nDist = 2 * C;
    double splitGain;
    Vector features(H), uFeatures(H);

    while (i < FN) {
        train.col(i, trainIdx, features);
        train.col(i, trainIdx, uFeatures);
        // Get unique features.
        std::sort(std::begin(uFeatures), std::end(uFeatures));
        double *pos = std::unique(std::begin(uFeatures), end(uFeatures));
        double *j = std::begin(uFeatures);

        while (j < pos) {
            // Get the distribution of left and right labels.
            double dist[nDist];
            int k = 0, l = 0, r = 0;
            while (k < nDist) {
                dist[k] = 0.0;
                ++k;
            }
            k = 0;
            // Range left: [0, C-1], right: [C, 2C-1]
            while (k < H) {
                int y = labels[k];
                if (*j >= features[k]) {
                    dist[y] += 1.0;
                    ++l;
                } else {
                    dist[y + C] += 1.0;
                    ++r;
                }
                ++k;
            }

            // Calculate split gain.
            k = 0;
            double lGini = 0.0, rGini = 0.0;
            while (k < C) {
                lGini += pow(dist[k], 2);
                ++k;
            }
            while (k < nDist) {
                rGini += pow(dist[k], 2);
                ++k;
            }
            splitGain = 1.0 - (lGini / l + rGini / r) / H;

            if (splitGain < best.splitGain) {
                best.splitGain = splitGain;
                best.splitValue = *j;
                best.splitFeature = featuresIdx[i];
                best.curFeature = i;
            }
            ++j;
        }
        ++i;
        features = 0.0;
        uFeatures = 0.0;
    }
}

double CART::_predict(const Vector &vec, const Node *node) {
    double y;

    if (!node->leftChild && !node->rightChild) return node->leaf;

    if (vec[node->splitFeature] <= node->splitValue)
         y = this->_predict(vec, this->nodes[node->leftChild]);
    else y = this->_predict(vec, this->nodes[node->rightChild]);

    return y;
}

Vector CART::_predictProb(const Vector &vec, const Node *node) {
    Vector res;

    if (!node->leftChild && !node->rightChild) return node->prob;

    if (vec[node->splitFeature] <= node->splitValue)
         res = this->_predictProb(vec, this->nodes[node->leftChild]);
    else res = this->_predictProb(vec, this->nodes[node->rightChild]);

    return res;
}
