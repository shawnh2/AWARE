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

void CART::fit(const Matrix &train, const Indexes &featuresIdx, int categories) {
    // Select the training samples.
    const int N = train.n;
    Indexes trainIdx(N);
    std::iota(trainIdx.begin(), trainIdx.end(), 0);

    this->nCategories = categories;
    this->buildTree(train, trainIdx, featuresIdx, 1, 1);
}

double CART::predict(const Vector &vec) {
    return this->predict(vec, this->nodes[1]);
}

Vector CART::predict(const Matrix &test) {
    Vector preds(0.0, test.n);
    for (int i = 0; i < test.n; ++i) {
        preds[i] = this->predict(test[i], this->nodes[1]);
    }
    return preds;
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

    // If labels are all the same or samples are less than minimum samples number.
    if (H <= this->minSamplesSplit || H == distLabels.max()) {
        int leaf = argmax(distLabels);
        this->nodes[iNode] = Node(-1, 0.0, leaf, 0, 0);
    }

    // Check depth
    if (depth < this->maxDepth) {
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
            int leaf = argmax(distLabels);
            this->nodes[iNode] = Node(-1, 0.0, leaf, 0, 0);
        }

        // Grow the left and right child.
        int leftChild = iNode * 2;
        int rightChild = leftChild + 1;
        this->buildTree(train, lti, featuresIdx, depth + 1, leftChild);
        this->buildTree(train, rti, featuresIdx, depth + 1, rightChild);

        this->nodes[iNode] = Node(best.splitFeature, best.splitValue, 0.0, leftChild, rightChild);
    }
    // Exceed the maximum depth.
    else {
        int leaf = argmax(distLabels);
        this->nodes[iNode] = Node(-1, 0.0, leaf, 0, 0);
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
            int k = 0, l = 0, r = 0;
            while (k < H) {
                int at = labels[k];
                if (*j >= features[k]) {
                    ++ldist[at];
                    ++l;
                }
                else {
                    ++rdist[at];
                    ++r;
                }
                ++k;
            }
            // Calculate split gain.
            // = l / H * (1.0 - sum(pow(ldist / l, 2))) + r / H * (1.0 - sum(pow(rdist / r, 2)));
            splitGain = 1.0 - (pow(ldist, 2) / l + pow(rdist, 2) / r).sum() / H;

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

double CART::predict(const Vector &vec, const Node &node) {
    double y;

    if (!node.leftChild && !node.rightChild) return node.leaf;

    if (vec[node.splitFeature] <= node.splitValue)
         y = this->predict(vec, this->nodes[node.leftChild]);
    else y = this->predict(vec, this->nodes[node.rightChild]);

    return y;
}
