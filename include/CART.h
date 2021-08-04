#ifndef WRF_CART_H
#define WRF_CART_H

#include "Matrix.h"

namespace wrf {

    // Binary tree node
    typedef struct Node {
        int splitFeature;
        double splitValue;
        double leaf;
        // The index of left and right child. 0 means none.
        int leftChild, rightChild;
    } Node;

    // Store the information about one best split feature.
    typedef struct bestSplit {
        double splitGain;
        double splitValue;
        int splitFeature;
        int curFeature;
    } bestSplit;

    // Classification And Regression Tree
    class CART {
    public:
        // The maximum depth of decision tree.
        int maxDepth;

        // The minimum number of samples required to split an internal node.
        int minSamplesSplit;

        // The minimum number of samples required to be at a leaf node.
        int minSamplesLeaf;

        // The minimum information gain of one split.
        double minSplitGain;

        CART(int maxDepth, int minSamplesSplit, int minSamplesLeaf, double minSplitGain);

        void fit(const Matrix &train, const Indexes &featuresIdx, int categories);

        // Predict label form a single vector.
        double predict(const Vector &vec);
        // Predict labels from a test set.
        Vector predict(const Matrix &test);

    private:
        // Store the node in a sequence.
        std::vector<Node*> nodes;

        // The categories number of current training dataset.
        int nCategories = 0;

        void buildTree(
            const Matrix &train,
            const Indexes &trainIdx,
            const Indexes &featuresIdx,
            int depth,
            int nodeIdx
        );

        // Get one best split, criterion: Gini Index.
        void getBestSplit(
            const Matrix &train,
            const Indexes &trainIdx,
            const Indexes &featuresIdx,
            const Vector &labels,
            bestSplit &best
        ) const;

        double predict(const Vector &vec, Node *node);
    };

}

#endif //WRF_CART_H
