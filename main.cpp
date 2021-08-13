#include <iostream>
#include <numeric>
#include <random>
#include <unistd.h>

#include "Dataset.h"
#include "RF.h"
#include "TWRF.h"
#include "WAVE.h"
#include "BTA.h"
#include "AWARE.h"

using namespace wrf;
using namespace std;

#define N_ESTIMATORS 100
#define MAX_DEPTH 8
#define EPOCH 100

// -d: The index of dataset.

int main(int argc, char **argv) {
    int DI, RS = 99, ch;

    while ((ch = getopt(argc, argv, "d:")) != -1) {
        if (ch == 'd') {
            DI = atoi(optarg);
            assert(DI >= 0 && DI < DatasetList.size());
        }
    }

    // Load dataset.
    DatasetInfo ds = DatasetList[DI];
    const int H = ds.height, W = ds.width, C = ds.categories;
    Matrix set(H, W, 0.0);
    loadDataset(ds, set, RS);

    // Assign fitting algorithm
    AWARE algor(N_ESTIMATORS, MAX_DEPTH);
    cout << "Fitting on dataset: " << ds.name << endl;

    // Perform 10-fold cross validation.
    Indexes fold(10, H / 10);
    fold[0] += H % 10;

    Vector ACC(0.0, EPOCH);
    for (int i = 0; i < EPOCH; ++i, ++RS) {
        // Shuffle the loaded dataset.
        int randomIndexes[H];
        std::iota(randomIndexes, randomIndexes + H, 0);
        std::shuffle(randomIndexes, randomIndexes + H, std::default_random_engine(RS));

        int a = 0, z;
        Vector foldACC(0.0, 10);
        for (int k = 0, foldRS = 99; k < 10; foldRS *= ++k) {
            const int testSize = fold[k];
            z = a + testSize - 1;

            Matrix test(testSize, W), train(H - testSize, W);
            trainTestSplit(set, randomIndexes, train, test, a, z);
            // Fitting and predict.
            algor.fit(train, C, foldRS);
            Vector Y = test.col(-1), acc(0.0, testSize);
            Vector y = algor.predict(test, train);
            acc[y == Y] = 1.0;
            foldACC[k] = acc.sum() / testSize;

            a = z + 1;
        }
        ACC[i] = foldACC.sum() / 10;
        cout << "epoch=" << i << " \t acc=" << ACC[i] << endl;
    }

    // Mean accuracy
    cout << "\nFinal Mean Accuracy on " << ds.name << ": " << ACC.sum() / EPOCH << endl;

    return 0;
}
