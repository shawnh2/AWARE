#include <iostream>
#include <chrono>
#include <unistd.h>

#include "Dataset.h"
#include "RF.h"
#include "TWRF.h"
#include "WAVE.h"
#include "AWARE.h"

using namespace wrf;
using namespace std;
using namespace chrono;

#define FIT(RF, ...) \
    auto *__rf = new RF(__VA_ARGS__); \
    __rf->fit(train, ds.categories);

#define PREDICT(...) \
    Vector __pred = __rf->predict(__VA_ARGS__); \
    Vector __acc(0.0, M); \
    __acc[__pred == labels] = 1.0;

#define TIME_BEGIN() auto __Ta = system_clock::now();

#define TIME_END() auto __Tz = system_clock::now(); \
    auto __Td = duration_cast<microseconds>(__Tz - __Ta);

#define ACC __acc.sum() / M
#define TIME __Td.count() / 10e6

/*
 Available args:
 -D dataset_index * (not optional)
 -n n_estimators
 -d max_depth
 -r random_state
 */

int main(int argc, char **argv) {
    int datasetIdx;
    int nEstimators = 100;
    int maxDepth = 5;
    unsigned randomState = 0;

    int ch;
    while ((ch = getopt(argc, argv, "D:n::d::r::")) != -1) {
        switch (ch) {
            case 'D':
                datasetIdx = atoi(optarg);
                assert(datasetIdx >= 0 && datasetIdx < 18);
                break;
            case 'n':
                nEstimators = atoi(optarg);
                break;
            case 'd':
                maxDepth = atoi(optarg);
                break;
            case 'r':
                randomState = atoi(optarg);
            default:
                break;
        }
    }

    DatasetInfo ds = DatasetList[datasetIdx];
    int N = int(ds.height * 0.75);
    int M = ds.height - N;
    Matrix train(N, ds.width, 0.0);
    Matrix test(M, ds.width, 0.0);
    loadDataset(ds, train, test);

    Vector labels = test.col(-1);

    cout << "Fitting on dataset: " << ds.name << " [";
    cout << "n_estimators=" << nEstimators;
    cout << ", max_depth=" << maxDepth;
    cout << ", random_state=" << randomState << "]\n";
    // RF
    {
        TIME_BEGIN()
        FIT(RandomForestClassifier, nEstimators, maxDepth, randomState)
        PREDICT(test)
        TIME_END()
        cout << " RF " << '\t' << ACC << '\t' << TIME << "s\n";
    }

    // TWRF
    {
        TIME_BEGIN()
        FIT(TWRF, nEstimators, maxDepth, randomState)
        PREDICT(test, train)
        TIME_END()
        cout << "TWRF" << '\t' << ACC << '\t' << TIME << "s\n";
    }

    // WAVE
    {
        TIME_BEGIN()
        FIT(WAVE, nEstimators, maxDepth, randomState)
        PREDICT(test, train)
        TIME_END()
        cout << "WAVE" << '\t' << ACC << '\t' << TIME << "s\n";
    }

    // AWARE
    {
        TIME_BEGIN()
        FIT(AWARE, nEstimators, maxDepth, randomState)
        PREDICT(test, train)
        TIME_END()
        cout << "AWARE" << '\t' << ACC << '\t' << TIME << "s\n";
    }

    return 0;
}
