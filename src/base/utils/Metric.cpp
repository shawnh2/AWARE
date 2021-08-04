#include "Metric.h"

using namespace wrf;

Vector wrf::distribution(const Vector &labels, int k) {
    int n = labels.size(), i = 0;
    Vector res(0.0, k);
    while (i < n) {
        res[labels[i]] += 1.0;
        ++i;
    }
    return res;
}

int wrf::argmax(const Vector &dist) {
    double val = dist.max();
    int i = 0, n = dist.size();
    while (i < n) {
        if (val == dist[i]) return i;
        ++i;
    }
    // Return the last index if not found.
    return i;
}
