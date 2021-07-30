#include "Metric.h"

void wrf::distribution(const Vector &labels, Vector &out) {
    int n = labels.size();
    for (int i = 0; i < n; ++i) out[labels[i]] += 1.0;
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
