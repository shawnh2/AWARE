#ifndef WRF_METRIC_H
#define WRF_METRIC_H

#include "Matrix.h"

namespace wrf {

    // Get the distribution of labels.
    void distribution(const Vector &labels, Vector &out);

    // Get the index of first max value in vector.
    int argmax(const Vector &dist);

}

#endif //WRF_METRIC_H
