
#ifndef _LC_DG_UPDATE_HPP_
#define _LC_DG_UPDATE_HPP_

#include <utility>
#include <iostream>
#include <algorithm>

#include "utility.hpp"

std::tuple<nice::row_vector, nice::row_vector>
LC_DG_optimization(float descent_rate,
                   size_t max_iteration,
                   float alphag,
                   float betag,
                   float cg,
                   float gamma,
                   blaze::DynamicMatrix<float> const& Pk,
                   blaze::DynamicMatrix<float> const& Qk,
                   nice::row_vector const& mu2,
                   nice::row_vector const& lambda2,
                   float kkt_threshold = 0.0001,
                   bool verbose = true);

#endif
