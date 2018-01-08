
#ifndef _OPTIMIZATION_PHASES_HPP_
#define _OPTIMIZATION_PHASES_HPP_

#include <utility>
#include <iostream>
#include <algorithm>

#include "utility.hpp"
#include <coin/IpTNLP.hpp>
#include <coin/IpIpoptApplication.hpp>

std::tuple<nice::row_vector, nice::row_vector>
LC_DG_optimization(Ipopt::IpoptApplication& app,
                   float alphag,
                   float betag,
                   float cg,
                   float gamma,
                   float Pg2_max,
                   float Sg2,
                   blaze::DynamicMatrix<float> const& Pk,
                   blaze::DynamicMatrix<float> const& Qk,
                   nice::row_vector const& mu2,
                   nice::row_vector const& lambda2);

std::tuple<nice::row_vector, nice::row_vector>
LC_DS_first_optimization(float descent_rate,
                         size_t max_iteration,
                         float gamma,
                         float gammab,
                         blaze::DynamicMatrix<float> const& Pk,
                         blaze::DynamicMatrix<float> const& Qk,
                         nice::row_vector& Eb8,
                         float Eb_min,
                         float Eb_max,
                         nice::row_vector const& mu8,
                         nice::row_vector const& lambda8,
                         float kkt_threshold = 0.0001);

#endif
