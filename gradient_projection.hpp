
#ifndef _NICE_GRADIENT_PROJECTION_HPP_
#define _NICE_GRADIENT_PROJECTION_HPP_

#include <iostream>
#include <algorithm>
#include <chrono>
#include <functional>
#include <optional>
#include <cmath>

#include "utility.hpp"

namespace nice
{
    inline blaze::DynamicMatrix<float>
    tangent_subspace(nice::matrix const& N)
    {
        //std::cout << "N: " << N << std::endl;
        return  inv(N * trans(N)) * N;
    }

    inline float
    compute_alpha(float gamma,
                  float max_alpha,
                  float objective_value, 
                  nice::row_vector const& s,
                  nice::row_vector const& gradient)
    {
        float alpha = ((-1) * gamma * objective_value) / blaze::dot(gradient, s);
        return std::min(alpha, max_alpha);
    }

    inline float
    is_parallel(nice::row_vector const& x,
                nice::row_vector const& y)
    {
        auto product = blaze::dot(x, y);
        return blaze::dot(x, x) * blaze::dot(y, y) - product * product;
    }

    inline nice::matrix
    projection_matrix(nice::sparse_matrix const& N,
                      nice::matrix const& tangent)
    {
        auto id = blaze::IdentityMatrix<float, blaze::rowMajor>(N.columns());
        return id - blaze::trans(N) * tangent;
    }

    inline nice::matrix
    projection_matrix(nice::matrix const& N,
                      nice::matrix const& tangent)
    {
        auto id = blaze::IdentityMatrix<float, blaze::rowMajor>(N.columns());
        return id - blaze::trans(N) * tangent;
    }

    using nonlinear_set =
        std::tuple<std::function<float(nice::row_vector const&)>,
                   std::function<nice::row_vector(nice::row_vector const&)>>;

    template<typename MatrixType>
    using linear_set = std::tuple<MatrixType, nice::column_vector>;


    std::optional<std::tuple<nice::matrix, nice::column_vector>>
    active_constraints(nice::row_vector const& x,
                       nice::linear_set<nice::sparse_matrix> const& linear_constraints,
                       std::vector<nice::nonlinear_set> const& nonlinear_constraints);

    std::optional<std::tuple<nice::matrix, nice::column_vector>>
    active_constraints(nice::row_vector const& x,
                       nice::linear_set<nice::matrix> const& linear_constraints,
                       std::vector<nice::nonlinear_set> const& nonlinear_constraints);

    template<typename F, typename d_F, typename MatrixType>
    inline nice::row_vector
    gradient_projection(F& function,
                        d_F& d_function,
                        float descent_rate,
                        size_t max_iterations,
                        float gamma,
                        float max_alpha,
                        nice::row_vector const& starting_point,
                        linear_set<MatrixType> const& linear_constraints,
                        std::vector<nice::nonlinear_set> const& nonlinear_constraints,
                        float kkt_threshold = 1e-5,
                        nice::verboseness verbose = nice::verboseness::very_verbose)
    {
        auto start = std::chrono::steady_clock::now();

        auto iteration = 0u;
        auto point = starting_point;

        for(iteration = 0; iteration < max_iterations; ++iteration)
        {
            auto objective = function(point);
            auto gradient = d_function(point);

            if(nice::norm_l2(gradient) < kkt_threshold)
                break;

            if(static_cast<size_t>(verbose) > 1u)
            {
                std::cout << '\n'
                          << "iteration: " << iteration + 1
                          << " objective: " << objective 
                          << std::endl;
            }
            if(verbose == nice::verboseness::log)
            {
                std::cout << "point: " << point;
                std::cout << "gradient: " << gradient;
                std::cout << "update: " << descent_rate * gradient;
            }

            auto active = nice::active_constraints(point,
                                                   linear_constraints,
                                                   nonlinear_constraints);
            if(!active)
                point = point - descent_rate * gradient;
            else
            {
                auto [N, g] = active.value();
                auto tangent = nice::tangent_subspace(N); 
                auto s = blaze::evaluate((-1) * gradient * nice::projection_matrix(N, tangent));

                auto a = nice::compute_alpha(gamma, max_alpha, objective, s, gradient);
                auto projection_move = a * s;
                auto restoration_move = (-1) * blaze::trans(g) * tangent;
                auto update = projection_move + restoration_move;

                point = point + update;

                if(nice::is_parallel(update, gradient) < kkt_threshold)
                    break;

                if(nice::norm_l2(s) < kkt_threshold)
                    break;

                if(verbose == nice::verboseness::log)
                {
                    std::cout << "active: " << std::get<0>(active.value());
                    std::cout << "a: " << a << std::endl;
                    std::cout << "s: " << s;
                    std::cout << "projection: " << projection_move;
                    std::cout << "resto: " << restoration_move;
                    std::cout << "kkt1: " << nice::norm_l2(s) << std::endl;
                    std::cout << "kkt2: " << nice::is_parallel(update, gradient) << std::endl;
                    std::cout << "update: " << update << std::endl;;
                }
            }
        }

        auto end= std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<
                std::chrono::microseconds>(end - start);

        if(static_cast<size_t>(verbose) > 0)
        {
            std::cout << "time: " << duration.count() << "us" << std::endl;
            std::cout << "iterations: " << iteration + 1 << std::endl;
            std::cout << "objective: " << function(point) << std::endl;
            std::cout << "optimal value: " << point;
        }

        return point;
    }
}

#endif
