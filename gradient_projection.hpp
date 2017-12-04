
#ifndef _NICE_GRADIENT_PROJECTION_HPP_
#define _NICE_GRADIENT_PROJECTION_HPP_

#include <iostream>
#include <algorithm>
#include <chrono>
#include <optional>
#include <cmath>

#include "utility.hpp"

namespace nice
{
    blaze::DynamicMatrix<float>
    projection_matrix(matrix const& N);

    std::optional<matrix>
    active_constraints(matrix const& A,
                       column_vector const& b,
                       row_vector const& x);

    float
    max_lambda(matrix const& A,
               column_vector const& b,
               row_vector const& x,
               row_vector const& s);

    template<typename F, typename d_F, typename L_F>
    inline nice::row_vector
    gradient_projection(F& function,
                        d_F& d_function,
                        L_F& best_lambda,
                        float descent_rate,
                        size_t max_iterations,
                        nice::row_vector const& starting_point,
                        nice::matrix const& A,
                        nice::column_vector const& b,
                        bool verbose = false,
                        float kkt_threshold = 1e-5)
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

            if(verbose)
            {
                std::cout << "iteration: " << iteration + 1
                          << " objective: " << objective 
                          << '\n';
            }

            auto next_point = point - descent_rate * gradient;
            auto active = nice::active_constraints(A, b, next_point);

            if(!active)
            {
                point = next_point;
                std::cout << "gradient: " <<  point << std::endl;
            }
            else
            {
                std::cout << "fuck" << std::endl;
                auto P = nice::projection_matrix(active.value());
                auto s = (-1) * gradient * blaze::trans(P);

                if(nice::norm_l2(s) < kkt_threshold)
                    break;

                float best = best_lambda(point, s);
                float max = nice::max_lambda(A, b, point, s);
                auto update = blaze::evaluate(std::min(best, max) * s);

                if(nice::norm_l2(update) < kkt_threshold)
                    break;

                point = point + update;
                std::cout << "projected grad: " << point << std::endl;
            }
        }

        auto end= std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<
                std::chrono::microseconds>(end - start);

        if(verbose)
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
