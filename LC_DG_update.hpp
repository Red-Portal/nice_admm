
#ifndef _LC_DG_UPDATE_HPP_
#define _LC_DG_UPDATE_HPP_

#include <utility>
#include <iostream>
#include <algorithm>

#include "utility.hpp"

inline std::tuple<row_vector, row_vector>
LC_DG_optimization(float descent_rate,
                   float alphag,
                   float betag,
                   float cg,
                   float gamma,
                   blaze::DynamicMatrix<float> const& Pk,
                   blaze::DynamicMatrix<float> const& Qk,
                   row_vector const& mu2,
                   row_vector const& lambda2)
{
    auto Pg2 = row_vector(24, 1);
    auto Qg2 = row_vector(24, 1);

    float LC_DG_loss = std::numeric_limits<float>::max();

    // cashed constants
    auto betag_constant = row_vector(24, betag);
    auto cg_constant = row_vector(24, cg);

    gradient_descent P_optimizer;
    gradient_descent Q_optimizer;

    std::cout << "LC_LG" << std::endl;
    for(auto i = 0u; i < 27; ++i)
    {                                       
        auto Cg_diesel2 = nice::sum(alphag * nice::power(Pg2) + betag * Pg2 + cg_constant);

        auto P_inner = blaze::evaluate((-1 * Pg2) - row(Pk, 0) + mu2);
        auto Q_inner = blaze::evaluate((-1 * Qg2) - row(Qk, 0) + lambda2);
            
        auto P = (1/(2 * gamma)) * dot(P_inner, P_inner);
        auto Q = (1/(2 * gamma)) * dot(Q_inner, Q_inner);
        LC_DG_loss = Cg_diesel2 + P + Q;

        auto Cg_diesel2_derivative = 2 * alphag * Pg2 + betag_constant;

        auto Pg_delta = descent_rate * (Cg_diesel2_derivative + (-1 / gamma) * (P_inner));
        auto Qg_delta = descent_rate * (-1 / gamma) * (Q_inner);

        //std::cout << "diesel delta " << Cg_diesel2_derivative[0] << std::endl;
        //std::cout << "diesel " << Cg_diesel2 << std::endl;

        //std::cout << "Pg delta " << Pg_delta[0] << std::endl;
        //std::cout << "Qg delta " << Qg_delta[0] << std::endl;

        //std::cout << "Pg2 " << Pg2[0] << std::endl;
        //std::cout << "Qg2 " << Qg2[0] << std::endl;

        std::cout << "loss: " << LC_DG_loss << std::endl;

        Pg2 = P_optimizer.update(Pg2, Pg_delta);
        Qg2 = Q_optimizer.update(Qg2, Qg_delta);

        //std::cout << "New Pg2 " << Pg2[0] << std::endl;
        //std::cout << "New Qg2 " << Qg2[0] << std::endl;
    }

    return {Pg2, Qg2};
} 

#endif
