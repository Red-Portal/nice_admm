
#include <optional>
#include <vector>

#include "optimization_phases.hpp"
#include "LC_DG_problem.hpp"

std::tuple<nice::row_vector, nice::row_vector>
LC_DG_optimization(Ipopt::IpoptApplication& app
                   float alphag,
                   float betag,
                   float cg,
                   float gamma,
                   float Pg2_max,
                   float Sg2,
                   blaze::DynamicMatrix<float> const& Pk,
                   blaze::DynamicMatrix<float> const& Qk,
                   nice::row_vector const& mu2,
                   nice::row_vector const& lambda2)
{
    auto betag_constant = nice::row_vector(24, betag);
    auto cg_constant = nice::row_vector(24, cg);

    std::cout << "LC_LG" << std::endl;

    auto objective = [&](nice::row_vector const& PQg) -> float
        {
            auto Pg = blaze::subvector(PQg, 0, 24);
            auto Qg = blaze::subvector(PQg, 24, 24);

            auto Cg_diesel2 = nice::sum(alphag * nice::power(Pg) + betag * Pg + cg_constant);

            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Q_inner = lambda2 - Qg - row(Qk, 0);

            auto P = (1/(2 * gamma)) * dot(P_inner, P_inner);
            auto Q = (1/(2 * gamma)) * dot(Q_inner, Q_inner);

            return 0.1 * Cg_diesel2 + P + Q;
        };

    auto d_PQg = [&](nice::row_vector const& PQg) -> nice::row_vector
        {
            auto d_PQg = nice::row_vector(48);

            auto Pg = blaze::subvector(PQg, 0, 24);
            auto Qg = blaze::subvector(PQg, 24, 24);

            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Cg_diesel2_derivative = 2 * alphag * Pg + betag_constant;
            auto d_Pg =  (0.1 *Cg_diesel2_derivative + (-1 / gamma) * (P_inner));
            blaze::subvector(d_PQg, 0, 24) = d_Pg;

            auto Q_inner = lambda2 - Qg - row(Qk, 0);
            auto d_Qg = (-1 / gamma) * (Q_inner);
            blaze::subvector(d_PQg, 24, 24) = d_Qg;

            return d_PQg;
        };

    static auto constraints = LC_DG_constraints(Pg2_max, Sg2);
    auto [linear, nonlinear] = constraints;

    auto result = nice::gradient_projection(objective,
                                            d_PQg,
                                            descent_rate,
                                            max_iteration,
                                            0.01,
                                            200,
                                            PQg_init,
                                            linear,
                                            nonlinear,
                                            kkt_threshold,
                                            nice::verboseness::log);



    //std::cout << "current Pg2: "<< Pg2[0] << std::endl;
    //std::cout << "current Qg2: "<< Qg2[0] << std::endl;


    auto Pg = blaze::subvector(result, 0, 24);
    auto Qg = blaze::subvector(result, 24, 24);

    return {Pg, Qg};
} 
