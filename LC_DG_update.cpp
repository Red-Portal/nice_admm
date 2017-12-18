#include <optional>

#include "gradient_projection.hpp"
#include "optimization_phases.hpp"

// template<typename d_F, typename bestL>
// inline std::optional<nice::row_vector>
// Pg2_update(float descent_rate,
//            float kkt_threshold,
//            nice::row_vector const& Pg2,
//            d_F& d_Pg2,
//            bestL& best_lambda,
//            nice::sparse_matrix const& A,
//            nice::column_vector const& b)
// {
//     auto Pg2_gradient = blaze::evaluate((-1) * d_Pg2(Pg2));
//     auto next_Pg2 = blaze::evaluate(Pg2 + descent_rate * Pg2_gradient);

//     auto active = nice::active_constraints(A, b, next_Pg2);

//     if(!active)
//     {
//         return next_Pg2;
//     }
//     else
//     {
//         auto P = nice::projection_matrix(active.value());
//         //std::cout << "P rows: "<< P.rows() << " cols: "<< P.columns() << std::endl;
//         auto s = Pg2_gradient * blaze::trans(P);
//         std::cout << "P: "<< s << std::endl;

//         if(nice::norm_l2(s) < kkt_threshold)
//         {
//             auto max_rescale = nice::max_lambda(A, b, Pg2, Pg2_gradient);
//             if(max_rescale < kkt_threshold)
//                 return {};
//             else
//                 return Pg2 + max_rescale * Pg2_gradient;
//         }

//         float best = best_lambda(Pg2, s);
//         float max = nice::max_lambda(A, b, Pg2, s);
//         auto update = blaze::evaluate(std::min(best, max) * s);

//         if(nice::norm_l2(update) < kkt_threshold)
//             return {};

//         return Pg2 + update;
//     }
// }

std::tuple<nice::sparse_matrix, nice::column_vector>
LC_DG_constraints(float Pg2_max)
{
    auto row_size = 23 * 2 + 24 * 2;
    auto constraints = nice::sparse_matrix(row_size, 24, row_size * 2);

    for(auto i = 0u; i < 23u; ++i)
    {
        constraints(i, i) = 1;
        constraints(i, i + 1) = -1;
    }

    for(auto i = 23u; i < 46u; ++i)
    {
        constraints(i, i - 23) = -1;
        constraints(i, i - 22) = 1;
    }

    for(auto i = 0u; i < 24u; ++i)
        constraints(i + 46, i) = -1; 

    for(auto i = 0u; i < 24u; ++i)
        constraints(i + 70, i) = +1; 

    auto constraint_range = nice::column_vector(row_size, 0);

    for(auto i = 0u; i < 46; ++i)
        constraint_range[i] = 1.8;

    for(auto i = 70; i < 94; ++i)
        constraint_range[i] = Pg2_max; // Pg2 max!

    return {constraints, constraint_range};
}

std::tuple<nice::row_vector, nice::row_vector>
LC_DG_optimization(float descent_rate,
                   size_t max_iteration,
                   float alphag,
                   float betag,
                   float cg,
                   float gamma,
                   float Pg2_max,
                   float Sg2,
                   blaze::DynamicMatrix<float> const& Pk,
                   blaze::DynamicMatrix<float> const& Qk,
                   nice::row_vector const& mu2,
                   nice::row_vector const& lambda2,
                   float kkt_threshold,
                   bool verbose)
{
    size_t Pg2_iter_count = 0;
    size_t Qg2_iter_count = 0;

    auto PQg = nice::row_vector(48, 1);

    // cashed constants
    auto betag_constant = nice::row_vector(24, betag);
    auto cg_constant = nice::row_vector(24, cg);

    std::cout << "LC_LG" << std::endl;

    auto objective = [&](nice::row_vector const& PQg)
        {
            auto Pg = blaze::subvector(PQg, 0, 24);
            auto Qg = blaze::subvector(PQg, 24, 48);

            auto Cg_diesel2 = nice::sum(alphag * nice::power(Pg) + betag * Pg + cg_constant);

            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Q_inner = lambda2 - Qg - row(Qk, 0);

            auto P = (1/(2 * gamma)) * dot(P_inner, P_inner);
            auto Q = (1/(2 * gamma)) * dot(Q_inner, Q_inner);

            return Cg_diesel2 + P + Q;
        };

    auto d_PQg = [&](nice::row_vector const& PQg)
        {
            auto d_PQg = nice::row_vector(48);

            auto Pg = blaze::subvector(PQg, 0, 24);
            auto Qg = blaze::subvector(PQg, 24, 48);

            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Cg_diesel2_derivative = 2 * alphag * Pg + betag_constant;
            auto d_Pg = descent_rate * (Cg_diesel2_derivative + (-1 / gamma) * (P_inner));
            blaze::subvector(d_PQg, 0, 24) = d_Pg;

            auto Q_inner = lambda2 - Qg - row(Qk, 0);
            auto d_Qg = descent_rate * (-1 / gamma) * (Q_inner);
            blaze::subvector(d_PQg, 24, 48) = d_Qg;
        };

    static auto constraints = LC_DG_constraints(Pg2_max, Sg2);
    auto [nonlinear, linear] = constraints;

    float objective_value = 0;

    auto result = nice::gradient_projection(objective,
                                            d_PQg,
                                            descent_rate,
                                            max_iteration,
                                            PQg);


    //std::cout << "current Pg2: "<< Pg2[0] << std::endl;
    //std::cout << "current Qg2: "<< Qg2[0] << std::endl;


    return {Pg2, Qg2};
} 
