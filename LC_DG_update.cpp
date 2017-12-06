#include <optional>

#include "gradient_projection.hpp"
#include "LC_DG_update.hpp"

template<typename d_F, typename bestL>
inline std::optional<nice::row_vector>
Pg2_update(float descent_rate,
           float kkt_threshold,
           nice::row_vector const& Pg2,
           d_F& d_Pg2,
           bestL& best_lambda,
           nice::sparse_matrix const& A,
           nice::column_vector const& b)
{
    auto Pg2_gradient = blaze::evaluate((-1) * d_Pg2(Pg2));
    auto next_Pg2 = blaze::evaluate(Pg2 + descent_rate * Pg2_gradient);

    auto active = nice::active_constraints(A, b, next_Pg2);

    if(!active)
    {
        return next_Pg2;
    }
    else
    {
        auto P = nice::projection_matrix(active.value());
        //std::cout << "P rows: "<< P.rows() << " cols: "<< P.columns() << std::endl;
        auto s = Pg2_gradient * blaze::trans(P);
        std::cout << "P: "<< s << std::endl;

        if(nice::norm_l2(s) < kkt_threshold)
        {
            auto max_rescale = nice::max_lambda(A, b, Pg2, Pg2_gradient);
            if(max_rescale < kkt_threshold)
                return {};
            else
                return Pg2 + max_rescale * Pg2_gradient;
        }

        float best = best_lambda(Pg2, s);
        float max = nice::max_lambda(A, b, Pg2, s);
        auto update = blaze::evaluate(std::min(best, max) * s);

        if(nice::norm_l2(update) < kkt_threshold)
            return {};

        return Pg2 + update;
    }
}

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
    bool Pg2_found = false;
    bool Qg2_found = false;

    size_t Pg2_iter_count = 0;
    size_t Qg2_iter_count = 0;

    auto Pg2 = nice::row_vector(24, 1);
    auto Qg2 = nice::row_vector(24, 1);

    // cashed constants
    auto betag_constant = nice::row_vector(24, betag);
    auto cg_constant = nice::row_vector(24, cg);

    std::cout << "LC_LG" << std::endl;

    auto objective = [&](nice::row_vector const& Pg,
                         nice::row_vector const& Qg)
        {
            auto Cg_diesel2 = nice::sum(alphag * nice::power(Pg) + betag * Pg + cg_constant);

            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Q_inner = lambda2 - Qg - row(Qk, 0);

            auto P = (1/(2 * gamma)) * dot(P_inner, P_inner);
            auto Q = (1/(2 * gamma)) * dot(Q_inner, Q_inner);

            return Cg_diesel2 + P + Q;
        };

    auto d_Pg2 = [&](nice::row_vector const& Pg)
        {
            auto P_inner = mu2 - Pg - row(Pk, 0);
            auto Cg_diesel2_derivative = 2 * alphag * Pg + betag_constant;
            return blaze::evaluate(descent_rate * (Cg_diesel2_derivative + (-1 / gamma) * (P_inner)));
        };

    auto d_Qg2 = [&](nice::row_vector const& Qg)
        {
            auto Q_inner = lambda2 - Qg - row(Qk, 0);
            return blaze::evaluate(descent_rate * (-1 / gamma) * (Q_inner));
        };

    auto best_lambda = [&](nice::row_vector const& Pg,
                           nice::row_vector const& s)->float
        {
            auto dividend = 2 * alphag * dot(Pg, s) - betag * dot(s, (Pg + row(Pk, 0) - mu2));
            auto dividor = dot(s, s) * (2 * alphag + (1 / gamma));
            return dividend / dividor;
        };

    static auto constraints = LC_DG_constraints(Pg2_max);
    auto [A, b] = constraints;

    float objective_value = 0;

    auto i = 0u;
    for(i = 0u; i < max_iteration; ++i)
    {                                       
        if(verbose)
            objective_value = objective(Pg2, Qg2);

        if(Pg2_found && Qg2_found)
            break;

        if(!Qg2_found)
        {
            ++Qg2_iter_count;
            auto Qg2_gradient = d_Qg2(Qg2);

            if(nice::norm_l2(Qg2_gradient) < kkt_threshold)
                Qg2_found = true;
            else
                Qg2 -= descent_rate * Qg2_gradient;
        }

        if(!Pg2_found)
        {
            ++Pg2_iter_count;
            auto result = Pg2_update(0.1 * descent_rate,
                                     kkt_threshold,
                                     Pg2,
                                     d_Pg2,
                                     best_lambda,
                                     A,
                                     b);
            if(result)
                Pg2 = result.value();
            else
                Pg2_found = true;
        }

        //std::cout << "current Pg2: "<< Pg2[0] << std::endl;
        //std::cout << "current Qg2: "<< Qg2[0] << std::endl;

        if(verbose)
        {
            std::cout << "iteration: " << i + 1
                      << " optimal value: " << objective_value
                      << '\n';
        }
    }

    std::cout << "Pg2 iteration count: " << Pg2_iter_count << std::endl;
    std::cout << "Qg2 iteration count: " << Qg2_iter_count << std::endl;
    
    return {Pg2, Qg2};
} 
