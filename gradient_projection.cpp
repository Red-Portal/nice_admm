
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>
#include <string>

#include "gradient_projection.hpp"

namespace nice
{
    std::optional<std::tuple<nice::matrix, nice::column_vector>>
    active_constraints(nice::row_vector const& x,
                       nice::linear_set<nice::sparse_matrix> const& linear_constraints,
                       std::vector<nice::nonlinear_set> const& nonlinear_constraints)
    {
        auto const& [A, b] = linear_constraints;

        auto linear_tight_values = std::vector<std::pair<size_t, float>>();
        auto nonlinear_tight_values = std::vector<std::pair<size_t, float>>();
        linear_tight_values.reserve(A.rows());
        nonlinear_tight_values.reserve(nonlinear_constraints.size());

        auto linear_constraint_values = blaze::evaluate(A * blaze::trans(x) - b);

        size_t idx = 0;
        std::for_each(nonlinear_constraints.begin(),
                      nonlinear_constraints.end(),
                      [&x, &idx, &nonlinear_tight_values](nice::nonlinear_set const& elem){
                          float value = std::get<0>(elem)(x);
                          if(value >= std::numeric_limits<float>::epsilon())
                              nonlinear_tight_values.emplace_back(idx, value); 
                          ++idx;
                      });

        idx = 0;
        std::for_each(linear_constraint_values.begin(),
                      linear_constraint_values.end(),
                      [&idx, &linear_tight_values](float elem){
                          if(elem >= std::numeric_limits<float>::epsilon())
                              linear_tight_values.emplace_back(idx, elem); 
                          ++idx;
                      });

        if(linear_tight_values.empty() && nonlinear_tight_values.empty())
            return {};
        else
        {
            size_t columns = x.size();

            auto tight_count = linear_tight_values.size() + nonlinear_tight_values.size();
            auto active_set = nice::matrix(tight_count, columns);
            auto active_value = nice::column_vector(tight_count);

            size_t global_idx = 0;

            for(auto const& i : nonlinear_tight_values)
            {
                auto const& grad = std::get<1>(nonlinear_constraints[i.first]);
                blaze::row(active_set, global_idx) = grad(x);
                active_value[global_idx] = i.second;
                    
                ++global_idx;
            }


            for(auto const& i : linear_tight_values)
            {
                blaze::row(active_set, global_idx) = blaze::row(A, i.first);
                active_value[global_idx] = i.second;

                ++global_idx;
            }
            return std::make_tuple(active_set, active_value);
        }
    }

    std::optional<std::tuple<nice::matrix, nice::column_vector>>
    active_constraints(nice::row_vector const& x,
                       nice::linear_set<nice::matrix> const& linear_constraints,
                       std::vector<nice::nonlinear_set> const& nonlinear_constraints)
    {
        auto const& [A, b] = linear_constraints;

        auto linear_tight_values = std::vector<std::pair<size_t, float>>();
        auto nonlinear_tight_values = std::vector<std::pair<size_t, float>>();
        linear_tight_values.reserve(A.rows());
        nonlinear_tight_values.reserve(nonlinear_constraints.size());

        auto linear_constraint_values = blaze::evaluate(A * blaze::trans(x) - b);

        size_t idx = 0;
        std::for_each(nonlinear_constraints.begin(),
                      nonlinear_constraints.end(),
                      [&x, &idx, &nonlinear_tight_values](nice::nonlinear_set const& elem){
                          float value = std::get<0>(elem)(x);
                          if(value >= std::numeric_limits<float>::epsilon())
                              nonlinear_tight_values.emplace_back(idx, value); 
                          ++idx;
                      });

        idx = 0;
        std::for_each(linear_constraint_values.begin(),
                      linear_constraint_values.end(),
                      [&idx, &linear_tight_values](float elem){
                          if(elem >= std::numeric_limits<float>::epsilon())
                              linear_tight_values.emplace_back(idx, elem); 
                          ++idx;
                      });

        if(linear_tight_values.empty() && nonlinear_tight_values.empty())
            return {};
        else
        {
            size_t columns = x.size();

            auto tight_count = linear_tight_values.size() + nonlinear_tight_values.size();
            auto active_set = nice::matrix(tight_count, columns);
            auto active_value = nice::column_vector(tight_count);

            size_t global_idx = 0;

            for(auto const& i : nonlinear_tight_values)
            {
                auto const& grad = std::get<1>(nonlinear_constraints[i.first]);
                blaze::row(active_set, global_idx) = grad(x);
                active_value[global_idx] = i.second;
                    
                ++global_idx;
            }


            for(auto const& i : linear_tight_values)
            {
                blaze::row(active_set, global_idx) = blaze::row(A, i.first);
                active_value[global_idx] = i.second;

                ++global_idx;
            }
            return std::make_tuple(active_set, active_value);
        }
    }
}

// int main(int argc, char** argv)
// {
//     std::ios::sync_with_stdio(false);
//     (void)argc;

//     auto f = [](nice::row_vector const& x)
//         {
//             auto first = x[0] - 1;
//             auto second = x[1] - 2;
//             auto third = x[2] - 3;
//             auto fourth = x[3] - 4;
//             return first * first + second * second + third * third + fourth * fourth;
//         };

//     auto d_f = [](nice::row_vector const& x)
//         {
//             auto result = nice::row_vector(x.size());
//             result[0] = 2 * x[0] - 2; 
//             result[1] = 2 * x[1] - 4;  
//             result[2] = 2 * x[2] - 6;  
//             result[3] = 2 * x[3] - 8;  
//             return result;
//         };

//     // auto f = [](nice::row_vector const& x)
//     //     {
//     //         return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 2*x[0] - 3*x[3];
//     //     };

//     // auto d_f = [](nice::row_vector const& x)
//     //     {
//     //         auto result = nice::row_vector(x.size());
//     //         result[0] = 2 * x[0] - 2; 
//     //         result[1] = 2 * x[1];  
//     //         result[2] = 2 * x[2];  
//     //         result[3] = 2 * x[3] - 3;  
//     //         return result;
//     //     };

//     auto const A = nice::sparse_matrix{{-1, -1, -1, -1},
//                                        {-3, -3, -2, -1},
//                                        {1, 0, 0, 0},
//                                        {0, 1, 0, 0},
//                                        {0, 0, 1, 0},
//                                        {0, 0, 0, 1}};

//     auto const b = nice::column_vector{-5,
//                                        -10,
//                                        0,
//                                        0,
//                                        0,
//                                        0};

//     // auto const A = nice::sparse_matrix{{2, 1, 1, 4},
//     //                                    {1, 0, 0, 0},
//     //                                    {0, 1, 0, 0},
//     //                                    {0, 0, 1, 0},
//     //                                    {0, 0, 0, 1}};

//     // auto const b = nice::column_vector{7,
//     //                                    0,
//     //                                    0,
//     //                                    0,
//     //                                    0};

//     auto linear = std::make_tuple(A, b);


//     auto nonlinear = std::vector<nice::nonlinear_set>(0);
//     //auto nonlinear = std::vector<nice::nonlinear_set>(1);

//     // auto g = std::function<float(nice::row_vector const&)>(
//     //     [](nice::row_vector const& x)
//     //     {
//     //         return x[0] + x[1] + x[2]*x[2] + x[3] - 5.1;
//     //     });

//     // auto d_g = std::function<nice::row_vector(nice::row_vector const&)>(
//     //     [](nice::row_vector const& x)
//     //     {
//     //         auto result = nice::row_vector(4);
//     //         result[0] = 1;
//     //         result[1] = 1;
//     //         result[2] = 2 * x[2];
//     //         result[3] = 1;
//     //         return result;
//     //     });

//     //nonlinear[0] = nice::nonlinear_set(g, d_g);

//     //auto starting_point = nice::row_vector{5., 5., 5., 5.};
//     auto starting_point = nice::row_vector{0.5, 0.5, 0.5, 0.5};

//     std::string lr_arg(argv[1]);
//     size_t idx = 0;
//     double lr = std::stod(lr_arg, &idx);

//     std::cout << nice::gradient_projection(f,
//                                            d_f,
//                                            lr,
//                                            10000,
//                                            starting_point,
//                                            linear,
//                                            nonlinear,
//                                            0.01,
//                                            nice::verboseness::verbose) << std::endl;
// }
