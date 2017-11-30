
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <limits>
#include <cmath>

#include "gradient_projection.hpp"


float
function(nice::row_vector const& x) 
{
    float first = x[0] - 1;
    float second = x[1] - 2;
    float third = x[2] - 3;
    float fourth = x[3] - 4;

    return first * first + second * second + third * third + fourth * fourth;
}

nice::row_vector
d_function(row_vector const& x) 
{
    auto gradient = row_vector(4);
    gradient[0] = 2 * x[0] - 2;
    gradient[1] = 2 * x[1] - 4;
    gradient[2] = 2 * x[2] - 6;
    gradient[3] = 2 * x[3] - 8;
    return gradient;
}

float
best_lambda(row_vector const& x,
            row_vector const& s) 
{
    auto dividend = s[0] * (1 - s[0])
        + s[1] * (2 - s[1])
        + s[2] * (3 - s[2])
        + s[3] * (4 - s[3]);

    auto divisor = s[0] * s[0]
        + s[1] * s[1]
        + s[2] * s[2]
        + s[3] * s[3];
    
    auto result = dividend / divisor;
    if(result >= 0)
        return result;
    else
        return std::numeric_limits<float>::max();
}

namespace nice
{
    blaze::DynamicMatrix<float>
    projection_matrix(matrix const& N)
    {
        auto id = blaze::IdentityMatrix<float>(4);
        return id - (trans(N) * inv(N * trans(N)) * N);
    }

    float
    max_lambda(matrix const& A,
               matrix const& b,
               row_vector const& x,
               row_vector const& s)
    {
        auto lambdas = ((-1) * A * trans(x) + b) / (A * trans(s));

        auto valid_lambdas = std::vector<float>();
        valid_lambdas.reserve(lambdas.size());

        for(auto i : lambdas)
        {
            if(i < 0)
                valid_lambdas.push_back(i);
        }

        if(valid_lambdas.size() == 0)
            return std::numeric_limits<float>::max();
        else if(valid_lambdas.size() == 1)
            return valid_lambdas[0];
        else
            return *std::min_element(valid_lambdas.begin(),
                                     valid_lambdas.end());
    }
}

int main()
{
    auto const N = matrix{{1, 1, 1, 1},
                          {3, 3, 2, 1},
                          {-1, 0, 0, 0},
                          {0, -1, 0, 0},
                          {0, 0, -1, 0},
                          {0, 0, 0, -1}};

    auto const b = column_vector{5, 10, 0, 0, 0, 0};

    auto in = blaze::DynamicMatrix<float>{{1, 1, 1, 1},
                                          {-1, 0, 0, 0}};

    auto start = std::chrono::steady_clock::now();
    auto result = nice::projection_matrix(in);
    auto end = std::chrono::steady_clock::now();

    std::cout << result << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "time: " << duration.count() << "us" << std::endl;
}
