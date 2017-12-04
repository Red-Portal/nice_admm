#include <vector>
#include <limits>

#include "gradient_projection.hpp"

namespace nice
{
    blaze::DynamicMatrix<float>
    projection_matrix(matrix const& N)
    {
        auto id = blaze::IdentityMatrix<float>(N.rows());
        return id - (trans(N) * inv(N * trans(N)) * N);
    }

    float
    max_lambda(matrix const& A,
               column_vector const& b,
               row_vector const& x,
               row_vector const& s)
    {
        auto lambdas = (((-1) * A * trans(x)) + b) / (A * trans(s));

        auto valid_lambdas = std::vector<float>();
        valid_lambdas.reserve(lambdas.size());

        for(auto i : blaze::evaluate(lambdas))
        {
            if(i > 0)
                valid_lambdas.push_back(i);
        }

        if(valid_lambdas.size() == 0)
            return std::numeric_limits<float>::max();
        else if(valid_lambdas.size() == 1)
        {
            float max_lambda_result = valid_lambdas[0];
            return max_lambda_result - std::numeric_limits<float>::epsilon();
        }
        else
        {
            float max_lambda_result = *std::min_element(valid_lambdas.begin(),
                                                        valid_lambdas.end());
            return max_lambda_result - std::numeric_limits<float>::epsilon();
        }
    }

    std::optional<matrix>
    active_constraints(matrix const& A,
                       column_vector const& b,
                       row_vector const& x)
    {
        auto status = blaze::evaluate(A * trans(x) - b);

        if (std::all_of(status.begin(), status.end(),
                        [](float elem)
                        {
                            return elem < 0;
                        }))
            return {};

        auto active_set = matrix();
        active_set.reserve(A.capacity());

        for(size_t i = 0u; i < status.size(); ++i)
        {
            if(status[i] > 0)
            {
                active_set.resize(active_set.rows() + 1, A.columns());
                blaze::row(active_set, active_set.rows() - 1) = blaze::row(A, i);
            }
        }
        return active_set;
    }
}

/*float
  function(nice::row_vector const& x) 
  {
  float first = x[0] - 1;
  float second = x[1] - 2;
  float third = x[2] - 3;
  float fourth = x[3] - 4;

  return first * first
  + second * second
  + third * third
  + fourth * fourth;
  }

  nice::row_vector
  d_function(nice::row_vector const& x) 
  {
  auto gradient = nice::row_vector(4);
  gradient[0] = 2 * x[0] - 2;
  gradient[1] = 2 * x[1] - 4;
  gradient[2] = 2 * x[2] - 6;
  gradient[3] = 2 * x[3] - 8;
  return gradient;
  }

  float
  best_lambda(nice::row_vector const& x,
  nice::row_vector const& s) 
  {
  auto dividend = s[0] * (1 - x[0])
  + s[1] * (2 - x[1])
  + s[2] * (3 - x[2])
  + s[3] * (4 - x[3]);

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


  int main()
  {
  std::ios::sync_with_stdio(false);

  auto starting_point = nice::row_vector{0., 0., 0., 0.};

  auto const A = nice::matrix{{1, 1, 1, 1},
  {3, 3, 2, 1},
  {-1, 0, 0, 0},
  {0, -1, 0, 0},
  {0, 0, -1, 0},
  {0, 0, 0, -1}};

  auto const b = nice::column_vector{5,
  10,
  0,
  0,
  0,
  0};

  std::cout << nice::gradient_projection(function,
  d_function,
  best_lambda,
  0.02,
  100,
  starting_point,
  A,
  b,
  true,
  0.00001) << std::endl;
  }*/
