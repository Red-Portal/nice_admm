
#ifndef _NICE_UTILITY_HPP_
#define _NICE_UTILITY_HPP_

#include <blaze/Blaze.h>

namespace nice
{
    using matrix = blaze::DynamicMatrix<float>;
    using column_vector = blaze::DynamicVector<float, blaze::columnVector>; 
    using row_vector = blaze::DynamicVector<float, blaze::rowVector>; 

    float sum(blaze::DynamicMatrix<float> const& mat)
    {
        float sum = 0;
        for(auto i = 0u; i < mat.rows(); ++i)
        {
            for(auto j = 0u; j < mat.columns(); ++j)
            {
                sum += mat(i, j);
            }
        }
        return sum;
    }

    template<typename Vec>
    float sum(Vec const& vec)
    {
        float sum = 0;
        for(auto i = 0u; i < vec.size(); ++i)
        {
            sum += vec[i];
        }
        return sum;
    }

    float mean(blaze::DynamicMatrix<float> const& mat)
    {
        return sum(mat) / (mat.rows() * mat.columns());
    }

    template<bool Allign>
    blaze::DynamicVector<float, Allign>
    power(blaze::DynamicVector<float, Allign> const& vec)
    {
        return vec * vec;
    }
}

#endif
