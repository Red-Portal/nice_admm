
#ifndef _NICE_UTILITY_HPP_
#define _NICE_UTILITY_HPP_

#define BLAZE_BLAS_MODE 1
#define BLAZE_BLAS_IS_PARALLEL 1
#define BLAZE_USE_VECTORIZATION 1

#include <blaze/Blaze.h>

namespace nice
{
    using matrix = blaze::DynamicMatrix<float>;
    using sparse_matrix = blaze::CompressedMatrix<float>;
    using column_vector = blaze::DynamicVector<float, blaze::columnVector>; 
    using row_vector = blaze::DynamicVector<float, blaze::rowVector>; 

    enum class verboseness : size_t {quiet = 0,
                                     verbose = 1,
                                     very_verbose = 2,
                                     log = 3};


    inline float
    sum(blaze::DynamicMatrix<float> const& mat)
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
    inline float
    sum(Vec const& vec)
    {
        float sum = 0;
        for(auto i = 0u; i < vec.size(); ++i)
        {
            sum += vec[i];
        }
        return sum;
    }

    inline float
    mean(blaze::DynamicMatrix<float> const& mat)
    {
        return sum(mat) / (mat.rows() * mat.columns());
    }

    template<typename VT, bool TF>
    inline decltype(auto)
    power(blaze::Vector<VT, TF> const& vec)
    {
        return vec * vec;
    }

    template< typename VT, bool TF >
    inline decltype(auto)
    norm_l2( blaze::Vector<VT,TF> const& vec )
    {
        return sqrt( blaze::dot( ~vec, ~vec ) );
    }
}

#endif
