#ifndef _NICE_ADMM_OPTIMIZER_HPP_
#define _NICE_ADMM_OPTIMIZER_HPP_

#include <blaze/math/dense/DynamicVector.h>

using column_vector = blaze::DynamicVector<float, blaze::columnVector>; 
using row_vector = blaze::DynamicVector<float, blaze::rowVector>; 

class gradient_descent
{
public:
    inline
    gradient_descent() = default;

    inline
    void clear()
    {
        return;
    }

    inline row_vector
    operator()(row_vector const& gradient)
    {
        return gradient;
    }
};

#endif
