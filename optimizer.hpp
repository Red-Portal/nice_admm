
#ifndef _NICE_ADMM_OPTIMIZER_HPP_
#define _NICE_ADMM_OPTIMIZER_HPP_

#include <blaze/Blaze.h>

using column_vector = blaze::DynamicVector<float, blaze::columnVector>; 
using row_vector = blaze::DynamicVector<float, blaze::rowVector>; 

class momentum_gradient_descent
{
private:
    row_vector _momentum;
    float _gamma;

public:
    inline
    momentum_gradient_descent() : _momentum(24, 0), _gamma(0.5) {};

    inline
    void clear()
    {
        return;
    }

    inline row_vector
    update(row_vector const& original, row_vector const& gradient)
    {
        _momentum = _momentum * _gamma + gradient;
        return original - _momentum;
    }
};

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
    update(row_vector const& original, row_vector const& gradient)
    {
        return original - gradient;
    }
};

#endif
