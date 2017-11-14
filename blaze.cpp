

#include <blaze/Blaze.h>
#include <limits>

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


int main()
{
    using column_vector = blaze::DynamicVector<float, blaze::columnVector>; 
    using row_vector = blaze::DynamicVector<float, blaze::rowVector>; 

    auto start = std::chrono::steady_clock::now();
    
    float alphag = 10;
    float betag = 70;
    float cg = 0;
    float Pg2_max = 6;
    float Pg10_max = 1;
    float Sg2 = 12;
    float Sg10 = 2;
    float alphab = 1;
    float betab = 0.75;
    float gammab = 0.5;
    float gamma = 0.75;
    
    auto rho = column_vector{50, 50, 50, 40,
                             40, 40, 50, 60,
                             60, 60, 70, 70,
                             70, 80, 90, 100,
                             120, 140, 120, 90,
                             90, 90, 80, 60};

    float deltab = 0.2;
    float etab = 0.95;
    float xi = 1;

    auto Eb_min = row_vector(25, 0.1);
    auto Eb_max = row_vector(25, 3);

    float v_min = 0.95;
    float v_max = 1.05;

    auto loss = row_vector(24, 0);

    float xi0 = 1;
    float xi1 = 1;
    float deltat = 1;

    auto mu2 = row_vector(24, 0);
    auto mu8 = row_vector(24, 0);
    auto mu10 = row_vector(24, 0);
    auto lambda2 = row_vector(24, 0);
    auto lambda8 = row_vector(24, 0);
    auto lambda10 = row_vector(24, 0);

    auto Pl3 = row_vector{0.3, 0.2, 0.2, 0.2,
                          0.2, 0.2, 0.1, 0.3,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3};
    auto Pl4 = row_vector{0.5, 0.6, 0.5, 1.0,
                          0, 0.2, 0.3, 0.3,
                          0.9, 0.9, 0.9, 1.0,
                          1.7, 1.5, 1.5, 1.0,
                          0.5, 0.5, 0.8, 1.3,
                          1.3, 1.5, 0.9, 0.9};
    auto Pl5 = row_vector{0.6, 0.4, 0.4, 0.3,
                          0.0, 0.3, 0.3, 0.3,
                          0.4, 0.5, 0.9, 0.9,
                          1.7, 1.2, 1.7, 1.2,
                          1.2, 1.0, 0.8, 1.0,
                          1.5, 1.2, 0.9, 0.5};
    auto Pl6 = row_vector{0.7, 0.5, 1.0, 1.1,
                          1.0, 0.8, 0.8, 1.0,
                          0.2, 0.6, 0.6, 0.5,
                          1.1, 1.7, 0.5, 0.5,
                          0.9, 0.9, 0.5, 1.2,
                          1.3, 1.2, 0.5, 0.5};
    auto Pl9 = row_vector{0.3, 0.2, 0.3, 0.2,
                          0.2, 0.0, 0.2, 0.2,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3};
    auto Pl11 = row_vector{0.6, 0.7, 0.0, 0.0,
                           1.1, 0.3, 0.3, 0.3,
                           0.8, 0.8, 0.8, 0.9,
                           1.1, 1.2, 1.8, 1.1,
                           1.1, 1.1, 1.2, 1.2,
                           1.3, 1.4, 0.8, 0.9};
    auto Pl12 = row_vector{0.2, 0.3, 0.3, 0,
                           0.0, 0.3, 0.1, 0.2,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3};
    auto Pl13 = row_vector{0.2, 0.3, 0.3, 0.0,
                           0.0, 0.2, 0.2, 0.0,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3,
                           0.3, 0.3, 0.3, 0.3};

    auto Ql3 = 0.01f * Pl3;
    auto Ql4 = 0.01f * Pl4;
    auto Ql5 = 0.01f * Pl5;
    auto Ql6 = 0.01f * Pl6;
    auto Ql9 = 0.01f * Pl9;
    auto Ql11 = 0.01f * Pl11;
    auto Ql12 = 0.01f * Pl12;
    auto Ql13 = 0.01f * Pl13;

    auto Pl = Pl3 + Pl4 + Pl5 + Pl6 + Pl9 + Pl11 + Pl12 + Pl13;
    auto Ql = Ql3 + Ql4 + Ql5 + Ql6 + Ql9 + Ql11 + Ql12 + Ql13;

    auto Pg14 = row_vector{0, 0, 0, 0,
                           0.0, 0.0, 0.0, 0.2,
                           0.5, 0.8, 0.9, 1.0,
                           1.0, 1, 0.95, 0.75,
                           0.5, 0.25, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0};
    auto Pg15 = row_vector{0.8, 0.6, 1.5, 1.9,
                           2.1, 2.2, 1.4, 0.0,
                           0.0, 0.8, 0.0, 1.5,
                           0.0, 1.4, 0.1, 0.0,
                           0.4, 0.3, 1.05, 1.15,
                           2.55, 2.8, 2.7, 2.8};

    auto Qg14 = 0.01f * Pg14;
    auto Qg15 = 0.01f * Pg15;

    auto P2k = row_vector(24, 0);
    auto Q2k = row_vector(24, 0);
    auto P8k = row_vector(24, 0);
    auto Q8k = row_vector(24, 0);
    auto P10k = row_vector(24, 0);
    auto Q10k = row_vector(24, 0);

    auto Pk = blaze::DynamicMatrix<float>(3, 24);
    row(Pk, 0) = P2k;
    row(Pk, 1) = P8k;
    row(Pk, 2) = P10k;

    auto Qk = blaze::DynamicMatrix<float>(3, 24);
    row(Qk, 0) = Q2k;
    row(Qk, 1) = Q8k;
    row(Qk, 2) = Q10k;

    auto Eb8 = row_vector(24, 0);
    Eb8[0] = 1.5;

    auto Pkg2 = row_vector(24, 0);
    auto Qkg2 = row_vector(24, 0);
    auto Pkg8 = row_vector(24, 0);
    auto Qkg8 = row_vector(24, 0);
    auto Pkg10 = row_vector(24, 0);
    auto Qkg10 = row_vector(24, 0);

    auto P_net = blaze::DynamicMatrix<float>(24, 3, 0);
    auto P_bus = blaze::DynamicMatrix<float>(3, 24, 0.1);
    auto P_main = row_vector(24, 0);

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);

    std::cout << "variable init done!" << std::endl;
    std::cout << "time: " << duration.count() << "us" << std::endl;

    start = std::chrono::steady_clock::now();

    float descent_rate = 0.01; 

    // while(mean(blaze::eval(abs(abs(P_bus) - abs(Pk)))) > 0.001)
    {
        auto Pg2 = row_vector(24, 1);
        auto Qg2 = row_vector(24, 1);

        float LC_DG_loss = std::numeric_limits<float>::max();

        // cashed constants
        auto betag_vector = row_vector(24, betag);

        std::cout << "LC_LG" << std::endl;
        for(auto i = 0u; i < 27; ++i)
        {                                       
            auto Cg_diesel2 = alphag * power(Pg2) + betag * Pg2 +  row_vector(24, 1) * cg ;

            auto P = (1/(2 * gamma)) * sum(power(blaze::evaluate((-1 * Pg2) - row(Pk, 0) + mu2)));
            auto Q = (1/(2 * gamma)) * sum(power(blaze::evaluate((-1 * Qg2) - row(Qk, 0) + lambda2)));
            LC_DG_loss = sum(blaze::eval(Cg_diesel2)) + P + Q;
            
            // gradient descent
            auto Cg_diesel2_derivative = 2 * alphag * Pg2 + betag_vector;

            auto Pg_delta = descent_rate * (Cg_diesel2_derivative - (1 / gamma) * ((-1 * Pg2) - row(Pk, 0) + mu2));
            auto Qg_delta = descent_rate * (-1 / gamma) * ((-1 * Qg2) - row(Qk, 0) + lambda2);

            //std::cout << "loss: " << LC_DG_loss << std::endl;

            Pg2 = Pg2 - Pg_delta;
            Qg2 = Qg2 - Qg_delta;

        }
        end = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "loss: " << LC_DG_loss << std::endl;
        std::cout << "Pg2: " << Pg2 << std::endl;
        std::cout << "Qg2: " << Qg2 << std::endl;

        //std::cout << "loss: " << loss << std::endl;
        std::cout << "time: " << duration.count() << "us" << std::endl;
    }
}

