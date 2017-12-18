
#include "utility.hpp"
#include "optimization_phases.hpp"

#include <iostream>
#include <chrono>
#include <limits>

int main()
{
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
    //float ksi_g = 0.001;
    
    auto rho = nice::column_vector{50, 50, 50, 40,
                                   40, 40, 50, 60,
                                   60, 60, 70, 70,
                                   70, 80, 90, 100,
                                   120, 140, 120, 90,
                                   90, 90, 80, 60};

    float deltab = 0.2;
    float etab = 0.95;
    float xi = 1;

    auto Eb_min = nice::row_vector(25, 0.1);
    auto Eb_max = nice::row_vector(25, 3);

    float v_min = 0.95;
    float v_max = 1.05;

    auto loss = nice::row_vector(24, 0);

    float xi0 = 1;
    float xi1 = 1;
    float deltat = 1;

    auto mu2 = nice::row_vector(24, 0);
    auto mu8 = nice::row_vector(24, 0);
    auto mu10 = nice::row_vector(24, 0);
    auto lambda2 = nice::row_vector(24, 0);
    auto lambda8 = nice::row_vector(24, 0);
    auto lambda10 = nice::row_vector(24, 0);

    auto Pl3 = nice::row_vector{0.3, 0.2, 0.2, 0.2,
                                0.2, 0.2, 0.1, 0.3,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3};
    auto Pl4 = nice::row_vector{0.5, 0.6, 0.5, 1.0,
                                0, 0.2, 0.3, 0.3,
                                0.9, 0.9, 0.9, 1.0,
                                1.7, 1.5, 1.5, 1.0,
                                0.5, 0.5, 0.8, 1.3,
                                1.3, 1.5, 0.9, 0.9};
    auto Pl5 = nice::row_vector{0.6, 0.4, 0.4, 0.3,
                                0.0, 0.3, 0.3, 0.3,
                                0.4, 0.5, 0.9, 0.9,
                                1.7, 1.2, 1.7, 1.2,
                                1.2, 1.0, 0.8, 1.0,
                                1.5, 1.2, 0.9, 0.5};
    auto Pl6 = nice::row_vector{0.7, 0.5, 1.0, 1.1,
                                1.0, 0.8, 0.8, 1.0,
                                0.2, 0.6, 0.6, 0.5,
                                1.1, 1.7, 0.5, 0.5,
                                0.9, 0.9, 0.5, 1.2,
                                1.3, 1.2, 0.5, 0.5};
    auto Pl9 = nice::row_vector{0.3, 0.2, 0.3, 0.2,
                                0.2, 0.0, 0.2, 0.2,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3,
                                0.3, 0.3, 0.3, 0.3};
    auto Pl11 = nice::row_vector{0.6, 0.7, 0.0, 0.0,
                                 1.1, 0.3, 0.3, 0.3,
                                 0.8, 0.8, 0.8, 0.9,
                                 1.1, 1.2, 1.8, 1.1,
                                 1.1, 1.1, 1.2, 1.2,
                                 1.3, 1.4, 0.8, 0.9};
    auto Pl12 = nice::row_vector{0.2, 0.3, 0.3, 0,
                                 0.0, 0.3, 0.1, 0.2,
                                 0.3, 0.3, 0.3, 0.3,
                                 0.3, 0.3, 0.3, 0.3,
                                 0.3, 0.3, 0.3, 0.3,
                                 0.3, 0.3, 0.3, 0.3};
    auto Pl13 = nice::row_vector{0.2, 0.3, 0.3, 0.0,
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

    auto Pg14 = nice::row_vector{0, 0, 0, 0,
                                 0.0, 0.0, 0.0, 0.2,
                                 0.5, 0.8, 0.9, 1.0,
                                 1.0, 1, 0.95, 0.75,
                                 0.5, 0.25, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0};
    auto Pg15 = nice::row_vector{0.8, 0.6, 1.5, 1.9,
                                 2.1, 2.2, 1.4, 0.0,
                                 0.0, 0.8, 0.0, 1.5,
                                 0.0, 1.4, 0.1, 0.0,
                                 0.4, 0.3, 1.05, 1.15,
                                 2.55, 2.8, 2.7, 2.8};

    auto Qg14 = 0.01f * Pg14;
    auto Qg15 = 0.01f * Pg15;

    //blaze::Rand<nice::row_vector> randomizer{};

    auto P2k = nice::row_vector(24, -1);
    auto Q2k = nice::row_vector(24, 0);
    auto P8k = nice::row_vector(24, 0);
    auto Q8k = nice::row_vector(24, 0);
    auto P10k = nice::row_vector(24, 0);
    auto Q10k = nice::row_vector(24, 0);

    auto Pk = blaze::DynamicMatrix<float>(3, 24);
    row(Pk, 0) = P2k;
    row(Pk, 1) = P8k;
    row(Pk, 2) = P10k;

    auto Qk = blaze::DynamicMatrix<float>(3, 24);
    row(Qk, 0) = Q2k;
    row(Qk, 1) = Q8k;
    row(Qk, 2) = Q10k;

    auto Eb8 = nice::row_vector(24, 0);
    Eb8[0] = 1.5;

    auto Pkg2 = nice::row_vector(24, 0);
    auto Qkg2 = nice::row_vector(24, 0);
    auto Pkg8 = nice::row_vector(24, 0);
    auto Qkg8 = nice::row_vector(24, 0);
    auto Pkg10 = nice::row_vector(24, 0);
    auto Qkg10 = nice::row_vector(24, 0);

    auto P_net = blaze::DynamicMatrix<float>(24, 3, 0);
    auto P_bus = blaze::DynamicMatrix<float>(3, 24, 0.1);
    auto Q_bus = blaze::DynamicMatrix<float>(3, 24, 0.1);
    auto P_main = nice::row_vector(24, 0);

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);

    std::cout << "variable init done!" << std::endl;
    std::cout << "time: " << duration.count() << "us" << std::endl;

    start = std::chrono::steady_clock::now();


    float descent_rate = 0.2;

    // while(mean(blaze::eval(abs(abs(P_bus) - abs(Pk)))) > 0.001)
    {
        auto [Pgk2, Qgk2] =
            LC_DG_optimization(descent_rate,
                               2000,
                               alphag,
                               betag,
                               cg,
                               gamma,
                               Pg2_max,
                               Sg2,
                               Pk,
                               Qk,
                               mu2,
                               lambda2,
                               std::numeric_limits<float>::epsilon(),
                               false);

        row(P_bus, 0) = Pgk2;
        row(Q_bus, 0) = Qgk2;

        end = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Pg2: " << Pgk2 << std::endl;
        std::cout << "Qg2: " << Qgk2 << std::endl;


        auto [Pbk8, Qbk8] =
            LC_DS_first_optimization(descent_rate,
                                     2000,
                                     gamma,
                                     gammab,
                                     Pk,
                                     Qk,
                                     mu8,
                                     lambda8,
                                     std::numeric_limits<float>::epsilon(),
                                     true);

        row(P_bus, 1) = Pgk8;
        row(Q_bus, 1) = Qgk8;

        std::cout << "time: " << duration.count() << "us" << std::endl;
    }
}

