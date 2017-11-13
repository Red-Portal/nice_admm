
#include <cstdlib>

#include <mgcpp/context/thread_guard.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>
#include <mgcpp/global/init.hpp>
#include <mgcpp/operations/abs.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/mean.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/sub.hpp>

#include <iostream>
#include <cassert>
#include <chrono>

int main()
{
    mgcpp::init();
    mgcpp::thread_guard guard(0, true);

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
    
    auto rho = mgcpp::device_vector<float>{50, 50, 50, 40,
                                           40, 40, 50, 60,
                                           60, 60, 70, 70,
                                           70, 80, 90, 100,
                                           120, 140, 120, 90,
                                           90, 90, 80, 60};
    assert(rho.shape() == 24);

    float deltab = 0.2;
    float etab = 0.95;
    float xi = 1;

    auto Eb_min = mgcpp::device_vector<float>(25, 0.1);
    auto Eb_max = mgcpp::device_vector<float>(25, 3);

    float v_min = 0.95;
    float v_max = 1.05;

    auto loss = mgcpp::device_vector<float>(24);
    loss.zero();

    float xi0 = 1;
    float xi1 = 1;
    float deltat = 1;

    auto mu2 = mgcpp::device_vector<float>(24); 
    mu2.zero();

    auto mu8 = mgcpp::device_vector<float>(24); 
    mu8.zero();

    auto mu10 = mgcpp::device_vector<float>(24); 
    mu10.zero();

    auto lambda2 = mgcpp::device_vector<float>(24); 
    lambda2.zero();

    auto lambda8 = mgcpp::device_vector<float>(24); 
    lambda8.zero();

    auto lambda10 = mgcpp::device_vector<float>(24); 
    lambda10.zero();

    auto Pl3 = mgcpp::device_vector<float>{0.3, 0.2, 0.2, 0.2,
                                           0.2, 0.2, 0.1, 0.3,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3};
    auto Pl4 = mgcpp::device_vector<float>{0.5, 0.6, 0.5, 1.0,
                                           0, 0.2, 0.3, 0.3,
                                           0.9, 0.9, 0.9, 1.0,
                                           1.7, 1.5, 1.5, 1.0,
                                           0.5, 0.5, 0.8, 1.3,
                                           1.3, 1.5, 0.9, 0.9};
    auto Pl5 = mgcpp::device_vector<float>{0.6, 0.4, 0.4, 0.3,
                                           0.0, 0.3, 0.3, 0.3,
                                           0.4, 0.5, 0.9, 0.9,
                                           1.7, 1.2, 1.7, 1.2,
                                           1.2, 1.0, 0.8, 1.0,
                                           1.5, 1.2, 0.9, 0.5};
    auto Pl6 = mgcpp::device_vector<float>{0.7, 0.5, 1.0, 1.1,
                                           1.0, 0.8, 0.8, 1.0,
                                           0.2, 0.6, 0.6, 0.5,
                                           1.1, 1.7, 0.5, 0.5,
                                           0.9, 0.9, 0.5, 1.2,
                                           1.3, 1.2, 0.5, 0.5};
    auto Pl9 = mgcpp::device_vector<float>{0.3, 0.2, 0.3, 0.2,
                                           0.2, 0.0, 0.2, 0.2,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3,
                                           0.3, 0.3, 0.3, 0.3};
    auto Pl11 = mgcpp::device_vector<float>{0.6, 0.7, 0.0, 0.0,
                                            1.1, 0.3, 0.3, 0.3,
                                            0.8, 0.8, 0.8, 0.9,
                                            1.1, 1.2, 1.8, 1.1,
                                            1.1, 1.1, 1.2, 1.2,
                                            1.3, 1.4, 0.8, 0.9};
    auto Pl12 = mgcpp::device_vector<float>{0.2, 0.3, 0.3, 0,
                                            0.0, 0.3, 0.1, 0.2,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3};
    auto Pl13 = mgcpp::device_vector<float>{0.2, 0.3, 0.3, 0.0,
                                            0.0, 0.2, 0.2, 0.0,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3,
                                            0.3, 0.3, 0.3, 0.3};

    using mgcpp::strict::mult;

    auto Ql3 = mult(0.01f, Pl3);
    auto Ql4 = mult(0.01f, Pl4);
    auto Ql5 = mult(0.01f, Pl5);
    auto Ql6 = mult(0.01f, Pl6);
    auto Ql9 = mult(0.01f, Pl9);
    auto Ql11 = mult(0.01f, Pl11);
    auto Ql12 = mult(0.01f, Pl12);
    auto Ql13 = mult(0.01f, Pl13);

    using mgcpp::strict::add;
    auto Pl =
        add(Pl3,
            add(Pl4,
                add(Pl5,
                    add(Pl6,
                        add(Pl9,
                            add(Pl11,
                                add(Pl12, Pl13)))))));
    auto Ql =
        add(Ql3,
            add(Ql4,
                add(Ql5,
                    add(Ql6,
                        add(Ql9,
                            add(Ql11,
                                add(Ql12, Ql13)))))));

    auto Pg14 = mgcpp::device_vector<float>{0, 0, 0, 0,
                                            0.0, 0.0, 0.0, 0.2,
                                            0.5, 0.8, 0.9, 1.0,
                                            1.0, 1, 0.95, 0.75,
                                            0.5, 0.25, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0};
    auto Pg15 = mgcpp::device_vector<float>{0.8, 0.6, 1.5, 1.9,
                                            2.1, 2.2, 1.4, 0.0,
                                            0.0, 0.8, 0.0, 1.5,
                                            0.0, 1.4, 0.1, 0.0,
                                            0.4, 0.3, 1.05, 1.15,
                                            2.55, 2.8, 2.7, 2.8};

    auto Qg14 = mult(0.01f, Pg14);
    auto Qg15 = mult(0.01f, Pg15);

    auto P2k = mgcpp::device_vector<float>(24);
    P2k.zero();

    auto Q2k = mgcpp::device_vector<float>(24);
    Q2k.zero();

    auto P8k = mgcpp::device_vector<float>(24);
    P8k.zero();

    auto Q8k = mgcpp::device_vector<float>(24);
    Q8k.zero();

    auto P10k = mgcpp::device_vector<float>(24);
    P10k.zero();

    auto Q10k = mgcpp::device_vector<float>(24);
    Q10k.zero();

    auto Pk = mgcpp::device_matrix<float>(3, 24);
    Pk.row(0) = P2k ;
    Pk.row(1) = P8k ;
    Pk.row(2) = P10k ;

    auto Qk = mgcpp::device_matrix<float>(3, 24);
    Qk.row(0) = Q2k ;
    Qk.row(1) = Q8k ;
    Qk.row(2) = Q10k ;
    // Pk = [P2k; P8k; P10k];
    // Qk = [Q2k; Q8k; Q10k];

    // Eb8 = (1).*[1.5, zeros(1,23)]; 

    auto Pgk2 = mgcpp::device_vector<float>(24);
    Pgk2.zero();

    auto Qgk2 = mgcpp::device_vector<float>(24);
    Qgk2.zero();

    auto Pbk8 = mgcpp::device_vector<float>(24);
    Pbk8.zero();

    auto Qbk8 = mgcpp::device_vector<float>(24);
    Qbk8.zero();

    auto Pgk10 = mgcpp::device_vector<float>(24);
    Pgk10.zero();

    auto P_net= mgcpp::device_matrix<float>(24, 3);
    P_net.zero();

    auto P_bus = mgcpp::device_matrix<float>(3, 24);
    P_bus.zero();

    auto P_main = mgcpp::device_vector<float>(24);
    P_main.zero();

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);

    std::cout << "variable init done!" << std::endl;
    std::cout << "time: " << duration.count() << "us" << std::endl;

    using mgcpp::strict::mean;
    using mgcpp::strict::abs;
    using mgcpp::strict::sub;

    start = std::chrono::steady_clock::now();

    auto loss_value = mean(abs(sub(abs(P_bus), abs(Pk))));

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start);

    std::cout << "loss: " << loss_value << std::endl;
    std::cout << "time: " << duration.count() << "us" << std::endl;
}
