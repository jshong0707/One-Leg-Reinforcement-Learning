#include "Controller.hpp"
#include <cmath>        // std::acos
#include <utility>      // std::move
#include <Eigen/Core>
#include <iostream>

using Eigen::Vector2d;

struct Controller::Impl
{
    // Sampling time
    double Ts = 0.001;

    // Low-pass filter cutoff frequency
    double cutoff_freq = 100.0;
    double tau = 0.0;

    //* Task Space Control
    Eigen::Vector2d P_term = Vector2d::Zero();
    Eigen::Vector2d P_term_old = Vector2d::Zero();

    Eigen::Vector2d I_term = Vector2d::Zero();
    Eigen::Vector2d I_term_old = Vector2d::Zero();

    Eigen::Vector2d D_term = Vector2d::Zero();
    Eigen::Vector2d D_term_old = Vector2d::Zero();

    Eigen::Vector2d PID_output = Vector2d::Zero();

    Eigen::Vector2d KP = Vector2d::Zero();
    Eigen::Vector2d KI = Vector2d::Zero();
    Eigen::Vector2d KD = Vector2d::Zero();

    //* Joint Space Control
    Eigen::Vector2d j_P_term = Vector2d::Zero();
    Eigen::Vector2d j_P_term_old = Vector2d::Zero();

    Eigen::Vector2d j_I_term = Vector2d::Zero();
    Eigen::Vector2d j_I_term_old = Vector2d::Zero();

    Eigen::Vector2d j_D_term = Vector2d::Zero();
    Eigen::Vector2d j_D_term_old = Vector2d::Zero();

    Eigen::Vector2d j_PID_output = Vector2d::Zero();

    Eigen::Vector2d j_KP = Vector2d::Zero();
    Eigen::Vector2d j_KI = Vector2d::Zero();
    Eigen::Vector2d j_KD = Vector2d::Zero();

    Eigen::Vector2d ctrl_input;
    Eigen::Vector2d Joint_input;

    //* Admittance Control
    double Fz = 0.0;
    double Fz_old = 0.0;
    double Fz_old2 = 0.0;
    double dz = 0.0;
    double dz_old = 0.0;
    double dz_old2 = 0.0;
    

    
    Impl()
    {        
        tau = 1.0 / (2.0 * M_PI * cutoff_freq);
    }


    Vector2d j_PID(const Vector2d& error, const Vector2d& error_old)
    {
 
        j_KP[0] = 20;
        j_KP[1] = 20;
        j_KI[0] = 1;
        j_KI[1] = 1;
        j_KD[0] = 0.01;
        j_KD[1] = 0.01;



        tau = 1.0/(2.0*M_PI*cutoff_freq);
        j_P_term = j_KP.cwiseProduct(error);
        I_term  = j_KI.cwiseProduct((Ts/2.0)*(error+error_old)) + I_term_old;
        j_D_term  = 2.0*j_KD.cwiseProduct((1.0/(2.0*tau+Ts))*(error-error_old))
                          -((Ts-2.0*tau)/(2.0*tau+Ts)) * j_D_term_old;
        
        j_PID_output = j_P_term + j_I_term + j_D_term ; //[Leg_num] + D_term[Leg_num];

        j_P_term_old = j_P_term;
        j_I_term_old = j_I_term;        
        j_D_term_old = j_D_term;
        
        return j_PID_output;
    
        
    }
    
    Vector2d PID(const Vector2d& error, const Vector2d& error_old)
    {
        //    x                z    
        KP[0] = 5000;     KP[1] = 10000;         
        KI[0] = 1000;      KI[1] = 1200; 
        KD[0] = 160;       KD[1] = 160;   

            

        tau = 1.0/(2.0*M_PI*cutoff_freq);
        P_term  = KP.cwiseProduct(error);
     
        I_term  = KI.cwiseProduct((Ts/2.0)*(error+error_old)) + I_term_old;
     
        D_term  = 2.0*KD.cwiseProduct((1.0/(2.0*tau+Ts))*(error-error_old))
                          -((Ts-2.0*tau)/(2.0*tau+Ts))*D_term_old;
     
                          PID_output = P_term + I_term + D_term;
        
        P_term_old = P_term;
        I_term_old = I_term; 
        D_term_old = D_term;

        
        return PID_output;
    
        
    }

    double Admittance(double omega_n, double zeta, double k, double Fz)
    {
        double ad_M = k/(pow(omega_n,2));
        double ad_B = 2*zeta*k/omega_n;
        double ad_K = k;
        
        double c1 = 4 * ad_M + 2 * ad_B * Ts + ad_K * pow(Ts, 2);
        double c2 = -8 * ad_M + 2 * ad_K * pow(Ts, 2);
        double c3 = 4 * ad_M - 2 * ad_B * Ts + ad_K * pow(Ts, 2);


        for(int i = 0; i < 4; i++)
        {
            dz = (pow(Ts, 2) * Fz + 2 * pow(Ts, 2) * Fz_old +
                        pow(Ts, 2) * Fz_old2 - c2 * dz_old - c3 * dz_old2) / c1;
        }

            dz_old2 = dz_old;
            dz_old = dz; 
            Fz_old2 = Fz_old;
            Fz_old = Fz;

        return dz;
    }


};



Controller::Controller()
    : pimpl_(std::make_unique<Impl>())
{}

Controller::~Controller() = default;

Vector2d Controller::PID(const Vector2d& error, const Vector2d& error_old)
{return pimpl_->PID(error, error_old);}

Vector2d Controller::j_PID(const Vector2d& error, const Vector2d& error_old)
{return pimpl_->j_PID(error, error_old);}

double Controller::Admittance(double omega_n, double zeta, double k, double Fz)
{return pimpl_->Admittance(omega_n, zeta, k, Fz);}