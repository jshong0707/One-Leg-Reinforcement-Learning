#pragma once

#include <memory>
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;

class Controller
{
public:
    Controller();
    ~Controller();              

    Vector2d j_PID(const Vector2d& error, const Vector2d& error_old);
    Vector2d PID(const Vector2d& error, const Vector2d& error_old);
    double Admittance(double omega_n, double zeta, double k, double Fz);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
