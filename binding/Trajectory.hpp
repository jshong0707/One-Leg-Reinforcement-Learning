#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <mujoco/mujoco.h>

using namespace Eigen;
using namespace std;

class FSM;

class Trajectory
{
public:
    Trajectory(std::shared_ptr<FSM> FSM_);
    ~Trajectory();
    
    Vector2d get_trajectory(double t, double z_vel, int Traj_mode);

    //! Contact Flag
    void contact_state_update(const mjModel* m, mjData* d);
 
private:    
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

};

