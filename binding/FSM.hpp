#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <mujoco/mujoco.h>
#include <Eigen/Core>

using namespace Eigen;
class FSM
{
public:
    FSM();
    ~FSM();
    bool get_contact_state(const mjModel* m, mjData* d);
    bool update_contact_event(const mjModel* m, mjData* d);
    double get_contact_time() const;
    const std::vector<double>& get_contact_values() const;
    double get_contact_elapsed(double sim_time) const;

    void get_pos(Vector2d pos);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};


