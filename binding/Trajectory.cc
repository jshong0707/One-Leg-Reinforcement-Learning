#include "Trajectory.hpp"
#include "FSM.hpp"

struct Trajectory::Impl{

    //! Contact State
        std::shared_ptr<FSM> FSM_;
        bool contact_flag = false; // true: contact, false: no contact
        bool contact_happened = false;
        double contact_time = 0;
        double t_contact = 0;
        std::vector<double> contact_values; // zvel, zpos
        
    
    Vector2d pos_ref = Vector2d::Zero();
    Vector2d error = Vector2d::Zero();
    

    
    Impl(std::shared_ptr<FSM> F_)
    :FSM_(std::move(F_))
    {
        contact_values.resize(2);
        contact_values = {0, 0};

    }
     Vector2d stance_traj(double t)
    {
        double t_c = t_contact;
        double z_vel_contact = contact_values[0];
        double z_pos_contact = contact_values[1];

        //! Debugging
        double T = 0.15;
        double z0 = z_pos_contact;
        // double z0 = -0.48;
        double zd0 = -z_vel_contact;
        double zf = -0.15;
        double zdf = 0.0;

        // 계수 계산
        // double a0 = z0;
        // double a1 = zd0;
        // double a2 = (3*(zf - z0))/(T*T) - (2*zd0 + zdf)/T;
        // double a3 = (2*(z0 - zf))/(T*T*T) + (zd0 + zdf)/(T*T);

        double tau = t_contact;
         // 다항식 계수
        double a0 = z0;
        double a1 = zd0;
        double a2 = (3*(zf - z0))/(T*T) - (2*zd0 + zdf)/T;
        double a3 = (2*(z0 - zf))/(T*T*T) + (zd0 + zdf)/(T*T);

    // trajectory
        double z;
        if (tau < T) {
            z = a0 + a1*tau + a2*tau*tau + a3*tau*tau*tau;
        } else {
            z = zf;
        }

        pos_ref[0] = 0.0;
        pos_ref[1] = z0;

        // pos_ref[1] = -0.48;
        return pos_ref;
    }
    
    Vector2d Manipulability_Shaping(double z_vel)
    {
        double qm;
        double qb;
        double q_max = 20;
        double L = 0.25;
        double vz_max;

        vz_max = -z_vel;

        qb = acos(-sqrt(2)/2 * vz_max/(q_max * L));
        qm = acos(-cos(qb));

        pos_ref[0] = -L*cos(qm) - L*cos(qb);
        pos_ref[1] = -L*sin(qm) - L*sin(qb);

        //! ROM Avoidance
        if(abs(pos_ref[1]) > 0.48)
        {
            pos_ref[1] = -0.48;
        }
        
        // pos_ref[1] = -0.48;
        return pos_ref;
    }

    void contact_state_update(const mjModel* m, mjData* d)
    {
        if(t_contact == 0 || t_contact > 0.2)
        {
        contact_happened = FSM_->update_contact_event(m, d);
        contact_flag = FSM_->get_contact_state(m, d);
        contact_time = FSM_->get_contact_time();
        }
        t_contact = FSM_->get_contact_elapsed(d->time);

        if(contact_happened)
        {
            contact_values = FSM_->get_contact_values();
            contact_values[1] = pos_ref[1];
            cout << contact_values[0] << "  " << contact_values[1] << endl;
        }


    }
};



Vector2d Trajectory::get_trajectory(double t, double z_vel, int Traj_mode)
{
    Traj_mode = pimpl_->contact_flag;
    // Traj_mode = 1;
    // cout << pimpl_->contact_flag << endl;

    if(Traj_mode == 0)
    {
        return pimpl_->Manipulability_Shaping(z_vel);
    }
    else if(Traj_mode == 1)
    {
        return pimpl_->stance_traj(t);   
    }
    else
    {
        return Vector2d::Zero();
    }
    
}



void Trajectory::contact_state_update(const mjModel* m, mjData* d)
{
    pimpl_->contact_state_update(m, d);
    
}

Trajectory::Trajectory(std::shared_ptr<FSM> FSM_)
:pimpl_(std::make_unique<Impl>(std::move(FSM_)))
{
    
}

Trajectory::~Trajectory() = default;
