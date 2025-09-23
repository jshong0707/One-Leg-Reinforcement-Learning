#include "FSM.hpp"
#include "EventTrigger.hpp"

struct FSM:: Impl
{
    mjModel* m;
    mjData* d;

    //! Contact State
        EventTrigger contact_event;
        bool contact_happened = false;
        bool contact_flag = false;
        Vector2d pos = Vector2d::Zero();

    Impl()
    {
        
    }

    ~Impl() = default;

    bool get_contact_state(const mjModel* m, mjData* d) {
        int foot_geom_id = mj_name2id(m, mjOBJ_GEOM, "foot_tip");
        if (foot_geom_id < 0) {
            std::cerr << "foot_tip geom not found!" << std::endl;
            return false;
        }

        for (int i = 0; i < d->ncon; i++) {
            const mjContact& con = d->contact[i];
            if (con.geom1 == foot_geom_id || con.geom2 == foot_geom_id) {
                return true;
            }
        }
        
        return false;
    }
    
    bool update_contact_event(const mjModel* m, mjData* d) {
        bool contact_flag = get_contact_state(m, d);
        
        return contact_event.update(contact_flag, d->time, [&]() {
            std::vector<double> vals;
            vals.push_back(d->sensordata[2]); // trunk_vel_z
            vals.push_back(pos[1]); // pos z
            // vals.push_back()
            return vals;
        });
    }
};

FSM::FSM()
: pimpl_(std::make_unique<Impl>())
{
} 

FSM::~FSM() = default;

//! Get Leg Pos
    void FSM:: get_pos(Vector2d pos)
    {
        pimpl_->pos = pos;
    }


//! Contact Event Trigger
    bool FSM::update_contact_event(const mjModel* m, mjData* d) {
        return pimpl_->update_contact_event(m, d);
    }
    bool FSM::get_contact_state(const mjModel* m, mjData* d)
    {
        return pimpl_->get_contact_state(m, d);
    }
    double FSM::get_contact_time() const {
        return pimpl_->contact_event.saved_time;
    }

    const std::vector<double>& FSM::get_contact_values() const {
        return pimpl_->contact_event.saved_values;
    }

    double FSM::get_contact_elapsed(double sim_time) const {
        return pimpl_->contact_event.elapsed(sim_time);
    }