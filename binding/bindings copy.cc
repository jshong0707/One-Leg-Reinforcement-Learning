// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>

#include <mujoco/mujoco.h>         // mjModel, mjData
#include "Controller.hpp"
// #include "Trajectory.hpp"
// #include "FSM.hpp"

namespace py = pybind11;

// ---- Python mujoco 객체에서 C 포인터 뽑기 (model/data .ptr) ----
static mjModel* as_mjmodel_ptr(const py::object& model_obj) {
    if (py::hasattr(model_obj, "ptr")) {
        py::object p = model_obj.attr("ptr");
        // int 형태, c_void_p.value 형태, capsule 등 다양하게 대비
        if (py::isinstance<py::int_>(p)) {
            auto addr = p.cast<uintptr_t>();
            return reinterpret_cast<mjModel*>(addr);
        }
        if (py::hasattr(p, "value")) {
            auto addr = p.attr("value").cast<uintptr_t>();
            return reinterpret_cast<mjModel*>(addr);
        }
        if (py::isinstance<py::capsule>(p)) {
            py::capsule cap = p.cast<py::capsule>();
            return reinterpret_cast<mjModel*>(cap.get_pointer());
        }
    }
    throw std::runtime_error("Failed to get mjModel* from Python model (.ptr missing or unsupported type).");
}

static mjData* as_mjdata_ptr(const py::object& data_obj) {
    if (py::hasattr(data_obj, "ptr")) {
        py::object p = data_obj.attr("ptr");
        if (py::isinstance<py::int_>(p)) {
            auto addr = p.cast<uintptr_t>();
            return reinterpret_cast<mjData*>(addr);
        }
        if (py::hasattr(p, "value")) {
            auto addr = p.attr("value").cast<uintptr_t>();
            return reinterpret_cast<mjData*>(addr);
        }
        if (py::isinstance<py::capsule>(p)) {
            py::capsule cap = p.cast<py::capsule>();
            return reinterpret_cast<mjData*>(cap.get_pointer());
        }
    }
    throw std::runtime_error("Failed to get mjData* from Python data (.ptr missing or unsupported type).");
}

PYBIND11_MODULE(ctrlbind, m) {
    m.doc() = "Bindings: Controller (PID/jPID/Admittance) + Trajectory + FSM + kinematics helpers";

    // ---------- Controller ----------
    py::class_<Controller>(m, "Controller")
        .def(py::init<>())
        .def("pid",
             [](Controller& self, const Eigen::Vector2d& err, const Eigen::Vector2d& err_old) {
                 return self.PID(err, err_old);
             },
             py::arg("error"), py::arg("error_old"),
             "Task-space PID output for 2D vector error (returns f_task)")
        .def("j_pid",
             [](Controller& self, const Eigen::Vector2d& err, const Eigen::Vector2d& err_old) {
                 return self.j_PID(err, err_old);
             },
             py::arg("error"), py::arg("error_old"),
             "Joint-space PID")
        .def("admittance",
             [](Controller& self, double omega_n, double zeta, double k, double Fz) {
                 return self.Admittance(omega_n, zeta, k, Fz);
             },
             py::arg("omega_n"), py::arg("zeta"), py::arg("k"), py::arg("Fz"),
             "Second-order admittance (returns dz)");

    // ---------- FSM ----------
    // 주의: FSM 은 C++에서 mjModel/mjData 를 직접 사용 (foot_tip 접촉 검사)합니다.  :contentReference[oaicite:2]{index=2}
    py::class_<FSM, std::shared_ptr<FSM>>(m, "FSM")
        .def(py::init<>())
        .def("get_pos", &FSM::get_pos, py::arg("pos"),
             "Update leg position (Vector2d) used for logging at contact")
        .def("get_contact_state",
             [](FSM& self, const py::object& model, const py::object& data) {
                 mjModel* mptr = as_mjmodel_ptr(model);
                 mjData*  dptr = as_mjdata_ptr(data);
                 return self.get_contact_state(mptr, dptr);
             },
             py::arg("model"), py::arg("data"),
             "Return current contact flag for geom 'foot_tip'")
        .def("update_contact_event",
             [](FSM& self, const py::object& model, const py::object& data) {
                 mjModel* mptr = as_mjmodel_ptr(model);
                 mjData*  dptr = as_mjdata_ptr(data);
                 return self.update_contact_event(mptr, dptr);
             },
             py::arg("model"), py::arg("data"),
             "Edge-triggered contact event; logs z-vel, z-pos at contact")
        .def("get_contact_time", &FSM::get_contact_time)
        .def("get_contact_values", &FSM::get_contact_values,
             py::return_value_policy::reference_internal,
             "Return saved values at contact (e.g., [zvel, zpos])")
        .def("get_contact_elapsed", &FSM::get_contact_elapsed, py::arg("sim_time"),
             "Elapsed time since last contact event (sim time)");

    // ---------- Trajectory ----------
    // Trajectory 는 FSM 공유포인터로 생성되고,
    // pre-contact: Manipulability_Shaping(z_vel), post-contact: stance_traj(t) 로 동작합니다. :contentReference[oaicite:3]{index=3}
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<std::shared_ptr<FSM>>(), py::arg("fsm"))
        .def("contact_state_update",
             [](Trajectory& self, const py::object& model, const py::object& data) {
                 mjModel* mptr = as_mjmodel_ptr(model);
                 mjData*  dptr = as_mjdata_ptr(data);
                 self.contact_state_update(mptr, dptr);
             },
             py::arg("model"), py::arg("data"),
             "Update internal contact flag/timestamps from MuJoCo model/data")
        .def("get_trajectory",
             &Trajectory::get_trajectory,
             py::arg("t"), py::arg("z_vel"), py::arg("traj_mode") = 0,
             "Return [x_ref, z_ref]; pre-contact uses Manipulability_Shaping(z_vel), post-contact uses stance_traj(t)")
        // 편의 함수: MuJoCo-루프에서 dz 를 더한 최종 reference 가 바로 필요할 때
        .def("reference_with_dz",
             [](Trajectory& self, double t, double z_vel, double dz, int traj_mode) {
                 Eigen::Vector2d ref = self.get_trajectory(t, z_vel, traj_mode);
                 ref[1] += dz;   // post-contact 에서 dz 로만 움직이게 할 때 편리
                 return ref;
             },
             py::arg("t"), py::arg("z_vel"), py::arg("dz"), py::arg("traj_mode") = 0,
             "Convenience: returns get_trajectory(t,z_vel,mode) with z_ref += dz");

    // ---------- FK / J helpers (디버깅/검증용) ----------
    m.def("fk_pos",
          [](double q0, double q1, double L) {
              Eigen::Vector2d x;
              x << -L*std::cos(q0) - L*std::cos(q1),
                   -L*std::sin(q0) - L*std::sin(q1);
              return x;
          },
          py::arg("q0"), py::arg("q1"), py::arg("L") = 0.25,
          "Planar 2R end-effector position [x, z]");

    m.def("jacobian",
          [](double q0, double q1, double L) {
              Eigen::Matrix2d J;
              J <<  L*std::sin(q0),  L*std::sin(q1),
                   -L*std::cos(q0), -L*std::cos(q1);
              return J;
          },
          py::arg("q0"), py::arg("q1"), py::arg("L") = 0.25,
          "Planar 2R Jacobian (2x2) for [x, z]");
}
