// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   // Eigen <-> NumPy 변환
#include "Controller.hpp"
#include <cmath> 

namespace py = pybind11;

PYBIND11_MODULE(ctrlbind, m) {
    m.doc() = "Python bindings for Controller (PID, j_PID, Admittance)";

    py::class_<Controller>(m, "Controller")
        .def(py::init<>())
        // 메서드 이름은 파이썬에서 스네이크케이스로 노출
        .def("pid",
             [](Controller& self, const Eigen::Vector2d& err, const Eigen::Vector2d& err_old) {
                 return self.PID(err, err_old);
             },
             py::arg("error"), py::arg("error_old"),
             "Task-space PID output for 2D vector error")
        .def("j_pid",
             [](Controller& self, const Eigen::Vector2d& err, const Eigen::Vector2d& err_old) {
                 return self.j_PID(err, err_old);
             },
             py::arg("error"), py::arg("error_old"),
             "Joint-space PID output for 2D vector error")
        .def("admittance",
             [](Controller& self, double omega_n, double zeta, double k, double Fz) {
                 return self.Admittance(omega_n, zeta, k, Fz);
             },
             py::arg("omega_n"), py::arg("zeta"), py::arg("k"), py::arg("Fz"),
             "Second-order admittance (returns dz)");


    auto fk_pos = [](double q0, double q1, double L) {
        Eigen::Vector2d x;
        x << -L*std::cos(q0) - L*std::cos(q1),
                -L*std::sin(q0) - L*std::sin(q1);
        return x; // [x, z]
    };

    // 2R 평면팔 야코비안(작업공간 [x,z] 기준)
    auto jacobian = [](double q0, double q1, double L) {
        Eigen::Matrix2d J;
        J <<  L*std::sin(q0),  L*std::sin(q1),
             -L*std::cos(q0), -L*std::cos(q1);
        return J;
    };

        // FK / J 를 파이썬에 노출 (디버깅/검증용)
    m.def("fk_pos",
          [=](double q0, double q1, double L){ return fk_pos(q0, q1, L); },
          py::arg("q0"), py::arg("q1"), py::arg("L") = 0.25,
          "Planar 2R end-effector position [x, z]");

    m.def("jacobian",
          [=](double q0, double q1, double L){ return jacobian(q0, q1, L); },
          py::arg("q0"), py::arg("q1"), py::arg("L") = 0.25,
          "Planar 2R Jacobian (2x2) for [x, z]");

    // 태스크공간 PID → 조인트 토크 (tau = J^T f)
    // 반환값: (tau[2], e_x[2])  — e_x 는 다음 스텝에서 error_old 로 넘기면 됨
    m.def("task_pid_tau",
          [=](Controller& ctrl,
              double q0, double q1,
              const Eigen::Vector2d& x_ref,
              const Eigen::Vector2d& e_x_old,
              double L) {
              // 1) 기구학
              Eigen::Vector2d x = fk_pos(q0, q1, L);
              Eigen::Matrix2d J = jacobian(q0, q1, L);

              // 2) 태스크 오차
              Eigen::Vector2d e = x_ref - x;

              // 3) 태스크 힘 (PID in task space)
              Eigen::Vector2d f = ctrl.PID(e, e_x_old);

              // 4) 조인트 토크로 변환
              Eigen::Vector2d tau = J.transpose() * f;

              return py::make_tuple(tau, e);
          },
          py::arg("controller"),
          py::arg("q0"), py::arg("q1"),
          py::arg("x_ref"),
          py::arg("e_x_old"),
          py::arg("L") = 0.25,
          "Compute joint torques from task-space PID via tau = J^T * f");
}
