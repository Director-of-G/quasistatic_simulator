#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "controllers/contact_controller.h"
#include "controllers/contact_controller_params.h"


namespace py = pybind11;

PYBIND11_MODULE(contact_ctrl_cpp, m) {

  {
    using Class = ContactControllerParameters;
    py::class_<Class>(m, "ContactControllerParameters")
        .def(py::init<>())
        .def_readwrite("weight_q", &Class::weight_q)
        .def_readwrite("weight_dq", &Class::weight_dq)
        .def_readwrite("weight_f",
                       &Class::weight_f)
        .def_readwrite("weight_u", &Class::weight_u)
        .def_readwrite("weight_du", &Class::weight_du)
        .def_readwrite("time_step", &Class::time_step)
        .def_readwrite("horizon_length", &Class::horizon_length)
        .def_readwrite("calc_torque_feedforward", &Class::calc_torque_feedforward)
        .def_readwrite("is_3d_floating", &Class::is_3d_floating)
        .def_readwrite("hand_base_trans", &Class::hand_base_trans)
        .def_readwrite("hand_base_rot", &Class::hand_base_rot)
        .def("__copy__",
             [](const Class& self) {
               return Class(self);
             })
        .def(
            "__deepcopy__",
            [](const Class& self, py::dict) {
              return Class(self);
            },
            "memo");
  }

  {
    using Class = ContactController;
    py::class_<Class>(m, "ContactControllerCpp")
        .def(py::init<std::string,
                      ContactControllerParameters>(),
             py::arg("model_path"), py::arg("controller_params"))
        .def("step",
              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&,
                               const Eigen::Ref<const Eigen::VectorXd>&>
              (&Class::Step),
                  py::arg("q"), py::arg("v"))
        .def("set_x0",
              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&>
              (&Class::Set_X0),
                  py::arg("x0"))
        .def("set_xref",
              py::overload_cast<const std::vector<Eigen::VectorXd>&>
              (&Class::Set_Xref),
                  py::arg("x_ref"))
        .def("set_fext_feedforward",
              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&>
              (&Class::Set_Fext),
                  py::arg("fext_ff"))
        .def("get_Ad", &Class::Get_Ad)
        .def("get_Bd", &Class::Get_Bd)
        .def("get_Ac", &Class::Get_Ac)
        .def("get_Bc", &Class::Get_Bc)
        .def("get_phi_vn_vt", &Class::Get_phi_vn_vt)
        .def("get_tau_feedforward", &Class::Get_tau_feedforward);
  }
}
