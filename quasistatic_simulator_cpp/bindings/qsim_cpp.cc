#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>

#include "diffcp/log_barrier_solver.h"
#include "diffcp/qp_derivatives.h"
#include "diffcp/socp_derivatives.h"
#include "qsim/batch_quasistatic_simulator.h"
#include "qsim/contact_jacobian_calculator.h"
#include "qsim/finite_differencing_gradient.h"
#include "qsim/quasistatic_simulator.h"
// #include "qsim/pinocchio_calculator.h"

namespace py = pybind11;

// // convert RowMajor Eigen::Tensor to np.ndarray
// py::array tensor_to_array(const Eigen::Tensor<double, 3, Eigen::RowMajor>& tensor) {
//     const auto shape = tensor.dimensions();
//     const auto stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
//         shape[1] * shape[2], shape[2]);

//     return py::array(
//         py::buffer_info(
//             const_cast<double*>(tensor.data()),
//             sizeof(double),
//             py::format_descriptor<double>::format(),
//             3,
//             {shape[0], shape[1], shape[2]},
//             {sizeof(double) * stride.outer(), sizeof(double) * stride.inner(), sizeof(double)}
//         )
//     );
// }

// // convert ColMajor Eigen::Tensor to np.ndarray
// py::array tensor_to_array(const Eigen::Tensor<double, 3, Eigen::ColMajor>& tensor) {
//     const auto dimensions = tensor.dimensions();
//     const size_t dim_x = dimensions[0];
//     const size_t dim_y = dimensions[1];
//     const size_t dim_z = dimensions[2];

//     const size_t stride_x = sizeof(double);
//     const size_t stride_y = stride_x * dim_x;
//     const size_t stride_z = stride_y * dim_y;

//     return py::array(
//         py::buffer_info(
//             const_cast<double*>(tensor.data()),
//             sizeof(double),
//             py::format_descriptor<double>::format(),
//             3,
//             {dim_x, dim_y, dim_z},
//             {stride_x, stride_y, stride_z}
//         )
//     );
// }

PYBIND11_MODULE(qsim_cpp, m) {
  py::enum_<GradientMode>(m, "GradientMode")
      .value("kNone", GradientMode::kNone)
      .value("kBOnly", GradientMode::kBOnly)
      .value("kAB", GradientMode::kAB);

  py::enum_<ForwardDynamicsMode>(m, "ForwardDynamicsMode")
      .value("kQpMp", ForwardDynamicsMode::kQpMp)
      .value("kQpCvx", ForwardDynamicsMode::kQpCvx)
      .value("kSocpMp", ForwardDynamicsMode::kSocpMp)
      .value("kLogPyramidMp", ForwardDynamicsMode::kLogPyramidMp)
      .value("kLogPyramidCvx", ForwardDynamicsMode::kLogPyramidCvx)
      .value("kLogPyramidMy", ForwardDynamicsMode::kLogPyramidMy)
      .value("kLogIcecream", ForwardDynamicsMode::kLogIcecream);

  py::enum_ <DfDxMode>(m, "DfDxMode")
      .value("kAutoDiff", DfDxMode::kAutoDiff)
      .value("kAnalyticWithFiniteDiff", DfDxMode::kAnalyticWithFiniteDiff);

  {
    using Class = QuasistaticSimParameters;
    py::class_<Class>(m, "QuasistaticSimParameters")
        .def(py::init<>())
        .def_readwrite("h", &Class::h)
        .def_readwrite("gravity", &Class::gravity)
        .def_readwrite("contact_detection_tolerance",
                       &Class::contact_detection_tolerance)
        .def_readwrite("is_quasi_dynamic", &Class::is_quasi_dynamic)
        .def_readwrite("log_barrier_weight", &Class::log_barrier_weight)
        .def_readwrite("unactuated_mass_scale", &Class::unactuated_mass_scale)
        .def_readwrite("calc_contact_forces", &Class::calc_contact_forces)
        .def_readwrite("forward_mode", &Class::forward_mode)
        .def_readwrite("gradient_mode", &Class::gradient_mode)
        .def_readwrite("gradient_dfdx_mode", &Class::gradient_dfdx_mode)
        .def_readwrite("gradient_lstsq_tolerance",
                       &Class::gradient_lstsq_tolerance)
        .def_readwrite("nd_per_contact", &Class::nd_per_contact)
        .def_readwrite("use_free_solvers", &Class::use_free_solvers)
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
    using Class = QuasistaticSimulator;
    py::class_<Class>(m, "QuasistaticSimulatorCpp")
        .def(py::init<std::string,
                      const std::unordered_map<std::string, Eigen::VectorXd>&,
                      const std::unordered_map<std::string, std::string>&,
                      QuasistaticSimParameters>(),
             py::arg("model_directive_path"), py::arg("robot_stiffness_str"),
             py::arg("object_sdf_paths"), py::arg("sim_params"))
        .def("update_mbp_positions",
             py::overload_cast<const ModelInstanceIndexToVecMap&>(
                 &Class::UpdateMbpPositions))
        .def("update_mbp_positions_from_vector",
             py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&>(
                 &Class::UpdateMbpPositions))
        .def("get_mbp_positions", &Class::GetMbpPositions)
        .def("get_positions", &Class::GetPositions)
        .def("get_mbp_positions_as_vec", &Class::GetMbpPositionsAsVec)
        .def("step",
             py::overload_cast<const ModelInstanceIndexToVecMap&,
                               const ModelInstanceIndexToVecMap&,
                               const QuasistaticSimParameters&>(&Class::Step),
             py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"),
             py::arg("sim_params"))
        .def("step_default",
             py::overload_cast<const ModelInstanceIndexToVecMap&,
                               const ModelInstanceIndexToVecMap&>(&Class::Step),
             py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"))
        .def("calc_dynamics",
             py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&,
                               const Eigen::Ref<const Eigen::VectorXd>&,
                               const QuasistaticSimParameters&>(
                 &Class::CalcDynamics),
             py::arg("q"), py::arg("u"), py::arg("sim_params"))

        // TODO(yongpeng): APIs for forward and backward dynamics
        .def("calc_dynamics_forward", 
                                        // [](Class &instance,
                                        //  const Eigen::Ref<const Eigen::VectorXd>& q,
                                        //  const Eigen::Ref<const Eigen::VectorXd>& u,
                                        //  const QuasistaticSimParameters& sim_params) {
                                        //  auto startTime = std::chrono::steady_clock::now();
                                        //  auto result = instance.CalcDynamicsForward(q, u, sim_params);
                                        //  auto endTime = std::chrono::steady_clock::now();
                                        //  double duration_millsecond = std::chrono::duration<double, std::milli>(endTime - startTime).count();
                                        //  std::cout << "> CalcDynamicsForward in C++ time: " << duration_millsecond << " ms" << std::endl;
                                        //  return result;
                                        // },
             py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&,
                               const Eigen::Ref<const Eigen::VectorXd>&,
                               const QuasistaticSimParameters&>(
                 &Class::CalcDynamicsForward),
             py::arg("q"), py::arg("u"), py::arg("sim_params"))

        .def("calc_dynamics_forward", 
             py::overload_cast<const Eigen::Ref<const Eigen::VectorXd>&,
                               const Eigen::Ref<const Eigen::VectorXd>&,
                               const Eigen::Ref<const Eigen::VectorXd>&,
                               const QuasistaticSimParameters&>(
                 &Class::CalcDynamicsForward),
             py::arg("q"), py::arg("u"), py::arg("tau_ext"), py::arg("sim_params"))

        .def("calc_dynamics_backward",
             py::overload_cast<const QuasistaticSimParameters&>(
                 &Class::CalcDynamicsBackward),
             py::arg("sim_params"))
        .def("update_contact_info",
             &Class::UpdateContactInformation,
             py::arg("q"), py::arg("sim_params"))

        .def("calc_scaled_mass_matrix", &Class::CalcScaledMassMatrix)
        .def("calc_tau_ext", &Class::CalcTauExt)
        .def("get_model_instance_name_to_index_map",
             &Class::GetModelInstanceNameToIndexMap)
        .def("get_all_models", &Class::get_all_models)
        .def("get_actuated_models", &Class::get_actuated_models)
        .def("get_unactuated_models", &Class::get_unactuated_models)
        .def("get_query_object", &Class::get_query_object,
             py::return_value_policy::reference_internal)
        .def("get_plant", &Class::get_plant,
             py::return_value_policy::reference_internal)
        .def("get_scene_graph", &Class::get_scene_graph,
             py::return_value_policy::reference_internal)
        .def("get_contact_results", &Class::get_contact_results,
             py::return_value_policy::reference_internal)
        .def("get_contact_results_copy", &Class::GetContactResultsCopy)
        .def("get_sim_params", &Class::get_sim_params,
             py::return_value_policy::reference_internal)
        .def("get_sim_params_copy", &Class::get_sim_params_copy)
        .def("num_actuated_dofs", &Class::num_actuated_dofs)
        .def("num_unactuated_dofs", &Class::num_unactuated_dofs)
        .def("num_dofs", &Class::num_dofs)
        .def("get_Dq_nextDq", &Class::get_Dq_nextDq)
        .def("get_Dq_nextDqa_cmd", &Class::get_Dq_nextDqa_cmd)
        .def("get_tau_Ac", &Class::get_tau_Ac)
        .def("get_tau_Bc", &Class::get_tau_Bc)
        .def("get_f_Ac", &Class::get_f_Ac)
        .def("get_f_Bc", &Class::get_f_Bc)
        .def("get_geom_names_Ac", &Class::get_geom_names_Ac)
        .def("get_geom_names_Bc", &Class::get_geom_names_Bc)
        .def("get_points_Ac", &Class::get_points_Ac)
        .def("get_points_Bc", &Class::get_points_Bc)
        .def("get_sdists", &Class::get_contact_sdists)
        .def("get_Nhat", &Class::get_Nhat)
        .def("get_Jn_list", &Class::get_Jn_list)
        .def("get_phi_list", &Class::get_phi_list)
        .def("get_phi", &Class::get_phi)
        .def("get_Jn", &Class::get_Jn)
        .def("get_velocity_indices", &Class::GetVelocityIndices)
        .def("get_position_indices", &Class::GetPositionIndices)
        .def("get_v_dict_from_vec", &Class::GetVdictFromVec)
        .def("get_q_dict_from_vec", &Class::GetQDictFromVec)
        .def("get_q_vec_from_dict", &Class::GetQVecFromDict)
        .def("get_q_a_cmd_vec_from_dict", &Class::GetQaCmdVecFromDict)
        .def("get_q_a_cmd_dict_from_vec", &Class::GetQaCmdDictFromVec)
        .def("get_q_a_indices_into_q", &Class::GetQaIndicesIntoQ)
        .def("get_q_u_indices_into_q", &Class::GetQuIndicesIntoQ)
        .def("get_actuated_joint_limits", &Class::GetActuatedJointLimits)
        .def("get_avg_forward_time", &Class::get_avg_forward_time)
        .def("get_avg_backward_time", &Class::get_avg_backward_time)
        .def("print_solver_info_for_default_params",
             &Class::print_solver_info_for_default_params)
        .def("set_manipuland_names", &Class::SetManipulandNames)
        .def("set_collision_body_names", &Class::SetCollisionBodyNames)
        .def("check_collision", &Class::CheckCollision);
  }

  {
    using Class = BatchQuasistaticSimulator;
    py::class_<Class>(m, "BatchQuasistaticSimulator")
        .def(py::init<std::string,
                      const std::unordered_map<std::string, Eigen::VectorXd>&,
                      const std::unordered_map<std::string, std::string>&,
                      QuasistaticSimParameters>(),
             py::arg("model_directive_path"), py::arg("robot_stiffness_str"),
             py::arg("object_sdf_paths"), py::arg("sim_params"))
        .def("calc_dynamics_parallel", &Class::CalcDynamicsParallel)
        .def("calc_bundled_ABc_trj", &Class::CalcBundledABcTrj)
        .def("sample_gaussian_matrix", &Class::SampleGaussianMatrix)
        .def("calc_Bc_lstsq", &Class::CalcBcLstsq)
        .def("get_num_max_parallel_executions",
             &Class::get_num_max_parallel_executions)
        .def("set_num_max_parallel_executions",
             &Class::set_num_max_parallel_executions);
  }

  {
    using Class = FiniteDiffGradientCalculator;
    py::class_<Class>(m, "FiniteDiffGradientCalculator")
        .def(py::init([](QuasistaticSimulator* q_sim) {
               return std::make_unique<Class>(q_sim);
             }),
             py::arg("q_sim"), py::keep_alive<1, 2>()
             // Keep alive,reference: "self" keeps "q_sim" alive.
             )
        .def("calc_A", &Class::CalcA)
        .def("calc_B", &Class::CalcB);
  }

  {
    using Class = QpDerivativesActive;
    py::class_<Class>(m, "QpDerivativesActive")
        .def(py::init<double>(), py::arg("tol"))
        .def("UpdateProblem", &Class::UpdateProblem)
        .def("get_DzDe", &Class::get_DzDe)
        .def("get_DzDb", &Class::get_DzDb)
        .def("get_DzDvecG_active", &Class::get_DzDvecG_active);
  }

  {
    using Class = SocpDerivatives;
    py::class_<Class>(m, "SocpDerivatives")
        .def(py::init<double>(), py::arg("tol"))
        .def("UpdateProblem", &Class::UpdateProblem)
        .def("get_DzDe", &Class::get_DzDe)
        .def("get_DzDb", &Class::get_DzDb)
        .def("get_DzDvecG_active", &Class::get_DzDvecG_active);
  }

  {
    using Class = QpLogBarrierSolver;
    py::class_<Class>(m, "QpLogBarrierSolver")
        .def(py::init<>())
        .def("solve", &Class::Solve);
  }

  // {
  //   using Class = PinocchioCalculator;
  //   py::class_<Class>(m, "PinocchioCalculator")
  //       .def(py::init<
  //             const std::string&,
  //             const std::string&>(), 
  //         py::arg("robot_urdf_filename"), 
  //         py::arg("object_urdf_filename"))
  //       .def("UpdateModelConfiguration", 
  //         &Class::UpdateModelConfiguration, 
  //         py::arg("q0"))
  //       .def("UpdateHessiansAndJacobians", 
  //         &Class::UpdateHessiansAndJacobians)
        
  //       .def("GetContactJacobian",
  //         &Class::GetContactJacobian,
  //         py::arg("T_C2F"),
  //         py::arg("N_hat"),
  //         py::arg("F_name"))

  //       .def("GetContactKinematicHessian", [](Class &instance,
  //                                             const Eigen::Matrix4d& T_C2F,
  //                                             const Eigen::Vector3d& N_hat,
  //                                             const std::string F_name) {
  //                                             auto tensor = instance.GetContactKinematicHessian(T_C2F, N_hat, F_name);
  //                                             return tensor_to_array(tensor);
  //                                            },
  //         py::arg("T_C2F"), 
  //         py::arg("N_hat"), 
  //         py::arg("F_name"));
  // }
}
