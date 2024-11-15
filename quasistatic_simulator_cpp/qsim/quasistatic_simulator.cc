#include "qsim/quasistatic_simulator.h"

#include <set>
#include <vector>
#include <chrono>
#include "math.h"

#include "drake/common/drake_path.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/parsing/process_model_directives.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/diagram_builder.h"
#include "qsim/get_model_paths.h"

using drake::AutoDiffXd;
using drake::Matrix3X;
using drake::MatrixX;
using drake::Vector3;
using drake::Vector4;
using drake::VectorX;
using drake::math::ExtractValue;
using drake::math::InitializeAutoDiff;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::Body;
using drake::multibody::Joint;
using drake::multibody::JointIndex;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
// using namespace mosek::fusion;
// using namespace monty;

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

void CreateMbp(
    drake::systems::DiagramBuilder<double>* builder,
    const string& model_directive_path,
    const std::unordered_map<string, VectorXd>& robot_stiffness_str,
    const std::unordered_map<string, string>& object_sdf_paths,
    const Eigen::Ref<const Vector3d>& gravity,
    drake::multibody::MultibodyPlant<double>** plant,
    drake::geometry::SceneGraph<double>** scene_graph,
    std::set<ModelInstanceIndex>* robot_models,
    std::set<ModelInstanceIndex>* object_models,
    std::unordered_map<ModelInstanceIndex, Eigen::VectorXd>* robot_stiffness) {
  std::tie(*plant, *scene_graph) =
      drake::multibody::AddMultibodyPlantSceneGraph(builder, 1e-3);
  // Set name so that MBP and SceneGraph can be accessed by name.
  (*plant)->set_name(kMultiBodyPlantName);
  (*scene_graph)->set_name(kSceneGraphName);
  auto parser = drake::multibody::Parser(*plant, *scene_graph);
  // TODO(pang): add package paths from yaml file? Hard-coding paths is clearly
  //  not the solution...
  parser.package_map().Add("quasistatic_simulator", GetQsimModelsPath());
  parser.package_map().Add(
      "drake_manipulation_models",
      drake::MaybeGetDrakePath().value() + "/manipulation/models");
  parser.package_map().Add("iiwa_controller", GetRoboticsUtilitiesModelsPath());
  // parser.package_map().Add("idto_models", "/home/yongpeng/research/projects/contact_rich/idto/examples/models");

  // Objects.
  // Use a Set to sort object names.
  std::set<std::string> object_names;
  for (const auto& item : object_sdf_paths) {
    object_names.insert(item.first);
  }
  for (const auto& name : object_names) {
    object_models->insert(
        parser.AddModelFromFile(object_sdf_paths.at(name), name));
  }

  // Robots.
  drake::multibody::parsing::ProcessModelDirectives(
      drake::multibody::parsing::LoadModelDirectives(model_directive_path),
      *plant, nullptr, &parser);
  for (const auto& [name, Kp] : robot_stiffness_str) {
    auto robot_model = (*plant)->GetModelInstanceByName(name);

    robot_models->insert(robot_model);
    (*robot_stiffness)[robot_model] = Kp;
  }

  // Gravity.
  (*plant)->mutable_gravity_field().set_gravity_vector(gravity);
  (*plant)->Finalize();
}

QuasistaticSimulator::QuasistaticSimulator(
    const std::string& model_directive_path,
    const std::unordered_map<std::string, Eigen::VectorXd>& robot_stiffness_str,
    const std::unordered_map<std::string, std::string>& object_sdf_paths,
    QuasistaticSimParameters sim_params)
    : sim_params_(std::move(sim_params)),
      solver_scs_(std::make_unique<drake::solvers::ScsSolver>()),
      solver_osqp_(std::make_unique<drake::solvers::OsqpSolver>()),
      solver_grb_(std::make_unique<drake::solvers::GurobiSolver>()),
      solver_msk_(std::make_unique<drake::solvers::MosekSolver>()),
      solver_log_pyramid_(std::make_unique<QpLogBarrierSolver>()),
      solver_log_icecream_(std::make_unique<SocpLogBarrierSolver>()) {
  auto builder = drake::systems::DiagramBuilder<double>();

  CreateMbp(&builder, model_directive_path, robot_stiffness_str,
            object_sdf_paths, sim_params_.gravity, &plant_, &sg_,
            &models_actuated_, &models_unactuated_, &robot_stiffness_);
  // All models instances.
  models_all_ = models_unactuated_;
  models_all_.insert(models_actuated_.begin(), models_actuated_.end());
  diagram_ = builder.Build();

  // Contexts.
  context_ = diagram_->CreateDefaultContext();
  context_plant_ =
      &(diagram_->GetMutableSubsystemContext(*plant_, context_.get()));
  context_sg_ = &(diagram_->GetMutableSubsystemContext(*sg_, context_.get()));

  // MBP introspection.
  n_q_ = plant_->num_positions();
  n_v_ = plant_->num_velocities();

  for (const auto& model : models_all_) {
    velocity_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kV);
    position_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kQ);
  }

  n_v_a_ = 0;
  for (const auto& model : models_actuated_) {
    auto n_v_a_i = plant_->num_velocities(model);
    DRAKE_THROW_UNLESS(n_v_a_i == robot_stiffness_[model].size());
    n_v_a_ += n_v_a_i;
  }

  n_v_u_ = 0;
  for (const auto& model : models_unactuated_) {
    n_v_u_ += plant_->num_velocities(model);
  }

  // Find planar model instances.
  /* Features of a 3D un-actuated model instance:
   *
   * 1. The model instance has only 1 rigid body.
   * 2. The model instance has a floating base.
   * 3. The model instance has 6 velocities and 7 positions.
   */
  for (const auto& model : models_unactuated_) {
    const auto n_v = plant_->num_velocities(model);
    const auto n_q = plant_->num_positions(model);
    if (n_v == 0) {
      is_model_fixed_[model] = true;
    } else {
      is_model_fixed_[model] = false;
    }
    if (n_v == 6 && n_q == 7) {
      const auto body_indices = plant_->GetBodyIndices(model);
      DRAKE_THROW_UNLESS(body_indices.size() == 1);
      DRAKE_THROW_UNLESS(plant_->get_body(body_indices.at(0)).is_floating());
      is_3d_floating_[model] = true;
    } else {
      is_3d_floating_[model] = false;
    }
  }

  for (const auto& model : models_actuated_) {
    is_3d_floating_[model] = false;
  }

  // QP derivative.
  dqp_ = std::make_unique<QpDerivativesActive>(
      sim_params_.gradient_lstsq_tolerance);
  dsocp_ =
      std::make_unique<SocpDerivatives>(sim_params_.gradient_lstsq_tolerance);

  // Find smallest stiffness.
  VectorXd min_stiffness_vec(models_actuated_.size());
  int i = 0;
  for (const auto& model : models_actuated_) {
    min_stiffness_vec[i] = robot_stiffness_.at(model).minCoeff();
    i++;
  }
  min_K_a_ = min_stiffness_vec.minCoeff();

  // AutoDiff plants.
  diagram_ad_ =
      drake::systems::System<double>::ToAutoDiffXd<drake::systems::Diagram>(
          *diagram_);
  plant_ad_ = dynamic_cast<const drake::multibody::MultibodyPlant<AutoDiffXd>*>(
      &(diagram_ad_->GetSubsystemByName(plant_->get_name())));
  sg_ad_ = dynamic_cast<const drake::geometry::SceneGraph<drake::AutoDiffXd>*>(
      &(diagram_ad_->GetSubsystemByName(sg_->get_name())));

  // AutoDiff contexts.
  context_ad_ = diagram_ad_->CreateDefaultContext();
  context_plant_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*plant_ad_, context_ad_.get()));
  context_sg_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*sg_ad_, context_ad_.get()));

  // ContactComputers.
  cjc_ = std::make_unique<ContactJacobianCalculator<double>>(diagram_.get(),
                                                             models_all_);
  cjc_ad_ = std::make_unique<ContactJacobianCalculator<AutoDiffXd>>(
      diagram_ad_.get(), models_all_);

  contact_results_.set_plant(plant_);
}

std::vector<int> QuasistaticSimulator::GetIndicesForModel(
    drake::multibody::ModelInstanceIndex idx, ModelIndicesMode mode) const {
  std::vector<double> selector;
  if (mode == ModelIndicesMode::kQ) {
    selector.resize(n_q_);
  } else {
    selector.resize(n_v_);
  }
  std::iota(selector.begin(), selector.end(), 0);
  Eigen::Map<VectorXd> selector_eigen(selector.data(), selector.size());

  VectorXd indices_d;
  if (mode == ModelIndicesMode::kQ) {
    indices_d = plant_->GetPositionsFromArray(idx, selector_eigen);
  } else {
    indices_d = plant_->GetVelocitiesFromArray(idx, selector_eigen);
  }
  std::vector<int> indices(indices_d.size());
  for (size_t i = 0; i < indices_d.size(); i++) {
    indices[i] = roundl(indices_d[i]);
  }
  return indices;
}

/*
 * Similar to the python implementation, this function updates context_plant_
 * and query_object_.
 */
void QuasistaticSimulator::UpdateMbpPositions(
    const ModelInstanceIndexToVecMap& q_dict) {
  for (const auto& model : models_all_) {
    plant_->SetPositions(context_plant_, model, q_dict.at(model));
  }

  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpPositions(
    const Eigen::Ref<const Eigen::VectorXd>& q) {
  plant_->SetPositions(context_plant_, q);
  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const ModelInstanceIndexToVecAdMap& q_dict) const {
  for (const auto& model : models_all_) {
    plant_ad_->SetPositions(context_plant_ad_, model, q_dict.at(model));
  }

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const Eigen::Ref<const drake::AutoDiffVecXd>& q) const {
  plant_ad_->SetPositions(context_plant_ad_, q);

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetMbpPositions() const {
  ModelInstanceIndexToVecMap q_dict;
  for (const auto& model : models_all_) {
    q_dict[model] = plant_->GetPositions(*context_plant_, model);
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetPositions(
    drake::multibody::ModelInstanceIndex model) const {
  return plant_->GetPositions(*context_plant_, model);
}

void QuasistaticSimulator::CalcQAndTauH(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict, const double h,
    MatrixXd* Q_ptr, VectorXd* tau_h_ptr,
    const double unactuated_mass_scale) const {
  MatrixXd& Q = *Q_ptr;
  Q = MatrixXd::Zero(n_v_, n_v_);
  VectorXd& tau_h = *tau_h_ptr;
  tau_h = VectorXd::Zero(n_v_);
  ModelInstanceIndexToMatrixMap M_u_dict;
  if (sim_params_.is_quasi_dynamic) {
    M_u_dict = CalcScaledMassMatrix(h, unactuated_mass_scale);
  }

  for (const auto& model : models_unactuated_) {
    if (is_model_fixed(model)) continue;
    const auto& idx_v = velocity_indices_.at(model);
    const auto n_v_i = idx_v.size();
    const VectorXd& tau_ext = tau_ext_dict.at(model);

    for (int i = 0; i < tau_ext.size(); i++) {
      tau_h(idx_v[i]) = tau_ext(i) * h;
    }

    if (sim_params_.is_quasi_dynamic) {
      for (int i = 0; i < n_v_i; i++) {
        for (int j = 0; j < n_v_i; j++) {
          Q(idx_v[i], idx_v[j]) = M_u_dict.at(model)(i, j);
        }
      }
    }
  }

  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    VectorXd dq_a_cmd = q_a_cmd_dict.at(model) - q_dict.at(model);
    const auto& Kp = robot_stiffness_.at(model);
    VectorXd tau_a_h = Kp.array() * dq_a_cmd.array();
    tau_a_h += tau_ext_dict.at(model);
    tau_a_h *= h;

    for (int i = 0; i < tau_a_h.size(); i++) {
      tau_h(idx_v[i]) = tau_a_h(i);
    }

    for (int i = 0; i < idx_v.size(); i++) {
      int idx = idx_v[i];
      Q(idx, idx) = Kp(i) * h * h;
    }
  }
}

void AddPointPairContactInfoFromForce(
    const ContactPairInfo<double>& cpi,
    const Eigen::Ref<const Vector3d>& f_Bc_W,
    drake::multibody::ContactResults<double>* contact_results) {
  drake::geometry::PenetrationAsPointPair<double> papp;
  papp.id_A = cpi.id_A;
  papp.id_B = cpi.id_B;
  papp.p_WCa = cpi.p_WCa;
  papp.p_WCb = cpi.p_WCb;
  papp.nhat_BA_W = cpi.nhat_BA_W;
  Vector3d p_WC = (papp.p_WCa + papp.p_WCb) / 2;
  contact_results->AddContactInfo(
      drake::multibody::PointPairContactInfo<double>(
          cpi.body_A_idx, cpi.body_B_idx, f_Bc_W, p_WC, 0, 0, papp));
}

void QuasistaticSimulator::CalcContactResultsQp(
    const std::vector<ContactPairInfo<double>>& contact_info_list,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star, const int n_d,
    const double h, drake::multibody::ContactResults<double>* contact_results) {
  const auto n_c = contact_info_list.size();
  DRAKE_ASSERT(beta_star.size() == n_c * n_d);
  contact_results->Clear();
  int i_beta = 0;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& cpi = contact_info_list[i_c];

    // Compute contact force.
    Vector3d f_Ac_W;
    f_Ac_W.setZero();
    for (int i = 0; i < n_d; i++) {
      f_Ac_W +=
          (cpi.nhat_BA_W + cpi.mu * cpi.t_W.col(i)) * beta_star[i_beta + i];
    }
    f_Ac_W /= h;

    // Assemble Contact info.
    AddPointPairContactInfoFromForce(cpi, -f_Ac_W, contact_results);

    i_beta += n_d;
  }
}

void QuasistaticSimulator::CalcContactResultsSocp(
    const std::vector<ContactPairInfo<double>>& contact_info_list,
    const vector<VectorXd>& lambda_star, const double h,
    drake::multibody::ContactResults<double>* contact_results) {
  const auto n_c = contact_info_list.size();
  DRAKE_ASSERT(n_c == lambda_star.size());
  contact_results->Clear();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& cpi = contact_info_list[i_c];

    // Compute contact force.
    Vector3d f_Ac_W = cpi.nhat_BA_W * lambda_star[i_c][0] / cpi.mu;
    f_Ac_W += cpi.t_W * lambda_star[i_c].tail(2);
    f_Ac_W /= h;

    // Assemble Contact info.
    AddPointPairContactInfoFromForce(cpi, -f_Ac_W, contact_results);
  }
}

void QuasistaticSimulator::CalcContactResultsLogIcecream(
  const std::vector<ContactPairInfo<double>>& contact_info_list,
  const std::vector<Eigen::VectorXd>& lambda_star,
  const double n_v,
  const double h
) {
  const auto n_c = contact_info_list.size();
  DRAKE_ASSERT(n_c == lambda_star.size());

  contact_geom_names_A_.clear();
  contact_geom_names_B_.clear();

  generalized_fA_.resize(n_c, n_v);
  generalized_fB_.resize(n_c, n_v);
  spatial_fA_.resize(n_c, 6);
  spatial_fB_.resize(n_c, 6);
  contact_sdists_.resize(n_c);
  contact_points_A_.resize(n_c, 3);
  contact_points_B_.resize(n_c, 3);

  generalized_fA_.setZero();
  generalized_fB_.setZero();
  spatial_fA_.setZero();
  spatial_fB_.setZero();
  contact_sdists_.setZero();
  contact_points_A_.setZero();
  contact_points_B_.setZero();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& cpi = contact_info_list[i_c];

    // Compute contact force (B exerts on A)
    Vector3d f_Ac_W = cpi.nhat_BA_W * lambda_star[i_c][0] / cpi.mu;
    f_Ac_W += cpi.t_W * lambda_star[i_c].tail(2);
    f_Ac_W /= h;

    // generalized force
    Eigen::VectorXd tau_Ac = cpi.JcA.transpose() * f_Ac_W;
    Eigen::VectorXd tau_Bc = -cpi.JcB.transpose() * f_Ac_W;

    generalized_fA_.row(i_c) = tau_Ac;
    generalized_fB_.row(i_c) = tau_Bc;
    spatial_fA_.row(i_c).head(3) = f_Ac_W;
    spatial_fB_.row(i_c).head(3) = -f_Ac_W;
    contact_points_A_.row(i_c) = cpi.p_ACa;
    contact_points_B_.row(i_c) = cpi.p_BCb;
    // contact_sdists_(i_c) = cpi.sdist;

    contact_geom_names_A_.push_back(cpi.geom_name_A);
    contact_geom_names_B_.push_back(cpi.geom_name_B);
  }
}

void QuasistaticSimulator::Step(const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                                const ModelInstanceIndexToVecMap& tau_ext_dict,
                                const QuasistaticSimParameters& params) {
  problem_updated_ = true;

  const auto fm = params.forward_mode;
  const auto q_dict = GetMbpPositions();
  auto q_next_dict(q_dict);

  // TODO(yongpeng): Count time cost
  auto stepStartTime = std::chrono::steady_clock::now();
  auto nowTime = std::chrono::steady_clock::now();
  double duration_millsecond = 0.0;

  if (kPyramidModes.find(fm) != kPyramidModes.end()) {
    // Optimization coefficient matrices and vectors.
    MatrixXd Q, Jn, J;
    VectorXd tau_h, phi, phi_constraints;
    // Primal and dual solutions.
    VectorXd v_star;
#ifdef VERBOSE_TIMECOST
    auto calcProblemDataStartTime = std::chrono::steady_clock::now();
#endif
    CalcPyramidMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                        &Jn, &J, &phi, &phi_constraints);

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - calcProblemDataStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the pyramid matrices" << std::endl;
#endif

    if (fm == ForwardDynamicsMode::kQpMp) {
      VectorXd beta_star;
#ifdef VERBOSE_TIMECOST
      auto qpMpStartTime = std::chrono::steady_clock::now();
#endif
      ForwardQp(Q, tau_h, J, phi_constraints, params, &q_next_dict, &v_star,
                &beta_star);
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - qpMpStartTime).count();
      std::cout << "> qpMp forward time: " << duration_millsecond << " ms" << std::endl;
      qpMpStartTime = std::chrono::steady_clock::now();
#endif

      if (params.calc_contact_forces) {
        CalcContactResultsQp(cjc_->get_contact_pair_info_list(), beta_star,
                             params.nd_per_contact, params.h,
                             &contact_results_);
        contact_results_.set_plant(plant_);
      }

      // BackwardQp(Q, tau_h, Jn, J, phi_constraints, q_dict, q_next_dict, v_star,
      //            beta_star, params);
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                          params, &solver_log_pyramid_->get_H_llt());
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - qpMpStartTime).count();
      std::cout << "> pyramidQp backward time: " << duration_millsecond << " ms" << std::endl;
#endif
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMp) {
#ifdef VERBOSE_TIMECOST
      auto pyramidMpStartTime = std::chrono::steady_clock::now();
#endif
      ForwardLogPyramid(Q, tau_h, J, phi_constraints, params, &q_next_dict,
                        &v_star);
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - pyramidMpStartTime).count();
      std::cout << "> pyramidMp forward time: " << duration_millsecond << " ms" << std::endl;
      pyramidMpStartTime = std::chrono::steady_clock::now();
#endif
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                         params, nullptr);
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - pyramidMpStartTime).count();
      std::cout << "> pyramidMp backward time: " << duration_millsecond << " ms" << std::endl;
#endif
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMy) {
#ifdef VERBOSE_TIMECOST
      auto pyramidMyStartTime = std::chrono::steady_clock::now();
#endif
      ForwardLogPyramidInHouse(Q, tau_h, J, phi_constraints, params,
                               &q_next_dict, &v_star);
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - pyramidMyStartTime).count();
      std::cout << "> pyramidMy forward time: " << duration_millsecond << " ms" << std::endl;
#endif

      if (J.rows() > 0) {
        BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                          params, &solver_log_pyramid_->get_H_llt());
      }
      else {
        BackwardLogPyramid(Q, q_dict, params, false);
      }

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - stepStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the backward(calcDiff) pass" << std::endl;
#endif

      return;
    }
  }

  if (kIcecreamModes.find(fm) != kIcecreamModes.end()) {
    MatrixXd Q, Jn;
    VectorXd tau_h, phi;
    std::vector<Eigen::Matrix3Xd> J_list;
    VectorXd v_star;
    CalcIcecreamMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                         &Jn, &J_list, &phi);

    if (fm == ForwardDynamicsMode::kSocpMp) {
      std::vector<Eigen::VectorXd> lambda_star_list;
      std::vector<Eigen::VectorXd> e_list;

      ForwardSocp(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star,
                  &lambda_star_list, &e_list);

      if (params.calc_contact_forces) {
        CalcContactResultsSocp(cjc_->get_contact_pair_info_list(),
                               lambda_star_list, params.h, &contact_results_);
        contact_results_.set_plant(plant_);
      }

      BackwardSocp(Q, tau_h, J_list, e_list, phi, q_dict, q_next_dict, v_star,
                   lambda_star_list, params);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogIcecream) {
      auto beforeTime = std::chrono::steady_clock::now();
      ForwardLogIcecream(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star);
      auto afterTime = std::chrono::steady_clock::now();
      double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();

#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - stepStartTime).count();
      std::cout << "> it took " << duration_millsecond << " ms to compute the forward(calc) pass" << std::endl;
#endif

      beforeTime = afterTime;
      if (J_list.size() > 0) {
        BackwardLogIcecream(q_dict, q_next_dict, v_star, params,
                          solver_log_icecream_->get_H_llt());
      }
      else {
        BackwardLogIcecream(Q, q_dict, params, false);
      }
      
#ifdef VERBOSE_TIMECOST
      nowTime = std::chrono::steady_clock::now();
      duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - stepStartTime).count();
      std::cout << "> it took " << duration_millsecond << " ms to compute the backward(calcDiff) pass" << std::endl;
#endif

      return;
    }
  }

  std::stringstream ss;
  ss << "Forward dynamics mode " << static_cast<int>(fm)
     << " is not supported in C++.";
  throw std::logic_error(ss.str());
}

void QuasistaticSimulator::CalcPyramidMatrices(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict,
    const QuasistaticSimParameters& params, Eigen::MatrixXd* Q,
    Eigen::VectorXd* tau_h_ptr, Eigen::MatrixXd* Jn_ptr, Eigen::MatrixXd* J_ptr,
    Eigen::VectorXd* phi_ptr, Eigen::VectorXd* phi_constraints_ptr) const {
  const auto sdps = CalcCollisionPairs(params.contact_detection_tolerance, false);
  std::vector<MatrixXd> J_list;
  const auto n_d = params.nd_per_contact;
  cjc_->CalcJacobianAndPhiQp(context_plant_, sdps, n_d, phi_ptr, Jn_ptr,
                             &J_list, &Nhat_);
  MatrixXd& J = *J_ptr;
  VectorXd& phi_constraints = *phi_constraints_ptr;

  const auto n_c = J_list.size();
  const auto n_f = n_c * n_d;
  J.resize(n_f, n_v_);
  phi_constraints.resize(n_f);
  for (int i_c = 0; i_c < n_c; i_c++) {
    J(Eigen::seqN(i_c * n_d, n_d), Eigen::all) = J_list[i_c];
    phi_constraints(Eigen::seqN(i_c * n_d, n_d)).setConstant((*phi_ptr)(i_c));
  }

  CalcQAndTauH(q_dict, q_a_cmd_dict, tau_ext_dict, params.h, Q, tau_h_ptr,
               params.unactuated_mass_scale);
}

void QuasistaticSimulator::CalcIcecreamMatrices(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict,
    const QuasistaticSimParameters& params, Eigen::MatrixXd* Q,
    Eigen::VectorXd* tau_h,
    Eigen::MatrixXd* Jn_ptr,
    std::vector<Eigen::Matrix3Xd>* J_list,
    Eigen::VectorXd* phi) const {
  const auto sdps = CalcCollisionPairs(params.contact_detection_tolerance, false);
  cjc_->CalcJacobianAndPhiSocp(context_plant_, sdps, phi, Jn_ptr, J_list, &Nhat_);
  CalcQAndTauH(q_dict, q_a_cmd_dict, tau_ext_dict, params.h, Q, tau_h,
               params.unactuated_mass_scale);
}

void QuasistaticSimulator::ForwardQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    Eigen::VectorXd* beta_star_ptr) {
  auto& q_dict = *q_dict_ptr;
  const auto n_f = phi_constraints.size();
  const auto h = params.h;

  // construct and solve MathematicalProgram.
  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  prog.AddQuadraticCost(Q, -tau_h, v, true);

  const VectorXd e = phi_constraints / h;
  auto constraints = prog.AddLinearConstraint(
      -J, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v);
  auto solver = PickBestQpSolver(params);
  solver->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics QP cannot be solved.");
  }

  *v_star_ptr = mp_result_.GetSolution(v);
  if (constraints.evaluator()->num_constraints() > 0) {
    *beta_star_ptr = -mp_result_.GetDualSolution(constraints);
  } else {
    *beta_star_ptr = Eigen::VectorXd(0);
  }

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    std::vector<Eigen::VectorXd>* lambda_star_ptr,
    std::vector<Eigen::VectorXd>* e_list) {
  auto& q_dict = *q_dict_ptr;
  VectorXd& v_star = *v_star_ptr;
  const auto h = params.h;
  const auto n_c = phi.size();

  //* solve with Mosek Fusion C++ API
  /*
    You can refer to test_mosek_solver.cpp
  */

  //* solve with Drake MP agent
  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");

  prog.AddQuadraticCost(Q, -tau_h, v, true);

  std::vector<drake::solvers::Binding<drake::solvers::LorentzConeConstraint>>
      constraints;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const double mu = cjc_->get_friction_coefficient(i_c);
    e_list->emplace_back(Vector3d(phi[i_c] / mu / h, 0, 0));
    constraints.push_back(
        prog.AddLorentzConeConstraint(J_list.at(i_c), e_list->back(), v));
  }

  auto solver = PickBestSocpSolver(params);
  solver->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    std::cout << "number of contacts: " << n_c << std::endl;
    throw std::runtime_error("Quasistatic dynamics SOCP cannot be solved.");
  }

  // Primal and dual solutions.
  v_star = mp_result_.GetSolution(v);
  if (is_socp_calculating_dual(params)) {
    for (int i = 0; i < n_c; i++) {
      lambda_star_ptr->emplace_back(mp_result_.GetDualSolution(constraints[i]));
    }
  } else {
    lambda_star_ptr->clear();
  }

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr) {
  auto& q_dict = *q_dict_ptr;
  VectorXd& v_star = *v_star_ptr;
  const auto n_f = J.rows();
  const auto h = params.h;

  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  auto s = prog.NewContinuousVariables(n_f, "s");

  prog.AddQuadraticCost(Q, -tau_h, v, true);
  prog.AddLinearCost(-VectorXd::Constant(n_f, 1 / params.log_barrier_weight), 0,
                     s);

  drake::solvers::VectorXDecisionVariable v_s_i(n_v_ + 1);
  v_s_i.head(n_v_) = v;
  for (int i = 0; i < n_f; i++) {
    MatrixXd A = MatrixXd::Zero(3, n_v_ + 1);
    A.row(0).head(n_v_) = J.row(i);
    A(2, n_v_) = 1;

    Vector3d b(phi_constraints[i] / h, 1, 0);

    v_s_i[n_v_] = s[i];
    prog.AddExponentialConeConstraint(A.sparseView(), b, v_s_i);
  }
  auto solver = PickBestConeSolver(params);
  solver->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error(
        "Quasistatic dynamics Log Pyramid cannot be solved.");
  }

  v_star = mp_result_.GetSolution(v);

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardLogPyramidInHouse(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr) {

  auto forwardStartTime = std::chrono::steady_clock::now();
  auto nowTime = std::chrono::steady_clock::now();
  double duration_millsecond = 0.0;

  auto& q_dict = *q_dict_ptr;

  const auto n_c = J.rows() / 2.0;
  const auto n_v = Q.rows();

  // std::cout << "matrix J has " << n_c << " rows" << std::endl;

  if (n_c > 0) {
    solver_log_pyramid_->Solve(Q, -tau_h, -J, phi_constraints / params.h,
                             params.log_barrier_weight, params.use_free_solvers,
                             v_star_ptr);
  }
  else {
    v_star_ptr->setZero(n_v);
  }

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - forwardStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to solve the implicit dynamics" << std::endl;
#endif

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - forwardStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to update the drake mbp" << std::endl;
#endif
}

void QuasistaticSimulator::ForwardLogIcecream(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    std::vector<Eigen::VectorXd>* lambda_star_ptr) {
  auto& q_dict = *q_dict_ptr;

  const auto h = params.h;
  const auto n_c = J_list.size();
  const auto n_v = Q.rows();

  MatrixXd J(n_c * 3, n_v);
  VectorXd phi_h_mu(n_c);
  for (int i = 0; i < n_c; i++) {
    J.block(i * 3, 0, 3, n_v) = J_list.at(i);
    phi_h_mu[i] = phi[i] / h / cjc_->get_friction_coefficient(i);
  }

  bool calc_dual_flag = params.calc_contact_forces;
  if (n_c > 0) {
    solver_log_icecream_->Solve(Q, -tau_h, -J, phi_h_mu,
                              params.log_barrier_weight,
                              params.use_free_solvers, v_star_ptr);
    calc_dual_flag = true;
  }
  else {
    // TODO(yongpeng): set dq*=0 to avoid infeasible phase 1 problems
    v_star_ptr->setZero(n_v);
    calc_dual_flag = false;
  }

  if(lambda_star_ptr && calc_dual_flag) {
    CalcDualSolutionLogIcecream(*v_star_ptr, J_list, phi, params, lambda_star_ptr);
  }

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::CalcDualSolutionLogIcecream(
  const Eigen::Ref<const Eigen::VectorXd>& v_star,
  const std::vector<Eigen::Matrix3Xd>& J_list,
  const Eigen::Ref<const Eigen::VectorXd>& phi,
  const QuasistaticSimParameters& params,
  std::vector<Eigen::VectorXd>* lambda_star_ptr
) {
  std::vector<Eigen::VectorXd>& lambda_star = *lambda_star_ptr;
  lambda_star.clear();

  const auto h = params.h;
  const auto n_c = J_list.size();
  const double kappa = params.log_barrier_weight;

  for (int i_c = 0; i_c < n_c; i_c++) {
    Eigen::Matrix3Xd J_i = J_list.at(i_c);
    double phi_h_mu_i = phi[i_c] / h / cjc_->get_friction_coefficient(i_c);
    Eigen::Vector3d generalized_v_i = J_i * v_star;
    generalized_v_i[0] += phi_h_mu_i;
    double alpha_i = pow(generalized_v_i[0], 2) - generalized_v_i.tail(2).squaredNorm();

    lambda_star.emplace_back(3);
    Eigen::VectorXd& lambda_star_i = lambda_star.back();
    lambda_star_i[0] = generalized_v_i[0];
    lambda_star_i.tail(2) = -generalized_v_i.tail(2);

    lambda_star_i *= 2 / (alpha_i * kappa);
  }
}

void QuasistaticSimulator::BackwardQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& Jn,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_dict_next,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const Eigen::Ref<const Eigen::VectorXd>& lambda_star,
    const QuasistaticSimParameters& params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  if (params.gradient_mode == GradientMode::kAB) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, lambda_star,
                        0.1 * params.h, true);
    const auto& Dv_nextDe = dqp_->get_DzDe();
    const auto& Dv_nextDb = dqp_->get_DzDb();

    Dq_nextDq_ =
        CalcDfDxQp(Dv_nextDb, Dv_nextDe, Jn, v_star, q_dict_next, h, n_d);
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    return;
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, lambda_star,
                        0.1 * params.h, false);
    const auto& Dv_nextDb = dqp_->get_DzDb();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const vector<Eigen::Matrix3Xd>& J_list,
    const std::vector<Eigen::VectorXd>& e_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_dict_next,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const std::vector<Eigen::VectorXd>& lambda_star_list,
    const QuasistaticSimParameters& params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  std::vector<Eigen::MatrixXd> G_list;
  for (const auto& J : J_list) {
    G_list.emplace_back(-J);
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    dsocp_->UpdateProblem(Q, -tau_h, G_list, e_list, v_star, lambda_star_list,
                          0.1 * params.h, false);
    const auto& Dv_nextDb = dsocp_->get_DzDb();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict_next);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  if (params.gradient_mode == GradientMode::kAB) {
    dsocp_->UpdateProblem(Q, -tau_h, G_list, e_list, v_star, lambda_star_list,
                          0.1 * params.h, true);
    const auto& Dv_nextDb = dsocp_->get_DzDb();
    const auto& Dv_nextDe = dsocp_->get_DzDe();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict_next);
    Dq_nextDq_ = CalcDfDxSocp(Dv_nextDb, Dv_nextDe, J_list, v_star, q_dict,
                              q_dict_next, params.h);

    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    Eigen::LLT<Eigen::MatrixXd> const* const H_llt) {

  auto backwardStartTime = std::chrono::steady_clock::now();
  auto nowTime = std::chrono::steady_clock::now();
  double duration_millsecond = 0.0;

  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  if (H_llt) {
    CalcUnconstrainedBFromHessian(*H_llt, params, q_dict, &Dq_nextDqa_cmd_);

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - backwardStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the Dq_nextDqa_cmd from hessian" << std::endl;
#endif

    if (params.gradient_mode == GradientMode::kAB) {
      Dq_nextDq_ =
          CalcDfDxLogPyramid(v_star, q_dict, q_next_dict, params, *H_llt);
    } else {
      Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    }

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - backwardStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the Dq_nextDq" << std::endl;
#endif

    return;
  }

  Eigen::MatrixXd H(n_v_, n_v_);
  // not used, but needed by CalcGradientAndHessian.
  Eigen::VectorXd Df(n_v_);
  solver_log_pyramid_->CalcGradientAndHessian(
      Q, VectorXd::Zero(n_v_), -J, phi_constraints / params.h, v_star,
      params.log_barrier_weight, &Df, &H);

  CalcUnconstrainedBFromHessian(H.llt(), params, q_dict, &Dq_nextDqa_cmd_);
  if (params.gradient_mode == GradientMode::kAB) {
    Dq_nextDq_ =
        CalcDfDxLogPyramid(v_star, q_dict, q_next_dict, params, H.llt());
  } else {
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
  }
}

void QuasistaticSimulator::BackwardLogPyramid(
  const Eigen::Ref<const Eigen::MatrixXd>& Q,
  const ModelInstanceIndexToVecMap& q_dict,
  const QuasistaticSimParameters& params,
  bool has_contact) {

  DRAKE_ASSERT(has_contact == false);

  Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
  for (const auto& model : models_actuated_) {
    const auto& idx_q = position_indices_.at(model);
    for (int i = 0; i < idx_q.size(); i++) {
      int idx = idx_q[i];
      Dq_nextDq_(idx, idx) = 1.0;
    }
  }

  MatrixXd Dv_nextDb(n_v_, n_v_);
  Dv_nextDb = -Q.inverse();
  Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict);

  return;

}

void QuasistaticSimulator::BackwardLogIcecream(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    const Eigen::LLT<Eigen::MatrixXd>& H_llt) {

  auto backwardStartTime = std::chrono::steady_clock::now();
  auto nowTime = std::chrono::steady_clock::now();
  double duration_millsecond = 0.0;

  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  // ********************************
  // * Compute the LLT of barrier Hessian
  // * This allows calling BackwardLogIcecream
  // * directly without calling Forward at first
  // ********************************

  // if (!H_llt) {
  
  /*
  // get problem data
  MatrixXd Q = CalcDiffProblemData_.Q;
  VectorXd tau_h = CalcDiffProblemData_.tau_h;
  VectorXd phi = CalcDiffProblemData_.phi;

  const auto h = params.h;
  const auto n_c = CalcDiffProblemData_.J_list.size();
  const auto n_v = Q.rows();

  MatrixXd J(n_c * 3, n_v);
  VectorXd phi_h_mu(n_c);
  for (int i_c = 0; i_c < n_c; i_c++) {
    J.block(i_c * 3, 0, 3, n_v) = CalcDiffProblemData_.J_list.at(i_c);
    phi_h_mu[i_c] = phi[i_c] / h / cjc_->get_friction_coefficient(i_c);
  }

  // may be different from forward, because backward is a penalty problem
  VectorXd v_star_backward;

  // solve for primal optimal (v_star)
  solver_log_icecream_->Solve(Q, -tau_h, -J, phi_h_mu,
                              params.log_barrier_weight,
                              params.use_free_solvers, &v_star_backward);

  MatrixXd H(n_v, n_v);
  VectorXd Df(n_v);

  solver_log_icecream_->CalcGradientAndHessian(
    Q, -tau_h, -J, phi_h_mu,
    v_star_backward, params.log_barrier_weight,
    &Df, &H
  );
  */
  
  // }

  if (params.gradient_mode == GradientMode::kBOnly) {
    CalcUnconstrainedBFromHessian(H_llt, params, q_next_dict, &Dq_nextDqa_cmd_);
    // CalcUnconstrainedBFromHessian(H.llt(), params, q_next_dict, &Dq_nextDqa_cmd_);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  if (params.gradient_mode == GradientMode::kAB) {
    CalcUnconstrainedBFromHessian(H_llt, params, q_dict, &Dq_nextDqa_cmd_);
    // CalcUnconstrainedBFromHessian(H.llt(), params, q_dict, &Dq_nextDqa_cmd_);

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - backwardStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the Dq_nextDqa_cmd from hessian" << std::endl;
#endif
    if (params.gradient_dfdx_mode == DfDxMode::kAutoDiff) {
      Dq_nextDq_ = CalcDfDxLogIcecream(v_star, q_dict, q_next_dict, params.h,
                                        params.log_barrier_weight, H_llt);
    } else if (params.gradient_dfdx_mode == DfDxMode::kAnalyticWithFiniteDiff) {
      Dq_nextDq_ = CalcDfDxLogIcecreamAnalytic(v_star, q_dict, q_next_dict, params, H_llt);
    } else {
      throw std::logic_error("Invalid gradient_dfdx_mode.");
    }

    /*
      // TODO(yongpeng): debug the following
      Test calc DfDx for arbitrary shape here
    */
    // Eigen::MatrixXd Dq_nextDq_a_ = CalcDfDxLogIcecreamAnalytic(v_star, q_dict, q_next_dict, params, H_llt);
    // const auto cosine = (Dq_nextDq_.reshaped().transpose() * Dq_nextDq_a_.reshaped()) / (Dq_nextDq_.norm() * Dq_nextDq_a_.norm());
    // std::cout << "Dq_nextDq: ||AutoDiff||=" << Dq_nextDq_.norm() << ", ||Analytic||=" << Dq_nextDq_a_.norm() << ", cosine=" << cosine << std::endl;

#ifdef VERBOSE_TIMECOST
    nowTime = std::chrono::steady_clock::now();
    duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - backwardStartTime).count();
    std::cout << "> it took " << duration_millsecond << " ms to compute the Dq_nextDq" << std::endl;
#endif

    return;
  }

  throw std::logic_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardLogIcecream(
  const Eigen::Ref<const Eigen::MatrixXd>& Q,
  const ModelInstanceIndexToVecMap& q_dict,
  const QuasistaticSimParameters& params,
  bool has_contact) {

  DRAKE_ASSERT(has_contact == false);

  Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
  for (const auto& model : models_actuated_) {
    const auto& idx_q = position_indices_.at(model);
    for (int i = 0; i < idx_q.size(); i++) {
      int idx = idx_q[i];
      Dq_nextDq_(idx, idx) = 1.0;
    }
  }

  MatrixXd Dv_nextDb(n_v_, n_v_);
  Dv_nextDb = -Q.inverse();
  Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict);

  return;

}

void QuasistaticSimulator::CalcUnconstrainedBFromHessian(
    const Eigen::LLT<Eigen::MatrixXd>& H_llt,
    const QuasistaticSimParameters& params,
    const ModelInstanceIndexToVecMap& q_dict, Eigen::MatrixXd* B_ptr) const {
  MatrixXd Dv_nextDb(n_v_, n_v_);
  Dv_nextDb.setIdentity();
  Dv_nextDb *= -params.log_barrier_weight;
  H_llt.solveInPlace(Dv_nextDb);
  *B_ptr = CalcDfDu(Dv_nextDb, params.h, q_dict);
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetVdictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& v) const {
  DRAKE_THROW_UNLESS(v.size() == n_v_);
  std::unordered_map<ModelInstanceIndex, VectorXd> v_dict;

  for (const auto& model : models_all_) {
    const auto& idx_v = velocity_indices_.at(model);
    v_dict[model] = v(idx_v);
  }
  return v_dict;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& q) const {
  DRAKE_THROW_UNLESS(q.size() == n_q_);
  ModelInstanceIndexToVecMap q_dict;

  for (const auto& model : models_all_) {
    const auto& idx_q = position_indices_.at(model);
    q_dict[model] = q(idx_q);
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetQVecFromDict(
    const ModelInstanceIndexToVecMap& q_dict) const {
  VectorXd q(n_q_);
  for (const auto& model : models_all_) {
    q(position_indices_.at(model)) = q_dict.at(model);
  }
  return q;
}

Eigen::VectorXd QuasistaticSimulator::GetQaCmdVecFromDict(
    const ModelInstanceIndexToVecMap& q_a_cmd_dict) const {
  int i_start = 0;
  VectorXd q_a_cmd(n_v_a_);
  for (const auto& model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd.segment(i_start, n_v_i) = q_a_cmd_dict.at(model);
    i_start += n_v_i;
  }

  return q_a_cmd;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQaCmdDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& q_a_cmd) const {
  ModelInstanceIndexToVecMap q_a_cmd_dict;
  int i_start = 0;
  for (const auto& model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd_dict[model] = q_a_cmd.segment(i_start, n_v_i);
    i_start += n_v_i;
  }

  return q_a_cmd_dict;
}

void QuasistaticSimulator::AddTauExtFromVec(
  const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
  ModelInstanceIndexToVecMap* tau_ext_dict
) const {
  DRAKE_THROW_UNLESS(tau_ext.size() == n_v_);
  // apply external torque to unactuated models
  for (const auto& model : models_unactuated_) {
    if (tau_ext_dict->find(model) == tau_ext_dict->end()) {
      const auto n_v_i = plant_->num_velocities(model);
      (*tau_ext_dict)[model] = VectorXd::Zero(n_v_i);
    }
    const auto& idx_v = velocity_indices_.at(model);
    (*tau_ext_dict)[model] += tau_ext(idx_v);
  }
}

Eigen::VectorXi QuasistaticSimulator::GetModelsIndicesIntoQ(
    const std::set<drake::multibody::ModelInstanceIndex>& models) const {
  const int n = std::accumulate(models.begin(), models.end(), 0,
                                [&position_indices = position_indices_](
                                    int n, const ModelInstanceIndex& model) {
                                  return n + position_indices.at(model).size();
                                });
  Eigen::VectorXi models_indices(n);
  Eigen::Index i_start = 0;
  for (const auto& model : models) {
    const auto& indices = position_indices_.at(model);
    const auto n_model = indices.size();
    models_indices(Eigen::seqN(i_start, n_model)) =
        Eigen::Map<const Eigen::VectorXi>(indices.data(), n_model);
    i_start += n_model;
  }
  return models_indices;
}

Eigen::VectorXi QuasistaticSimulator::GetQaIndicesIntoQ() const {
  return GetModelsIndicesIntoQ(models_actuated_);
}

Eigen::VectorXi QuasistaticSimulator::GetQuIndicesIntoQ() const {
  return GetModelsIndicesIntoQ(models_unactuated_);
}

void QuasistaticSimulator::Step(
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict) {
  Step(q_a_cmd_dict, tau_ext_dict, sim_params_);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDu(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb, const double h,
    const ModelInstanceIndexToVecMap& q_dict) const {
  MatrixXd DbDqa_cmd = MatrixXd::Zero(n_v_, n_v_a_);
  int j_start = 0;
  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    const int n_v_i = idx_v.size();
    const auto& Kq_i = robot_stiffness_.at(model);

    for (int k = 0; k < n_v_i; k++) {
      int i = idx_v[k];
      int j = j_start + k;
      DbDqa_cmd(i, j) = -h * Kq_i[k];
    }

    j_start += n_v_i;
  }

  const MatrixXd Dv_nextDqa_cmd = Dv_nextDb * DbDqa_cmd;

  // 2D systems.
  if (n_v_ == n_q_) {
    return h * Dv_nextDqa_cmd;
  }

  // 3D systems.
  return h * ConvertRowVToQdot(q_dict, Dv_nextDqa_cmd);
}

/*
 * Used to provide the optional input to CalcDGactiveDqFromJActiveList, when
 * the dynamics is a QP.
 */
std::vector<std::vector<int>> CalcRelativeActiveIndicesList(
    const std::vector<int>& lambda_star_active_indices, const int n_d) {
  int i_c_current = -1;
  std::vector<std::vector<int>> relative_active_indices_list;
  for (const auto i : lambda_star_active_indices) {
    const int i_c = i / n_d;
    if (i_c_current != i_c) {
      relative_active_indices_list.emplace_back();
      i_c_current = i_c;
    }
    relative_active_indices_list.back().push_back(i % n_d);
  }
  return relative_active_indices_list;
}

/*
 * J_active_ad_list is a list of (n_d, n_v) matrices.
 * For QP contact dynamics, n_d is number of extreme rays in the polyhedral
 *  friction cone.
 * For SOCP contact dynamics, n_d is 3.
 *
 * For QP dynamics, for contact Jacobian in J_active_ad_list, it is possible
 *  that only some of its n_d rows are active. This is when the optional
 *  relative_active_indices_list becomes useful: for J_active_ad_list[i],
 *  relative_active_indices_list[i] stores the indices of its active rows,
 *  ranging from 0 to n_d - 1.
 *
 * This function returns DG_active_vecDq, a matrix of shape
 *  (n_lambda_active * n_v, n_q).
 *
 * NOTE THAT G_active = -J_active!!!
 */
template <Eigen::Index M>
MatrixXd CalcDGactiveDqFromJActiveList(
    const std::vector<Eigen::Matrix<AutoDiffXd, M, -1>>& J_active_ad_list,
    const std::vector<std::vector<int>>* relative_active_indices_list) {
  const int m = J_active_ad_list.front().rows();
  const auto n_v = J_active_ad_list.front().cols();
  const auto n_q = J_active_ad_list.front()(0, 0).derivatives().size();
  int n_la;  // Total number of active rows in G_active.
  if (relative_active_indices_list) {
    n_la = std::accumulate(relative_active_indices_list->begin(),
                           relative_active_indices_list->end(), 0,
                           [](int a, const std::vector<int>& b) {
                             return a + b.size();
                           });
  } else {
    n_la = J_active_ad_list.size() * m;
  }

  std::vector<int> row_indices_all(m);
  std::iota(row_indices_all.begin(), row_indices_all.end(), 0);

  MatrixXd DvecG_activeDq(n_la * n_v, n_q);
  for (int i_q = 0; i_q < n_q; i_q++) {
    // Fill one column of DvecG_activeDq.
    int i_G = 0;  // row index into DvecG_activeDq.
    for (int j = 0; j < n_v; j++) {
      for (int i_c = 0; i_c < J_active_ad_list.size(); i_c++) {
        const auto& J_i = J_active_ad_list[i_c];

        // Find indices of active rows of the current J_i.
        const std::vector<int>* row_indices{nullptr};
        if (relative_active_indices_list) {
          row_indices = &(relative_active_indices_list->at(i_c));
        } else {
          row_indices = &row_indices_all;
        }

        for (const auto& i : *row_indices) {
          DvecG_activeDq(i_G, i_q) = -J_i(i, j).derivatives()[i_q];
          i_G += 1;
        }
      }
    }
  }
  return DvecG_activeDq;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
    const Eigen::Ref<const Eigen::MatrixXd>& Jn,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict, const double h,
    const size_t n_d) const {
  MatrixXd Dv_nextDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(Dv_nextDb, h, &Dv_nextDq);

  /*----------------------------------------------------------------*/
  // Compute Dv_nextDvecG from the KKT conditions of the QP.
  const auto& [Dv_nextDvecG_active, lambda_star_active_indices] =
      dqp_->get_DzDvecG_active();
  const auto n_la = lambda_star_active_indices.size();

  /*----------------------------------------------------------------*/
  // e := phi_constraints / h.
  MatrixXd De_active_Dq(n_la, n_q_);
  std::vector<int> active_contact_indices;
  for (int i = 0; i < n_la; i++) {
    const size_t i_c = lambda_star_active_indices[i] / n_d;
    De_active_Dq.row(i) = ConvertColVToQdot(q_dict, Jn.row(i_c)) / h;

    if (active_contact_indices.empty() ||
        active_contact_indices.back() != i_c) {
      active_contact_indices.push_back(i_c);
    }
  }

  Dv_nextDq += Dv_nextDe(Eigen::all, lambda_star_active_indices) * De_active_Dq;

  /*----------------------------------------------------------------*/
  if (!lambda_star_active_indices.empty()) {
    // This is skipped if there is no contact.
    // Compute DvecGDq using Autodiff through MBP.
    const auto q = GetQVecFromDict(q_dict);
    const auto q_ad = InitializeAutoDiff(q);
    UpdateMbpAdPositions(q_ad);
    const auto sdps_active =
        CalcSignedDistancePairsFromCollisionPairs(&active_contact_indices);
    // TODO(pang): only J_active_ad is used. Think of a less wasteful interface?
    std::vector<MatrixX<AutoDiffXd>> J_active_ad_list;
    MatrixX<AutoDiffXd> Jn_active_ad;
    VectorX<AutoDiffXd> phi_active_ad;
    cjc_ad_->CalcJacobianAndPhiQp(context_plant_ad_, sdps_active, n_d,
                                  &phi_active_ad, &Jn_active_ad,
                                  &J_active_ad_list);

    const auto relative_active_indices_list =
        CalcRelativeActiveIndicesList(lambda_star_active_indices, n_d);
    const auto DvecG_activeDq = CalcDGactiveDqFromJActiveList<-1>(
        J_active_ad_list, &relative_active_indices_list);

    Dv_nextDq += Dv_nextDvecG_active * DvecG_activeDq;
  }

  return CalcDq_nextDqFromDv_nextDq(Dv_nextDq, q_dict, v_star, h);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict, double h) const {
  static constexpr int m{3};  // Dimension of 2nd order cones.

  MatrixXd Dv_nextDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(Dv_nextDb, h, &Dv_nextDq);

  const auto& [Dv_nextDvecG_active, lambda_star_active_indices] =
      dsocp_->get_DzDvecG_active();
  const auto n_la = lambda_star_active_indices.size();

  /*-------------------------------------------------------------------*/
  // e[i] := phi[i] / h / mu[i].
  MatrixXd De_active_Dq(n_la, n_q_);
  // The vector e, as defined in the SocpDerivatives class, e is an (m * n_l)
  // vector, where n_l == J_list.size(). But we know that for every m-length
  // segment of e, only the first element is a function of q.
  vector<int> active_indices_into_e;
  for (int i = 0; i < n_la; i++) {
    const int i_c = lambda_star_active_indices[i];
    De_active_Dq.row(i) = ConvertColVToQdot(q_dict, J_list[i_c].row(0)) / h;
    active_indices_into_e.push_back(i_c * m);
  }

  Dv_nextDq += Dv_nextDe(Eigen::all, active_indices_into_e) * De_active_Dq;
  /*----------------------------------------------------------------*/
  if (!lambda_star_active_indices.empty()) {
    const auto q = GetQVecFromDict(q_next_dict);
    UpdateMbpAdPositions(InitializeAutoDiff(q));
    const auto sdps_active =
        CalcSignedDistancePairsFromCollisionPairs(&lambda_star_active_indices);
    // TODO(pang): only J_active_ad is used. Think of a less wasteful interface?
    std::vector<Matrix3X<AutoDiffXd>> J_active_ad_list;
    VectorX<AutoDiffXd> phi_active_ad;
    MatrixX<AutoDiffXd> Jn_active_ad;
    cjc_ad_->CalcJacobianAndPhiSocp(context_plant_ad_, sdps_active,
                                    &phi_active_ad, &Jn_active_ad, &J_active_ad_list, nullptr);

    const auto DvecG_activeDq =
        CalcDGactiveDqFromJActiveList<3>(J_active_ad_list, nullptr);

    Dv_nextDq += Dv_nextDvecG_active * DvecG_activeDq;
  }

  return CalcDq_nextDqFromDv_nextDq(Dv_nextDq, q_next_dict, v_star, h);
}

void QuasistaticSimulator::CalcDv_nextDbDq(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb, const double h,
    drake::EigenPtr<Eigen::MatrixXd> Dv_nextDq_ptr) const {
  MatrixXd DbDq = MatrixXd::Zero(n_v_, n_q_);
  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    const auto& idx_q = position_indices_.at(model);
    const auto& Kq_i = robot_stiffness_.at(model);
    // TODO(pang): This needs double check!
    DbDq(idx_q, idx_v).diagonal() = h * Kq_i;
  }
  *Dv_nextDq_ptr += Dv_nextDb * DbDq;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDq_nextDqFromDv_nextDq(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDq,
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star, const double h) const {
  if (n_v_ == n_q_) {
    return MatrixXd::Identity(n_v_, n_v_) + h * Dv_nextDq;
  }

  MatrixXd A = ConvertRowVToQdot(q_dict, Dv_nextDq);
  AddDNDq2A(v_star, &A);
  A *= h;
  A.diagonal() += VectorXd::Ones(n_q_);
  return A;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxLogIcecream(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict, const double h,
    const double kappa, const Eigen::LLT<MatrixXd>& H_llt) const {
  MatrixXd DyDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(MatrixXd::Identity(n_v_, n_v_) * kappa, h, &DyDq);

  /*----------------------------------------------------------------*/
  const auto q = GetQVecFromDict(q_dict);
  const auto q_ad = InitializeAutoDiff(q);
  UpdateMbpAdPositions(q_ad);
  const auto sdps = CalcSignedDistancePairsFromCollisionPairs();
  const auto n_c = sdps.size();
  std::vector<Matrix3X<AutoDiffXd>> J_ad_list;
  VectorX<AutoDiffXd> phi_ad;
  MatrixX<AutoDiffXd> Jn_list;
  cjc_ad_->CalcJacobianAndPhiSocp(context_plant_ad_, sdps, &phi_ad, &Jn_list, &J_ad_list, nullptr);

  //  cout << "DyDq\n" << DyDq << endl;

  VectorX<AutoDiffXd> y(n_v_);
  y.setZero();
  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& J = J_ad_list[i_c];
    Vector3<AutoDiffXd> w = J * v_star;
    w[0] += phi_ad[i_c] / h / cjc_->get_friction_coefficient(i_c);
    AutoDiffXd d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    VectorX<AutoDiffXd> thing_to_add =
        2 * J.transpose() * Vector3<AutoDiffXd>(w[0] / d, -w[1] / d, -w[2] / d);
    y += thing_to_add;

    //    const auto A_to_add = drake::math::ExtractGradient(thing_to_add);
    //    cout << i_c;
    //    cout << " d " << d;
    //    cout << " phi " << phi_ad[i_c];
    //    cout << " A_max " << A_to_add.array().abs().maxCoeff();
    //    cout << "\n";
  }

  DyDq += drake::math::ExtractGradient(y);
  DyDq *= -1;
  H_llt.solveInPlace(DyDq);  // Now it becomes Dv_nextDq.

  return CalcDq_nextDqFromDv_nextDq(DyDq, q_next_dict, v_star, h);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxLogPyramid(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const QuasistaticSimParameters& params,
    const Eigen::LLT<Eigen::MatrixXd>& H_llt) const {

  auto DfDxStartTime = std::chrono::steady_clock::now();
  auto nowTime = std::chrono::steady_clock::now();
  double duration_millsecond = 0.0;

  const auto kappa = params.log_barrier_weight;
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  MatrixXd DyDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(MatrixXd::Identity(n_v_, n_v_) * kappa, h, &DyDq);

  /*----------------------------------------------------------------*/
  const auto q = GetQVecFromDict(q_dict);
  UpdateMbpAdPositions(InitializeAutoDiff(q));

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - DfDxStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to initialize the AutoDiff" << std::endl;
#endif

  const auto sdps = CalcSignedDistancePairsFromCollisionPairs();
  std::vector<MatrixX<AutoDiffXd>> J_ad_list;
  MatrixX<AutoDiffXd> Jn_ad;
  VectorX<AutoDiffXd> phi_ad;
  cjc_ad_->CalcJacobianAndPhiQp(context_plant_ad_, sdps, n_d, &phi_ad, &Jn_ad,
                                &J_ad_list);

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - DfDxStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to do the collision check" << std::endl;
#endif

  const auto n_c = sdps.size();
  VectorX<AutoDiffXd> y(n_v_);
  y.setZero();
  for (int i = 0; i < n_c; i++) {
    for (int j = 0; j < n_d; j++) {
      const Eigen::RowVectorX<AutoDiffXd>& J_ij = J_ad_list[i].row(j);
      const auto d = J_ij.dot(v_star) + phi_ad[i] / h;
      y -= J_ij.transpose() / d;
    }
  }

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - DfDxStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to prepare the AutoDiff" << std::endl;
#endif

  DyDq += drake::math::ExtractGradient(y);
  DyDq *= -1;
  H_llt.solveInPlace(DyDq);  // Now it becomes Dv_nextDq.

#ifdef VERBOSE_TIMECOST
  nowTime = std::chrono::steady_clock::now();
  duration_millsecond = std::chrono::duration<double, std::milli>(nowTime - DfDxStartTime).count();
  std::cout << "> it took " << duration_millsecond << " ms to solve the AutoDiff" << std::endl;
#endif

  return CalcDq_nextDqFromDv_nextDq(DyDq, q_next_dict, v_star, h);
}

void QuasistaticSimulator::GetGeneralizedForceFromExternalSpatialForce(
    const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>&
        easf,
    ModelInstanceIndexToVecMap* tau_ext) const {
  // TODO(pang): actually process externally applied spatial force.
  for (const auto& model : models_actuated_) {
    (*tau_ext)[model] = Eigen::VectorXd::Zero(plant_->num_velocities(model));
  }
}

void QuasistaticSimulator::CalcGravityForUnactuatedModels(
    ModelInstanceIndexToVecMap* tau_ext) const {
  const auto gravity_all =
      plant_->CalcGravityGeneralizedForces(*context_plant_);

  for (const auto& model : models_unactuated_) {
    if (is_model_fixed(model)) continue;
    const auto& indices = velocity_indices_.at(model);
    const int n_v_i = indices.size();
    (*tau_ext)[model] = VectorXd(n_v_i);
    for (int i = 0; i < n_v_i; i++) {
      (*tau_ext)[model][i] = gravity_all[indices[i]];
    }
  }
}

ModelInstanceIndexToVecMap QuasistaticSimulator::CalcTauExt(
  const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>& easf_list

) const {
  ModelInstanceIndexToVecMap tau_ext;
  GetGeneralizedForceFromExternalSpatialForce(easf_list, &tau_ext);
  CalcGravityForUnactuatedModels(&tau_ext);
  return tau_ext;
}

ModelInstanceNameToIndexMap
QuasistaticSimulator::GetModelInstanceNameToIndexMap() const {
  ModelInstanceNameToIndexMap name_to_index_map;
  for (const auto& model : models_all_) {
    name_to_index_map[plant_->GetModelInstanceName(model)] = model;
  }
  return name_to_index_map;
}

inline Eigen::Matrix3d MakeSkewSymmetricFromVec(
    const Eigen::Ref<const Vector3d>& v) {
  return Eigen::Matrix3d{{0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0}};
}

Eigen::Matrix<double, 4, 3> QuasistaticSimulator::CalcNW2Qdot(
    const Eigen::Ref<const Eigen::Vector4d>& Q) {
  Eigen::Matrix<double, 4, 3> E;
  //  E.row(0) << -Q[1], -Q[2], -Q[3];
  //  E.row(1) << Q[0], Q[3], -Q[2];
  //  E.row(2) << -Q[3], Q[0], Q[1];
  //  E.row(3) << Q[2], -Q[1], Q[0];
  E.row(0) = -Q.tail(3);
  E.bottomRows(3) = -MakeSkewSymmetricFromVec(Q.tail(3));
  E.bottomRows(3).diagonal().setConstant(Q[0]);
  E *= 0.5;
  return E;
}

Eigen::Matrix<double, 3, 4> QuasistaticSimulator::CalcNQdot2W(
    const Eigen::Ref<const Eigen::Vector4d>& Q) {
  Eigen::Matrix<double, 3, 4> E;
  E.col(0) = -Q.tail(3);
  E.rightCols(3) = MakeSkewSymmetricFromVec(Q.tail(3));
  E.rightCols(3).diagonal().setConstant(Q[0]);
  E *= 2;
  return E;
}

Eigen::Map<const Eigen::VectorXi> QuasistaticSimulator::GetIndicesAsVec(
    const drake::multibody::ModelInstanceIndex& model,
    ModelIndicesMode mode) const {
  std::vector<int> const* indices{nullptr};
  if (mode == ModelIndicesMode::kQ) {
    indices = &position_indices_.at(model);
  } else {
    indices = &velocity_indices_.at(model);
  }

  return {indices->data(), static_cast<Eigen::Index>(indices->size())};
}

Eigen::MatrixXd QuasistaticSimulator::ConvertRowVToQdot(
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::MatrixXd>& M_v) const {
  MatrixXd M_qdot(n_q_, M_v.cols());
  for (const auto& model : models_all_) {
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);

    if (is_model_floating(model)) {
      // If q contains a quaternion.
      const Eigen::Vector4d& Q_WB = q_dict.at(model).head(4);

      // Rotation.
      M_qdot(idx_q_model.head(4), Eigen::all) =
          CalcNW2Qdot(Q_WB) * M_v(idx_v_model.head(3), Eigen::all);
      // Translation.
      M_qdot(idx_q_model.tail(3), Eigen::all) =
          M_v(idx_v_model.tail(3), Eigen::all);
    } else {
      M_qdot(idx_q_model, Eigen::all) = M_v(idx_v_model, Eigen::all);
    }
  }

  return M_qdot;
}

Eigen::MatrixXd QuasistaticSimulator::ConvertColVToQdot(
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::MatrixXd>& M_v) const {
  MatrixXd M_qdot(M_v.rows(), n_q_);
  for (const auto& model : models_all_) {
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);

    if (is_model_floating(model)) {
      const Eigen::Vector4d& Q_WB = q_dict.at(model).head(4);

      // Rotation.
      M_qdot(Eigen::all, idx_q_model.head(4)) =
          M_v(Eigen::all, idx_v_model.head(3)) * CalcNQdot2W(Q_WB);

      // Translation.
      M_qdot(Eigen::all, idx_q_model.tail(3)) =
          M_v(Eigen::all, idx_v_model.tail(3));
    } else {
      M_qdot(Eigen::all, idx_q_model) = M_v(Eigen::all, idx_v_model);
    }
  }

  return M_qdot;
}

void QuasistaticSimulator::AddDNDq2A(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    drake::EigenPtr<Eigen::MatrixXd> A_ptr) const {
  Eigen::Matrix4d E;
  for (const auto& model : models_unactuated_) {
    if (!is_model_floating(model) || is_model_fixed(model)) {
      continue;
    }
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);
    const Vector3d& w = v_star(idx_v_model.head(3));  // angular velocity.

    E.row(0) << 0, -w[0], -w[1], -w[2];
    E.row(1) << w[0], 0, -w[2], w[1];
    E.row(2) << w[1], w[2], 0, -w[0];
    E.row(3) << w[2], -w[1], w[0], 0;

    (*A_ptr)(idx_q_model.head(4), idx_q_model.head(4)) += E;
  }
}

std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>>
QuasistaticSimulator::CalcSignedDistancePairsFromCollisionPairs(
    std::vector<int> const* active_contact_indices) const {
  std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>> sdps_ad;
  std::vector<int> all_indices;
  if (active_contact_indices == nullptr) {
    all_indices.resize(ordered_collision_pairs_.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    active_contact_indices = &all_indices;
  }

  for (const auto i : *active_contact_indices) {
    const auto& collision_pair = ordered_collision_pairs_[i];
    sdps_ad.push_back(query_object_ad_->ComputeSignedDistancePairClosestPoints(
        collision_pair.first, collision_pair.second));
  }
  return sdps_ad;
}

std::vector<drake::geometry::SignedDistancePair<double>>
QuasistaticSimulator::CalcCollisionPairs(
  double contact_detection_tolerance,
  bool in_order
) const {
  auto sdps = query_object_->ComputeSignedDistancePairwiseClosestPoints(
      contact_detection_tolerance);\
  GetFingerGeomidsFromSdps(sdps, &collision_pairs_);
  // collision_pairs_.clear();

  // // Save collision pairs, which may later be used in gradient computation by
  // // the AutoDiff MBP.
  // for (const auto& sdp : sdps) {
  //   collision_pairs_.emplace_back(sdp.id_A, sdp.id_B);
  // }

  if (in_order) {
    std::vector<drake::geometry::SignedDistancePair<double>> sdps_sorted;
    std::vector<CollisionPair> found_collision_pairs;
    GetFingerGeomidsFromSdps(sdps, &found_collision_pairs);
    for (const auto& pair : ordered_collision_pairs_) {
      auto it = std::find(found_collision_pairs.begin(), found_collision_pairs.end(), pair);
      if (it != found_collision_pairs.end()) {
        int index = std::distance(found_collision_pairs.begin(), it);
        sdps_sorted.push_back(sdps[index]);
      }
    }
    return sdps_sorted;
  } else {
    GetFingerGeomidsFromSdps(sdps, &ordered_collision_pairs_);
    return sdps;
  }
}

ModelInstanceIndexToMatrixMap QuasistaticSimulator::CalcScaledMassMatrix(
    double h, double unactuated_mass_scale) const {
  MatrixXd M(n_v_, n_v_);
  plant_->CalcMassMatrix(*context_plant_, &M);

  ModelInstanceIndexToMatrixMap M_u_dict;
  double counter = 0;
  for (const auto& model : models_unactuated_) {
    if (is_model_fixed(model)) continue;
    const auto& idx_v_model = velocity_indices_.at(model);
    if (idx_v_model.size() == 0) continue;
    M_u_dict[model] = M(idx_v_model, idx_v_model);
  }

  if (unactuated_mass_scale == 0 || std::isnan(unactuated_mass_scale)) {
    return M_u_dict;
  }

  std::unordered_map<drake::multibody::ModelInstanceIndex, double>
      max_eigen_value_M_u;
  for (const auto& model : models_unactuated_) {
    if (is_model_fixed(model)) continue;
    // TODO(pang): use the eigen value instead of maximum
    max_eigen_value_M_u[model] = M_u_dict.at(model).diagonal().maxCoeff();
  }

  const double min_K_a_h2 = min_K_a_ * h * h;

  for (auto& [model, M_u] : M_u_dict) {
    auto scale =
        min_K_a_h2 / max_eigen_value_M_u[model] / unactuated_mass_scale;
    M_u *= scale;
  }

  return M_u_dict;
}

void QuasistaticSimulator::UpdateQdictFromV(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr) const {
  const auto v_dict = GetVdictFromVec(v_star);
  auto& q_dict = *q_dict_ptr;
  const auto h = params.h;

  std::unordered_map<ModelInstanceIndex, VectorXd> dq_dict;
  for (const auto& model : models_all_) {
    const auto& idx_v = velocity_indices_.at(model);
    const auto n_q_i = plant_->num_positions(model);

    if (is_3d_floating_.at(model)) {
      // Positions of the model contains a quaternion. Conversion from
      // angular velocities to quaternion dot is necessary.
      const auto& q_u = q_dict[model];
      const Eigen::Vector4d Q(q_u.head(4));

      VectorXd dq_u(7);
      const auto& v_u = v_dict.at(model);
      dq_u.head(4) = CalcNW2Qdot(Q) * v_u.head(3) * h;
      dq_u.tail(3) = v_u.tail(3) * h;

      dq_dict[model] = dq_u;
    } else {
      dq_dict[model] = v_dict.at(model) * h;
    }
  }

  // TODO(pang): not updating unactuated object poses can lead to penetration at
  //  the next time step. A better solution is needed.
  if (params.unactuated_mass_scale > 0 ||
      std::isnan(params.unactuated_mass_scale)) {
    for (const auto& model : models_all_) {
      auto& q_model = q_dict[model];
      q_model += dq_dict[model];

      if (is_3d_floating_.at(model)) {
        // Normalize quaternion.
        q_model.head(4).normalize();
      }
    }
  } else {
    // un-actuated objects remain fixed.
    for (const auto& model : models_actuated_) {
      auto& q_model = q_dict[model];
      q_model += dq_dict[model];
    }
  }
}

VectorXd QuasistaticSimulator::CalcDynamics(
    QuasistaticSimulator* q_sim, const Eigen::Ref<const VectorXd>& q,
    const Eigen::Ref<const VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u);
  q_sim->Step(q_a_cmd_dict, tau_ext_dict, sim_params);
  return q_sim->GetMbpPositionsAsVec();
}

VectorXd QuasistaticSimulator::CalcDynamics(
    const Eigen::Ref<const VectorXd>& q, const Eigen::Ref<const VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  return CalcDynamics(this, q, u, sim_params);
}

// TODO(yongpeng): added the splition of forward and backward dynamics
// -----------------------------------------------------------------------------
void QuasistaticSimulator::Calc(const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                                const ModelInstanceIndexToVecMap& tau_ext_dict,
                                const QuasistaticSimParameters& params) {
  problem_updated_ = true;

  const auto fm = params.forward_mode;
  const auto q_dict = GetMbpPositions();
  auto q_next_dict(q_dict);

  MatrixXd Q;
  VectorXd tau_h, phi, v_star;

  if (kPyramidModes.find(fm) != kPyramidModes.end()) {
    // Optimization coefficient matrices and vectors.
    MatrixXd Jn, J;
    VectorXd phi_constraints;
    // Primal and dual solutions.
    CalcPyramidMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                        &Jn, &J, &phi, &phi_constraints);

    if (fm == ForwardDynamicsMode::kQpMp) {
      VectorXd beta_star;
      ForwardQp(Q, tau_h, J, phi_constraints, params, &q_next_dict, &v_star,
                &beta_star);

      if (params.calc_contact_forces) {
        CalcContactResultsQp(cjc_->get_contact_pair_info_list(), beta_star,
                             params.nd_per_contact, params.h,
                             &contact_results_);
        contact_results_.set_plant(plant_);
      }
      // beta_star
      CalcDiffProblemData_.beta_star = beta_star;
    }

    else if (fm == ForwardDynamicsMode::kLogPyramidMp) {
      ForwardLogPyramid(Q, tau_h, J, phi_constraints, params, &q_next_dict,
                        &v_star);
    }

    else if (fm == ForwardDynamicsMode::kLogPyramidMy) {
      ForwardLogPyramidInHouse(Q, tau_h, J, phi_constraints, params,
                               &q_next_dict, &v_star);
    }

    // save problem data for calcDiff
    CalcDiffProblemData_.Jn = Jn;
    CalcDiffProblemData_.J = J;
    CalcDiffProblemData_.phi_constraints = phi_constraints;
  }

  else if (kIcecreamModes.find(fm) != kIcecreamModes.end()) {
    
    MatrixXd Jn;
    std::vector<Eigen::Matrix3Xd> J_list;

    // auto startTime = std::chrono::steady_clock::now();
    // auto endTime = std::chrono::steady_clock::now();
    // double duration_millsecond = 0.0;

    CalcIcecreamMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                         &Jn, &J_list, &phi);

    // endTime = std::chrono::steady_clock::now();
    // duration_millsecond = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    // std::cout << "> CalcPyramidMatrices time: " << duration_millsecond << " ms" << std::endl;

    if (fm == ForwardDynamicsMode::kSocpMp) {
      std::vector<Eigen::VectorXd> lambda_star_list;
      std::vector<Eigen::VectorXd> e_list;

      ForwardSocp(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star,
                  &lambda_star_list, &e_list);
      if (params.calc_contact_forces) {
        CalcContactResultsSocp(cjc_->get_contact_pair_info_list(),
                               lambda_star_list, params.h, &contact_results_);
        contact_results_.set_plant(plant_);
      }

      // save e_list
      CalcDiffProblemData_.e_list.clear();
      for (auto e_i : e_list) {
        CalcDiffProblemData_.e_list.push_back(e_i);
      }

      // save lambda_star_list
      CalcDiffProblemData_.lambda_star_list.clear();
      for (auto lambda_star_i : lambda_star_list) {
        CalcDiffProblemData_.lambda_star_list.push_back(lambda_star_i);
      }
    }

    else if (fm == ForwardDynamicsMode::kLogIcecream) {
      // startTime = std::chrono::steady_clock::now();
      std::vector<Eigen::VectorXd> lambda_star_list;

      ForwardLogIcecream(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star, &lambda_star_list);

      const auto n_v = Q.rows();
      const auto n_c = J_list.size();
      if (params.calc_contact_forces) {
        CalcContactResultsLogIcecream(
          cjc_->get_contact_pair_info_list(), lambda_star_list, n_v, params.h
        );
        contact_sdists_ = phi;
      }
      else {
        generalized_fA_.resize(0, n_v);
        generalized_fB_.resize(0, n_v);
        spatial_fA_.resize(0, 6);
        spatial_fB_.resize(0, 6);
        contact_points_A_.resize(0, 3);
        contact_points_B_.resize(0, 3);
        contact_sdists_.resize(0);
        contact_geom_names_A_.clear();
        contact_geom_names_B_.clear();
      }
      // endTime = std::chrono::steady_clock::now();
      // duration_millsecond = std::chrono::duration<double, std::milli>(endTime - startTime).count();
      // std::cout << "> ForwardLogIcecream time: " << duration_millsecond << " ms" << std::endl;
    }

    // save J_list and Jn_list
    CalcDiffProblemData_.Jn = Jn;
    CalcDiffProblemData_.J_list.clear();
    for (auto J_i : J_list){
      CalcDiffProblemData_.J_list.push_back(J_i);
    }
  }

  // save problem data for calcDiff
  CalcDiffProblemData_.Q = Q;
  CalcDiffProblemData_.tau_h = tau_h;
  CalcDiffProblemData_.phi = phi;
  CalcDiffProblemData_.v_star = v_star;
  CalcDiffProblemData_.q_dict = q_dict;
  CalcDiffProblemData_.q_next_dict = q_next_dict;

}

void QuasistaticSimulator::CalcDiff(const QuasistaticSimParameters& params) {
  if (!problem_updated_) {
    std::stringstream ss;
    ss << "You shold call Calc first!";
    throw std::logic_error(ss.str());
  }
  problem_updated_ = false;

  const auto fm = params.forward_mode;
  auto q_dict(CalcDiffProblemData_.q_dict);
  auto q_next_dict(CalcDiffProblemData_.q_next_dict);

  MatrixXd Q(CalcDiffProblemData_.Q);
  VectorXd tau_h(CalcDiffProblemData_.tau_h), phi(CalcDiffProblemData_.phi), v_star(CalcDiffProblemData_.v_star);

  if (kPyramidModes.find(fm) != kPyramidModes.end()) {
    // Optimization coefficient matrices and vectors.
    MatrixXd Jn(CalcDiffProblemData_.Jn), J(CalcDiffProblemData_.J);
    VectorXd phi_constraints(CalcDiffProblemData_.phi_constraints);

    if (fm == ForwardDynamicsMode::kQpMp) {
      VectorXd beta_star(CalcDiffProblemData_.beta_star);

      // BackwardQp(Q, tau_h, Jn, J, phi_constraints, q_dict, q_next_dict, v_star,
      //            beta_star, params);
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                          params, &solver_log_pyramid_->get_H_llt());
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMp) {
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                         params, nullptr);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMy) {
      if (J.rows() > 0) {
        BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                          params, &solver_log_pyramid_->get_H_llt());
      }
      else {
        BackwardLogPyramid(Q, q_dict, params, false);
      }
      return;
    }
  }

  if (kIcecreamModes.find(fm) != kIcecreamModes.end()) {
    std::vector<Eigen::Matrix3Xd> J_list(CalcDiffProblemData_.J_list);

    if (fm == ForwardDynamicsMode::kSocpMp) {
      std::vector<Eigen::VectorXd> lambda_star_list(CalcDiffProblemData_.lambda_star_list);
      std::vector<Eigen::VectorXd> e_list(CalcDiffProblemData_.e_list);

      BackwardSocp(Q, tau_h, J_list, e_list, phi, q_dict, q_next_dict, v_star,
                   lambda_star_list, params);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogIcecream) {
      if (J_list.size() > 0) {
        BackwardLogIcecream(q_dict, q_next_dict, v_star, params,
                          solver_log_icecream_->get_H_llt());
      }
      else {
        BackwardLogIcecream(Q, q_dict, params, false);
      }
      return;
    }
  }

  std::stringstream ss;
  ss << "Forward dynamics mode " << static_cast<int>(fm)
     << " is not supported in C++.";
  throw std::logic_error(ss.str());
}

VectorXd QuasistaticSimulator::CalcDynamicsForward(
    QuasistaticSimulator* q_sim, const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u);
  q_sim->Calc(q_a_cmd_dict, tau_ext_dict, sim_params);
  return q_sim->GetMbpPositionsAsVec();
}

VectorXd QuasistaticSimulator::CalcDynamicsForward(
    QuasistaticSimulator* q_sim, const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& tau_u,
    const QuasistaticSimParameters& sim_params) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u);
  q_sim->AddTauExtFromVec(tau_u, &tau_ext_dict);
  q_sim->Calc(q_a_cmd_dict, tau_ext_dict, sim_params);
  return q_sim->GetMbpPositionsAsVec();
}

/*
  The functions for binding with Python
*/
VectorXd QuasistaticSimulator::CalcDynamicsForward(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  auto t_start = std::chrono::steady_clock::now();
  auto q_next = CalcDynamicsForward(this, q, u, sim_params);
  auto t_end = std::chrono::steady_clock::now();
  
  n_forward_calls_ += 1;
  total_forward_time_ += std::chrono::duration<double>(t_end - t_start).count();
  
  return q_next;
}

VectorXd QuasistaticSimulator::CalcDynamicsForward(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const QuasistaticSimParameters& sim_params) {
  auto t_start = std::chrono::steady_clock::now();
  auto q_next = CalcDynamicsForward(this, q, u, tau_ext, sim_params);
  auto t_end = std::chrono::steady_clock::now();
  
  n_forward_calls_ += 1;
  total_forward_time_ += std::chrono::duration<double>(t_end - t_start).count();
  
  return q_next;
}

void QuasistaticSimulator::CalcDynamicsBackward(
    QuasistaticSimulator* q_sim,
    const QuasistaticSimParameters& sim_params) {
  q_sim->CalcDiff(sim_params);
}

void QuasistaticSimulator::CalcDynamicsBackward(
    const QuasistaticSimParameters& sim_params) {
  auto t_start = std::chrono::steady_clock::now();
  CalcDynamicsBackward(this, sim_params);
  auto t_end = std::chrono::steady_clock::now();

  n_backward_calls_ += 1;
  total_backward_time_ += std::chrono::duration<double>(t_end - t_start).count();
}

void QuasistaticSimulator::SetManipulandNames(
  const std::vector<std::string>& manipuland_names) {
  if (cjc_) {
    cjc_->SetManipulandNames(manipuland_names);
  }
  else {
    std::stringstream ss;
    ss << "You shold initialize ContactJacobianCalculator first!";
    throw std::logic_error(ss.str());
  }
}

void QuasistaticSimulator::UpdateContactInformation(
  const Eigen::Ref<const Eigen::VectorXd>& q,
  const QuasistaticSimParameters& params
) {
  this->UpdateMbpPositions(q);
  const auto q_dict = this->GetMbpPositions();
  const auto fm = params.forward_mode;
  
  const auto sdps = this->CalcCollisionPairs(params.contact_detection_tolerance, true);

  VectorXd phi;
  MatrixXd Jn, Nhat;

  if (kPyramidModes.find(fm) != kPyramidModes.end()) {
    // J_list is used by CalcJacobianAndPhiQp, but not stored
    std::vector<Eigen::MatrixXd> J_list;
    this->cjc_->CalcJacobianAndPhiQp(
      this->context_plant_, sdps, params.nd_per_contact, &phi, &Jn, &J_list, &Nhat
    );
  }

  if (kIcecreamModes.find(fm) != kIcecreamModes.end()) {
    // J_list is used by CalcJacobianAndPhiSocp, but not stored
    std::vector<Eigen::Matrix3Xd> J_list;
    this->cjc_->CalcJacobianAndPhiSocp(
      this->context_plant_, sdps, &phi, &Jn, &J_list, &Nhat
    );
  }

  MutableContactInfoData_.phi = phi;
  MutableContactInfoData_.Jn = Jn;

  // std::cout << "q: " << q.format(CommaInitFmt) << std::endl;
}

// -----------------------------------------------------------------------------

std::unordered_map<drake::multibody::ModelInstanceIndex,
                   std::unordered_map<std::string, Eigen::VectorXd>>
QuasistaticSimulator::GetActuatedJointLimits() const {
  std::unordered_map<drake::multibody::ModelInstanceIndex,
                     std::unordered_map<std::string, Eigen::VectorXd>>
      joint_limits;
  for (const auto& model : models_actuated_) {
    const auto n_q = plant_->num_positions(model);
    joint_limits[model]["lower"] = Eigen::VectorXd(n_q);
    joint_limits[model]["upper"] = Eigen::VectorXd(n_q);
    int n_dofs = 0;
    for (const auto& joint_idx : plant_->GetJointIndices(model)) {
      const auto& joint = plant_->get_joint(joint_idx);
      const auto n_dof = joint.num_positions();
      if (n_dof != 1) {
        continue;
      }
      for (int j = 0; j < n_dof; j++) {
        auto lower = joint.position_lower_limits();
        auto upper = joint.position_upper_limits();
        joint_limits[model]["lower"][n_dofs] = lower[0];
        joint_limits[model]["upper"][n_dofs] = upper[0];
      }
      n_dofs += n_dof;
    }
    // No floating joints in the robots.
    DRAKE_THROW_UNLESS(n_q == n_dofs);
  }
  return joint_limits;
}

drake::solvers::SolverBase* QuasistaticSimulator::PickBestSocpSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_scs_.get();
  }
  // Commercial solvers.
  if (is_socp_calculating_dual(params)) {
    return solver_msk_.get();
  }
  return solver_grb_.get();
}

drake::solvers::SolverBase* QuasistaticSimulator::PickBestQpSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_osqp_.get();
  }
  return solver_grb_.get();
}

drake::solvers::SolverBase* QuasistaticSimulator::PickBestConeSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_scs_.get();
  }
  return solver_msk_.get();
}

void QuasistaticSimulator::print_solver_info_for_default_params() const {
  const auto socp_solver_name =
      PickBestConeSolver(sim_params_)->solver_id().name();
  const auto qp_solver_name = PickBestQpSolver(sim_params_)->solver_id().name();
  const auto cone_solver_name =
      PickBestConeSolver(sim_params_)->solver_id().name();

  cout << "=========== Solver Info ===========" << endl;
  cout << "Using free solvers? " << sim_params_.use_free_solvers << endl;
  cout << "SOCP solver: " << socp_solver_name << endl;
  cout << "QP solver: " << qp_solver_name << endl;
  cout << "Cone solver: " << cone_solver_name << endl;
  cout << "===================================" << endl;
}

/*
CalcDfDxLogPyramid computes DfDx using Drake's AutoDiffXd.
Here we derive the derivatives analytically.
*/

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxLogIcecreamAnalytic(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const QuasistaticSimParameters& params,
    const Eigen::LLT<Eigen::MatrixXd>& H_llt){

  std::vector<Eigen::MatrixXd> Dv_next_DJ, Dv_next_DPhi;

  const auto kappa = params.log_barrier_weight;
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  // Dδq*/Db * Db/Dq
  MatrixXd DyDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(MatrixXd::Identity(n_v_, n_v_) * kappa, h, &DyDq);
  // H_llt.solveInPlace(DyDq);

  this->UpdateMbpPositions(q_dict);

  const auto sdps = CalcCollisionPairs(params.contact_detection_tolerance, true);
  // GetFingerNamesFromSdps(sdps, &ordered_fingers_in_contact_);
  
  const auto n_c = sdps.size();
  std::vector<Eigen::Matrix3Xd> J_list;
  MatrixXd Jn, Nhat;
  VectorXd phi;
  cjc_->CalcJacobianAndPhiSocp(context_plant_, sdps, &phi, &Jn, &J_list, &Nhat);

  static const Eigen::Matrix3d W{{1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
  for (size_t i_c = 0; i_c < n_c; i_c++) {
    std::vector<std::string> names;
    GetBodyNameFromSdp(sdps[i_c], &names);
    // std::cout << "finger " << i_c << " when DfDx: " << names[1] << std::endl;

    Eigen::MatrixXd J_i = J_list[i_c];
    VectorXd Jn_i = Jn.row(i_c);
    VectorXd Jt1_i = J_i.row(1);
    VectorXd Jt2_i = J_i.row(2);
    double phi_i = phi[i_c];

    // calc canonical values
    double mu = cjc_->get_friction_coefficient(i_c);
    Jn_i = Jn_i / mu;
    phi_i = phi_i / h / mu;

    // calc auxiliary values
    Vector3d w = J_i * v_star;
    w[0] += phi_i;
    const double d = w.transpose() * W * w;
    VectorXd m = (phi_i + Jn_i.transpose() * v_star) * Jn_i - \
                  (Jt1_i.transpose() *  v_star) * Jt1_i - \
                  (Jt2_i.transpose() *  v_star) * Jt2_i;

    // calculate Dv_next_DJ
    Eigen::MatrixXd Dg_DJ(n_v_, 3*n_v_);
    Dg_DJ.block(0, 0, n_v_, n_v_) = (2 / d / d) * (-d * (Jn_i * v_star.transpose() + w[0] * Eigen::MatrixXd::Identity(n_v_, n_v_)) \
                                                  + 2 * w[0] * m * v_star.transpose());
    Dg_DJ.block(0, n_v_, n_v_, n_v_) = (2 / d / d) * (d*(Jt1_i * v_star.transpose() + w[1] * Eigen::MatrixXd::Identity(n_v_, n_v_)) \
                                                      - 2 * w[1] * m * v_star.transpose());
    Dg_DJ.block(0, 2 * n_v_, n_v_, n_v_) = (2 / d / d) * (d*(Jt2_i * v_star.transpose() + w[2] * Eigen::MatrixXd::Identity(n_v_, n_v_)) \
                                                          - 2 * w[2] * m * v_star.transpose());

    // Dv_next_DJ.emplace_back(-H_llt.solve(Dg_DJ));
    Dv_next_DJ.emplace_back(Dg_DJ);

    // TODO(yongpeng): debug cosine similarity between DKKT/DJ and DKKT/DPhi
    // std::cout << "Comparing the " << i_c << "th contact!" << std::endl;
    // AutoDiffXd d_ad;
    // Vector3<AutoDiffXd> w_ad;
    // VectorX<AutoDiffXd> y_ad;
    // const auto J_i_ad = InitializeAutoDiff(J_i.reshaped<Eigen::RowMajor>());
    // Matrix3X<AutoDiffXd> J_i_ad_mat = J_i_ad.reshaped<Eigen::RowMajor>(3, n_v_);
    // w_ad = J_i_ad_mat * v_star;
    // w_ad[0] += phi_i;
    // d_ad = -w_ad[0] * w_ad[0] + w_ad[1] * w_ad[1] + w_ad[2] * w_ad[2];
    // y_ad = (2 / d_ad) * J_i_ad_mat.transpose() * Vector3<AutoDiffXd>(w_ad[0], -w_ad[1], -w_ad[2]);
    // Eigen::MatrixXd Dg_DJ_ad = drake::math::ExtractGradient(y_ad);
    // std::cout << "Dg_DJ: ||AutoDiff||=" << Dg_DJ_ad.norm() << ", ||Analytic||=" << Dg_DJ.norm() << std::endl;
    // std::cout << "cosine similarity: " << (Dg_DJ_ad.reshaped().transpose() * Dg_DJ.reshaped()) / (Dg_DJ_ad.norm() * Dg_DJ.norm()) << std::endl;

    // calculate Dv_next_DPhi
    Eigen::VectorXd Dg_DPhi = (2 / d / d) * (-d * Jn_i + 2 * w[0] * m);
    // Dv_next_DPhi.emplace_back(-H_llt.solve(Dg_DPhi));
    Dv_next_DPhi.emplace_back(Dg_DPhi);

    // TODO(yongpeng): remove these after debug
    // Eigen::VectorXd phi_i_vec(1);
    // phi_i_vec(0) = phi_i;
    // const auto phi_i_ad = InitializeAutoDiff(phi_i_vec);
    // w_ad = J_i * v_star;
    // w_ad[0] += phi_i_ad(0, 0);
    // d_ad = -w_ad[0] * w_ad[0] + w_ad[1] * w_ad[1] + w_ad[2] * w_ad[2];
    // y_ad = (2 / d_ad) * J_i.transpose() * Vector3<AutoDiffXd>(w_ad[0], -w_ad[1], -w_ad[2]);
    // Eigen::VectorXd Dg_DPhi_ad = drake::math::ExtractGradient(y_ad);
    // std::cout << "Dg_DPhi: ||AutoDiff||=" << Dg_DPhi_ad.norm() << ", ||Analytic||=" << Dg_DPhi.norm() << std::endl;
    // std::cout << "cosine similarity: " << (Dg_DPhi_ad.transpose() * Dg_DPhi) / (Dg_DPhi_ad.norm() * Dg_DPhi.norm()) << std::endl;
  }

  // calc dJ/dq
  // std::cout << "calc dJ/dq ..." << std::endl;
  std::vector<Eigen::MatrixXd> H_list;
  Eigen::VectorXd q = GetQVecFromDict(q_dict);
  CalcContactHessianFiniteDiff(params, q, &H_list);

  // calc Dv_nextDq
  // std::cout << "calc Dv_nextDq ..." << std::endl;
  Eigen::MatrixXd DyDq_part2 = MatrixXd::Zero(n_v_, n_q_);
  for (size_t i_c = 0; i_c < n_c; i_c++) {
    VectorXd Jn_i = Jn.row(i_c);
    Jn_i = Jn_i / h / cjc_->get_friction_coefficient(i_c);
    DyDq_part2 += Dv_next_DJ[i_c] * H_list[i_c] + Dv_next_DPhi[i_c] * Jn_i.transpose();
  }
  // H_llt.solveInPlace(DyDq_part2);
  DyDq += DyDq_part2;
  DyDq *= -1;
  H_llt.solveInPlace(DyDq);

  return CalcDq_nextDqFromDv_nextDq(DyDq, q_next_dict, v_star, h);
}

/**************************************************************/

/*
  Get contact Jacobian in the order of ordered_fingers_in_contact_.
  This is to ensure dJ/dq and dv_next/dJ are multiplied in the correct order.
  Jn in the returned J_list_ptr is divided by mu.
*/
void QuasistaticSimulator::GetOrderedContactJacobian(
  const Eigen::VectorXd& q,
  double tolerance,
  std::vector<Eigen::Matrix3Xd>* J_list_ptr,
  bool in_order
) {
  // unused placeholders
  MatrixXd Jn, Nhat;
  VectorXd phi;
  this->UpdateMbpPositions(q);
  const auto sdps = CalcCollisionPairs(tolerance, in_order);

  // // sort sdps
  // std::vector<drake::geometry::SignedDistancePair<double>> sdps_sorted;
  // std::vector<std::string> finger_names_from_sdps;
  // GetFingerNamesFromSdps(sdps, &finger_names_from_sdps);
  // for (const auto& name : ordered_fingers_in_contact_) {
  //   auto it = std::find(finger_names_from_sdps.begin(), finger_names_from_sdps.end(), name);
  //   if (it != finger_names_from_sdps.end()) {
  //     int index = std::distance(finger_names_from_sdps.begin(), it);
  //     sdps_sorted.push_back(sdps[index]);
  //   }
  // }

  cjc_->CalcJacobianAndPhiSocp(context_plant_, sdps, &phi, &Jn, J_list_ptr, &Nhat);
}

drake::multibody::JointIndex QuasistaticSimulator::GetParentJointIndex(
  const drake::multibody::Body<double>& body
) const {
  for (int i = 0; i < plant_->num_joints(); ++i) {
    const auto& joint = plant_->get_joint(JointIndex(i));
    if (joint.child_body().index() == body.index()) {
      return joint.index();
    }
  }
}

void QuasistaticSimulator::SetCollisionBodyNames(
  const std::vector<std::string>& fingertip_names,
  const std::string& object_name
) {
  body_to_q_indices_.clear();
  for (const auto& name : fingertip_names) {
    std::vector<long> q_indices;
    FindDofIndicesToRoot(name, &q_indices);
    body_to_q_indices_[name] = q_indices;
  }
  std::vector<long> q_indices;
  FindDofIndicesToRoot(object_name, &q_indices);
  body_to_q_indices_["object"] = q_indices;
}

void QuasistaticSimulator::FindDofIndicesToRoot(
  const std::string& body_name,
  std::vector<long>* q_indices_ptr
) const {
  const Body<double>& body = plant_->GetBodyByName(body_name);

  const Body<double>* current_body = &body;
  q_indices_ptr->clear();

  while (current_body->index() != plant_->world_body().index()) {
      const auto& joint = plant_->get_joint(GetParentJointIndex(*current_body));
      if (joint.num_positions() >= 1) {
        for (int i = 0; i < joint.num_positions(); i++) {
          q_indices_ptr->push_back(joint.position_start() + i);
        }
      }
      current_body = &joint.parent_body();
  }
}

void QuasistaticSimulator::GetBodyNameFromSdp(
  const drake::geometry::SignedDistancePair<double>& sdp,
  std::vector<std::string>* body_names_ptr
) const {
  body_names_ptr->clear();
  const auto& insp = sg_->model_inspector();
  const Body<double>* body_A = plant_->GetBodyFromFrameId(insp.GetFrameId(sdp.id_A));
  const Body<double>* body_B = plant_->GetBodyFromFrameId(insp.GetFrameId(sdp.id_B));
  body_names_ptr->push_back(body_A->name());
  body_names_ptr->push_back(body_B->name());
}

void QuasistaticSimulator::GetFingerNamesFromSdps(
  const std::vector<drake::geometry::SignedDistancePair<double>>& sdps,
  std::vector<std::string>* finger_names_ptr
) const {
  finger_names_ptr->clear();
  for (const auto& sdp : sdps) {
    std::vector<std::string> body_names;
    GetBodyNameFromSdp(sdp, &body_names);
    finger_names_ptr->push_back(body_names[1]);   // assume fingertip is the second geom
  }
}

void QuasistaticSimulator::GetFingerGeomidsFromSdps(
    const std::vector<drake::geometry::SignedDistancePair<double>>& sdps,
    std::vector<CollisionPair>* collision_pairs_ptr
  ) const {
    collision_pairs_ptr->clear();

    // Save collision pairs, which may later be used in gradient computation by
    // the AutoDiff MBP.
    for (const auto& sdp : sdps) {
      collision_pairs_ptr->emplace_back(sdp.id_A, sdp.id_B);
    }
  }

void QuasistaticSimulator::CalcContactHessianFiniteDiff(
  QuasistaticSimParameters sim_params,
  const Eigen::VectorXd& q,
  std::vector<Eigen::MatrixXd>* H_list_ptr
) {
  if (body_to_q_indices_.empty()) {
    std::stringstream ss;
    ss << "You should call SetCollisionBodyNames first!";
    throw std::logic_error(ss.str());
  }
  // PrintAllJointWithNames();

  int n_v = plant_->num_velocities();
  double dq = 1e-5; // perturbation in each dim
  H_list_ptr->clear();

  this->UpdateMbpPositions(q);
  const auto sdps = CalcCollisionPairs(sim_params.contact_detection_tolerance, true);
  // PrintAllContactWithNames(sdps);

  std::vector<Eigen::Matrix3Xd> J_list_unperturbed;
  GetOrderedContactJacobian(q, sim_params.contact_detection_tolerance, &J_list_unperturbed, true);

  // for (const auto&sdp : sdps){
  for (size_t i = 0; i < sdps.size(); i++) {
    const auto& sdp = sdps[i];
    const auto& J = J_list_unperturbed[i];

    std::vector<std::string> body_names;
    GetBodyNameFromSdp(sdp, &body_names);

    // include indices of the object and the corresponding finger
    const std::string& finger_name = body_names[1];
    // std::cout << "finger " << i << " when diff(H): " << finger_name << std::endl;
    std::vector<long> q_indices;
    // if (finger_name == "fingertip") {
    //   q_indices = {1, 2, 3, 4};
    //   // q_indices = {5, 6, 7, 8};
    // }
    // else if (finger_name == "fingertip_2") {
    //   q_indices = {9, 10, 11, 12};
    //   // q_indices = {13, 14, 15, 16};
    // }
    // else if (finger_name == "fingertip_3") {
    //   q_indices = {13, 14, 15, 16};
    //   // q_indices = {9, 10, 11, 12};
    // }
    // else if (finger_name == "thumb_fingertip") {
    //   q_indices = {5, 6, 7, 8};
    //   // q_indices = {1, 2, 3, 4};
    // }
    // else {
    //   std::cout << "Unknown finger name: " << finger_name << std::endl;
    //   continue;
    // }
    q_indices = body_to_q_indices_.at(finger_name);
    q_indices.insert(q_indices.end(), body_to_q_indices_.at("object").begin(), body_to_q_indices_.at("object").end());

    Eigen::MatrixXd H(3*n_v, n_v);
    H.setZero();

    for (const auto j : q_indices) {
    // for (size_t j = 0; j <= 16; j++) {
      // perturb each finger joint and object joint
      auto q_perturbed = q;
      q_perturbed[j] += dq;
      std::vector<Eigen::Matrix3Xd> J_list_perturbed;
      GetOrderedContactJacobian(q_perturbed, sim_params.contact_detection_tolerance, &J_list_perturbed, true);


      DRAKE_ASSERT(J_list_unperturbed.size() == J_list_perturbed.size());
      const auto J_finite_diff = (J_list_perturbed[i] - J) / dq;

      // std::cout << "||J||:" << J.norm() << " ||J_list_perturbed||:" << J_list_perturbed[i].norm() << std::endl;
      // std::cout << "||J_finite_diff||" << J_finite_diff.norm() << std::endl;

      H.block(0, j, n_v, 1) = J_finite_diff.row(0);
      H.block(n_v, j, n_v, 1) = J_finite_diff.row(1);
      H.block(2*n_v, j, n_v, 1) = J_finite_diff.row(2);
    }
    H_list_ptr->push_back(H);
    // std::cout << "||H||:" << H.norm() << std::endl;
  }

  // TODO(yongpeng): remove after debugging
  // // compute dJ/dq using AutoDiff
  // // std::cout << "calc dJ/dq with AutoDiff" << std::endl;
  // const auto q_ad = InitializeAutoDiff(q);
  // UpdateMbpAdPositions(q_ad);
  // const auto sdps_ad = CalcSignedDistancePairsFromCollisionPairs();
  // std::vector<Matrix3X<AutoDiffXd>> J_ad_list;
  // VectorX<AutoDiffXd> phi_ad;
  // MatrixX<AutoDiffXd> Jn_ad_list;
  // cjc_ad_->CalcJacobianAndPhiSocp(context_plant_ad_, sdps_ad, &phi_ad, &Jn_ad_list, &J_ad_list, nullptr);
  // std::vector<VectorX<AutoDiffXd>> J_ad_vec_list;
  // for (size_t i = 0; i < sdps.size(); i++) {
  //   VectorX<AutoDiffXd> J_ad_vec(3*n_v_);
  //   for (size_t j = 0; j < 3; j++) {
  //     J_ad_vec.block(j*n_v_, 0, n_v_, 1) = J_ad_list[i].row(j);
  //   }
  //   J_ad_vec_list.push_back(J_ad_vec);
  // }

  // // compare finite diff and AutoDiff
  // // std::cout << "compare dJ/dq with AutoDiff and finite diff" << std::endl;
  // for (size_t i = 0; i < sdps.size(); i++) {
  //   // std::cout << "the " << i << "th contact" << std::endl;
  //   const auto& J_ad_vec = J_ad_vec_list[i];
  //   const Eigen::VectorXd H_AutoDiff = drake::math::ExtractGradient(J_ad_vec).reshaped();
  //   const Eigen::VectorXd H_FiniteDiff = H_list_ptr->at(i).reshaped();

  //   DRAKE_ASSERT((H_AutoDiff.norm() > 1e-5) && (H_FiniteDiff.norm() > 1e-5));
  //   DRAKE_ASSERT(H_AutoDiff.size() == H_FiniteDiff.size());

  //   double cosine = H_AutoDiff.dot(H_FiniteDiff) / (H_AutoDiff.norm() * H_FiniteDiff.norm());

  //   // std::cout << "AutoDiff H dimensions: " << H_AutoDiff.rows() << "x" << H_AutoDiff.cols() << std::endl;
  //   // std::cout << "FiniteDiff H dimensions: " << H_FiniteDiff.rows() << "x" << H_FiniteDiff.cols() << std::endl;
  //   // std::cout << "||H_AutoDiff||=" << H_AutoDiff.norm() \
  //   //           << " ||H_FiniteDiff||=" << H_FiniteDiff.norm() \
  //   //           << " cosine similarity:" << cosine << std::endl;
  //   DRAKE_ASSERT(std::abs(cosine) > 0.99);
  // }
}

void QuasistaticSimulator::PrintAllJointWithNames() const {
  for (int i = 0; i < plant_->num_joints(); ++i) {
    const auto& joint = plant_->get_joint(JointIndex(i));
    if (joint.num_positions() == 0) {
      std::cout << "Joint " << i << ": " << joint.name() << std::endl;
    }
    else {
      const int i_start = joint.position_start();
      const int i_end = i_start + joint.num_positions();
      std::cout << "Joint " << i << ": " << joint.name() << " index: [" << i_start << ", " << i_end << "]" << std::endl;
    }
  }
}

void QuasistaticSimulator::PrintAllContactWithNames(
  const std::vector<drake::geometry::SignedDistancePair<double>>& sdps
) const{
  for (const auto sdp : sdps) {
    std::vector<std::string> body_names;
    GetBodyNameFromSdp(sdp, &body_names);
    std::cout << "Contact: " << body_names[0] << " and " << body_names[1] << std::endl;
  }
}

bool QuasistaticSimulator::CheckCollision() const {
  return query_object_->HasCollisions();
}
