#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <unordered_set>


// This class defines possible object geometry
enum class ObjectGeom {
  kZRotSphere,
  kFreeRotSphere,
  kFreeMoveSphere,
  kCapsuleValve
};


// This class defines the type of contact model
enum class ContactModelType {
  kConstant,  // Constant stiffness and damping
  kVariant    // Stiffness and damping depend on the distance & velocity
};


struct ContactModelParameters {
  double sigma{0.001};     
  double k{500};            // stiffness
  double d{100};            // damping
  double vd{0.1};           // normal dissipation
  double vs{0.1};           // tangential dissipation
  double mu{1.0};           // friction coefficient
  ContactModelType type;    // contact model type
};


struct ContactControllerParameters {
  double weight_q{1e-3};
  double weight_dq{0.5};
  double weight_f{1.0};
  double weight_u{0.5};
  double weight_du{5.0};

  double time_step{0.01};
  int horizon_length{5};

  double robot_kp{0.5};
  double robot_kd{1e-3};

  bool calc_torque_feedforward{false};
  // bool is_3d_floating{false};
  // bool is_valve{false};
  ObjectGeom object_geom{ObjectGeom::kZRotSphere};
  bool enable_multi_contact{false};

  Eigen::Vector3d hand_base_trans, hand_base_rot;

  struct ContactModelParameters model_params;
};