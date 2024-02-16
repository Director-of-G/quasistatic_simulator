#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <unordered_set>


struct ContactControllerParameters {
  double weight_q{1e-3};
  double weight_dq{0.5};
  double weight_f{1.0};
  double weight_u{0.5};
  double weight_du{5.0};

  double time_step{0.01};
  int horizon_length{5};

  bool calc_torque_feedforward{false};
  bool is_3d_floating{false};

  Eigen::Vector3d hand_base_trans, hand_base_rot;
};