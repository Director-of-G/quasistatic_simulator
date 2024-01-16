// This file is modified from contact_jacobian_calculator.h
// Jacobian calculation and differentiation is replaced by pinocchio
// to speed up dynamics

#pragma once

#ifndef PINOCCHIO_WITH_HPP_FCL
    #define PINOCCHIO_WITH_HPP_FCL
#endif

#include <string>
#include <Eigen/Dense>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

class PinocchioCalculator {
 public:
  PinocchioCalculator(
    const std::string& robot_urdf_filename,
    const std::string& object_urdf_filename
  );

  void UpdateModelConfiguration(
    const Eigen::VectorXd& q0
  );

  void UpdateHessiansAndJacobians();

  Eigen::MatrixXd GetContactJacobian(
    const Eigen::Matrix4d& T_C2F,
    const Eigen::Vector3d& N_hat,
    const std::string& F_name
  ) const;

  Eigen::Tensor<double, 3, Eigen::ColMajor, Eigen::DenseIndex> GetContactKinematicHessian(
    const Eigen::Matrix4d& T_C2F,
    const Eigen::Vector3d& N_hat,
    const std::string& F_name
  ) const;

  // TODO(yongpeng): private attributes
  pinocchio::Model model;
  pinocchio::GeometryModel geom_model;
  pinocchio::Data data;
  pinocchio::GeometryData geom_data;
};

