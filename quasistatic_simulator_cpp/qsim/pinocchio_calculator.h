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
#include <pinocchio/algorithm/model.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include <pinocchio/algorithm/jacobian.hpp>

template <typename T>
struct PinContactPairInfo {
  // modified from ContactPairInfo from contact_jacobian_calculator.h

  typedef Eigen::Matrix<T, 3, 1> Vector3T;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;
  typedef Eigen::Matrix<T, 3, Eigen::Dynamic> Matrix3XT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;

  // Contact normal pointing to body A from body B.
  Vector3T nhat_BA_W;

  // Tangents which are perpendicular to nhat_BA_W. Each column of the matrix
  // is a tangent vector. For QP dynamics, there are n_d tangent vectors; for
  // SOCP, there are two.
  Matrix3XT t_W;

  // The (3, n_v) contact Jacobian defined in the docs.
  Matrix3XT Jc;
  double mu{0};  // coefficient of friction.

  // For contact force visualization.
  Vector3T p_WCa;
  Vector3T p_WCb;

  // geometry indices
  pinocchio::GeomIndex id_A;
  pinocchio::GeomIndex id_B;

  // TODO(yongpeng): For contact normal.
  bool is_A_manipuland{false};
};

template <typename T>
class PinocchioCalculator {
 public:
  PinocchioCalculator(
    const std::string& robot_urdf_filename,
    const std::string& object_urdf_filename
  );

  void UpdateModelConfiguration(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& q0
  );

  void UpdateCollisionPairs(
    double contact_detection_tolerance,
    std::vector<PinContactPairInfo<T>>* contact_pairs_ptr
  );

  void CalcJacobianAndPhiQp(
    double contact_detection_tolerance,
    const int n_d,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* phi_ptr,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* Jn_ptr,
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>* J_list_ptr,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* Nhat_list_ptr
  );

  // TODO(yongpeng): private attributes
  pinocchio::Model model;
  pinocchio::GeometryModel geom_model;
  pinocchio::Data data;
  pinocchio::GeometryData geom_data;
};

extern template class PinocchioCalculator<double>;
