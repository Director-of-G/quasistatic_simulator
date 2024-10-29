#include "qsim/pinocchio_calculator.h"

#include <iostream>

using std::vector;

using std::cout;
using std::endl;

PinocchioCalculator::PinocchioCalculator(
  const std::string& robot_urdf_filename,
  const std::string& object_urdf_filename
) {
  pinocchio::SE3 se3_I = pinocchio::SE3::Identity();

  pinocchio::Model object_model;
  pinocchio::GeometryModel object_geom_model;

  pinocchio::urdf::buildModel(robot_urdf_filename, model);
  pinocchio::urdf::buildModel(object_urdf_filename, object_model);
  
  pinocchio::urdf::buildGeom(model, robot_urdf_filename, pinocchio::COLLISION, geom_model);
  pinocchio::urdf::buildGeom(object_model, object_urdf_filename, pinocchio::COLLISION, object_geom_model);

  // append manipuland to model
  pinocchio::JointIndex object_joint_idx = model.addJoint(0, pinocchio::JointModelFreeFlyer(), se3_I, "object_root_joint");
  model.addJointFrame(object_joint_idx, -1);

  model.appendBodyToJoint(object_joint_idx, object_model.inertias.back(), se3_I);
  pinocchio::FrameIndex object_frame_idx = model.addBodyFrame("object_link", object_joint_idx, se3_I, -1);

  // append collision geometry to model
  pinocchio::GeometryObject object_collision_geom = object_geom_model.geometryObjects.back();
  object_collision_geom.parentFrame = object_frame_idx;
  object_collision_geom.parentJoint = object_joint_idx;
  geom_model.addGeometryObject(object_collision_geom);

  // collision detection
  data = pinocchio::Data(model);
  geom_model.addAllCollisionPairs();
  geom_data = pinocchio::GeometryData(geom_model);
}

void PinocchioCalculator::UpdateModelConfiguration(
  const Eigen::VectorXd& q0
) {
  pinocchio::forwardKinematics(model, data, q0);
  pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data);
}

void PinocchioCalculator::UpdateHessiansAndJacobians() {
  pinocchio::computeJointJacobians(model, data);
  pinocchio::computeJointKinematicHessians(model, data);
}

Eigen::MatrixXd PinocchioCalculator::GetContactJacobian(
    const Eigen::Matrix4d& T_C2F,
    const Eigen::Vector3d& N_hat,
    const std::string& F_name
) const {
  pinocchio::FrameIndex F_idx = model.getFrameId(F_name);
  pinocchio::JointIndex J_idx = model.frames.at(F_idx).parent;

  pinocchio::Data::Matrix6x J(6, model.nv);
  J.setZero();
  pinocchio::getJointJacobian(model, data, J_idx, pinocchio::LOCAL_WORLD_ALIGNED, J);

  return J;

}

Eigen::Tensor<double, 3, Eigen::ColMajor, Eigen::DenseIndex> PinocchioCalculator::GetContactKinematicHessian(
  const Eigen::Matrix4d& T_C2F,
  const Eigen::Vector3d& N_hat,
  const std::string& F_name
) const {

  pinocchio::SE3 SE3_T_C2F(T_C2F);
  pinocchio::FrameIndex F_idx = model.getFrameId(F_name);
  pinocchio::JointIndex J_idx = model.frames.at(F_idx).parent;
  
  pinocchio::Tensor<double, 3> kin_H(6, model.nv, model.nv);
  kin_H.setZero();
  pinocchio::getJointKinematicHessian(model, data, J_idx, pinocchio::LOCAL_WORLD_ALIGNED, kin_H);

  return kin_H;

}

