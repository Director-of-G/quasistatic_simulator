#include "qsim/pinocchio_calculator.h"

#include <iostream>

using std::vector;

using std::cout;
using std::endl;

template <class T>
PinocchioCalculator<T>::PinocchioCalculator(
  const std::string& robot_urdf_filename,
  const std::string& object_urdf_filename
) {
  pinocchio::SE3 se3_I = pinocchio::SE3::Identity();

  pinocchio::Model object_model;
  pinocchio::GeometryModel object_geom_model;

  pinocchio::urdf::buildModel(robot_urdf_filename, pinocchio::JointModelPlanar(), model);
  pinocchio::urdf::buildModel(object_urdf_filename, pinocchio::JointModelPlanar(), object_model);
  
  pinocchio::urdf::buildGeom(model, robot_urdf_filename, pinocchio::COLLISION, geom_model);
  pinocchio::urdf::buildGeom(object_model, object_urdf_filename, pinocchio::COLLISION, object_geom_model);

  // append manipuland to model
  pinocchio::JointIndex object_joint_idx = model.addJoint(0, pinocchio::JointModelPlanar(), se3_I, "object_root_joint");
  model.addJointFrame(object_joint_idx, -1);

  model.appendBodyToJoint(object_joint_idx, object_model.inertias.back(), se3_I);
  pinocchio::FrameIndex object_frame_idx = model.addBodyFrame("object_link", object_joint_idx, se3_I, -1);

  // append collision geometry to model
  pinocchio::GeometryObject object_collision_geom = object_collision_model.geometryObjects.back();
  object_collision_geom.parentFrame = object_frame_idx;
  object_collision_geom.parentJoint = object_joint_idx;
  collision_model.addGeometryObject(object_collision_geom);

  // collision detection
  pinocchio::Data data(model);
  collision_model.addAllCollisionPairs();
  pinocchio::GeometryData geom_data(collision_model);
}

template <class T>
void PinocchioCalculator<T>::UpdateModelConfiguration(
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& q0
) {
  pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data, q0);
}

template <class T>
void PinocchioCalculator<T>::UpdateCollisionPairs(
    double contact_detection_tolerance,
    std::vector<PinContactPairInfo<T>>* contact_pairs_ptr
) {
  // clear previous result
  contact_pairs_ptr->clear();

  // compute collisions between all pairs
  pinocchio::computeCollisions(geom_model, geom_data, false);

  for(size_t k = 0; k < geom_model.collisionPairs.size(); ++k)
  {
    const pinocchio::CollisionPair & cp = geom_model.collisionPairs[k];
    const hpp::fcl::CollisionResult & cr = geom_data.collisionResults[k];
    
    PinContactPairInfo cp_info;
    cp_info.id_A = cp.first;
    cp_info.id_B = cp.second;

  }
}

template <class T>
void PinocchioCalculator<T>::CalcJacobianAndPhiQp(
    double contact_detection_tolerance,
    const int n_d,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* phi_ptr,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* Jn_ptr,
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>* J_list_ptr,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* Nhat_list_ptr
  ) const {

  const auto n_c = sdps.size();
  const int n_v = plant_->num_velocities();
  const auto n_f = n_d * n_c;

  VectorX<T>& phi = *phi_ptr;
  MatrixX<T>& Jn = *Jn_ptr;
  phi.resize(n_c);
  Jn.resize(n_c, n_v);
  if (Nhat_list_ptr) {
    Nhat_list_ptr->resize(n_c, 3);
  }
  J_list_ptr->clear();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& sdp = sdps[i_c];
    const auto& cpi = contact_pairs_[i_c];
    const auto mu = get_friction_coefficient(i_c);

    // contact normal
    if (Nhat_list_ptr) {
      if (cpi.is_A_manipuland) {
        Nhat_list_ptr->row(i_c) = sdp.distance * sdp.nhat_BA_W.transpose();
      }
      else {
        Nhat_list_ptr->row(i_c) = - sdp.distance * sdp.nhat_BA_W.transpose();
      }
    }

    phi[i_c] = sdp.distance;
    Jn.row(i_c) = sdp.nhat_BA_W.transpose() * cpi.Jc;

    contact_pairs_[i_c].t_W = CalcTangentVectors<T>(sdp.nhat_BA_W, n_d);
    const auto& d_W = contact_pairs_[i_c].t_W;
    J_list_ptr->template emplace_back(n_d, n_v);
    MatrixX<T>& J_i_c = J_list_ptr->back();
    for (int j = 0; j < n_d; j++) {
      J_i_c.row(j) = Jn.row(i_c) + mu * d_W.col(j).transpose() * cpi.Jc;
    }
  }
}

template class PinocchioCalculator<double>;
