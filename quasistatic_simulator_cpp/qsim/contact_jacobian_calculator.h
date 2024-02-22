#pragma once

#include <Eigen/Dense>

#include "drake/common/default_scalars.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "qsim/quasistatic_sim_params.h"

template <typename T>
struct ContactPairInfo {
  // Contact normal pointing to body A from body B.
  // double sdist;     // signed distance
  drake::Vector3<T> nhat_BA_W;

  // Tangents which are perpendicular to nhat_BA_W. Each column of the matrix
  // is a tangent vector. For QP dynamics, there are n_d tangent vectors; for
  // SOCP, there are two.
  drake::Matrix3X<T> t_W;

  // The (3, n_v) contact Jacobian defined in the docs.
  drake::Matrix3X<T> Jc;
  double mu{0};  // coefficient of friction.

  // For contact force visualization.
  // contact points location in world frame
  drake::Vector3<T> p_WCa;
  drake::Vector3<T> p_WCb;
  drake::multibody::BodyIndex body_A_idx;
  drake::multibody::BodyIndex body_B_idx;
  drake::geometry::GeometryId id_A;
  drake::geometry::GeometryId id_B;

  // TODO(yongpeng): some new features
  // contact point in geometry's body frame
  // this is different from p_WCa and p_WCb
  drake::Vector3<T> p_ACa;
  drake::Vector3<T> p_BCb;

  // contact jacobian
  drake::Matrix3X<T> JcA;
  drake::Matrix3X<T> JcB;

  // geometry name
  std::string geom_name_A;
  std::string geom_name_B;

  // TODO(yongpeng): For contact normal.
  bool is_A_manipuland{false};
};

template <typename T>
class ContactJacobianCalculator {
 public:
  ContactJacobianCalculator(
      const drake::systems::Diagram<T>* diagram,
      std::set<drake::multibody::ModelInstanceIndex> models_all);

  // TODO(yongpeng): For contact normal.
  void SetManipulandNames(const std::vector<std::string>& manipuland_names);
  /*
   * Computes contact Jacobians for the list of SignedDistancePairs. This
   * should be the first function that gets called when computing Jacobians
   * for a given state, before computing normal / tangent Jacobians.
   */
  void UpdateContactPairInfo(
      const drake::systems::Context<T>* context_plant,
      const std::vector<drake::geometry::SignedDistancePair<T>>& sdps) const;

  /*
   *  Retrieve coefficient of friction from the cached list of contact info.
   */
  double get_friction_coefficient(int i_c) const {
    return contact_pairs_[i_c].mu;
  }

  const std::vector<ContactPairInfo<T>>& get_contact_pair_info_list() const {
    return contact_pairs_;
  }

  void CalcJacobianAndPhiQp(
      const drake::systems::Context<T>* context_plant,
      const std::vector<drake::geometry::SignedDistancePair<T>>& sdps, int n_d,
      drake::VectorX<T>* phi_ptr, drake::MatrixX<T>* Jn_ptr,
      std::vector<drake::MatrixX<T>>* J_list_ptr,
      drake::MatrixX<T>* Nhat_list_ptr=nullptr) const;

//   // TODO(yongpeng): calculate normal as an oracle to maintain contact
//   void CalcJacobianAndPhiQpAndNorm(
//       const drake::systems::Context<T>* context_plant,
//       const std::vector<drake::geometry::SignedDistancePair<T>>& sdps, int n_d,
//       drake::VectorX<T>* phi_ptr, drake::MatrixX<T>* Jn_ptr,
//       std::vector<drake::MatrixX<T>>* J_list_ptr,
//       drake::MatrixX<T>* Nhat_list_ptr) const;

  void CalcJacobianAndPhiSocp(
      const drake::systems::Context<T>* context_plant,
      const std::vector<drake::geometry::SignedDistancePair<T>>& sdps,
      drake::VectorX<T>* phi_ptr, drake::MatrixX<T>* Jn_ptr,
      std::vector<drake::Matrix3X<T>>* J_list_ptr,
      drake::MatrixX<T>* Nhat_list_ptr=nullptr) const;

  
  /*
   * GetContactPointsAndJacobian
   * @param[in]: p_ACa_ptr, contact point in bodyA's frame
   * @param[in]: p_BCb_ptr, contact point in bodyB's frame
   * @param[in]: JcA_list_ptr, contact jacobian in world frame
   * @param[in]: JcB_list_ptr, contact jacobian in world frame
   */
  void GetContactPointsAndJacobians(
    std::vector<drake::Vector3<T>>* p_ACa_ptr,
    std::vector<drake::Vector3<T>>* p_BCb_ptr,
    std::vector<drake::Matrix3X<T>>* JcA_list_ptr,
    std::vector<drake::Matrix3X<T>>* JcB_list_ptr
  ) const;

 private:
  double GetFrictionCoefficientForSignedDistancePair(
      drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const;

  std::unique_ptr<drake::multibody::ModelInstanceIndex> FindModelForBody(
      drake::multibody::BodyIndex body_idx) const;

  drake::multibody::BodyIndex GetMbpBodyFromGeometry(
      drake::geometry::GeometryId g_id) const;

  /*
   * Each contact Jacobian is the subtraction of the Jacobians of two points
   * on the two bodies in the contact pair. This function computes the
   * contribution from one of the two bodies.
   *
   * Jc_ptr must have shape (3, n_v).
   */
  drake::Matrix3X<T> CalcContactJaocibanFromPoint(
      const drake::systems::Context<T>* context_plant,
      const drake::multibody::BodyIndex& body_idx,
      const drake::VectorX<T>& pC_Body) const;

  const drake::multibody::MultibodyPlant<T>* plant_{nullptr};
  const drake::geometry::SceneGraph<T>* sg_{nullptr};

  // MBP.
  const std::set<drake::multibody::ModelInstanceIndex> models_all_;

  // friction_coefficients[g_idA][g_idB] and friction_coefficients[g_idB][g_idA]
  //  gives the coefficient of friction between contact geometries g_idA
  //  and g_idB.
  std::unordered_map<drake::geometry::GeometryId,
                     std::unordered_map<drake::geometry::GeometryId, double>>
      friction_coefficients_;

  // Mutable storage for the current contact.
  mutable std::vector<ContactPairInfo<T>> contact_pairs_;

  // TODO(yongpeng): For contact normal.
  std::vector<std::string> manipuland_names_;
};

extern template class ContactJacobianCalculator<double>;
extern template class ContactJacobianCalculator<drake::AutoDiffXd>;
