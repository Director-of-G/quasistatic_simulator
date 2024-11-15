#pragma once
#include <iostream>

#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <drake/common/find_resource.h>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/scene_graph_inspector.h>
#include <drake/geometry/geometry_set.h>
#include <drake/geometry/query_object.h>
#include <drake/geometry/query_results/signed_distance_pair.h>
#include <drake/geometry/collision_filter_manager.h>
#include <drake/geometry/collision_filter_declaration.h>

#include <drake/math/rotation_matrix.h>

#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/math/spatial_velocity.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/plant/multibody_plant_config_functions.h>
#include <drake/multibody/tree/fixed_offset_frame.h>
#include <drake/multibody/tree/revolute_joint.h>
#include <drake/multibody/tree/ball_rpy_joint.h>
#include <drake/multibody/tree/weld_joint.h>

#include "drake/solvers/decision_variable.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/osqp_solver.h"

#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

#include "controllers/contact_controller_params.h"
#include "controllers/contact_model.h"

using std::string;
using drake::multibody::MultibodyPlant;


class ContactController {
    public:
        ContactController(
            const std::string& model_path,
            ContactControllerParameters controller_params
        );

        ContactController(
            const std::string& robot_sdf_path,
            const std::unordered_map<string, string>& object_sdf_paths,
            const std::unordered_map<string, string>& package_paths,
            ContactControllerParameters controller_params
        );

        void SetFingerGeomNames(
            const std::vector<std::string>& finger_geom_name_list
        );

        Eigen::VectorXd Step(
            const Eigen::Ref<const Eigen::VectorXd>& q,
            const Eigen::Ref<const Eigen::VectorXd>& v
        );

        void Set_X0(
            const Eigen::Ref<const Eigen::VectorXd>& x0
        );

        void Set_Xref(
            const std::vector<Eigen::VectorXd>& x_ref
        );

        void Set_Fext(
            const Eigen::Ref<const Eigen::VectorXd>& f_ext
        );

        Eigen::MatrixXd Get_Ad () const { return A_discrete_; }
        Eigen::MatrixXd Get_Bd () const { return B_discrete_; }
        Eigen::MatrixXd Get_Ac () const { return A_continuous_; }
        Eigen::MatrixXd Get_Bc () const { return B_continuous_; }
        std::vector<Eigen::VectorXd> Get_phi_vn_vt () const { return contact_measurements_; }
        Eigen::VectorXd Get_tau_feedforward() const { return tau_feed_forward_; }

    private:

        void SetPlantPositionsAndVelocities(
            const Eigen::Ref<const Eigen::VectorXd>& q,
            const Eigen::Ref<const Eigen::VectorXd>& v
        );

        void CalcStiffAndJacobian(
            std::vector<Eigen::MatrixXd>* Kbar_all_ptr,
            std::vector<Eigen::MatrixXd>* J_all_ptr,
            std::vector<Eigen::MatrixXd>* G_all_ptr,
            std::vector<Eigen::MatrixXd>* Ke_all_ptr
        );

        void CalcActuationMatrix(
            const std::vector<Eigen::MatrixXd>& J_all,
            const std::vector<Eigen::MatrixXd>& G_all,
            const std::vector<Eigen::MatrixXd>& Ke_all,
            Eigen::MatrixXd* B_Fe_ptr
        );

        void CalcMPCProblemMatrices(
            const std::vector<Eigen::MatrixXd>& Kbar_all,
            const std::vector<Eigen::MatrixXd>& J_all,
            const Eigen::MatrixXd& B_Fe,
            Eigen::MatrixXd* A_mat, Eigen::MatrixXd* B_mat,
            Eigen::MatrixXd* Q_mat, Eigen::MatrixXd* R_mat,
            Eigen::MatrixXd* delR_mat
        );

        void ContinuousToDiscrete(
            const Eigen::Ref<const Eigen::MatrixXd>& Ac,
            const Eigen::Ref<const Eigen::MatrixXd>& Bc,
            double h,
            Eigen::MatrixXd* Ad_ptr,
            Eigen::MatrixXd* Bd_ptr
        );

        void SolveSparseMPC(
            const Eigen::Ref<const Eigen::MatrixXd>& A,
            const Eigen::Ref<const Eigen::MatrixXd>& B,
            const Eigen::Ref<const Eigen::MatrixXd>& Q,
            const Eigen::Ref<const Eigen::MatrixXd>& R,
            const Eigen::Ref<const Eigen::MatrixXd>& delR,
            const std::vector<Eigen::VectorXd>& x_ref,
            const Eigen::Ref<const Eigen::VectorXd>& x0,
            Eigen::VectorXd* u0
        );

        ContactControllerParameters params_;

        const drake::multibody::MultibodyPlant<double>& plant() const { return *plant_; }

        std::unique_ptr<drake::systems::Diagram<double>> diagram_;
        drake::multibody::MultibodyPlant<double>* plant_{nullptr};
        drake::geometry::SceneGraph<double>* scene_graph_{nullptr};

        std::unique_ptr<drake::systems::Context<double>> diagram_context_;
        drake::systems::Context<double>* plant_context_{nullptr};

        std::unique_ptr<CompliantContactModel> contact_model_;

        std::unique_ptr<drake::solvers::OsqpSolver> solver_osqp_;
        mutable drake::solvers::MathematicalProgramResult mp_result_;

        // const std::vector<std::string> finger_geom_name_list_{"allegro_hand_right::link_15_tip_collision_2",
        //                                                       "allegro_hand_right::link_3_tip_collision_1",
        //                                                       "allegro_hand_right::link_7_tip_collision_1",
        //                                                       "allegro_hand_right::link_11_tip_collision_1"};
        std::vector<std::string> finger_geom_name_list_;

        Eigen::VectorXd q_;
        Eigen::VectorXd v_;

        Eigen::MatrixXd Kpa_;
        Eigen::MatrixXd Kda_, Kda_inv_;

        Eigen::MatrixXd A_discrete_, B_discrete_;
        Eigen::MatrixXd A_continuous_, B_continuous_;

        std::vector<Eigen::VectorXd> x_ref_;
        Eigen::VectorXd x0_;
        Eigen::VectorXd fext_feed_forward_;

        std::vector<Eigen::VectorXd> contact_measurements_;
        Eigen::VectorXd tau_feed_forward_;

        int nc_{4};
        int nqa_{16};
        int nvu_{6};

        double kpa_{0.5};
        double kda_{1e-3};
};
