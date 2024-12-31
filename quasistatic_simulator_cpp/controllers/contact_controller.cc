#include "controllers/contact_controller.h"
#include "drake/multibody/parsing/process_model_directives.h"


using std::string;
using Eigen::Vector3d;
using drake::geometry::AddCompliantHydroelasticProperties;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::multibody::Body;
using drake::multibody::BodyIndex;
using drake::geometry::Cylinder;
using drake::geometry::GeometryId;
using drake::geometry::GeometrySet;
using drake::geometry::ProximityProperties;
using drake::geometry::QueryObject;
using drake::geometry::Rgba;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::geometry::SignedDistancePair;
using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::SpatialVelocity;
using drake::multibody::CoulombFriction;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");


/*
    The following creates a multibody plant model for the low level.
    Only kinematics is needed in low level.
*/
void CreateMbpLowlevel(
    drake::systems::DiagramBuilder<double>* builder,
    drake::multibody::MultibodyPlant<double>** plant,
    drake::geometry::SceneGraph<double>** scene_graph,
    const string& robot_sdf_path,
    const std::unordered_map<string, string>& object_sdf_paths,
    const std::unordered_map<string, string>& package_paths,
    double time_step
) {
    // Combine plant, scene graph and builder
    std::tie(*plant, *scene_graph) =
      drake::multibody::AddMultibodyPlantSceneGraph(builder, time_step);
    (*plant)->set_name("MultiBodyPlant");
    (*scene_graph)->set_name("SceneGraph");
    auto parser = drake::multibody::Parser(*plant, *scene_graph);
    
    // Add package paths
    for (const auto& item : package_paths) {
        parser.package_map().Add(item.first, item.second);
    }

    // TODO(yongpeng): remember add the object before robot, so that q will be [qu;qa]
    // Add object
    for (const auto& item : object_sdf_paths) {
        const auto& name = item.first;
        const auto& path = item.second;
        parser.AddModelFromFile(path, name);
    }

    // Add dex hand
    drake::multibody::parsing::ProcessModelDirectives(
      drake::multibody::parsing::LoadModelDirectives(robot_sdf_path),
      *plant, nullptr, &parser);

    // Finalize
    (*plant)->Finalize();
}

void AddFreeFloatingSphereToPlant(MultibodyPlant<double>* plant_ptr) {

    const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);

    MultibodyPlant<double>& plant = *plant_ptr;

    // Add a free-floating ball
    ModelInstanceIndex ball_idx = plant.AddModelInstance("ball");

    // Not matters, since we only use the kinematics at this time
    const double mass = 0.05;
    const double radius = 0.06;

    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidSphere(radius));
    const RigidBody<double>& ball = plant.AddRigidBody("ball", ball_idx, I);

    plant.RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                  Sphere(radius), "ball_visual", blue);
    plant.RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                     Sphere(radius), "ball_collision",
                                     CoulombFriction<double>());

}

void AddRotationOnlySphereToPlant(MultibodyPlant<double>* plant_ptr) {
    const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);

    MultibodyPlant<double>& plant = *plant_ptr;
    
    // Add a free-floating ball
    ModelInstanceIndex ball_idx = plant.AddModelInstance("ball");

    // Not matters, since we only use the kinematics at this time
    const double mass = 0.05;
    const double radius = 0.06;

    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidSphere(radius));
    const RigidBody<double>& ball = plant.AddRigidBody("ball", ball_idx, I);

    plant.RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                  Sphere(radius), "ball_visual", blue);
    plant.RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                     Sphere(radius), "ball_collision",
                                     CoulombFriction<double>());

    // Add rotation joint
    auto sphere_mount_transform = drake::math::RigidTransform<double>(drake::Vector3<double>(-0.06, 0.0, 0.072));
    const drake::multibody::RevoluteJoint<double>& sphere_root_joint = plant.AddJoint<drake::multibody::RevoluteJoint>(
        "sphere_root_joint",
        plant.world_body(),
        sphere_mount_transform,
        ball,
        {},
        Eigen::Vector3d::UnitZ()
    );
}

void AddFreeRotationSphereToPlant(MultibodyPlant<double>* plant_ptr) {
    const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);

    MultibodyPlant<double>& plant = *plant_ptr;
    
    // Add a free-floating ball
    ModelInstanceIndex ball_idx = plant.AddModelInstance("ball");

    // Not matters, since we only use the kinematics at this time
    const double mass = 0.05;
    const double radius = 0.06;

    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidSphere(radius));
    const RigidBody<double>& ball = plant.AddRigidBody("ball", ball_idx, I);

    plant.RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                  Sphere(radius), "ball_visual", blue);
    plant.RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                     Sphere(radius), "ball_collision",
                                     CoulombFriction<double>());

    // Add rotation joint
    auto sphere_mount_transform = drake::math::RigidTransform<double>(drake::Vector3<double>(-0.06, 0.0, 0.072));
    const drake::multibody::BallRpyJoint<double>& sphere_root_joint = plant.AddJoint<drake::multibody::BallRpyJoint>(
        "sphere_root_joint",
        plant.world_body(),
        sphere_mount_transform,
        ball,
        {}
    );
}

void AddRotationOnlyValveToPlant(MultibodyPlant<double>* plant_ptr) {
    const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);

    MultibodyPlant<double>& plant = *plant_ptr;
    
    // Add a free-floating ball
    ModelInstanceIndex valve_idx = plant.AddModelInstance("valve");

    // Not matters, since we only use the kinematics at this time
    const double mass = 0.05;
    Eigen::Vector3d size(0.04, 0.20, 0.04);

    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidBox(size[0], size[1], size[2]));
    const RigidBody<double>& valve = plant.AddRigidBody("valve", valve_idx, I);

    plant.RegisterVisualGeometry(valve, RigidTransformd::Identity(),
                                  Box(size[0], size[1], size[2]), "valve_visual", blue);
    plant.RegisterCollisionGeometry(valve, RigidTransformd::Identity(),
                                     Box(size[0], size[1], size[2]), "valve_collision",
                                     CoulombFriction<double>());

    // Add rotation joint
    auto valve_mount_transform = drake::math::RigidTransform<double>(drake::Vector3<double>(-0.08, 0.0, 0.11));
    const drake::multibody::RevoluteJoint<double>& valve_root_joint = plant.AddJoint<drake::multibody::RevoluteJoint>(
        "valve_root_joint",
        plant.world_body(),
        valve_mount_transform,
        valve,
        {},
        Eigen::Vector3d::UnitZ()
    );
}

void AddEnvironmentsToPlant(MultibodyPlant<double>* plant_ptr) {
    const drake::Vector4<double> brown(0.871, 0.722, 0.529, 0.8);

    MultibodyPlant<double>& plant = *plant_ptr;

    // Add table
    ModelInstanceIndex table_idx = plant.AddModelInstance("table");

    const double mass = 1.0;
    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidBox(0.5, 0.5, 0.0235));

    const RigidBody<double>& table = plant.AddRigidBody("table_link", table_idx, I);

    plant.RegisterVisualGeometry(table, RigidTransformd::Identity(),
                                  Box(Vector3d(0.5, 0.5, 0.0235)), "visual", brown);
    plant.RegisterCollisionGeometry(table, RigidTransformd::Identity(),
                                     Box(Vector3d(0.5, 0.5, 0.0235)), "collision",
                                     CoulombFriction<double>());

    RigidTransformd X_table;
    const drake::multibody::WeldJoint<double>& table_root_joint = plant.AddJoint<drake::multibody::WeldJoint>(
        "table_weld_joint",
        plant.world_body(),
        {},
        table,
        {},
        X_table
    );
}

void ExcludeRobotCollisionWithEnvs(
    MultibodyPlant<double>* plant_ptr,
    SceneGraph<double>* scene_graph_ptr
) {
    GeometrySet robot_geom_set, env_geom_set;

    // Get the geometry set of robot
    int counter = 0;
    ModelInstanceIndex robot = plant_ptr->GetModelInstanceByName("allegro_hand_right");
    for (BodyIndex body_index : plant_ptr->GetBodyIndices(robot)) {
        const auto& body = plant_ptr->get_body(body_index);

        for (const auto& geometry_id : plant_ptr->GetCollisionGeometriesForBody(body)) {
            robot_geom_set.Add(geometry_id);
            counter++;
        }
    }

    // Get the geometry set of environment
    counter = 0;
    ModelInstanceIndex env = plant_ptr->GetModelInstanceByName("table");
    for (BodyIndex body_index : plant_ptr->GetBodyIndices(env)) {
        const auto& body = plant_ptr->get_body(body_index);

        for (const auto& geometry_id : plant_ptr->GetCollisionGeometriesForBody(body)) {
            env_geom_set.Add(geometry_id);
            counter++;
        }
    }

    // Exclude collisions
    drake::geometry::CollisionFilterManager manager = scene_graph_ptr->collision_filter_manager();

    manager.Apply(
        drake::geometry::CollisionFilterDeclaration()
            .ExcludeBetween(robot_geom_set, env_geom_set)
    );
}

ContactController::ContactController(
    const std::string& model_path,
    ContactControllerParameters controller_params
): params_(std::move(controller_params)),
   solver_osqp_(std::make_unique<drake::solvers::OsqpSolver>()) {
    // Set up the system diagram for the simulator
    drake::systems::DiagramBuilder<double> builder;
    drake::multibody::MultibodyPlantConfig config;
    config.time_step = 0.001;
    auto [plant, scene_graph] =
      drake::multibody::AddMultibodyPlant(config, &builder);

    // Create plant model
    const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);
    const drake::Vector4<double> black(0.0, 0.0, 0.0, 1.0);

    // // Define gravity (so we can turn the hand upside down)
    // if (FLAGS_upside_down) {
    //   plant->mutable_gravity_field().set_gravity_vector(Vector3d(0, 0, 9.81));
    // }

    // Add manipuland and environment
    switch (controller_params.object_geom)
    {
        case ObjectGeom::kZRotSphere:
            {
                AddRotationOnlySphereToPlant(&plant);
            }
            break;
        case ObjectGeom::kFreeRotSphere:
            {
                AddFreeRotationSphereToPlant(&plant);
            }
            break;
        case ObjectGeom::kFreeMoveSphere:
            {
                AddFreeFloatingSphereToPlant(&plant);
                AddEnvironmentsToPlant(&plant);
            }
            break;
        case ObjectGeom::kCapsuleValve:
            {
                AddRotationOnlyValveToPlant(&plant);
            }
            break;
        default:
            break;
    }

    // Add a model of the hand
    Parser(&plant).AddModels(model_path);

    Vector3d hand_trans(controller_params.hand_base_trans);
    Vector3d hand_rot(controller_params.hand_base_rot);
    RigidTransformd X_hand(
        RollPitchYawd(hand_rot),
        hand_trans
    );
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("hand_root"),
                      X_hand);

    // Disable gravity
    plant.mutable_gravity_field().set_gravity_vector(Vector3d(0, 0, 0));

    // Finalize and build
    plant.Finalize();

    if (controller_params.object_geom == ObjectGeom::kFreeMoveSphere) {
        ExcludeRobotCollisionWithEnvs(&plant, &scene_graph);
    }

    diagram_ = builder.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ = &(diagram_->GetMutableSubsystemContext(plant, diagram_context_.get()));

    plant_ = &plant;
    scene_graph_ = &scene_graph;

    // others
    contact_model_ = std::make_unique<CompliantContactModel>(controller_params.model_params);
    Kpa_ = 0.5 * Eigen::MatrixXd::Identity(16, 16);
    Kda_ = 1e-3 * Eigen::MatrixXd::Identity(16, 16);
    Kda_inv_ = Kda_.inverse();
    nc_ = 4;
    nqa_ = plant_->num_actuated_dofs();
    nvu_ = plant_->num_velocities() - nqa_;

    tau_feed_forward_.resize(nqa_);
    fext_feed_forward_.resize(3 * nc_);
    fext_feed_forward_.setZero();
}

ContactController::ContactController(
    const std::string& robot_sdf_path,
    const std::unordered_map<string, string>& object_sdf_paths,
    const std::unordered_map<string, string>& package_paths,
    ContactControllerParameters controller_params
): params_(std::move(controller_params)),
   solver_osqp_(std::make_unique<drake::solvers::OsqpSolver>()) {
    // Set up the system diagram for the simulator
    drake::systems::DiagramBuilder<double> builder;
    drake::multibody::MultibodyPlantConfig config;
    config.time_step = 0.001;
    
    // Create Mbp
    CreateMbpLowlevel(&builder, &plant_, &scene_graph_, robot_sdf_path, object_sdf_paths, package_paths, config.time_step);

    // Build
    if (controller_params.object_geom == ObjectGeom::kFreeMoveSphere) {
        ExcludeRobotCollisionWithEnvs(plant_, scene_graph_);
    }

    diagram_ = builder.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ = &(diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get()));

    // others
    contact_model_ = std::make_unique<CompliantContactModel>(controller_params.model_params);
    Kpa_ = 0.5 * Eigen::MatrixXd::Identity(16, 16);
    Kda_ = 1e-3 * Eigen::MatrixXd::Identity(16, 16);
    Kda_inv_ = Kda_.inverse();
    nc_ = 4;
    nqa_ = plant_->num_actuated_dofs();
    nvu_ = plant_->num_velocities() - nqa_;

    tau_feed_forward_.resize(nqa_);
    fext_feed_forward_.resize(3 * nc_);
    fext_feed_forward_.setZero();
}

void ContactController::SetFingerGeomNames(
    const std::vector<std::string>& finger_geom_name_list
) {
    finger_geom_name_list_ = std::move(finger_geom_name_list);
}

Eigen::VectorXd ContactController::Step(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& v
) {
    if (finger_geom_name_list_.empty()) {
        throw std::runtime_error("Finger geom names are not set.");
    }

    SetPlantPositionsAndVelocities(q, v);

    std::vector<Eigen::MatrixXd> Kbar_list, J_list;
    std::vector<Eigen::MatrixXd> G_list, Ke_list;
    CalcStiffAndJacobian(&Kbar_list, &J_list, &G_list, &Ke_list);

    J_all = J_list;
    G_all = G_list;

    Eigen::MatrixXd B_Fe;
    B_Fe.setZero(3*nc_, nqa_);
    if (params_.enable_multi_contact) {
        CalcActuationMatrix(J_list, G_list, Ke_list, &B_Fe);
    }

    Eigen::MatrixXd Q, R, delR;
    CalcMPCProblemMatrices(Kbar_list, J_list, B_Fe, &A_continuous_, &B_continuous_, &Q, &R, &delR);

    // for (const auto& item : Kbar_list) {
    //     std::cout << "Kbar " << item.format(CleanFmt) << std::endl;
    // }

    // for (const auto& item : J_list) {
    //     std::cout << "J " << item.format(CleanFmt) << std::endl;
    // }

    // std::cout << "B_Fe " << B_Fe.format(CleanFmt) << std::endl;
    
    // std::cout << "continuous A mat " << A_continuous_.format(CleanFmt) << std::endl;
    // std::cout << "continuous B mat " << B_continuous_.format(CleanFmt) << std::endl;

    ContinuousToDiscrete(A_continuous_, B_continuous_, params_.time_step, &A_discrete_, &B_discrete_);

    // std::cout << "discrete A mat " << A_discrete_.format(CleanFmt) << std::endl;
    // std::cout << "discrete B mat " << B_discrete_.format(CleanFmt) << std::endl;

    Eigen::VectorXd u0;
    SolveSparseMPC(A_discrete_, B_discrete_, Q, R, delR, x_ref_, x0_, &u0);

    return u0;
}

void ContactController::Set_X0(
    const Eigen::Ref<const Eigen::VectorXd>& x0
) {
    x0_ = x0;
}

void ContactController::Set_Xref(
    const std::vector<Eigen::VectorXd>& x_ref
) {
    x_ref_ = x_ref;
}

void ContactController::Set_Fext(
    const Eigen::Ref<const Eigen::VectorXd>& f_ext
) {
    DRAKE_ASSERT(fext_feed_forward_.size() == f_ext.size());
    fext_feed_forward_ = f_ext;
}

void ContactController::SolveSparseMPC(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& delR,
    const std::vector<Eigen::VectorXd>& x_ref,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    Eigen::VectorXd* u0
) {
    int n_steps = params_.horizon_length;
    int n_x = B.rows();
    int n_u = B.cols();

    Eigen::VectorXd u_ref;
    u_ref.setZero(n_u);

    drake::solvers::MathematicalProgram prog;
    std::vector<drake::solvers::VectorXDecisionVariable> x_list;
    std::vector<drake::solvers::VectorXDecisionVariable> u_list;
    for (int i = 0; i < n_steps; i++) {
        auto x = prog.NewContinuousVariables(n_x);
        auto u = prog.NewContinuousVariables(n_u);

        // dynamic constraints
        // x(k+1) = A*x(k)+B*u(k)
        if (i == 0) {
            Eigen::VectorXd x_prev = x0;
            prog.AddLinearEqualityConstraint(x - A * x0 - B * u, Eigen::VectorXd::Constant(n_x, 0));
        }
        else {
            auto x_prev = x_list.back();
            prog.AddLinearEqualityConstraint(x - A * x_prev - B * u, Eigen::VectorXd::Constant(n_x, 0));
        }

        prog.AddQuadraticErrorCost(Q, x_ref.at(i), x);
        prog.AddQuadraticErrorCost(R, u_ref, u);

        x_list.push_back(x);
        u_list.push_back(u);
    }

    auto solver = solver_osqp_.get();
    solver->Solve(prog, {}, {}, &mp_result_);
    if (!mp_result_.is_success()) {
        throw std::runtime_error("QP cannot be solved.");
    }
    *u0 = mp_result_.GetSolution(u_list.front());

}

void ContactController::SetPlantPositionsAndVelocities(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& v
) {
    Eigen::VectorXd q_v(q.size()+v.size());
    q_v << q, v;
    plant().SetPositionsAndVelocities(plant_context_, q_v);
}

/*
    TODO: assume A is object and B is robot
*/
void ContactController::CalcStiffAndJacobian(
    std::vector<Eigen::MatrixXd>* Kbar_all_ptr,
    std::vector<Eigen::MatrixXd>* J_all_ptr,
    std::vector<Eigen::MatrixXd>* G_all_ptr,
    std::vector<Eigen::MatrixXd>* Ke_all_ptr
) {
    std::vector<Eigen::MatrixXd>& Kbar_all = *Kbar_all_ptr;
    std::vector<Eigen::MatrixXd>& J_all = *J_all_ptr;   // robot Jacobian (actuated)
    std::vector<Eigen::MatrixXd>& G_all = *G_all_ptr;   // grasping matrix (unactuated)
    std::vector<Eigen::MatrixXd>& Ke_all = *Ke_all_ptr; // contact stiffness matrix

    Kbar_all.clear();
    J_all.clear();
    G_all.clear();
    Ke_all.clear();
    contact_measurements_.clear();

    for (int i_c=0; i_c<nc_; i_c++) {
        Kbar_all.emplace_back(3, 3);
        J_all.emplace_back(3, nqa_);
        G_all.emplace_back(6, nvu_);
        Ke_all.emplace_back(3, 3);
        contact_measurements_.emplace_back(3);
        contact_measurements_.back().setZero();
    }

    tau_feed_forward_.setZero();

    const drake::geometry::QueryObject<double>& query_object =
      plant()
          .get_geometry_query_input_port()
          .template Eval<drake::geometry::QueryObject<double>>(*plant_context_);
    const drake::geometry::SceneGraphInspector<double>& inspector =
        query_object.inspector();
    // TODO(yongpeng): the default threshold is 0.1, need to be tuned
    const std::vector<SignedDistancePair<double>>& signed_distance_pairs =
        query_object.ComputeSignedDistancePairwiseClosestPoints(1.0);

    // std::cout << "sdp size: " << signed_distance_pairs.size() << std::endl;
    for (const SignedDistancePair<double>& pair : signed_distance_pairs) {
        int index_ = -1;

        // Normal outwards from B (robot).
        const drake::Vector3<double> nhat = pair.nhat_BA_W;
        const drake::math::RotationMatrixd R_WC = 
            drake::math::RotationMatrixd::MakeFromOneVector(nhat, 2);

        // Get geometry and transformation data for the witness points
        const GeometryId geometryA_id = pair.id_A;
        const GeometryId geometryB_id = pair.id_B;

        std::string geomA_name = inspector.GetName(geometryA_id);
        std::string geomB_name = inspector.GetName(geometryB_id);
        // std::cout << "signed distance: " << pair.distance << std::endl;
        if (std::find(finger_geom_name_list_.begin(), finger_geom_name_list_.end(), geomA_name) != finger_geom_name_list_.end()) {
            index_ = std::find(finger_geom_name_list_.begin(), finger_geom_name_list_.end(), geomA_name) - finger_geom_name_list_.begin();
        }
        else if (std::find(finger_geom_name_list_.begin(), finger_geom_name_list_.end(), geomB_name) != finger_geom_name_list_.end()) {
            index_ = std::find(finger_geom_name_list_.begin(), finger_geom_name_list_.end(), geomB_name) - finger_geom_name_list_.begin();
        }
        else {
            continue;
        }
        // std::cout << "name: A is " << geomA_name << ", and B is " << inspector.GetName(geometryB_id) << " index: " << index_ << std::endl;

        const Body<double>& bodyA =
            *(plant().GetBodyFromFrameId(inspector.GetFrameId(geometryA_id)));
        const Body<double>& bodyB =
            *(plant().GetBodyFromFrameId(inspector.GetFrameId(geometryB_id)));

        // Body poses in world.
        const drake::math::RigidTransform<double>& X_WA =
            plant().EvalBodyPoseInWorld(*plant_context_, bodyA);
        const drake::math::RigidTransform<double>& X_WB =
            plant().EvalBodyPoseInWorld(*plant_context_, bodyB);

        // Geometry poses in body frames.
        const drake::math::RigidTransform<double> X_AGa =
            inspector.GetPoseInFrame(geometryA_id).template cast<double>();
        const drake::math::RigidTransform<double> X_BGb =
            inspector.GetPoseInFrame(geometryB_id).template cast<double>();

        // Position of the witness points in the world frame.
        const auto& p_GaCa_Ga = pair.p_ACa;
        const RigidTransform<double> X_WGa = X_WA * X_AGa;
        const drake::Vector3<double> p_WCa_W = X_WGa * p_GaCa_Ga;
        const auto& p_GbCb_Gb = pair.p_BCb;
        const RigidTransform<double> X_WGb = X_WB * X_BGb;
        const drake::Vector3<double> p_WCb_W = X_WGb * p_GbCb_Gb;

        // We define the (common, unique) contact point C as the midpoint between
        // witness points Ca and Cb.
        const drake::Vector3<double> p_WC = 0.5 * (p_WCa_W + p_WCb_W);

        // Shift vectors.
        const drake::Vector3<double> p_AC_W = p_WC - X_WA.translation();
        const drake::Vector3<double> p_BC_W = p_WC - X_WB.translation();

        // Velocities.
        const SpatialVelocity<double>& V_WA =
            plant().EvalBodySpatialVelocityInWorld(*plant_context_, bodyA);
        const SpatialVelocity<double>& V_WB =
            plant().EvalBodySpatialVelocityInWorld(*plant_context_, bodyB);
        const SpatialVelocity<double> V_WAc = V_WA.Shift(p_AC_W);
        const SpatialVelocity<double> V_WBc = V_WB.Shift(p_BC_W);

        // Relative contact velocity (object - robot).
        const drake::Vector3<double> v_BcAc_W =
            V_WAc.translational() - V_WBc.translational();

        // Split into normal and tangential components.
        const double vn = nhat.dot(v_BcAc_W);
        const drake::Vector3<double> vt = v_BcAc_W - vn * nhat;

        // Contact Jacobian (robot side)
        // J_all.emplace_back(3, nqa_);
        Eigen::MatrixXd& Ja = J_all.at(index_);
        Eigen::MatrixXd Jqd_vWBc_W;
        Jqd_vWBc_W.setZero(3, plant().num_velocities());
        plant().CalcJacobianTranslationalVelocity(
            *plant_context_,
            drake::multibody::JacobianWrtVariable::kV,
            bodyB.body_frame(),
            X_WB.rotation().inverse() * p_BC_W,
            plant().world_frame(),
            plant().world_frame(),
            &Jqd_vWBc_W
        );
        Ja = Jqd_vWBc_W.rightCols(nqa_);

        // Grasping Matrix (object side)
        Eigen::MatrixXd& Gu = G_all.at(index_);
        Eigen::MatrixXd Jqd_vWAc_W;;
        Jqd_vWAc_W.setZero(6, plant().num_velocities());
        plant().CalcJacobianSpatialVelocity(
            *plant_context_,
            drake::multibody::JacobianWrtVariable::kV,
            bodyA.body_frame(),
            X_WA.rotation().inverse() * p_AC_W,
            plant().world_frame(),
            plant().world_frame(),
            &Jqd_vWAc_W
        );
        Gu = Jqd_vWAc_W.leftCols(nvu_);

        // contact stiffness and damping
        // Kbar_all.emplace_back(3, 3);
        Eigen::MatrixXd& Kbar = Kbar_all.at(index_);
        Eigen::MatrixXd Ke_, De_;
        
        contact_model_->CalcStiffnessAndDampingMatrices(
            pair.distance, vn, vt, &Ke_, &De_
        );
        Ke_ = R_WC.matrix() * Ke_ * R_WC.matrix().transpose();
        Ke_all.at(index_) = Ke_;
        Eigen::MatrixXd Kp_inv_i = Ja * Kpa_.inverse() * Ja.transpose();
        Kbar = (Eigen::MatrixXd::Identity(3, 3) + Ke_ * Kp_inv_i).inverse() * Ke_;

        // contact measurements
        Eigen::VectorXd& contact_meas = contact_measurements_.at(index_);
        contact_meas << pair.distance, vn, vt.norm();

        if (params_.calc_torque_feedforward) {
            Eigen::VectorXd f_ext_i = fext_feed_forward_(Eigen::seqN(3*index_, 3));
            tau_feed_forward_ += Ja.transpose() * f_ext_i;
        }

    }
}

void ContactController::CalcActuationMatrix(
    const std::vector<Eigen::MatrixXd>& J_all,
    const std::vector<Eigen::MatrixXd>& G_all,
    const std::vector<Eigen::MatrixXd>& Ke_all,
    Eigen::MatrixXd* B_Fe_ptr
) {
    Eigen::MatrixXd Ko, Ko_inv;
    Ko.setZero(nvu_, nvu_);
    Ko_inv.setZero(nvu_, nvu_);

    for (int i_c=0; i_c<nc_; i_c++) {
        Eigen::MatrixXd G_trans_i = G_all.at(i_c).topRows(3);
        Ko += G_trans_i.transpose() * Ke_all.at(i_c) * G_trans_i;
    }
    // TODO(yongpeng) Ko is actually singular, how to deal with this?
    // Ko_inv = Ko.inverse();
    // TODO(yongpeng) Currently set Ko^{-1} manually
    Ko_inv.setIdentity();
    Ko_inv *= 1e-2;
    // std::cout << "Calculated Ko=" << Ko.format(CleanFmt) << std::endl;
    // std::cout << "Calculated Ko_inv=" << Ko_inv.format(CleanFmt) << std::endl;

    Eigen::MatrixXd G_stack, K_bar, J_stack;
    G_stack.setZero(3*nc_, nvu_);
    K_bar.setZero(3*nc_, nqa_);
    J_stack.setZero(3*nc_, nqa_);

    for (int i_c=0; i_c<nc_; i_c++) {
        G_stack.block(3*i_c, 0, 3, nvu_) = G_all.at(i_c).topRows(3);
        K_bar.block(3*i_c, 0, 3, nqa_) = Ke_all.at(i_c) * J_all.at(i_c);
        J_stack.block(3*i_c, 0, 3, nqa_) = J_all.at(i_c);
    }

    // TODO: maybe B_Fe is not correct
    // Eigen::MatrixXd B_Fe = Eigen::MatrixXd::Identity(3*nc_, 3*nc_) + (1./kpa_) * K_bar * J_stack.transpose() + G_stack * Ko_inv * G_stack.transpose();
    Eigen::MatrixXd B_Fe = -Eigen::MatrixXd::Identity(3*nc_, 3*nc_) + (1./kpa_) * K_bar * J_stack.transpose();
    B_Fe = B_Fe.inverse() * K_bar;
    *B_Fe_ptr = B_Fe;
}

void ContactController::CalcMPCProblemMatrices(
    const std::vector<Eigen::MatrixXd>& Kbar_all,
    const std::vector<Eigen::MatrixXd>& J_all,
    const Eigen::MatrixXd& B_Fe,
    Eigen::MatrixXd* A_mat_ptr, Eigen::MatrixXd* B_mat_ptr,
    Eigen::MatrixXd* Q_mat_ptr, Eigen::MatrixXd* R_mat_ptr,
    Eigen::MatrixXd* delR_mat_ptr
) {
    int n_x = nqa_ * 2 + nc_ * 3;
    int n_u = nqa_;

    Eigen::MatrixXd& A_mat = *A_mat_ptr;
    Eigen::MatrixXd& B_mat = *B_mat_ptr;
    Eigen::MatrixXd& Q_mat = *Q_mat_ptr;
    Eigen::MatrixXd& R_mat = *R_mat_ptr;
    Eigen::MatrixXd& delR_mat = *delR_mat_ptr;

    A_mat.setZero(n_x, n_x);
    B_mat.setZero(n_x, n_u);
    Q_mat.setZero(n_x, n_x);
    R_mat.setZero(n_u, n_u);
    delR_mat.setZero(n_u, n_u);

    Eigen::MatrixXd Kdinv_x_Kp_ = Kda_inv_ * Kpa_;
    A_mat.block(0, 0, nqa_, nqa_) = -Kdinv_x_Kp_;
    A_mat.block(0, nqa_, nqa_, nqa_) = Kdinv_x_Kp_;

    B_mat.block(0, 0, nqa_, nqa_) = Eigen::MatrixXd::Identity(nqa_, nqa_);
    B_mat.block(nqa_, 0, nqa_, nqa_) = Eigen::MatrixXd::Identity(nqa_, nqa_);

    for (int i_c = 0; i_c < nc_; i_c++) {
        A_mat.block(0, 2*nqa_+3*i_c, nqa_, 3) = -Kda_inv_ * J_all.at(i_c).transpose();
    }

    if (!params_.enable_multi_contact) {
        // treat each contact separately
        for (int i_c = 0; i_c < nc_; i_c++) {
            B_mat.block(2*nqa_+3*i_c, 0, 3, n_u) = Kbar_all.at(i_c) * J_all.at(i_c);
        }
    } else {
        B_mat.block(2*nqa_, 0, 3*nc_, n_u) = B_Fe;
    }

    Q_mat.block(0, 0, nqa_, nqa_) = params_.weight_q * Eigen::MatrixXd::Identity(nqa_, nqa_);
    Q_mat.block(nqa_, nqa_, nqa_, nqa_) = params_.weight_dq * Eigen::MatrixXd::Identity(nqa_, nqa_);
    Q_mat.block(2*nqa_, 2*nqa_, 3*nc_, 3*nc_) = params_.weight_f * Eigen::MatrixXd::Identity(3*nc_, 3*nc_);

    R_mat = params_.weight_u * Eigen::MatrixXd::Identity(n_u, n_u);
    delR_mat = params_.weight_du * Eigen::MatrixXd::Identity(n_u, n_u);

}

void ContactController::ContinuousToDiscrete(
    const Eigen::Ref<const Eigen::MatrixXd>& Ac,
    const Eigen::Ref<const Eigen::MatrixXd>& Bc,
    double h,
    Eigen::MatrixXd* Ad_ptr,
    Eigen::MatrixXd* Bd_ptr
) {
    int n_x = Bc.rows();
    int n_u = Bc.cols();

    Eigen::MatrixXd& Ad = *Ad_ptr;
    Eigen::MatrixXd& Bd = *Bd_ptr;
    Ad.setZero(n_x, n_x);
    Bd.setZero(n_x, n_u);

    DRAKE_ASSERT(Ac.rows() == n_x && Ac.cols() == n_x);

    Eigen::Matrix<double, 60, 60> AB_, AB_expm_;
    AB_.setZero();

    AB_.block(0, 0, n_x, n_x) = Ac;
    AB_.block(0, n_x, n_x, n_u) = Bc;
    AB_ *= h;

    AB_expm_ = AB_.exp();

    Ad = AB_expm_.block(0, 0, n_x, n_x);
    Bd = AB_expm_.block(0, n_x, n_x, n_u);
}
