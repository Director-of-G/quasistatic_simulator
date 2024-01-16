#ifndef PINOCCHIO_WITH_HPP_FCL
    #define PINOCCHIO_WITH_HPP_FCL
#endif

#include <math.h>
#include <string>
#include <Eigen/Dense>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include <pinocchio/algorithm/jacobian.hpp>
#include <iostream> 

using std::string;
using Eigen::VectorXd;

int main(int argc, char **argv)
{
    pinocchio::SE3 se3_I = pinocchio::SE3::Identity();

    string robot_urdf_path="/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/robot_single/allegro_hand_description_right.urdf";
    string object_urdf_path="/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/object_single/sphere_r0.06m.urdf";

    pinocchio::Model model, object_model;
    pinocchio::GeometryModel collision_model, object_collision_model;
    pinocchio::GeometryModel visual_model, object_visual_model;

    // pinocchio::urdf::buildModel(robot_urdf_path, pinocchio::JointModelPlanar(), model);
    // pinocchio::urdf::buildModel(object_urdf_path, pinocchio::JointModelPlanar(), object_model);

    pinocchio::urdf::buildModel(robot_urdf_path, model);
    pinocchio::urdf::buildModel(object_urdf_path, object_model);
    
    pinocchio::urdf::buildGeom(model, robot_urdf_path, pinocchio::COLLISION, collision_model);
    pinocchio::urdf::buildGeom(object_model, object_urdf_path, pinocchio::COLLISION, object_collision_model);

    // append manipuland to model
    pinocchio::JointIndex object_joint_idx = model.addJoint(0, pinocchio::JointModelFreeFlyer(), se3_I, "object_root_joint");
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

    pinocchio::Model::ConfigVectorType q0;
    q0.resize(model.nq);
    q0 << 0.03501504, 0.75276565, 0.74146232, 0.83261002,
         0.63256269, 1.02378254, 0.64089555, 0.82444782,
        -0.1438725, 0.74696812, 0.61908827, 0.70064279,
        -0.06922541, 0.78533142, 0.82942863, 0.90415436,
        0.016, 0.001, 0.071,
        1, 0, 0, 0;

    pinocchio::forwardKinematics(model, data, q0);
    pinocchio::computeCollisions(model, data, collision_model, geom_data, q0);

    std::cout << "There are " << geom_data.collisionResults.size() << " collision pairs." << std::endl;
    std::cout << "Distance: " << geom_data.collisionResults.at(0).distance_lower_bound << std::endl;

    pinocchio::computeJointKinematicHessians(model, data, q0);
    // pinocchio::Tensor<double, 3> hessian(6, model.nq, model.nq);
    auto hessian = pinocchio::getJointKinematicHessian(model, data, 10, pinocchio::WORLD);

    return 0;
}
