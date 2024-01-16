#ifndef PINOCCHIO_WITH_HPP_FCL
    #define PINOCCHIO_WITH_HPP_FCL
#endif

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
 
#include <iostream>

using namespace pinocchio;

int main(int argc, char** argv) {
    const std::string urdf_filename = \
        "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/robot_obj_combined/allegro_hand_description_right_ball_scene.urdf";

    const std::string robots_model_path = \
        "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description";

    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    std::cout << "model name: " << model.name << std::endl;
    std::cout << "model configurations: " << model.nq << std::endl;
    std::cout << "model velocities: " << model.nv << std::endl;

    // Build the data associated to the model
    Data data(model);
    
    // Load the geometries associated to model which are contained in the URDF file
    GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model, robots_model_path);
    
    // Add all possible collision pairs and remove the ones collected in the SRDF file
    geom_model.addAllCollisionPairs();
    
    // Build the data associated to the geom_model
    GeometryData geom_data(geom_model); // contained the intermediate computations, like the placement of all the geometries with respect to the world frame
    
    // Load the reference configuration of the robots (this one should be collision free)
    Model::ConfigVectorType q;
    q.resize(23);
    q << 0.03501504, 0.75276565, 0.74146232, 0.83261002,
         0.63256269, 1.02378254, 0.64089555, 0.82444782,
        -0.1438725, 0.74696812, 0.61908827, 0.70064279,
        -0.06922541, 0.78533142, 0.82942863, 0.90415436,
        0, 0, 10,
        1, 0, 0, 0;
    
    // And test all the collision pairs
    pinocchio::computeCollisions(model,data,geom_model,geom_data,q);
    
    // Print the status of all the collision pairs
    for(size_t k = 0; k < geom_model.collisionPairs.size(); ++k)
    {
        const CollisionPair & cp = geom_model.collisionPairs[k];
        const hpp::fcl::CollisionResult & cr = geom_data.collisionResults[k];
        
        std::cout << "collision pair: " << cp.first << " , " << cp.second << " - collision: ";
        std::cout << (cr.isCollision() ? "yes" : "no") << std::endl;
    }
    
    // If you want to stop as soon as a collision is encounter, just add false for the final default argument stopAtFirstCollision
    computeCollisions(model,data,geom_model,geom_data,q,true);
    
    // And if you to check only one collision pair, e.g. the third one, at the neutral element of the Configuration Space of the robot
    const PairIndex pair_id = 2;
    const Model::ConfigVectorType q_neutral = neutral(model);
    updateGeometryPlacements(model, data, geom_model, geom_data, q_neutral); // performs a forward kinematics over the whole kinematics model + update the placement of all the geometries contained inside geom_model
    computeCollision(geom_model, geom_data, pair_id);
    computeJointKinematicHessians(model, data, q);
    // pinocchio::Tensor<double, 3> hessian(6, model.nq, model.nq);
    auto hessian = getJointKinematicHessian(model, data, 10, pinocchio::WORLD);
    std::cout << "model.nq=" << model.nq << std::endl;
    std::cout << "model.nv=" << model.nv << std::endl;
    const Eigen::Tensor<double, 3>::Dimensions &d = hessian.dimensions();
    std::cout << "dim0=" << d[0] << ", dim2=" << d[1] << ", dim3=" << d[2] << std::endl;
    
    return 0;

}
