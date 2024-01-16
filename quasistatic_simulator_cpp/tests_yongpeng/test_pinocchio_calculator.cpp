#include "qsim/pinocchio_calculator.h"

#include <Eigen/Dense>
#include <iostream>

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
Eigen::IOFormat HeavyFmt(4, 0, ", ", ";\n", "[", "]", "[", "]");


int main(char** argv, int argc) {
    const std::string robot_urdf_filename = \
            "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/robot_single/allegro_hand_description_right_v2.urdf";

    const std::string object_urdf_filename = \
        "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/object_single/sphere_r0.06m.urdf";

    PinocchioCalculator pin(
        robot_urdf_filename,
        object_urdf_filename
    );

    std::cout << "PinocchioCalculator is created." << std::endl;
    std::cout << "The pinocchio model has " << pin.model.nq << " configurations." << std::endl;

    pinocchio::Model::ConfigVectorType q;
    q.resize(23);
    q << 0.03501504, 0.75276565, 0.74146232, 0.83261002,
         0.63256269, 1.02378254, 0.64089555, 0.82444782,
        -0.1438725, 0.74696812, 0.61908827, 0.70064279,
        -0.06922541, 0.78533142, 0.82942863, 0.90415436,
        0, 0, 10,
        1, 0, 0, 0;

    pin.UpdateModelConfiguration(q);
    pin.UpdateHessiansAndJacobians();

    // Eigen::Tensor<double, 3, 0, Eigen::DenseIndex> kin_H(6, pin.model.nv, pin.model.nv);
    Eigen::Tensor<double, 3, Eigen::ColMajor, Eigen::DenseIndex> kin_H(2, 2, 2);

    Eigen::MatrixXd J;

    kin_H = pin.GetContactKinematicHessian(
        Eigen::Matrix4d::Identity(),
        Eigen::Vector3d::Zero(),
        "link_15"
    );

    J = pin.GetContactJacobian(
        Eigen::Matrix4d::Identity(),
        Eigen::Vector3d::Zero(),
        "link_15"
    );

    // kin_H.resize(6, pin.model.nv * pin.model.nv, 1);

    const Eigen::Tensor<double, 3>::Dimensions &d = kin_H.dimensions();
    std::cout << "kin_H dim0=" << d[0] << " dim1=" << d[1] << " dim2=" << d[2] << std::endl;

    Eigen::MatrixXd matrix(d[0], d[1]*d[2]);

    // 复制数据
    for (int i = 0; i < d[0]; ++i) {
        for (int j = 0; j < d[1]; ++j) {
            for (int k=0; k<d[2]; ++k) {
                matrix(i, j*d[1]+k) = kin_H(i, j, k);
                std::cout << kin_H(i, j, k) << std::endl;
            }
        }
    }

    std::cout << "matrix: " << std::endl;
    std::cout << matrix.format(HeavyFmt);

    std::cout << "J: " << std::endl;
    std::cout << J.format(HeavyFmt);

}
