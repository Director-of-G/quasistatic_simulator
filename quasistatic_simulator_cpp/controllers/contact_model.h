#pragma once
#include <iostream>

#include <math.h>

#include <Eigen/Dense>


class CompliantContactModel {
    public:
        
        CompliantContactModel() {};

        void CalcStiffnessAndDampingMatrices(
            const double& dist, const double& vn, const Eigen::Vector3d& vt,
            Eigen::MatrixXd* stiffness, Eigen::MatrixXd* damping
        );

    private:

        double sigma_{0.001};

        double k_{500};

        double vd_{0.1};

        double vs_{0.1};

        double mu_{1.0};

};
