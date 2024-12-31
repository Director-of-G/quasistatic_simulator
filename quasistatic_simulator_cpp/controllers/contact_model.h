#pragma once
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include "controllers/contact_controller_params.h"


class CompliantContactModel {
    public:
        
        CompliantContactModel(struct ContactModelParameters params)
            :sigma_(params.sigma), k_(params.k), d_(params.d),
            vd_(params.vd), vs_(params.vs), mu_(params.mu), type_(params.type) {};

        void CalcStiffnessAndDampingMatrices(
            const double& dist, const double& vn, const Eigen::Vector3d& vt,
            Eigen::MatrixXd* stiffness, Eigen::MatrixXd* damping
        );

    private:

        double sigma_{0.001};

        double k_{500};

        double d_{100};

        double vd_{0.1};

        double vs_{0.1};

        double mu_{1.0};

        ContactModelType type_{ContactModelType::kConstant};

};
