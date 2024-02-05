#include "controllers/contact_model.h"


void CompliantContactModel::CalcStiffnessAndDampingMatrices(
    const double& dist, const double& vn, const Eigen::Vector3d& vt,
    Eigen::MatrixXd* stiffness, Eigen::MatrixXd* damping
) {
    stiffness->setZero(3, 3);
    damping->setZero(3, 3);

    double compliant_fn = sigma_ * k_ * std::log(1 + std::exp(-dist/sigma_));
    double vn_vd = vn / vd_;
    double dissipation_factor = 0;
    if (vn_vd < 0) {
        dissipation_factor = 1 - vn_vd;
    } else if (vn_vd < 2) {
        dissipation_factor = 0.25 * (vn_vd - 2) * (vn_vd - 2);
    }

    double stiff_n = std::abs(-k_ * (1 - 1/(1+std::exp(-dist/sigma_))) * dissipation_factor);

    double damping_n = 0;
    if (vn_vd < 0) {
        dissipation_factor = -1;
    } else if (vn_vd < 2) {
        dissipation_factor = 0.5 * (vn_vd - 2);
    }
    damping_n = std::abs(damping_n / (vd_ * compliant_fn));

    double damping_t = std::pow(vs_ * vs_ + vt.squaredNorm(), 1.5);
    damping_t = std::abs(-mu_ * compliant_fn * dissipation_factor * vs_ * vs_ / damping_t);

    stiffness->diagonal() << 0.0, 0.0, stiff_n;
    damping->diagonal() << damping_t, damping_t, damping_n;

}
