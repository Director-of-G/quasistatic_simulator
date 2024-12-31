#include "controllers/contact_model.h"


void CompliantContactModel::CalcStiffnessAndDampingMatrices(
    const double& dist, const double& vn, const Eigen::Vector3d& vt,
    Eigen::MatrixXd* stiffness, Eigen::MatrixXd* damping
) {
    stiffness->setZero(3, 3);
    damping->setZero(3, 3);

    double stiff_n, stiff_t, damping_n, damping_t;

    if (type_ == ContactModelType::kVariant) {
        double compliant_fn = sigma_ * k_ * std::log(1 + std::exp(-dist/sigma_));
        double vn_vd = vn / vd_;
        double dissipation_factor = 0;
        if (vn_vd < 0) {
            dissipation_factor = 1 - vn_vd;
        } else if (vn_vd < 2) {
            dissipation_factor = 0.25 * (vn_vd - 2) * (vn_vd - 2);
        }

        stiff_n = std::abs(-k_ * (1 - 1/(1+std::exp(-dist/sigma_))) * dissipation_factor);

        damping_n = 0;
        if (vn_vd < 0) {
            dissipation_factor = -1;
        } else if (vn_vd < 2) {
            dissipation_factor = 0.5 * (vn_vd - 2);
        }
        damping_n = std::abs(damping_n / (vd_ * compliant_fn));
        stiff_t = std::abs(mu_ * stiff_n);

        damping_t = std::pow(vs_ * vs_ + vt.squaredNorm(), 1.5);
        damping_t = std::abs(-mu_ * compliant_fn * dissipation_factor * vs_ * vs_ / damping_t);
    } else {
        stiff_n = k_;
        stiff_t = mu_ * stiff_n;
        damping_n = d_;
        damping_t = mu_ * damping_n;
    }

    stiffness->diagonal() << stiff_t, stiff_t, stiff_n;
    damping->diagonal() << damping_t, damping_t, damping_n;

}
