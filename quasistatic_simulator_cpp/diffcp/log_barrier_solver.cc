#include "diffcp/log_barrier_solver.h"

#include <iostream>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/scs_solver.h"

using Eigen::Matrix3Xd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

LogBarrierSolverBase::LogBarrierSolverBase()
    : solver_scs_(std::make_unique<drake::solvers::ScsSolver>()),
      solver_grb_(std::make_unique<drake::solvers::GurobiSolver>()) {}

double LogBarrierSolverBase::BackStepLineSearch(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e,
    const Eigen::Ref<const Eigen::VectorXd>& v,
    const Eigen::Ref<const Eigen::VectorXd>& dv,
    const Eigen::Ref<const Eigen::VectorXd>& Df, const double kappa) const {
  double t = 1;
  int line_search_iters = 0;
  bool line_search_success = false;
  double f0 = CalcF(Q, b, G, e, kappa, v);

  while (line_search_iters < line_search_iter_limit_) {
    double f = CalcF(Q, b, G, e, kappa, v + t * dv);
    double f1 = f0 + alpha_ * t * Df.transpose() * dv;
    if (f < f1) {
      line_search_success = true;
      break;
    }
    t *= beta_;
    line_search_iters++;
  }

  if (!line_search_success) {
    std::stringstream msg;
    msg << "Back stepping Line search exceeded iteration limit. ";
    msg << "Gradient norm: ";
    msg << Df.norm();
    throw std::runtime_error(msg.str());
  }

  return t;
}

void LogBarrierSolverBase::SolveOneNewtonStep(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, double kappa,
    drake::EigenPtr<Eigen::VectorXd> v_star_ptr) const {
  const auto n_v = Q.rows();
  MatrixXd H(n_v, n_v);
  VectorXd Df(n_v);
  int n_iters = 0;
  bool converged = false;

  while (n_iters < newton_steps_limit_) {
    CalcGradientAndHessian(Q, b, G, e, *v_star_ptr, kappa, &Df, &H);
    H_llt_.compute(H);
    VectorXd dv = -H_llt_.solve(Df);
    double lambda_squared = -Df.transpose() * dv;
    if (lambda_squared / 2 < tol_) {
      converged = true;
      // std::cout << "Log barrier solver breaks at " << n_iters << " Newton iterations." << std::endl;
      break;
    }
    double t;
    try {
      t = BackStepLineSearch(Q, b, G, e, *v_star_ptr, dv, Df, kappa);
    } catch (std::runtime_error& err) {
      std::stringstream ss;
      ss << err.what();
      ss << ". Current kappa " << kappa;
      throw std::runtime_error(ss.str());
    }
    *v_star_ptr += t * dv;
    n_iters++;
  }

  if (!converged) {
    std::stringstream ss;
    ss << "QpLogBarrier Newton's method did not converge for barrier weight ";
    ss << kappa;
    throw std::runtime_error(ss.str());
  }
}

void LogBarrierSolverBase::SolveMultipleNewtonSteps(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, double kappa_max,
    drake::EigenPtr<VectorXd> v_star_ptr) const {
  double kappa = 1;
  double n_iters = 0;
  while (true) {
    SolveOneNewtonStep(Q, b, G, e, kappa, v_star_ptr);
    kappa = std::min(kappa_max, kappa * 2);
    if (kappa == kappa_max) {
      // std::cout << "> it took " << n_iters << " to solve the log barrier problem" << std::endl;
      break;
    }
  }
}

void LogBarrierSolverBase::SolveGradientDescent(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, const double kappa,
    drake::EigenPtr<Eigen::VectorXd> v_star_ptr) const {
  const auto n_v = Q.rows();
  MatrixXd H(n_v, n_v);
  VectorXd Df(n_v);
  int n_iters = 0;
  bool converged = false;

  while (n_iters < gradient_steps_limit_) {
    CalcGradientAndHessian(Q, b, G, e, *v_star_ptr, kappa, &Df, &H);
    VectorXd dv = -Df;
    if (dv.norm() < tol_) {
      converged = true;
      break;
    }
    double t;
    try {
      t = BackStepLineSearch(Q, b, G, e, *v_star_ptr, dv, Df, kappa);
    } catch (std::runtime_error& err) {
      std::stringstream ss;
      ss << err.what();
      ss << ". Current gradient norm " << dv.norm();
      throw std::runtime_error(ss.str());
    }
    *v_star_ptr += t * dv;
    n_iters++;
  }

  if (!converged) {
    std::stringstream ss;
    ss << "LogBarrier gradient descent did not converge. Final gradient norm ";
    ss << Df.norm();
    throw std::runtime_error(ss.str());
  }
}

void LogBarrierSolverBase::Solve(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                                 const Eigen::Ref<const Eigen::VectorXd>& b,
                                 const Eigen::Ref<const Eigen::MatrixXd>& G,
                                 const Eigen::Ref<const Eigen::VectorXd>& e,
                                 const double kappa_max,
                                 const bool use_free_solver,
                                 Eigen::VectorXd* v_star_ptr) const {
  VectorXd v(Q.rows());
  SolvePhaseOne(G, e, use_free_solver, &v);

  try {
    SolveOneNewtonStep(Q, b, G, e, kappa_max, &v);
  } catch (std::runtime_error& exception) {
    // std::cout << "Try to solve one newton step, failed!" << std::endl;
    SolveMultipleNewtonSteps(Q, b, G, e, kappa_max, &v);
  }

  *v_star_ptr = v;
}

void LogBarrierSolverBase::GetPhaseOneSolution(
    const drake::solvers::VectorXDecisionVariable& v,
    const drake::solvers::DecisionVariable& s,
    drake::EigenPtr<Eigen::VectorXd> v0_ptr) const {
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Phase 1 program cannot be solved.");
  }

  const auto s_value = mp_result_.GetSolution(s);
  if (s_value > -1e-6) {
    v0_ptr = nullptr;
    std::stringstream ss;
    ss << "Phase 1 cannot find a feasible solution. s = " << s_value << endl;
    throw std::runtime_error(ss.str());
  }

  *v0_ptr = mp_result_.GetSolution(v);
}

drake::solvers::SolverBase* LogBarrierSolverBase::get_solver(
    bool use_free_solver) const {
  if (use_free_solver) {
    return solver_scs_.get();
  }
  return solver_grb_.get();
}

void QpLogBarrierSolver::SolvePhaseOne(
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, const bool use_free_solver,
    drake::EigenPtr<Eigen::VectorXd> v0_ptr) const {
  const auto n_f = G.rows();
  const auto n_v = G.cols();
  auto prog = drake::solvers::MathematicalProgram();
  // v_s is the concatenation of [v, s], where v is the vector of generalized
  // velocities of the system and s is the scalar slack variables.
  auto v_s = prog.NewContinuousVariables(n_v + 1, "v");
  const auto& v = v_s.head(n_v);
  const auto& s = v_s[n_v];

  // G * v - e <= s  <==> [G, -1] * v_s <= e.
  MatrixXd G_1(n_f, n_v + 1);
  G_1.leftCols(n_v) = G;
  G_1.rightCols(1) = -VectorXd::Ones(n_f);

  prog.AddLinearCost(s);
  prog.AddLinearConstraint(
      G_1, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v_s);
  prog.AddBoundingBoxConstraint(-1, 1, v);

  get_solver(use_free_solver)->Solve(prog, {}, {}, &mp_result_);
  GetPhaseOneSolution(v, s, v0_ptr);
}

QpLogBarrierSolver::QpLogBarrierSolver() : LogBarrierSolverBase() {}

double QpLogBarrierSolver::CalcF(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, const double kappa,
    const Eigen::Ref<const Eigen::VectorXd>& v) const {
  double f = kappa * 0.5 * v.transpose() * Q * v;
  f += kappa * b.transpose() * v;
  for (int i = 0; i < G.rows(); i++) {
    double d = G.row(i) * v - e[i];
    if (d > 0) {
      // Out of domain of log(.), i.e. one of the inequality constraints is
      // infeasible.
      return std::numeric_limits<double>::infinity();
    }
    f -= log(-d);
  }
  return f;
}

void QpLogBarrierSolver::CalcGradientAndHessian(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e,
    const Eigen::Ref<const Eigen::VectorXd>& v, const double kappa,
    drake::EigenPtr<Eigen::VectorXd> Df_ptr,
    drake::EigenPtr<Eigen::MatrixXd> H_ptr) const {
  *H_ptr = Q * kappa;
  *Df_ptr = (Q * v + b) * kappa;
  for (int i = 0; i < G.rows(); i++) {
    double d = G.row(i) * v - e[i];
    *H_ptr += G.row(i).transpose() * G.row(i) / d / d;
    *Df_ptr += -G.row(i) / d;
  }
}

SocpLogBarrierSolver::SocpLogBarrierSolver() : LogBarrierSolverBase() {}

void SocpLogBarrierSolver::SolvePhaseOne(
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, const bool use_free_solver,
    drake::EigenPtr<Eigen::VectorXd> v0_ptr) const {
  const auto n_c = G.rows() / 3;
  const auto n_v = G.cols();
  DRAKE_THROW_UNLESS(G.rows() % 3 == 0);
  DRAKE_THROW_UNLESS(e.size() == n_c);

  auto prog = drake::solvers::MathematicalProgram();
  auto v_s = prog.NewContinuousVariables(n_v + 1, "v");
  const auto& v = v_s.head(n_v);
  const auto& s = v_s[n_v];

  prog.AddLinearCost(s);

  Matrix3Xd A(3, n_v + 1);
  A.rightCols(1) = Vector3d(1, 0, 0);

  // std::cout << "Matrix G has " << G.rows() << " rows and " << G.cols() << " columns." << std::endl;
  // std::cout << "There are " << n_c << " contact points." << std::endl;

  for (int i = 0; i < n_c; i++) {
    A.leftCols(n_v) = -G.block(i * 3, 0, 3, n_v);
    Vector3d b(e[i], 0, 0);
    prog.AddLorentzConeConstraint(A, b, v_s);
  }

  prog.AddBoundingBoxConstraint(-1, 1, v);

  get_solver(use_free_solver)->Solve(prog, {}, {}, &mp_result_);
  GetPhaseOneSolution(v, s, v0_ptr);
}

double SocpLogBarrierSolver::CalcF(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e, const double kappa,
    const Eigen::Ref<const Eigen::VectorXd>& v) const {
  return DoCalcF<double>(Q, b, G, e, kappa, v);
}

void SocpLogBarrierSolver::CalcGradientAndHessian(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::VectorXd>& e,
    const Eigen::Ref<const Eigen::VectorXd>& v, double kappa,
    drake::EigenPtr<Eigen::VectorXd> Df_ptr,
    drake::EigenPtr<Eigen::MatrixXd> H_ptr) const {
  *H_ptr = Q * kappa;
  *Df_ptr = (Q * v + b) * kappa;
  const int n_c = G.rows() / 3;
  const int n_v = G.cols();

  // Hessian of generalized log w.r.t. w.
  static const Eigen::Matrix3d A{{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Eigen::Matrix3d D2w;
  Eigen::Vector3d w_bar;

  for (int i = 0; i < n_c; i++) {
    const Eigen::Matrix3Xd& G_i = G.block(i * 3, 0, 3, n_v);
    Vector3d w = CalcWi<double>(G_i, e[i], v);
    const double d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    w_bar << w[0], -w[1], -w[2];

    // TODO(yongpeng): the first term should be 4 / d / d * A * w_bar * w_bar.transpose() * A.transpose()?
    D2w = 4 / d / d * w_bar * w_bar.transpose() - 2 / d * A;
    
    *Df_ptr -= 2 / d * G_i.transpose() * w_bar;
    *H_ptr += G_i.transpose() * D2w * G_i;
  }
}
