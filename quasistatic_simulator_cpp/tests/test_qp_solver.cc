#include <iostream>
#include <Eigen/Dense>

#include "drake/solvers/osqp_solver.h"


void QuasistaticSimulator::ForwardQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    Eigen::VectorXd* beta_star_ptr) {
  auto& q_dict = *q_dict_ptr;
  const auto n_f = phi_constraints.size();
  const auto h = params.h;

  // construct and solve MathematicalProgram.
  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  prog.AddQuadraticCost(Q, -tau_h, v, true);

  const VectorXd e = phi_constraints / h;
  auto constraints = prog.AddLinearConstraint(
      -J, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v);
  auto solver = PickBestQpSolver(params);
  solver->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics QP cannot be solved.");
  }

  *v_star_ptr = mp_result_.GetSolution(v);
  if (constraints.evaluator()->num_constraints() > 0) {
    *beta_star_ptr = -mp_result_.GetDualSolution(constraints);
  } else {
    *beta_star_ptr = Eigen::VectorXd(0);
  }

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

int main(int argc, char** argv) {
  auto drake_qp_solver = std::make_unique<drake::solvers::OsqpSolver>();
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
