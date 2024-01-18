#include "time.h"
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include "fusion.h"

#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/mathematical_program_result.h"

using namespace mosek::fusion;
using namespace monty;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;
Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

int main(int argc, char ** argv)
{
  // set random seed
  srand((int)time(0));

  // initialize problem data
  std::vector<Eigen::Matrix3Xd> J_list;
  Eigen::VectorXd phi, tau;
  Eigen::MatrixXd Q;
  int n_c = 5;
  int n_v_ = 6;
  double h = 0.025;

  phi.resize(n_c);
  phi << 0.001, 0.002, 0.001, 0.002, 0.001;
  tau = Eigen::VectorXd::Random(n_v_);

  Q = Eigen::MatrixXd::Random(n_v_, n_v_);
  Q = Q * Q.transpose();

  for (int i_c = 0; i_c < n_c; i_c++) {
    J_list.emplace_back();
    Eigen::Matrix3Xd & J_i = J_list.back();
    J_i = Eigen::Matrix3Xd::Random(3, n_v_);
  }

  std::cout << "Problem data: " << std::endl;
  std::cout << "Q: " << Q.format(CleanFmt) << std::endl;
  std::cout << "tau: " << tau.format(CleanFmt) << std::endl;
  for (int i_c = 0; i_c < n_c; i_c++) {
    std::cout << "J" << i_c << ": " << J_list[i_c].format(CleanFmt) << std::endl;
  }

  // Eigen::MatrixXd eig = Eigen::MatrixXd(3, 6);

  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 6; j++) {
  //     eig(i, j) = i * 6 + j;
  //   }
  // }

  // ***************************
  // * solve with Mosek Fusion API
  // ***************************

  auto start = std::chrono::system_clock::now();

  Model::t M = new Model("qp");
  auto _M = finally([&]() { M->dispose(); });
  Variable::t v_aug = M->variable("v_aug", n_v_+1, Domain::unbounded());

  for (int i_c = 0; i_c < n_c; i_c++) {
    const double mu = 1;
    RowMatrixXf J_i = J_list[i_c];
    // J_i.row(0) = J_i.row(0) / mu;
    std::vector<double> vec(&J_i(0, 0), J_i.data() + J_i.cols()*J_i.rows());
    std::shared_ptr<ndarray<double, 1>> ndarr(new ndarray<double, 1>(vec.data(), vec.size()));
    Matrix::t J_msk = Matrix::dense(3, n_v_, ndarr);

    auto phi_aug = new_array_ptr<double, 1>({phi[i_c] / mu / h, 0, 0});
    
    auto z_i = Expr::add(
        Expr::mul(J_msk, v_aug->slice(1, n_v_+1)),
        phi_aug
    );

    std::stringstream c_name;
    c_name << "qc" << i_c;
    M->constraint(c_name.str(), z_i, Domain::inQCone());
  }

  Eigen::LLT<Eigen::MatrixXd> lltOfQ(Q);
  RowMatrixXf LT = lltOfQ.matrixL().transpose();
  std::vector<double> vec(&LT(0, 0), LT.data() + LT.cols()*LT.rows());
  std::shared_ptr<ndarray<double, 1>> ndarr(new ndarray<double, 1>(vec.data(), vec.size()));
  Matrix::t LT_msk = Matrix::dense(n_v_, n_v_, ndarr);

  auto z_obj = Expr::vstack(
    v_aug->index(0),
    1,
    Expr::mul(LT_msk, v_aug->slice(1, n_v_+1))
  );

  M->constraint("rc0", z_obj, Domain::inRotatedQCone());

  Eigen::VectorXd tau_aug(n_v_+1);
  tau_aug(0) = 1;
  tau_aug.segment(1, n_v_) = -tau;
  std::vector<double> vec2(&tau_aug[0], tau_aug.data()+tau_aug.cols()*tau_aug.rows());
  auto tau_msk = new_array_ptr<double, 1>(vec2.size());
  std::copy(vec2.begin(), vec2.end(), tau_msk->begin());

  M->objective(
    "obj", ObjectiveSense::Minimize,
    Expr::dot(tau_msk, v_aug)
  );

  M->solve();

  ndarray<double, 1> v_aug_sol = *(v_aug->level());
  Eigen::VectorXd v_aug_sol_eig = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v_aug_sol.raw(), v_aug_sol.size());

  std::cout << "Mosek Fusion API solution: " << v_aug_sol_eig.segment(1, n_v_).format(CleanFmt) << std::endl;

  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout <<  "Mosek Fusion API cost " 
     << duration.count() 
     << " seconds" << std::endl;

  // ***************************
  // * solve with Drake MP agent
  // ***************************

  start = std::chrono::system_clock::now();

  std::unique_ptr<drake::solvers::MosekSolver> solver_msk_ =
      std::make_unique<drake::solvers::MosekSolver>();

  drake::solvers::MathematicalProgramResult mp_result_;

  std::vector<Eigen::VectorXd> elist;
  std::vector<Eigen::VectorXd>* e_list_ptr = &elist;

  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");

  prog.AddQuadraticCost(Q, -tau, v, true);

  std::vector<drake::solvers::Binding<drake::solvers::LorentzConeConstraint>>
      constraints;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const double mu = 1;
    e_list_ptr->emplace_back(Eigen::Vector3d(phi[i_c] / mu / h, 0, 0));
    constraints.push_back(
        prog.AddLorentzConeConstraint(J_list.at(i_c), e_list_ptr->back(), v));
  }

  auto solver = solver_msk_.get();
  solver->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics SOCP cannot be solved.");
  }

  // Primal and dual solutions.
  Eigen::VectorXd v_star;
  v_star = mp_result_.GetSolution(v);
  std::cout << "Drake MP agent solution: " << v_star.format(CleanFmt);

  end = std::chrono::system_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout <<  "Drake MP agent cost " 
     << duration.count() 
     << " seconds" << std::endl;

  // std::cout << "Eigen form: " << eig.format(CleanFmt) << std::endl;

  // std::vector<double> vec(&eigc(0, 0), eigc.data() + eigc.cols()*eigc.rows());

  // // std::shared_ptr<ndarray<double, 1>> ndarr = new_array_ptr<double, 1>(vec);
  // // std::shared_ptr<ndarray<double, 1>> ndarr(new ndarray<double, 1>(vec.data(), vec.size())); // Use the correct constructor for ndarra

  // Matrix::t msk_mat = Matrix::dense(3, 6, ndarr);

  // std::cout << "Mosek form: " << std::endl;
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 6; j++) {
  //     std::cout << msk_mat->get(i, j) << " ";
  //   }
  //   std::cout << "\n";
  // }

  // Model::t M = new Model("cqo1"); auto _M = finally([&]() { M->dispose(); });

  // Variable::t x  = M->variable("x", 3, Domain::greaterThan(0.0));
  // Variable::t y  = M->variable("y", 3, Domain::unbounded());

  // // Create the aliases
  // //      z1 = [ y[0],x[0],x[1] ]
  // //  and z2 = [ y[1],y[2],x[2] ]
  // Variable::t z1 = Var::vstack(y->index(0),  x->slice(0, 2));
  // Variable::t z2 = Var::vstack(y->slice(1, 3), x->index(2));

  // // Create the constraint
  // //      x[0] + x[1] + 2.0 x[2] = 1.0
  // auto aval = new_array_ptr<double, 1>({1.0, 1.0, 2.0});
  // M->constraint("lc", Expr::dot(aval, x), Domain::equalsTo(1.0));

  // // Create the constraints
  // //      z1 belongs to C_3
  // //      z2 belongs to K_3
  // // where C_3 and K_3 are respectively the quadratic and
  // // rotated quadratic cone of size 3, i.e.
  // //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
  // //  and  2.0 z2[0] z2[1] >= z2[2]^2
  // Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());
  // Constraint::t qc2 = M->constraint("qc2", z2, Domain::inRotatedQCone());

  // // Set the objective function to (y[0] + y[1] + y[2])
  // M->objective("obj", ObjectiveSense::Minimize, Expr::sum(y));

  // // Solve the problem
  // M->solve();

  // // Get the linear solution values
  // ndarray<double, 1> xlvl   = *(x->level());
  // ndarray<double, 1> ylvl   = *(y->level());
  // // Get conic solution of qc1
  // ndarray<double, 1> qc1lvl = *(qc1->level());
  // ndarray<double, 1> qc1dl  = *(qc1->dual());

  // std::cout << "x1,x2,x2 = " << xlvl << std::endl;
  // std::cout << "y1,y2,y3 = " << ylvl << std::endl;
  // std::cout << "qc1 levels = " << qc1lvl << std::endl;
  // std::cout << "qc1 dual conic var levels = " << qc1dl << std::endl;



}