#include "TimeIntegrator.h"

#include <exception>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <cinder/Log.h>

namespace ar
{
    VectorX TimeIntegrator::solveDense(const MatrixX& A, const VectorX& b) const
    {
        switch (denseSolver)
        {
        case DenseLinearSolver::PartialPivLU: {return A.partialPivLu().solve(b); }
        case DenseLinearSolver::FullPivLU:
            {
                Eigen::FullPivLU<MatrixX> decomp = A.fullPivLu();
                decomp.setThreshold(1e-5);
                int rank = decomp.rank();
                if (rank == std::min(A.rows(), A.cols()))
                    CI_LOG_D("rank is " << decomp.rank() << " (of " << std::min(A.rows(), A.cols()) << ")");
                else
                    CI_LOG_W("Matrix is not of full rank, only " << rank << " of " << std::min(A.rows(), A.cols()));
                return decomp.solve(b);
            }
        case DenseLinearSolver::HouseholderQR: {return A.householderQr().solve(b); }
        case DenseLinearSolver::ColPivHousholderQR:
            {
                Eigen::ColPivHouseholderQR<MatrixX> decomp = A.colPivHouseholderQr();
                int rank = decomp.rank();
                if (rank == std::min(A.rows(), A.cols()))
                    CI_LOG_D("rank is " << decomp.rank() << " (of " << std::min(A.rows(), A.cols()) << ")");
                else
                    CI_LOG_W("Matrix is not of full rank, only " << rank << " of " << std::min(A.rows(), A.cols()));
                return decomp.solve(b);
            }
        case DenseLinearSolver::FullPivHouseholderQR:
            {
                Eigen::FullPivHouseholderQR<MatrixX> decomp = A.fullPivHouseholderQr();
                int rank = decomp.rank();
                if (rank == std::min(A.rows(), A.cols()))
                    CI_LOG_D("rank is " << decomp.rank() << " (of " << std::min(A.rows(), A.cols()) << ")");
                else
                    CI_LOG_W("Matrix is not of full rank, only " << rank << " of " << std::min(A.rows(), A.cols()));
                return decomp.solve(b);
            }
        case DenseLinearSolver::CompleteOrthogonalDecomposition: {return A.completeOrthogonalDecomposition().solve(b); }
        case DenseLinearSolver::LLT: {return A.llt().solve(b); }
        case DenseLinearSolver::LDLT: {return A.ldlt().solve(b); }
        default: throw std::exception("unsupported solver");
        }
    }

    VectorX TimeIntegrator::solveSparse(const SparseMatrixRowMajor & A, const VectorX & b, const VectorX& guess) const
    {
        //TODO: do I want to support custom number of iterations and thesholds?
        switch (sparseSolver)
        {
        case SparseLinearSolver::ConjugateGradient:
            {
                Eigen::ConjugateGradient<SparseMatrixRowMajor, Eigen::Lower | Eigen::Upper> cg;
                if (sparseSolverIterations > 0) cg.setMaxIterations(sparseSolverIterations);
                if (sparseSolverTolerance > 0) cg.setTolerance(sparseSolverTolerance);
                cg.compute(A);
                if (guess.size() > 0)
                    return cg.solveWithGuess(b, guess);
                else
                    return cg.solve(b);
            }
        case SparseLinearSolver::BiCGSTAB:
        {
            Eigen::BiCGSTAB<SparseMatrixRowMajor> cg;
            if (sparseSolverIterations > 0) cg.setMaxIterations(sparseSolverIterations);
            if (sparseSolverTolerance > 0) cg.setTolerance(sparseSolverTolerance);
            cg.compute(A);
            if (guess.size() > 0)
                return cg.solveWithGuess(b, guess);
            else
                return cg.solve(b);
        }
        case SparseLinearSolver::LU:
        {
            CI_LOG_E("Sparse LU does only work with column-major matrices");
            return VectorX::Zero(b.size());
        }
        default: throw std::exception("unsupported solver");
        }
    }

    VectorX TimeIntegrator::solveSparse(const SparseMatrixColumnMajor & A, const VectorX & b, const VectorX& guess) const
    {
        //TODO: do I want to support custom number of iterations and thesholds?
        switch (sparseSolver)
        {
        case SparseLinearSolver::ConjugateGradient:
        {
            Eigen::ConjugateGradient<SparseMatrixColumnMajor, Eigen::Lower | Eigen::Upper> cg;
            if (sparseSolverIterations > 0) cg.setMaxIterations(sparseSolverIterations);
            if (sparseSolverTolerance > 0) cg.setTolerance(sparseSolverTolerance);
            cg.compute(A);
            if (guess.size() > 0)
                return cg.solveWithGuess(b, guess);
            else
                return cg.solve(b);
        }
        case SparseLinearSolver::BiCGSTAB:
        {
            Eigen::BiCGSTAB<SparseMatrixColumnMajor> cg;
            if (sparseSolverIterations > 0) cg.setMaxIterations(sparseSolverIterations);
            if (sparseSolverTolerance > 0) cg.setTolerance(sparseSolverTolerance);
            cg.compute(A);
            if (guess.size() > 0)
                return cg.solveWithGuess(b, guess);
            else
                return cg.solve(b);
        }
        case SparseLinearSolver::LU:
        {
            Eigen::SparseLU<SparseMatrixColumnMajor, Eigen::COLAMDOrdering<int>> lu;
            lu.analyzePattern(A);
            lu.factorize(A);
            return lu.solve(b);
        }
        default: throw std::exception("unsupported solver");
        }
    }

    std::shared_ptr<TimeIntegrator> TimeIntegrator::createIntegrator(Integrator type, Eigen::Index dof)
    {
        switch (type)
        {
        case Integrator::Newmark1: return std::make_shared<TimeIntegrator_Newmark1>(dof);
        case Integrator::Newmark2: return std::make_shared<TimeIntegrator_Newmark2>(dof);
        case Integrator::ExplicitCentralDifferences: return std::make_shared<TimeIntegrator_ExplicitCentralDifferences>(dof);
        case Integrator::ImplicitLinearAcceleartion: return std::make_shared<TimeIntegrator_ImplicitLinearAcceleration>(dof);
        case Integrator::Newmark3: return std::make_shared<TimeIntegrator_Newmark3>(dof);
        case Integrator::HHTAlpha: return std::make_shared<TimeIntegrator_HHTalpha>(dof);
        default: throw std::exception("unknown integrator");
        }
    }
}
