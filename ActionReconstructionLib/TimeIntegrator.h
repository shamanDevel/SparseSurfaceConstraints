#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cassert>
#include <memory>

#include "Commons.h"

namespace ar
{
    class TimeIntegrator
    {
    public:
        enum class DenseLinearSolver
        {
            PartialPivLU,
            FullPivLU,
            HouseholderQR,
            ColPivHousholderQR,
            FullPivHouseholderQR,
            CompleteOrthogonalDecomposition,
            LLT,
            LDLT
        };
        enum class SparseLinearSolver
        {
            ConjugateGradient,
            BiCGSTAB,
            LU
        };
        enum class Integrator
        {
            Newmark1,
            Newmark2,
            ExplicitCentralDifferences,
            ImplicitLinearAcceleartion,
            Newmark3,
            HHTAlpha
        };

        typedef Eigen::SparseMatrix<real, Eigen::RowMajor> SparseMatrixRowMajor;
        typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SparseMatrixColumnMajor;

    protected:
        Eigen::Index dof;
        VectorX currentU;
		VectorX currentUDot;
        DenseLinearSolver denseSolver;
        SparseLinearSolver sparseSolver;
        int sparseSolverIterations;
        real sparseSolverTolerance;

    public:
        explicit TimeIntegrator(Eigen::Index dof)
            : dof(dof)
            , currentU(VectorX::Zero(dof))
			, currentUDot(VectorX::Zero(dof))
            , denseSolver(DenseLinearSolver::LDLT)
            , sparseSolver(SparseLinearSolver::BiCGSTAB)
            , sparseSolverIterations(0)
            , sparseSolverTolerance(0)
        {}
        virtual ~TimeIntegrator() {};

        /**
         * \brief Returns the current displacements
         */
        const VectorX& getCurrentU() const { return currentU; }
		void setCurrentU(const VectorX& u) { currentU = u; }

		/**
		* \brief Returns the current velocities
		*/
		const VectorX& getCurrentUDot() const { return currentUDot; }
		void setCurrentUDot(const VectorX& uDot) { currentUDot = uDot; }

        /**
         * \brief Performs a time step integration of the ODE
         * diag(Mass)*u'' + damping*u' + stiffness*u = b
         * with timestep size deltaT.
         * The results can be obtained by \ref getCurrentU().
         * \param deltaT the time step
         * \param mass the diagonal entries of the mass matrix (lumped mass)
         * \param damping the damping matrix
         * \param stiffness the stiffness matrix
         * \param load the load vector
         */
        virtual void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) = 0;

        /**
         * \brief Sets the solver that is used to solve the dense linear systems.
         * \param s the solver
         */
        void setDenseLinearSolver(DenseLinearSolver s) { denseSolver = s; }
        DenseLinearSolver getDenseLinearSolver() const { return denseSolver; }

        /**
        * \brief Sets the solver that is used to solve the sparse linear systems.
        * \param s the solver
        */
        void setSparseLinearSolver(SparseLinearSolver s) { sparseSolver = s; }
        SparseLinearSolver getSparseLinearSolver() const { return sparseSolver; }

        void setSparseSolveIterations(int iterations) { sparseSolverIterations = iterations; }
        void setSparseSolveTolerance(real tolerance) { sparseSolverTolerance = tolerance; }
        int getSparseSolverIterations() const { return sparseSolverIterations; }
        real getSparseSolverTolerance() const { return sparseSolverTolerance; }

        /**
         * \brief Solves the linear system Ax=b
         * \param A 
         * \param b 
         * \return 
         */
        VectorX solveDense(const MatrixX& A, const VectorX& b) const;

        VectorX solveSparse(const SparseMatrixRowMajor& A, const VectorX& b, const VectorX& guess = VectorX()) const;
        VectorX solveSparse(const SparseMatrixColumnMajor& A, const VectorX& b, const VectorX& guess = VectorX()) const;

        VectorX solve(const MatrixX& A, const VectorX& b, const VectorX& guess = VectorX()) const { return solveDense(A, b); }
        VectorX solve(const SparseMatrixRowMajor& A, const VectorX& b, const VectorX& guess = VectorX()) const { return solveSparse(A, b, guess); }
        VectorX solve(const SparseMatrixColumnMajor& A, const VectorX& b, const VectorX& guess = VectorX()) const { return solveSparse(A, b, guess); }

    protected:
        void assertSize(const MatrixX& A) const { assert(A.rows() == dof); assert(A.cols() == dof); }
        void assertSize(const VectorX& b) const { assert(b.size() == dof); }

    public:
        static std::shared_ptr<TimeIntegrator> createIntegrator(Integrator type, Eigen::Index dof);

        template<typename MatrixType, typename MassVectorType>
        static MatrixType rayleighDamping(real massDamping, const MassVectorType& massVector, real stiffnessDamping, const MatrixType& stiffnessMatrix)
        {
            MatrixType tmp = (massDamping * massVector).asDiagonal(); //Limitation of Eigen: can't add a DiagonalWrapper to a Matrix
            return tmp + stiffnessDamping * stiffnessMatrix;
        }
    };

    /**
     * \brief First version of the newmark integration method.
     * Taken from "Newmarks Method of Direct Integration"
     * 
     * Given the stiffness matrix K, damping matrix D (rayleigh damping), mass vector (diagonal of mass matrix M),
     * and right hand side b, it computes
     * \f$ A u^n = B u^{n-1} + C \dot{u}^{n-1} + d \f$
     * with
     * \f$ A = \frac{1}{\theta \Delta t}M + D + \theta \Delta t K \$f,
     * \f$ B = \frac{1}{\theta \Delta t}M + D + (1-\theta) \Delta t K \$f,
     * \f$ C = \frac{1}{\theta} M \$f,
     * \f$ d = \Delta t b \$f.
     */
    class TimeIntegrator_Newmark1 : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevUDot;
        real theta;

    public:
		static constexpr real DefaultTheta = 0.6;
        /**
         * \brief First version of the Newmark integration
         * \param dof the degrees of freedom (size of vectors and matrices)
         * \param theta parameter theta with 0.5 <= theta < 1
         */
        explicit TimeIntegrator_Newmark1(Eigen::Index dof, real theta = DefaultTheta)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
    		, prevUDot(VectorX::Zero(dof))
    		, theta(theta)
        {}
        virtual ~TimeIntegrator_Newmark1() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;

        template<typename MatrixType, typename MassVectorType, typename RhsVectorType>
        static MatrixType getMatrixPartA(const MatrixType& K, const MatrixType& D, const MassVectorType& Mvec, const RhsVectorType& b, real deltaT, real theta)
        {
            MatrixType tmp = ((1 / (theta*deltaT)) * Mvec).asDiagonal(); //Limitation of Eigen: can't add a DiagonalWrapper to a Matrix
            return tmp + D + (theta*deltaT) * K;
        }

        template<typename MatrixType, typename MassVectorType, typename RhsVectorType>
        static MatrixType getMatrixPartB(const MatrixType& K, const MatrixType& D, const MassVectorType& Mvec, const RhsVectorType& b, real deltaT, real theta)
        {
            MatrixType tmp = ((1 / (theta*deltaT)) * Mvec).asDiagonal(); //Limitation of Eigen: can't add a DiagonalWrapper to a Matrix
            return tmp + D - (1 - theta)*deltaT * K;
        }

        template<typename MatrixType, typename MassVectorType, typename RhsVectorType>
        static MatrixType getMatrixPartC(const MatrixType& K, const MatrixType& D, const MassVectorType& Mvec, const RhsVectorType& b, real deltaT, real theta)
        {
            return ((1 / theta) * Mvec).asDiagonal();
        }

        template<typename MatrixType, typename MassVectorType, typename RhsVectorType>
        static RhsVectorType getMatrixPartD(const MatrixType& K, const MatrixType& D, const MassVectorType& Mvec, const RhsVectorType& b, real deltaT, real theta)
        {
            return deltaT * b;
        }

        template<typename VectorType>
        static VectorType computeUdot(const VectorType& currentU, const VectorType& prevU, const VectorType& prevUDot, real deltaT, real theta)
        {
            return (1 / (theta*deltaT)) * (currentU - prevU) - ((1 - theta) / theta) * prevUDot;
        }
    };

    /**
    * \brief Second version of the newmark integration method.
    * Taken from "Christian Dick: Computational Steering for Implantant Planning in Orthopedics"
    */
    class TimeIntegrator_Newmark2 : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevUDot;
        VectorX prevUDotDot;
        VectorX currentUDotDot;

    public:
        /**
        * \brief Second version of the Newmark integration
        * \param dof the degrees of freedom (size of vectors and matrices)
        */
        explicit TimeIntegrator_Newmark2(Eigen::Index dof)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
            , prevUDot(VectorX::Zero(dof))
            , prevUDotDot(VectorX::Zero(dof))
            , currentUDotDot(VectorX::Zero(dof))
        {}
        virtual ~TimeIntegrator_Newmark2() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;
    };

    /**
    * \brief Explicit central differences.
    * Taken from "Numerical Integration in Structural Dynamics"
    */
    class TimeIntegrator_ExplicitCentralDifferences : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevPrevU;

    public:
        explicit TimeIntegrator_ExplicitCentralDifferences(Eigen::Index dof)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
            , prevPrevU(VectorX::Zero(dof))
        {}
        virtual ~TimeIntegrator_ExplicitCentralDifferences() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;
    };

    /**
    * \brief Implicit linear acceleration method.
    * Taken from "Numerical Integration in Structural Dynamics"
    * \tparam real the scalar type, float or double
    */
    class TimeIntegrator_ImplicitLinearAcceleration : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevUDot;
        VectorX prevUDotDot;
        VectorX currentUDotDot;

    public:
        explicit TimeIntegrator_ImplicitLinearAcceleration(Eigen::Index dof)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
            , prevUDot(VectorX::Zero(dof))
            , prevUDotDot(VectorX::Zero(dof))
            , currentUDotDot(VectorX::Zero(dof))
        {}
        virtual ~TimeIntegrator_ImplicitLinearAcceleration() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;
    };

    /**
    * \brief Third version of the newmark method.
    * Taken from "Numerical Integration in Structural Dynamics"
    */
    class TimeIntegrator_Newmark3 : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevUDot;
        VectorX prevUDotDot;
        VectorX currentUDotDot;

    public:
        explicit TimeIntegrator_Newmark3(Eigen::Index dof)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
            , prevUDot(VectorX::Zero(dof))
            , prevUDotDot(VectorX::Zero(dof))
            , currentUDotDot(VectorX::Zero(dof))
        {}
        virtual ~TimeIntegrator_Newmark3() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;
    };

    /**
    * \brief HHT-alpha method.
    * Taken from "Numerical Integration in Structural Dynamics"
    */
    class TimeIntegrator_HHTalpha : public TimeIntegrator
    {
    private:
        VectorX prevU;
        VectorX prevUDot;
        VectorX prevUDotDot;
        VectorX currentUDotDot;
        real alpha, beta, gamma;

    public:
        /**
         * \brief HHT-alpha method
         * \param dof degrees of fredom
         * \param alpha alpha parameter, 0 <= alpha <= 1/3
         */
        explicit TimeIntegrator_HHTalpha(Eigen::Index dof, real alpha = 0.25)
            : TimeIntegrator(dof)
            , prevU(VectorX::Zero(dof))
            , prevUDot(VectorX::Zero(dof))
            , prevUDotDot(VectorX::Zero(dof))
            , currentUDotDot(VectorX::Zero(dof))
            , alpha(alpha)
            , beta( (1+alpha)*(1+alpha)/4 )
            , gamma( 0.5 + alpha )
        {}
        virtual ~TimeIntegrator_HHTalpha() {};

    public:
        void performStep(real deltaT, const VectorX& mass, const MatrixX& damping, const MatrixX& stiffness, const VectorX& load) override;
    };
}