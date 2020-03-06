#pragma once

#include <boost/serialization/access.hpp>
#include <boost/serialization/version.hpp>
#include <cassert>
#include <ostream>

#include "Commons.h"
#include "BackgroundWorker.h"
#include "TimeIntegrator.h"

namespace ar
{
    //Parent class for all soft body simulation implementations
    class SoftBodySimulation
    {
    public:
        /**
         * \brief The computation mode to handle the rotational component in the elasticity equations
         */
        enum class RotationCorrection
        {
            /**
             * \brief No handling of rotation
             */
            None,
            /**
             * \brief Corotational formulation
             */
            Corotation,
            //Nonlinear_StVernant

            _COUNT_
        };
        static const std::string RotationCorrectionNames[int(RotationCorrection::_COUNT_)];

		enum class CollisionResolution
		{
			/**
			 * \brief The positions and velocities are repaired after the time integration
			 */
			POST_REPAIR,
			/**
			 * \brief On collisions, impulse responses are added to the neumann boundaries in the next iteration.
			 */
			SIMPLE_NEUMANN,

			REPAIR_PLUS_NEUMANN,
            /**
             * \brief Forces modelled by virtual springs, implicitly in the Newmark Integration
             */
            SPRING_IMPLICIT,
            /**
             * \brief Forces modelled by virtual springs, explicit in the Newmark Integration
             */
            SPRING_EXPLICIT,

			_COUNT_
		};
		static const std::string CollisionResolutionNames[int(CollisionResolution::_COUNT_)];

		struct Settings
		{
		private:
			friend class boost::serialization::access;
		public:
			Vector2 gravity_;
			real youngsModulus_;
			real poissonsRatio_;
			real mass_;
			real dampingAlpha_;
			real dampingBeta_;
			real materialLambda_;
			real materialMu_;
			RotationCorrection rotationCorrection_;
            real timestep_;

			real groundPlaneHeight_; //center height from below
			real groundPlaneAngle_;  //angle of the plane (0 = horizontal)
			bool enableCollision_;
			CollisionResolution collisionResolution_;
			real collisionVelocityDamping_; //damping factor
            real groundStiffness_; //the stiffness of the spring that models the ground collision
            real softmaxAlpha_; //The smootheness of the softmax. As it goes to infinity, the hard max is approximated

            Settings();
			Settings(const Vector2& gravity, real youngs_modulus, real poissons_ratio, real mass, real damping_alpha,
				real damping_beta, RotationCorrection rotation_correction,
				real timestep)
				: gravity_(gravity),
				  youngsModulus_(youngs_modulus),
				  poissonsRatio_(poissons_ratio),
				  mass_(mass),
				  dampingAlpha_(damping_alpha),
				  dampingBeta_(damping_beta),
				  rotationCorrection_(rotation_correction),
				  timestep_(timestep)
			{
				SoftBodySimulation::computeMaterialParameters(youngsModulus_, poissonsRatio_, materialMu_, materialLambda_);
			}

		    friend std::ostream& operator<<(std::ostream& os, const Settings& obj)
		    {
				return os
					<< "gravity: (" << obj.gravity_.transpose() << ")"
					<< " youngsModulus: " << obj.youngsModulus_
					<< " poissonsRatio: " << obj.poissonsRatio_
					<< " mass: " << obj.mass_
					<< " dampingAlpha: " << obj.dampingAlpha_
					<< " dampingBeta: " << obj.dampingBeta_
					<< " materialLambda: " << obj.materialLambda_
					<< " materialMu: " << obj.materialMu_
					<< " rotationCorrection: " << RotationCorrectionNames[int(obj.rotationCorrection_)]
					<< " timestep: " << obj.timestep_
					<< " groundPlaneHeight: " << obj.groundPlaneHeight_
					<< " groundPlaneAngle: " << obj.groundPlaneAngle_
					<< " enableCollision:" << obj.enableCollision_
					<< " collisionResolution: " << CollisionResolutionNames[int(obj.collisionResolution_)]
					<< " collisionVelocityDamping: " << obj.collisionVelocityDamping_;
		    }

		private:
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & gravity_;
				ar & youngsModulus_;
				ar & poissonsRatio_;
				ar & mass_;
				ar & dampingAlpha_;
				ar & dampingBeta_;
				ar & materialLambda_;
				ar & materialMu_;
				ar & rotationCorrection_;
                if (version > 0) ar & timestep_;
			}
		};

    private:
		Settings settings_;
        
        TimeIntegrator::DenseLinearSolver denseLinearSolverType_;
        TimeIntegrator::SparseLinearSolver sparseLinearSolverType_;
        TimeIntegrator::Integrator integratorType_;
        bool hasSolution_;
        bool useSparseMatrixes_;
        int sparseSolverIterations_;
        real sparseSolverTolerance_;

    public:
        SoftBodySimulation();
        virtual ~SoftBodySimulation() {}

    public:
		const Settings& getSettings() const { return settings_; }
		void setSettings(const Settings& settings) { settings_ = settings; }

        void setGravity(const Vector2& gravity) { settings_.gravity_ = gravity; }
        const Vector2& getGravity() const { return settings_.gravity_; }

        static void computeMaterialParameters(real youngsModulus, real poissonsRatio,
                                              real& materialMu, real& materialLambda);

        void setMaterialParameters(real youngsModulus, real poissonsRatio);
        void setMass(real mass);
        void setDamping(real dampingOnMass, real dampingOnStiffness);
        void setTimestep(real timestep);

        real getYoungsModulus() const { return settings_.youngsModulus_; }
        real getPoissonsRatio() const { return settings_.poissonsRatio_; }
        real getMass() const { return settings_.mass_; }
        real getDampingAlpha() const { return settings_.dampingAlpha_; }
        real getDampingBeta() const { return settings_.dampingBeta_; }
        real getMaterialLambda() const { return settings_.materialLambda_; }
        real getMaterialMu() const { return settings_.materialMu_; }
        real getTimestep() const { return settings_.timestep_; }

        void setDenseLinearSolver(TimeIntegrator::DenseLinearSolver type) { denseLinearSolverType_ = type; }
        void setSparseLinearSolver(TimeIntegrator::SparseLinearSolver type) { sparseLinearSolverType_ = type; }
        void setTimeIntegrator(TimeIntegrator::Integrator type) { integratorType_ = type; }
        void setUseSparseMatrices(bool sparse) { useSparseMatrixes_ = sparse; }
        void setSparseSolveIterations(int iterations) { sparseSolverIterations_ = iterations; }
        void setSparseSolveTolerance(real tolerance) { sparseSolverTolerance_ = tolerance; }
        TimeIntegrator::DenseLinearSolver getDenseLinearSolver() const { return denseLinearSolverType_; }
        TimeIntegrator::SparseLinearSolver getSparseLinearSolver() const { return sparseLinearSolverType_; }
        TimeIntegrator::Integrator getTimeIntegrator() const { return integratorType_; }
        bool isUseSparseMatrices() const { return useSparseMatrixes_; }
        int getSparseSolverIterations() const { return sparseSolverIterations_; }
        real getSparseSolverTolerance() const { return sparseSolverTolerance_; }

        void setRotationCorrection(RotationCorrection mode) { settings_.rotationCorrection_ = mode; }
        RotationCorrection getRotationCorrection() const { return settings_.rotationCorrection_; }

		void setGroundPlane(real height, real angle) { settings_.groundPlaneHeight_ = height; settings_.groundPlaneAngle_ = angle; }
		real getGroundPlaneHeight() const { return settings_.groundPlaneHeight_; }
		real getGroundPlaneAngle() const { return settings_.groundPlaneAngle_; }
		void setEnableCollision(bool enable) { settings_.enableCollision_ = enable; }
		bool isEnableCollision() const { return settings_.enableCollision_; }
		void setCollisionResolution(CollisionResolution mode) { settings_.collisionResolution_ = mode; }
		CollisionResolution getCollisionResolution() const { return settings_.collisionResolution_; }
		void setCollisionVelocityDamping(real damping) { settings_.collisionVelocityDamping_ = damping; }
		real getCollisionVelocityDamping() const { return settings_.collisionVelocityDamping_; }
        void setGroundStiffness(real stiffness) { settings_.groundStiffness_ = stiffness; }
        real getGroundStiffness() const { return settings_.groundStiffness_; }
        void setCollisionSoftmaxAlpha(real alpha) { settings_.softmaxAlpha_ = alpha; }
        real getCollisionSoftmaxAlpha() const { return settings_.softmaxAlpha_; }

        bool hasSolution() const { return hasSolution_; }

    protected:
        void setHasSolution(bool solution) { hasSolution_ = solution; }

    public:
        /**
         * \brief Solves the simulation
         * \param dynamic 
         *  true -> dynamic simulation, based on the current timestep.
         *  false -> static solution
         * \param worker the background worker
         */
        virtual void solve(bool dynamic, BackgroundWorker* worker) = 0;
        void solveStaticSolution(BackgroundWorker* worker) { solve(false, worker); }
        void solveDynamicSolution(BackgroundWorker* worker) { solve(true, worker); }
        virtual void resetSolution() = 0;

    public:
        //Helpers

	    /**
		 * \brief Collision of point p with the ground.
		 * \param p the position
		 * \return the signed distance to the surface (positive outside, negative inside); the normal
		 */
		std::pair<real, Vector2> groundCollision(const Vector2& p) const;
		static std::pair<real, Vector2> groundCollision(const Vector2& p, real groundHeight, real groundAngle);
        static real groundCollisionDt(const Vector2& pDot, real groundHeight, real groundAngle);

        /**
         * \brief Computes the material matrix C from the given Lamé coefficients
         * \param mu the first lamé coefficient mu
         * \param lambda the second lamé coefficient lambda
         * \return the material matrix C
         */
        static Matrix3 computeMaterialMatrix(real mu, real lambda);

        /**
         * \brief Computes the 2D polar decomposition F = RS.
         * \param F An arbitrary, non-singular 2x2 matrix
         * \return R: the rotational component of F (a pure rotation matrix)
         */
        static Matrix2 polarDecomposition(const Matrix2& F);
        /**
         * \brief Computes the 3D polar decomposition F = RS.
         * \param F An arbitrary, non-singular 2x2 matrix
         * \param iterations the number of iterations to peform. 5 seems to be enough
         * \return R: the rotational component of F (a pure rotation matrix)
         */
        static Matrix3 polarDecomposition(const Matrix3& F, int iterations = 5);
    };

}
BOOST_CLASS_VERSION(ar::SoftBodySimulation::Settings, 1)
