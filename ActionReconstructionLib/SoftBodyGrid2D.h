#pragma once

#include <vector>
#include <Eigen/StdVector>
#include <string>
#include <ostream>
#include <optional>

#include "SoftBodySimulation.h"
#include "Integration.h"

namespace ar
{
    class SoftBodyGrid2D : public SoftBodySimulation
    {
    public:
        typedef Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic> grid_t;
        typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> bgrid_t;

        enum class AdvectionMode
        {
            SEMI_LAGRANGE_ONLY,
            SEMI_LAGRANGE_SHEPARD_INVERSION,
            DIRECT_FORWARD,

            _COUNT_
        };
        static const std::string& advectionModeName(AdvectionMode mode);

        static constexpr bool recoverSdf = false;

        struct GridSettings
        {
            bool explicitDiffusion_ = true;
            bool hardDirichletBoundaries_ = false;
            AdvectionMode advectionMode_ = AdvectionMode::SEMI_LAGRANGE_SHEPARD_INVERSION;
			int sdfRecoveryIterations_ = 20;

			GridSettings(){}
	        GridSettings(bool explicit_diffusion, bool hard_dirichlet_boundaries, AdvectionMode advection_mode,
		        int sdf_recovery_iterations)
		        : explicitDiffusion_(explicit_diffusion),
		          hardDirichletBoundaries_(hard_dirichlet_boundaries),
		          advectionMode_(advection_mode),
		          sdfRecoveryIterations_(sdf_recovery_iterations)
	        {
	        }

            friend std::ostream& operator<<(std::ostream& os, const GridSettings& obj)
            {
				return os
					<< "explicitDiffusion: " << obj.explicitDiffusion_
					<< " hardDirichletBoundaries: " << obj.hardDirichletBoundaries_
					<< " advectionMode: " << advectionModeName(obj.advectionMode_)
					<< " sdfRecoveryIteterations: " << obj.sdfRecoveryIterations_;
            }

        };

		typedef std::map<int, Eigen::Vector2i, std::less<>,
			Eigen::aligned_allocator<std::pair<const int, Eigen::Vector2i> > > indexToPos_t;

    private:
        int resolution_;
        real h_;
        Vector2 size_;

        grid_t sdf_;
        grid_t gridNeumannX_;
        grid_t gridNeumannY_;
        bgrid_t gridDirichlet_;

		real totalMass_;
		VectorX currentU; //displacements only on the free nodes, needed for rotation correction
        VectorX currentUDot;
		VectorX collisionForces_;
        grid_t uGridX_;
        grid_t uGridY_;
        grid_t uGridInvX_;
        grid_t uGridInvY_;
        grid_t sdfSolution_;

        GridSettings gridSettings_;
        static const std::string advectionModeNames[static_cast<int>(AdvectionMode::_COUNT_)];

        std::shared_ptr<TimeIntegrator> integrator_;

        Matrix28 Phi1, Phi2, Phi3, Phi4;
        Matrix38 B1, B2, B3, B4;

		Eigen::MatrixXi posToIndex_;
		indexToPos_t indexToPos_;
		int dof_;

    public:
        SoftBodyGrid2D();
        virtual ~SoftBodyGrid2D() {}

        void setGridResolution(int resolution);
        int getGridResolution() const { return resolution_; }
		//Sets the reference SDF,
		//and computes the degrees of freedom with \ref findDoF()
        void setSDF(const grid_t& sdf);

		const Vector2& getCellSize() const { return size_; } //for the inverse problems
		const Matrix38& getB1() const { return B1; }
		const Matrix38& getB2() const { return B2; }
		const Matrix38& getB3() const { return B3; }
		const Matrix38& getB4() const { return B4; }
		const Matrix28& getPhi1() const { return Phi1; }
		const Matrix28& getPhi2() const { return Phi2; }
		const Matrix28& getPhi3() const { return Phi3; }
		const Matrix28& getPhi4() const { return Phi4; }

		//Returns the mapping from grid position to the index of the degree of freedom (internal node)
		//Or -1 if that grid cell is outside
		const Eigen::MatrixXi& getPosToIndex() const { return posToIndex_; }
		//Returns the mapping from degree of freedom (internal node) to the grid position
		const indexToPos_t& getIndexToPos() const { return indexToPos_; }
		//Returns the degrees of freedom / number of internal nodes in the elasticity simulation
		int getDoF() const { return dof_; }

	    void setExplicitDiffusion(bool enable);
        bool isExplicitDiffusion() const { return gridSettings_.explicitDiffusion_; }
        void setHardDirichletBoundaries(bool enable) { gridSettings_.hardDirichletBoundaries_ = enable; }
        bool isHardDirichletBoundaries() const { return gridSettings_.hardDirichletBoundaries_; }
        void setAdvectionMode(AdvectionMode mode) { gridSettings_.advectionMode_ = mode; }
        AdvectionMode getAdvectionMode() const { return gridSettings_.advectionMode_; }
		void setSdfRecoveryIterations(int count) { gridSettings_.sdfRecoveryIterations_ = count; }
		void disableSdfRecovery() { gridSettings_.sdfRecoveryIterations_ = 0; }
		int getSdfRecoveryIterations() const { return gridSettings_.sdfRecoveryIterations_; }
        const GridSettings& getGridSettings() const { return gridSettings_; }
		void setGridSettings(const GridSettings& settings) { gridSettings_ = settings; }

        void resetBoundaries();
        void addDirichletBoundary(int x, int y, const Vector2& displacement);
        void addNeumannBoundary(int x, int y, const Vector2& force);

        void resetSolution() override;
        void solve(bool dynamic, BackgroundWorker* worker) override;

        const grid_t& getSdfReference() const { return sdf_; }
        const grid_t& getGridNeumannX() const { return gridNeumannX_; }
        const grid_t& getGridNeumannY() const { return gridNeumannY_; }
        const bgrid_t& getGridDirichlet() const { return gridDirichlet_; }

        const grid_t& getUGridX() const { return uGridX_; }
        const grid_t& getUGridY() const { return uGridY_; }
        const grid_t& getUGridInvX() const { return uGridInvX_; }
        const grid_t& getUGridInvY() const { return uGridInvY_; }
        const grid_t& getSdfSolution() const { return sdfSolution_; }
		const VectorX& getUSolution() const { return currentU; }

		//Computes the elastic energy for the given displacements
		real computeElasticEnergy(const grid_t& uGridX, const grid_t& uGridY);

    public: //for adjoint problem

        /**
         * \brief Finds the degrees of freedom.
         * This is called in \ref setSDF(const grid_t& sdf) and the results can be obtained by
         * \ref getPosToIndex(), \ref getIndexToPos() and \ref getDoF()
         * 
         * \param posToIndex mapping of position to index, only filled if explicit diffusion is used
         * \param indexToPos mapping of index to position, only filled if explicit diffusion is used
         * \return the degrees of freedom
         */
        int findDoF(Eigen::MatrixXi& posToIndex, indexToPos_t& indexToPos) const;

        [[deprecated("Deprecated: Replaced by assembleForceVector, assembleStiffnessMatrix, assembleMassMatrix")]]
        bool computeElementMatrix_old(int x, int y, const Vector2& size, const Matrix3& C, Matrix8& Ke, Vector8& MeVec, Vector8& Fe, const VectorX& currentDisplacements) const;

        /**
         * \brief Assembles the force vector
         * \param f the force vector to be filled
         * \param posToIndex the mapping of grid position to degree of freedom
         * \param extraForces additional forces on the active nodes (from collisions)
         */
        void assembleForceVector(VectorX& f, const Eigen::MatrixXi& posToIndex, const VectorX& extraForces) const;
        
		bool computeElementMatrix(int x, int y, Matrix8& Ke, Vector8& Fe,
			real materialMu, real materialLambda,
			RotationCorrection rotationCorrection, bool hardDirichlet,
			const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements) const;

		void computeNietscheDirichletBoundaries(
			int x, int y, Matrix8& KeDirichlet, Vector8& FeDirichlet,
			real materialMu, real materialLambda,
			const Integration2D<real>::IntegrationWeights& boundaryWeights) const;

    	/**
         * \brief Assembles the stiffness matrix.
         * The two Lamé coefficients are needed again to compute the matrix D in the Nietsche boundaries,
         * a variation of C that depends on the normal.
         * \param C the material matrix C
         * \param materialMu the Lamé coefficient mu, also used for the computation of C
         * \param materialLambda the Lamé coefficient lambda, also used for the computation of C
         * \param K the stiffness matrix to be filled
         * \param F the force vector to be filled (Nietsche boundaries)
         * \param posToIndex the  mapping of grid position to degree of freedom
         * \param currentDisplacements the current displacements for the rotation correction
         */
        void assembleStiffnessMatrix(const Matrix3& C, real materialMu, real materialLambda, MatrixX& K, VectorX* F, const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements) const;
        /**
         * \brief Assembles the (diagonal) mass matrix
         * \param Mvec the diagonal of the mass matrix to be filled
         * \param posToIndex the mapping of grid position to degree of freedom
         * \return the total mass
         */
        real assembleMassMatrix(VectorX& Mvec, const Eigen::MatrixXi& posToIndex) const;

        bgrid_t computeValidCells(const grid_t& uGridX, const grid_t& uGridY, const Eigen::MatrixXi& posToIndex) const;

		[[deprecated("Deprecated: Explicit diffusion is much faster and more stable")]]
        void augmentMatrixWithImplicitDiffusion(MatrixX& K) const;

	    /**
		 * \brief Computes the collision of the current element's boundary with the ground.
		 * \param x the cell lower-left x index
		 * \param y the cell lower-left y index
		 * \param phi the four values of the SDF
		 * \param ue the four 2D-displacements at the nodes packed together
		 * \return if there is collision, returns the boundary length that is inside and the collision normal
		 */
		[[deprecated("Deprecated: Replaced by implicit spring forces (implicitCollisionForces)")]]
		std::optional<std::pair<real, Vector2>> getGroundCollisionSimpleNeumann(int x, int y, const std::array<real, 4>& phi, const Vector8& ue) const;

		std::array<Vector2, 4> implicitCollisionForces(int x, int y, const std::array<real, 4>& sdfs,
			const std::array<Vector2, 4>& positions, const std::array<Vector2, 4>& velocities,
			bool implicit, real groundHeight, real groundAngle,
			real groundStiffness, real softminAlpha, real timestep, real newmarkTheta) const;

	    /**
		 * \brief Computes the Neumann Forces that are the resolut of the contact impulses with the ground.
		 * \param posToIndex
		 * \param dof number of degrees of freedom (max index in posToIndex) 
		 * \param currentDisplacements the current displacements
		 * \return the neumann forces on the grid (linearized by posToIndex)
		 */
		VectorX resolveCollisions(const Eigen::MatrixXi& posToIndex, int dof, 
            const VectorX& currentDisplacements, const VectorX& currentVelocities) const;

		/**
         * \brief Solves for the free nodes (dense matrices).
         * The solution contains only the displacements on the inner nodes.
         * They have to be mapped back to the whole grid and the SDF has to be advected.
         * This is handled in \ref solve(bool dynamic, BackgroundWorker* worker)
         * \param dynamic true: dynamic case, include time integration; false: static case
         * \param dof the number of inner nodes
         * \param posToIndex the mapping from grid position to degree of freedom
         * \param currentDisplacements the current displacements for the rotation correction
         * \param worker background worker for interruption test
         * \return the displacements and velocities
         */
        std::pair<VectorX, VectorX> solveImplDense(bool dynamic, int dof, const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements, BackgroundWorker* worker);


#if SOFT_BODY_SUPPORT_SPARSE_MATRICES==1
        [[deprecated("Deprecated: Explicit diffusion is much faster and more stable")]]
        void augmentMatrixWithImplicitDiffusion(const std::vector<bool>& insideCells, real epsilon, std::vector<Eigen::Triplet<real>>& Kentries) const;

		/**
		* \brief Solves for the free nodes (sparse matrices).
		* The solution contains only the displacements on the inner nodes.
		* They have to be mapped back to the whole grid and the SDF has to be advected.
		* This is handled in \ref solve(bool dynamic, BackgroundWorker* worker)
		* \param dynamic true: dynamic case, include time integration; false: static case
		* \param dof the number of inner nodes
		* \param posToIndex the mapping from grid position to degree of freedom
		* \param currentDisplacements the current displacements for the rotation correction
		* \param worker background worker for interruption test
		* \return the displacements
		*/
        std::pair<VectorX, VectorX> solveImplSparse(bool dynamic, int dof, const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements, BackgroundWorker* worker);
#endif

    public: //Factories
        static SoftBodyGrid2D CreateBar(
            const Vector2& rectCenter, const Vector2& rectHalfSize,
            int resolution, bool fitObjectToGrid,
			bool enableDirichlet, const Vector2& dirichletBoundaries, const Vector2& neumannBoundaries
        );

        static SoftBodyGrid2D CreateTorus(
            real torusOuterRadius, real torusInnerRadius,
            int resolution,
			bool enableDirichlet, const Vector2& dirichletBoundaries, const Vector2& neumannBoundaries
        );
    };
}
