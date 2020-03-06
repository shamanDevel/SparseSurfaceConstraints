#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <optional>
#include <variant>

#include "IInverseProblem.h"
#include "SoftBodyMesh2D.h"
#include "AdjointHelper.h"
#include "GridUtils.h"
#include "TimeIntegrator.h"

namespace ar {

    /**
     * \brief Solver for general adjoint problems on the dynamic simulation.
     * Includes corotational corrections
     */
    class InverseProblem_Adjoint_Dynamic
        : public IInverseProblem
    {
    private:
        static constexpr real newmarkTheta = TimeIntegrator_Newmark1::DefaultTheta;

		//AntTweakBar Properties
		int numIterations;
        bool optimizeMass;
		real initialMass;
        bool optimizeMassDamping;
		real initialMassDamping;
        bool optimizeStiffnessDamping;
		real initialStiffnessDamping;
        bool optimizeInitialPosition;
		real volumePrior;
		bool optimizeGroundPosition;
		real initialGroundPlaneHeight;
		real initialGroundPlaneAngle;
		bool optimizeYoungsModulus;
		real initialYoungsModulus;
		bool optimizePoissonRatio;
		real initialPoissonRatio;

		std::string timestepWeights;
		real costPositionWeighting;
		real costVelocityWeighting;
		bool costUseMeanAmplitude;
		bool gridBypassAdvection;
        bool gridUsePartialObservation;

    public: //public API
        InverseProblem_Adjoint_Dynamic();

        void setGridUsePartialObservation(bool enable) {gridUsePartialObservation = enable;}
        InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
        InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;

        void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) override;
        void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const override;

    public:
		//for integration tests
		void setInitialMass(real mass) { initialMass = mass; optimizeMass = true; }
		void setInitialMassDamping(real damping) { initialMassDamping = damping; optimizeMassDamping = true; }
		void setInitialStiffnessDamping(real damping) { initialStiffnessDamping = damping; optimizeStiffnessDamping = true; }

    public:
        //Main gradient computation

        /**
         * \brief Defines the keyframes for the cost function plus weighting.
         * If a keyframe should not participate in the cost,
         * the list of target positions must be empty.
         */
        struct MeshCostFunctionDefinition
        {
			int numSteps = 0;
			enum {SQUARED_DIFFERENCE, MEAN_AMPLITUDE} dataTerm = SQUARED_DIFFERENCE;

			struct Keyframe
			{
				VectorX position;
				VectorX velocity;
			};
            //vector of 
            //1. weighting
            //2. keyframes
            std::vector<std::pair<real, Keyframe> > keyframes;
			//Extra Weight for the position in a keyframe
			real positionWeighting = 1;
			//Extra weight for the velocity in a keyframe
			real velocityWeighting = 0.2;

			VectorX meanAmplitudeReference;

            //priors
			real volumePrior; //The volume should be kept
			real referenceVolume;
			Eigen::VectorXi windingOrder;
        };

	    /**
		 * \brief Defines the keyframes for the cost function plus weighting.
		 * If a keyframe should not participate in the cost, set the weight to zero.
		 * Then the associated sdf is not used and can be set to an empty grid.
		 */
		struct GridCostFunctionDefinition
		{
			int numSteps = 0;

			struct Keyframe
			{
				real weight = 0;
				GridUtils2D::grid_t sdf;
                VectorX u;
                Matrix2X observations;
			};
			std::vector<Keyframe> keyframes;
			enum {SOURCE_SDF, SOURCE_UXY, SOURCE_OBSERVATIONS} dataSource = SOURCE_SDF;

			//priors
			real volumePrior; //The volume should be kept
			real referenceVolume;
		};

        /**
         * \brief Defines which parameters are optimized and their current value.
         * The gradient will then have the gradient defined for these properties.
         */
        struct GradientInput
        {
            std::optional<real> mass;
            std::optional<real> dampingMass;
            std::optional<real> dampingStiffness;
			std::optional<real> groundHeight;
			std::optional<real> groundAngle;
			std::optional<real> youngsModulus;
			std::optional<real> poissonRatio;
            //Either initial position (mesh) or initial sdf (grid)
            std::optional<std::variant<VectorX, GridUtils2D::grid_t> > initialConfiguration;
        };

        struct GradientOutput
        {
            real finalCost = 0;
            std::optional<real> massGradient;
            std::optional<real> dampingMassGradient;
            std::optional<real> dampingStiffnessGradient;
			std::optional<real> groundHeightGradient;
			std::optional<real> groundAngleGradient;
			std::optional<real> youngsModulusGradient;
			std::optional<real> poissonRatioGradient;
            //Either initial position (mesh) or initial sdf (grid)
            std::optional<std::variant<VectorX, GridUtils2D::grid_t> > initialConfigurationGradient;
        };

        /**
         * \brief Computes the full gradient for the dynamic mesh simulation
         * \param input the definition of free parameters and their initial values
         * \param costFunction definition of the cost function
         * \param simulation the simulation with the vertices, triangles and non-free parameters
         * \param worker background worker
         * \param onlyCost set to true if only the cost should be computed, no gradient
         * \return the final cost and the gradients of each free parameter
         */
        static GradientOutput gradientMesh(
            const GradientInput& input,
            const MeshCostFunctionDefinition& costFunction,
            SoftBodyMesh2D simulation, //passed by value because it's modified
            BackgroundWorker* worker, bool onlyCost);

		static GradientOutput gradientMesh_Numerical(
			const GradientInput& input,
			const MeshCostFunctionDefinition& costFunction,
			SoftBodyMesh2D simulation, //passed by value because it's modified
			BackgroundWorker* worker, bool onlyCost);

		/**
		* \brief Computes the full gradient for the dynamic grid simulation
		* \param input the definition of free parameters and their initial values
		* \param costFunction definition of the cost function
		* \param simulation the simulation with the reference sdf and free-node-mapping
		* \param worker background worker
		* \param onlyCost set to true if only the cost should be computed, no gradient
		* \return the final cost and the gradients of each free parameter
		*/
		static GradientOutput gradientGrid(
			const GradientInput& input,
			const GridCostFunctionDefinition& costFunction,
			SoftBodyGrid2D simulation,
			BackgroundWorker* worker, bool onlyCost);

    private: //private helpers

	    /**
		 * \brief Parses the weights. Syntax:
		 * "<frameIndex:Weight> <frameIndex:Weight> ... "
		 * frame index is one-based
		 * \param weights the weight definition string
		 * \return the cost function definition
		 */
		MeshCostFunctionDefinition meshParseWeights(const std::string& weights, const SoftBodyMesh2D& simulation) const;

		/**
		* \brief Parses the weights. Syntax:
		* "<frameIndex:Weight> <frameIndex:Weight> ... "
		* frame index is one-based
		* \param weights the weight definition string
		* \return the cost function definition
		*/
		GridCostFunctionDefinition gridParseWeights(const std::string& weights, const SoftBodyGrid2D& simulation) const;

    public:
        //Static helper functions

        //Precomputed values to speed up the computation
        //of the individual forward and adjoint passes
        struct MeshPrecomputedValues
        {
            //material matrix
            Matrix3 matC;
            //diagonal of the mass matrix
            VectorX Mvec;
            //Stiffness matrix K, only used if corotational mode is off
            MatrixX K;
            //Matrices of the time integration, only used if corotational mode is off
            MatrixX A, B, C;
            //Vectors of the time integration, only used if corotational mode is off
            VectorX d;
            //Lu-decomposition of A
            Eigen::PartialPivLU<MatrixX> ALu;
        };

        /**
         * \brief Performs precomputations for a 
         * simulation (forward + adjoint) on the given mesh simulation.
         * 
         * Parameters used from the simulation:
         * - Degrees of freedom
         * - Boundary conditions
         * - Simulation parameters (material + rotation correction mode)
         * - Time integration modes (timestep + damping)
         * These parameters have to be constant during a forward + backward pass
         * 
         * \param simulation the simulation
         * \return a struct with precomputed values
         */
        static MeshPrecomputedValues meshPrecomputeValues(
            const SoftBodyMesh2D& simulation);

        //Additional stored values from the forward pass 
        //needed in the adjoint pass
        struct MeshForwardStorage
        {
            MatrixX K; //stiffness matrix, for corotational mode
            MatrixX A, B, C; //integration matrices, for corotational mode
            Eigen::PartialPivLU<MatrixX> ALu;
            VectorX rhs;
            ConjugateGradient_StorageForAdjoint<VectorX> cgStorage;
        };
        //forward state
        struct MeshForwardResult
        {
            VectorX u;
            VectorX uDot;
        };
        //backward / adjoint state
        struct MeshBackwardResult
        {
            VectorX adjU;
            VectorX adjUTmp;
            VectorX adjUDot;
            VectorX adjUDotTmp;
        };

	    /**
		 * \brief Direct adjoint variables of the parameters
		 */
		struct MeshParamAdjoint
		{
			real adjYoungsModulus = 0;
			real adjPoissonRatio = 0;
			real adjMass = 0;
			real adjMassDamping = 0;
			real adjStiffnessDamping = 0;
			real adjGroundPlaneHeight = 0;
			real adjGroundPlaneAngle = 0;
			VectorX adjInitialPosition;
		};

        /**
         * \brief Performs one forward step in the mesh simulation
         * \param currentState the current displacements \f$ u^{(n)} \f$ and current velocities \f$ \dot{u}^{(n)} \f$
         * \param simulation the simulation
         * \param precomputed precomputed values, see \ref meshPrecomputeValues
         * \return next state (displacements \f$ u^{(n+1)} \f$ and velocities \f$ \dot{u}^{(n+1)} \f$,
         *  additional storage for the adjoint pass
         */
        static std::tuple<MeshForwardResult, MeshForwardStorage> meshForwardStep(
            const MeshForwardResult& currentState,
            const SoftBodyMesh2D& simulation, const MeshPrecomputedValues& precomputed
        );

        /**
         * \brief Computes the adjoint step to the respective forward step
         *        \ref meshForwardStep .
         * \param prevResult the inputs to the associated forward step
		 * \param currentResult the outputs from the associated forward step
		 * \param currentAdj current adjoint variables. 
		 *	The variables \c adjSdfTmp, \c adjUTmp and \c adjUDotTmp are taken as input, might be
		 *	initialized with the derivatives of the cost function, and are also modified.
		 *	The variables \c adjU and \c adjUDot are the output adjoint variables
	 	 * \param prevAdj updates the temporarly adjoint variables (\c adjSdfTmp, \c adjUTmp and \c adjUDotTmp)
		 *   that are needed in the next adjoint step.
		 *   The variables \c adjU and \c adjUDot are not modified.
         * \param simulation the simulation
         * \param precomputed precomputed values, see \ref meshPrecomputeValues 
         * \param forwardStorage additional storage from the forward pass
         */
        static void meshAdjointStep(
            const MeshForwardResult& prevResult,
            const MeshForwardResult& currentResult,
            MeshBackwardResult& currentAdj,
            MeshBackwardResult& prevAdj,
			MeshParamAdjoint& paramAdj,
            const SoftBodyMesh2D& simulation, const MeshPrecomputedValues& precomputed,
            const MeshForwardStorage& forwardStorage
        );

        static void meshComputeElementMatrix(
            const Matrix3 & C, Matrix6 & Ke, Vector6& Fe,
            const Vector6& posE, const Vector6& currentUe,
            SoftBodySimulation::RotationCorrection rotationCorrection);

        static void meshComputeElementMatrixAdjoint(
            const Matrix3 & C, const SoftBodyMesh2D& simulation,
            const Vector6& posE, const Vector6& currentUe,
            SoftBodySimulation::RotationCorrection rotationCorrection,
            const Matrix6& adjKe, const Vector6& adjFe,
            Vector6& adjCurrentUe,
            Vector6* adjInitialPosition, real* adjYoungsModulus, real* adjPoissonRatio);

		static Vector2 meshImplicitCollisionForces(
			const Vector2& refPos, const Vector2& disp, const Vector2& vel,
			real groundHeight, real groundAngle,
			real groundStiffness, real softminAlpha, real timestep, real newmarkTheta);

		static void meshImplicitCollisionForcesAdjoint(
			const Vector2& refPos, const Vector2& disp, const Vector2& vel,
			real groundHeight, real groundAngle,
			real groundStiffness, real softminAlpha, real timestep, real newmarkTheta,
			const Vector2& adjForce,
			Vector2& adjRefPos, Vector2& adjDisp, Vector2& adjVel,
			real& adjGroundHeight, real& adjGroundAngle);

	    /**
		 * \brief Reads the keyframes from the cost function and computes the weighted average volume as a reference volume.
		 * Additionally, it computes the winding order per element.
		 * \param costFunction the cost function
		 * \return the reference volume and the winding order
		 */
		static std::pair<real, Eigen::VectorXi> meshComputeReferenceVolume(
			const MeshCostFunctionDefinition& costFunction, 
			const SoftBodyMesh2D& simulation);

	    /**
		 * \brief Computes the derivative of the volume prior with respect to the single coordinates of the initial position
		 * \param currentInitialPosition 
		 * \param costFunction 
		 * \return the cost added by this prior, the derivatives with respect to the coordinates
		 */
		static std::pair<real, VectorX> meshComputeVolumePriorDerivative(
			const VectorX& currentInitialPosition, 
			const MeshCostFunctionDefinition& costFunction,
			const SoftBodyMesh2D& simulation);

    private: //Other mesh helpers

        /**
         * \brief Computes the product \f$ y^T F + r^T\f, of the adjoint states
         * and the derivatives of the operations with respect to the initial positions,
         * plus the derivatives of the prior
         * \param forwardResults 
         * \param forwardStorages 
         * \param adjointResults 
         * \param precomputed 
         * \param costFunction the cost function for the gradient r^T
         * \param simulation 
         * \return the part for the gradient containing the gradients for the initial positions (size is 2*N, N=number of nodes)
         */
        static VectorX meshParameterDerivativeInitialPosition(
            const std::vector<MeshForwardResult>& forwardResults,
            const std::vector<MeshForwardStorage>& forwardStorages,
            const std::vector<MeshBackwardResult>& adjointResults,
			const MeshParamAdjoint& adjParam,
            const MeshPrecomputedValues& precomputed,
            const MeshCostFunctionDefinition& costFunction,
            const SoftBodyMesh2D& simulation);

    public:

		//Precomputed values to speed up the computation
		//of the individual forward and adjoint passes in the grid case.
		//For now, the same as for the mesh.
		typedef MeshPrecomputedValues GridPrecomputedValues;

		/**
		* \brief Direct adjoint variables of the parameters
		*/
		struct GridParamAdjoint
		{
			real adjYoungsModulus = 0;
			real adjPoissonRatio = 0;
			real adjMass = 0;
			real adjMassDamping = 0;
			real adjStiffnessDamping = 0;
			real adjGroundPlaneHeight = 0;
			real adjGroundPlaneAngle = 0;
			GridUtils2D::grid_t adjInitialSdf;
		};

		/**
		* \brief Performs precomputations for a
		* simulation (forward + adjoint) on the given mesh simulation.
		*
		* Parameters used from the simulation:
		* - Degrees of freedom
		* - Boundary conditions
		* - Simulation parameters (material + rotation correction mode)
		* - Time integration modes (timestep + damping)
		* These parameters have to be constant during a forward + backward pass
		*
		* \param simulation the simulation
		* \return a struct with precomputed values
		*/
		static GridPrecomputedValues gridPrecomputeValues(
			const SoftBodyGrid2D& simulation);

	    /**
		 * \brief The results of a single forward step.
		 */
		struct GridForwardResult
		{
			VectorX u;
			VectorX uDot;
			GridUtils2D::grid_t sdf;
            RowVectorX observationSdfValues;
		};

		//Additional stored values from the forward pass 
		//needed in the adjoint pass
		struct GridForwardStorage
		{
			MatrixX K; //stiffness matrix, for corotational mode
			MatrixX A, B, C; //integration matrices, for corotational mode
			Eigen::PartialPivLU<MatrixX> ALu;
			VectorX rhs;
			ConjugateGradient_StorageForAdjoint<VectorX> cgStorage;

			GridUtils2D::bgrid_t validCells;
			GridUtils2D::grid_t uGridXDiffused;
			GridUtils2D::grid_t uGridYDiffused;
            GridUtils2D::grid_t sdfPreReconstruction;
			GridUtils2D::grid_t advectionWeights;

            Eigen::Matrix2Xi observationCells;
            Matrix2X observationCellWeights;
		};

	    /**
		 * \brief The input and output adjoints of a single backward step.
		 *  Inputs: all variables prefixed with "-tmp", might be initialized with the derivatives of the cost function,
		 *   and are also changed for the next (previous) adjoint step
		 *  Output: all other variables. These are the final adjoint variables.
		 */
		struct GridBackwardResult
		{
			GridUtils2D::grid_t adjSdfTmp;
            RowVectorX adjObservationSdfValues;

			VectorX adjUTmp;
			VectorX adjUDotTmp;

			VectorX adjU;
			VectorX adjUDot;
		};

		/**
		* \brief Performs one forward step in the grid simulation
		* \param currentResult the current results, \f$ u^{(n)}, \dot{u}^{(n)}, \phi^{(n)} \f$
		* \param simulation the simulation (needed for resulution, timestep, boundary conditions and initial SDF)
		* \param precomputed precomputed values, see \ref gridPrecomputeValues()
		* \param partialObservationPoints NULL->full SDF, !NULL->SDF value at these positions are evaluated
		* \return the results for the next timestep and additional storage for the adjoint pass
		*/
		static std::pair<GridForwardResult, GridForwardStorage> gridForwardStep(
			const GridForwardResult& currentResult,
			const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed,
            const Matrix2X* partialObservationPoints);

		/**
		* \brief Computes the adjoint step to the respective forward step
		*        \ref gridForwardStep .
		* \param prevResult the inputs to the associated forward step
		* \param currentResult the outputs from the associated forward step
		* \param currentAdj current adjoint variables. 
		*	The variables \c adjSdfTmp, \c adjUTmp and \c adjUDotTmp are taken as input, might be
		*	initialized with the derivatives of the cost function, and are also modified.
		*	The variables \c adjU and \c adjUDot are the output adjoint variables
		* \param prevAdj updates the temporarly adjoint variables (\c adjSdfTmp, \c adjUTmp and \c adjUDotTmp)
		*   that are needed in the next adjoint step.
		*   The variables \c adjU and \c adjUDot are not modified.
		* \param simulation the simulation (needed for resulution, timestep, boundary conditions and initial SDF)
		* \param precomputed precomputed values, see \ref gridPrecomputeValues()
		* \param forwardStorage additional storage from the forward pass
		*/
		static void gridAdjointStep(
			const GridForwardResult& prevResult,
			const GridForwardResult& currentResult,
			GridBackwardResult& currentAdj,
			GridBackwardResult& prevAdj,
			GridParamAdjoint& paramAdj,
			const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed,
			const GridForwardStorage& forwardStorage
		);

		static void gridComputeElementMatrixAdjoint(
			int x, int y,
			const std::array<real, 4>& sdfs,
			const Vector8& currentUe, const Vector8& currentUeDot,
			SoftBodySimulation::RotationCorrection rotationCorrection,
			const Matrix8& adjKe, const Vector8& adjFe,
			Vector8& adjCurrentUe, Vector8& adjCurrentUeDot, std::array<real, 4>& adjSdf,
			GridParamAdjoint& paramAdj,
			const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed);

		//Adjoint version of SoftBodyGrid2D::implicitCollisionForces
		static void gridImplicitCollisionForcesAdjoint(
			const std::array<real, 4>& sdfs,
			const std::array<Vector2, 4>& positions, const std::array<Vector2, 4>& velocities,
			bool implicit, real groundHeight, real groundAngle, real groundStiffness, real softminAlpha,
			real timestep, real newmarkTheta,
			const std::array<ar::Vector2, 4>& adjForce,
			std::array<Vector2, 4>& adjPosition, std::array<Vector2, 4>& adjVelocity, std::array<real, 4>& adjSdf,
			real& adjGroundHeight, real& adjGroundAngle);

		/**
		* \brief Reads the keyframes from the cost function and computes the weighted average volume as a reference volume.
		* \param costFunction the cost function
		* \return the reference volume
		*/
		static real gridComputeReferenceVolume(
			const GridCostFunctionDefinition& costFunction,
			const SoftBodyGrid2D& simulation);

		/**
		* \brief Computes the derivative of the volume prior with respect to the single coordinates of the initial position
		* \param currentInitialSDF
		* \param costFunction
		* \return the cost added by this prior, the derivatives with respect to the SDF values
		*/
		static std::pair<real, GridUtils2D::grid_t> gridComputeVolumePriorDerivative(
			const GridUtils2D::grid_t& currentInitialSDF,
			const GridCostFunctionDefinition& costFunction,
			const SoftBodyGrid2D& simulation);

    public: //Testing
		void testPlot(int deformedTimestep, BackgroundWorker* worker) override;
    };



}
