#pragma once

#include "Commons3D.h"
#include "SoftBodyGrid3D.h"
#include "CostFunctions.h"
#include <ostream>
#include <numeric>

namespace ar3d
{

    /**
     * \brief Solver for the adjoint problem of the grid simulation.
     */
    class AdjointSolver
    {
    public:
	//--------------------------------------
	// structures and logic
	//--------------------------------------

        typedef SoftBodyGrid3D::Input Input;
        /**
         * \brief Precomputed values that don't change over the timesteps
         * and are used in the forward simulation
         */
        struct PrecomputedValues : SoftBodyGrid3D::Precomputed
        {
			Vector3X initialVelocity_;
        };
        static PrecomputedValues allocatePrecomputedValues(const Input& input);

        /**
         * \brief Results of the forward simulation
         */
        struct ForwardState
        {
            /**
		    * \brief Displacements of the active nodes.
		    */
            Vector3X displacements_;
            /**
            * \brief velocities of the active nodes
            */
            Vector3X velocities_;
            /**
             * \brief Displacements on the whole grid
             */
            WorldGridData<real3>::DeviceArray_t gridDisplacements_;
        };
        static ForwardState allocateForwardState(const Input& input);
        /**
         * \brief Intermediate storage of the forward simulation.
         * In memory-saving mode, all this data is reused
         */
        struct ForwardStorage
        {
            Vector3X forces_;
		    SMatrix3x3 stiffness_;
		    SMatrix3x3 newmarkA_;
		    Vector3X newmarkB_;
        };
        static ForwardStorage allocateForwardStorage(const Input& input);
        /**
         * State of the backward/adjoint step.
         * Since they are only used in between two steps, their memory are reused.
         */
        struct BackwardState
        {
            Vector3X adjDisplacements_;
            Vector3X adjVelocities_;
            WorldGridData<real3>::DeviceArray_t adjGridDisplacements_;
            double scale_ = 1;
            void reset();
        };
        static BackwardState allocateBackwardState(const Input& input, int costFunctionInput);
        /**
         * \brief Preallocated memory that is used in every backward step
         */
        struct BackwardStorage
        {
            VectorX unaryLumpedMass_;
            Vector3X unaryBodyForces_;

            SMatrix3x3 adjNewmarkA_;
            Vector3X adjNewmarkB_;
            SMatrix3x3 adjStiffness_;
            Vector3X adjForces_;
            VectorX adjMass_;
            DeviceScalar adjLambda_;
            DeviceScalar adjMu_;
        };
        static BackwardStorage allocateBackwardStorage(const Input& input);
        /**
         * The adjoint of the parameters
         */
        struct AdjointVariables
        {
            double3 adjGravity_ = {0};
            double adjYoungsModulus_ = 0;
			double adjPoissonRatio_ = 0;
			double adjMass_ = 0;
			double adjMassDamping_ = 0;
            double adjStiffnessDamping_ = 0;
			double4 adjGroundPlane_ = {0};
			double3 adjInitialLinearVelocity = { 0 };
			double3 adjInitialAngularVelocity = { 0 };

            AdjointVariables& operator*=(double scaling);
            friend std::ostream& operator<<(std::ostream& os, const AdjointVariables& obj);
        };
		/**
		 * Current states of the variables and which variables are optimized for
		 */
		struct InputVariables
		{
			bool optimizeGravity_;
			real3 currentGravity_;
			bool optimizeYoungsModulus_;
			real currentYoungsModulus_;
			bool optimizePoissonRatio_;
			real currentPoissonRatio_;
			bool optimizeMass_;
			real currentMass_;
			bool optimizeMassDamping_;
			real currentMassDamping_;
			bool optimizeStiffnessDamping_;
			real currentStiffnessDamping_;
			bool optimizeInitialLinearVelocity_;
			real3 currentInitialLinearVelocity_;
			bool optimizeInitialAngularVelocity_;
			real3 currentInitialAngularVelocity_;
			bool optimizeGroundPlane_;
			real4 currentGroundPlane_;

			InputVariables();

		    friend std::ostream& operator<<(std::ostream& os, const InputVariables& obj);
		};

	    /**
		 * \brief Temmporar storage for the cost function evaluation
		 */
		struct CostFunctionTmp
		{
			ICostFunction::Input costInput_;
			ICostFunction::Output costOutput_;
		};
		static CostFunctionTmp allocateCostFunctionTmp(const Input& input, CostFunctionPtr costFunction);

        /**
         * \brief Performs a forward step
         * \param prevState 
         * \param nextStateOut 
         * \param nextStorageOut 
         * \param input 
         * \param precomputed 
         * \param settings 
         * \param costFunctionRequiredInput 
         * \param memorySaving 
         * \return false if the linear solver did not converge
         */
        static bool performForwardStep(
			const ForwardState& prevState, ForwardState& nextStateOut, ForwardStorage& nextStorageOut,
			const Input& input, const PrecomputedValues& precomputed, const SoftBodySimulation3D::Settings& settings,
            int costFunctionRequiredInput, bool memorySaving);

        static real evaluateCostFunction(
            CostFunctionPtr costFunction, int timestep, CostFunctionTmp& tmp,
			const ForwardState& forwardState, BackwardState& backwardStateOut, const Input& input);

        static void performBackwardStep(
            const ForwardState& prevState, const ForwardState& nextState,
            const BackwardState& adjNextState, BackwardState& adjPrevStateOut,
            AdjointVariables& adjVariablesOut,
            const Input& input, const PrecomputedValues& precomputed, 
            ForwardStorage& nextStorage, BackwardStorage& adjStorage,
            const SoftBodySimulation3D::Settings& settings,
            int costFunctionRequiredInput, bool memorySaving);

        struct Statistics
        {
            int numActiveNodes = 0;
            int numEmptyNodes = 0;
            int numActiveElements = 0;
            std::vector<double> forwardTime;
            std::vector<double> costTime;
            std::vector<double> backwardTime;

            friend std::ostream& operator<<(std::ostream& os, const Statistics& obj)
            {
                os << "Statistics:{";
                os << "\n  numActiveNodes: " << obj.numActiveNodes;
                os << "\n  numEmptyNodes: " << obj.numEmptyNodes;
                os << "\n  numActiveElements: " << obj.numActiveElements;
				if (!obj.forwardTime.empty()) {
					os << "\n  forward time: avg=" << (std::accumulate(obj.forwardTime.begin(), obj.forwardTime.end(), 0.0) / obj.forwardTime.size())
						<< ", min=" << *std::min_element(obj.forwardTime.begin(), obj.forwardTime.end())
						<< ", max=" << *std::max_element(obj.forwardTime.begin(), obj.forwardTime.end());
				}
				if (!obj.costTime.empty()) {
					os << "\n  cost evaluation time: avg=" << (std::accumulate(obj.costTime.begin(), obj.costTime.end(), 0.0) / obj.costTime.size())
						<< ", min=" << *std::min_element(obj.costTime.begin(), obj.costTime.end())
						<< ", max=" << *std::max_element(obj.costTime.begin(), obj.costTime.end());
				}
				if (!obj.backwardTime.empty()) {
					os << "\n  backward time: avg=" << (std::accumulate(obj.backwardTime.begin(), obj.backwardTime.end(), 0.0) / obj.backwardTime.size())
						<< ", min=" << *std::min_element(obj.backwardTime.begin(), obj.backwardTime.end())
						<< ", max=" << *std::max_element(obj.backwardTime.begin(), obj.backwardTime.end());
				}
                os << "}";
                return os;
            }
        };

        /**
		 * \brief Computes the full gradient (forward + cost function + adjoint)
		 * \param input the reference configuration
		 * \param settings elasticity settings
		 * \param variables the current parameters and which to optimize for
		 * \param costFunction the cost function
		 * \param adjointVariablesOut output gradients
		 * \param memorySaving true: memory saving mode, less memory saved, but more recomputations
		 * \param worker optional worker to capture interrupts
		 * \param statistics timings
		 * \return the value of the cost function
		 */
		static real computeGradient(
			const Input& input, const SoftBodySimulation3D::Settings& settings, 
			const InputVariables& variables, CostFunctionPtr costFunction,
			AdjointVariables& adjointVariablesOut, bool memorySaving,
			BackgroundWorker2* worker = nullptr,
            Statistics* statistics = nullptr);

		/**
		 * \brief Computes the full gradient (forward + cost function + adjoint)
		 * \param input the reference configuration
		 * \param settings elasticity settings
		 * \param variables the current parameters and which to optimize for
		 * \param costFunction the cost function
		 * \param adjointVariablesOut output gradients
		 * \param finiteDifferencesDelta for testing the adjoint code versus simpler finite differences.
		 *   finiteDifferencesDelta=0 -> adjoint code is used
		 *   finiteDifferencesDelta>0 -> forward finite differences with this specified delta for every parameter
		 *   finiteDifferencesDelta<0 -> backward finite differences
		 * \param worker optional worker to capture interrupts
		 * \param statistics timings
		 * \return the value of the cost function
		 */
		static real computeGradientFiniteDifferences(
			const Input& input, const SoftBodySimulation3D::Settings& settings,
			const InputVariables& variables, CostFunctionPtr costFunction,
			AdjointVariables& adjointVariablesOut,
			real finiteDifferencesDelta,
			BackgroundWorker2* worker = nullptr,
			Statistics* statistics = nullptr);

        /**
         * \brief Computes the adjoint of the material parameters
         * \param youngsModulus 
         * \param poissonRatio 
         * \param adjMu 
         * \param adjLambda 
         * \param adjYoungOut 
         * \param adjPoissonOut 
         */
        static void adjointComputeMaterialParameters(
            double youngsModulus, double poissonRatio,
            double adjMu, double adjLambda,
            double& adjYoungOut, double& adjPoissonOut);

        /**
         * \brief Adjoint of \ref SoftBodyGrid3D::computeStiffnessMatrix().
         * \param input [In] the input specification
         * \param lastDisplacements [In] the previous displacements (for corotation)
         * \param settings [In] the current elasticity settings
         * \param adjStiffnessMatrix [In] the adjoint of the output stiffnes matrix
         * \param adjForce [In] the adjoint of the output force
         * \param adjLastDisplacementsOut [Out] the adjoint of the previous displacements
         * \param adjLambdaOut [Out] The adjoint of the Lamé coefficient \f$ \lambda $\f
         * \param adjMuOut [Out] The adjoint of the Lamé coefficient \f$ \mu $\f
         */
        static void adjointComputeStiffnessMatrix(
            const Input& input, const Vector3X& lastDisplacements, const SoftBodySimulation3D::Settings& settings,
            const SMatrix3x3& adjStiffnessMatrix, const Vector3X& adjForce,
            Vector3X& adjLastDisplacementsOut, DeviceScalar& adjLambdaOut, DeviceScalar& adjMuOut);

        /**
         * \brief Adjoint of \ref SoftBodyGrid3D::applyCollisionForces().
         * \param input  [In] the input specification
         * \param settings  [In] the elasticity settings
         * \param displacements [In] the current displacements
         * \param velocities [In] the current velocities
         * \param adjBodyForces [In] the adjoint of the computed body forces
         * \param adjDisplacementsOut [Out] the adjoint of the current displacements
         * \param adjVelocitiesOut [Out] the adjonit of the current velocities
         * \param adjGroundPlaneOut [Out] the adjoint of the ground plane settings
         */
        static void adjointApplyCollisionForces(
            const Input& input, const SoftBodySimulation3D::Settings& settings,
            const Vector3X& displacements, const Vector3X& velocities,
            const Vector3X& adjBodyForces,
            Vector3X& adjDisplacementsOut, Vector3X& adjVelocitiesOut, double4& adjGroundPlaneOut);

        /**
         * \brief Adjoint of \ref SoftBodyGrid3D::diffuseDisplacements().
         * \param input [In] the input specification
         * \param adjGridDisplacements [In] the adjoint of the displacements on the whole grid
         * \param adjDisplacementsOut [Out] the adjoint of the displacements on the active nodes
         */
        static void adjointDiffuseDisplacements(const Input& input,
            const WorldGridData<real3>::DeviceArray_t& adjGridDisplacements, Vector3X& adjDisplacementsOut);

		static void adjointComputeInitialVelocity(const Input& input, 
			const real3& linearVelocity, const real3& angularVelocity,
			const Vector3X& adjVelocities,
			double3& adjLinearVelocity, double3& adjAngularVelocity);

		struct Settings
		{
			int numIterations_;
            bool memorySaving_;
			bool normalizeUnits_;
            enum Optimizer
            {
                GRADIENT_DESCENT,
				RPROP,
                LBFGS
            } optimizer_;
            static Optimizer ToOptimizer(const std::string& str) { 
				if (str == "GradientDescent") return Optimizer::GRADIENT_DESCENT;
				else if (str == "Rprop") return Optimizer::RPROP;
				else return Optimizer::LBFGS;
			}
			static std::string FromOptimizer(const Optimizer& mode) {
				switch (mode)
				{
				case Optimizer::GRADIENT_DESCENT: return "GradientDescent";
				case Optimizer::RPROP: return "Rprop";
				case Optimizer::LBFGS: return "LBFGS";
				}
			}
            struct GradientDescentSettings
            {
                std::string epsilon_;
                std::string linearStepsize_;
                std::string maxStepsize_;
                std::string minStepsize_;
            } gradientDescentSettings_;
			struct RpropSettings
			{
				std::string epsilon_;
				std::string initialStepsize_;
			} rpropSettings_;
            struct LbfgsSettings
            {
                std::string epsilon_;
                int past_;
                std::string delta_;
                enum LineSearchAlgorithm
                {
                    Armijo,
                    Wolfe,
                    StrongWolfe
                } lineSearchAlg_;
                static inline LineSearchAlgorithm ToLineSearchAlg(const std::string& str)
                {
                    if (str == "Armijo") return Armijo;
                    else if (str == "Wolfe") return Wolfe;
                    return StrongWolfe;
                }
                static inline std::string FromLineSearchAlg(const LineSearchAlgorithm& mode)
                {
                    if (mode == Armijo) return "Armijo";
                    else if (mode == Wolfe) return "Wolfe";
                    return "StrongWolfe";
                }
                int linesearchMaxTrials_;
                std::string linesearchMinStep_;
                std::string linesearchMaxStep_;
                std::string linesearchTol_;
            } lbfgsSettings_;
			InputVariables variables_;
			Settings();
		};

	//--------------------------------------
	// GUI
	//--------------------------------------

		class GUI
		{
		private:
			Settings settings_;
            cinder::params::InterfaceGlRef params_;
		public:
			GUI();
			void initParams(cinder::params::InterfaceGlRef params, const std::string& group, bool noInitialValues = false);
			const Settings& getSettings() const { return settings_; }
            void load(const cinder::JsonTree& parent, bool noInitialValues = false);
            void save(cinder::JsonTree& parent, bool noInitialValues = false) const;
		};

    //--------------------------------------
    // the actual instance
    //--------------------------------------

    private:
		SimulationResults3DPtr reference_;
		Settings settings_;
		CostFunctionPtr costFunction_;
        real finalCost_;
        SoftBodySimulation3D::Settings finalVariables_;
		real finiteDifferencesDelta_ = 0;

    public:
		AdjointSolver(SimulationResults3DPtr reference, const Settings& settings, CostFunctionPtr costFunction);

		// callback(current variables, gradient, cost)
        typedef std::function<void(const SoftBodySimulation3D::Settings&, const SoftBodySimulation3D::Settings&, real)> Callback_t;
        /**
		 * \brief Solves for the missing parameters
		 * \param callback callback for the intermediate and final values
		 * \param worker the background worker
		 * \return true if completed, false if not (e.g. interrupted)
		 * \see getFinalCost()
		 * \see getFinalVariables()
		 */
		bool solve(const Callback_t& callback, BackgroundWorker2* worker);
        /**
         * \brief Returns the final cost after the optimization
         */
        real getFinalCost() const { return finalCost_; }
        /**
         * \brief Returns the reconstructed variables after the optimization
         */
        const SoftBodySimulation3D::Settings& getFinalVariables() const { return finalVariables_; }

		void useAdjoint() { finiteDifferencesDelta_ = 0; }
		void useFiniteDifferences(real delta) { finiteDifferencesDelta_ = delta; }
		bool isUseAdjoint() const { return finiteDifferencesDelta_ == 0; }

        void testGradient(BackgroundWorker2* worker);
    };
}
