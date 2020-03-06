#pragma once

#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <vector_types.h>
#include <glm/glm.hpp>
#include <ostream>
#include <Eigen/Core>

//NO INCLUDES TO ar3d packages

namespace ar3d_wrapper
{
	typedef double real;
	typedef double4 real3;
	typedef double4 real4;

	//Initialize the context
	void initialize();
	//Cleanup everything
	void cleanup();

	struct IInputSettings
	{
		bool enableDirichlet; //enable dirichlet boundaries
		real3 centerDirichlet, halfsizeDirichlet; //points in these boundaries are dirichlet boundaries
		bool sampleIntegrals;
		int diffusionDistance;
		real zeroCutoff;

		friend std::ostream& operator<<(std::ostream& os, const IInputSettings& obj)
		{
			return os
				<< "enableDirichlet: " << obj.enableDirichlet
				<< " centerDirichlet: " << obj.centerDirichlet.x << "," << obj.centerDirichlet.y << "," << obj.centerDirichlet.z
				<< " halfsizeDirichlet: " << obj.halfsizeDirichlet.x << "," << obj.halfsizeDirichlet.y << "," << obj.halfsizeDirichlet.z
				<< " sampleIntegrals: " << obj.sampleIntegrals
				<< " diffusionDistance: " << obj.diffusionDistance
				<< " zeroCutoff: " << obj.zeroCutoff;
		}
	};
	struct InputTorusSettings : IInputSettings
	{
		int resolution;
		real3 center;
		glm::vec3 orientation;
		real innerRadius, outerRadius;

		friend std::ostream& operator<<(std::ostream& os, const InputTorusSettings& obj)
		{
			return os
				<< static_cast<const IInputSettings&>(obj)
				<< " resolution: " << obj.resolution
				<< " center: " << obj.center.x << "," << obj.center.y << "," << obj.center.z
				<< " orientation: " << obj.orientation.x << "," << obj.orientation.y << "," << obj.orientation.z
				<< " innerRadius: " << obj.innerRadius
				<< " outerRadius: " << obj.outerRadius;
		}
	};
	struct InputSdfSettings : IInputSettings
	{
		std::string file = "";
		int voxelResolution = 0;
		glm::ivec3 offset = { 0,0,0 };
		glm::ivec3 size = { 0,0,0 };
		bool filledCells = false;

		friend std::ostream& operator<<(std::ostream& os, const InputSdfSettings& obj)
		{
			return os
				<< static_cast<const IInputSettings&>(obj)
				<< " file: " << obj.file
				<< " voxelResolution: " << obj.voxelResolution
				<< " offset: " << obj.offset.x << "," << obj.offset.y << "," << obj.offset.z
				<< " size: " << obj.size.x << "," << obj.size.y << "," << obj.size.z
				<< " filledCells: " << obj.filledCells;
		}
	};
	enum class InputType
	{
		TORUS,
		SDF
	};

	struct SimulationSettings
	{
		real3 gravity;
		real youngsModulus;
		real poissonsRatio;
		real mass;
		real dampingAlpha;
		real dampingBeta;
		real materialLambda;
		real materialMu;
		bool enableCorotation;
		real timestep;

		real3 initialLinearVelocity;
		real3 initialAngularVelocity;

		real4 groundPlane; //nx, ny, nz, dist
		bool enableCollision;
		real groundStiffness; //the stiffness of the spring that models the ground collision
		real softmaxAlpha; //The smootheness of the softmax. As it goes to infinity, the hard max is approximated
		bool stableCollision;

		int solverIterations;
		real solverTolerance;

		real newmarkTheta;

		bool debugSaveMatrices_ = false;

		friend std::ostream& operator<<(std::ostream& os, const SimulationSettings& obj)
		{
			return os
				<< "gravity: " << obj.gravity.x << "," << obj.gravity.y << "," << obj.gravity.z
				<< " youngsModulus: " << obj.youngsModulus
				<< " poissonsRatio: " << obj.poissonsRatio
				<< " mass: " << obj.mass
				<< " dampingAlpha: " << obj.dampingAlpha
				<< " dampingBeta: " << obj.dampingBeta
				<< " materialLambda: " << obj.materialLambda
				<< " materialMu: " << obj.materialMu
				<< " enableCorotation: " << obj.enableCorotation
				<< " timestep: " << obj.timestep
				<< " initialLinearVelocity: " << obj.initialLinearVelocity.x << "," << obj.initialLinearVelocity.y << "," << obj.initialLinearVelocity.z
				<< " initialAngularVelocity: " << obj.initialAngularVelocity.x << "," << obj.initialAngularVelocity.y << "," << obj.initialAngularVelocity.z
				<< " groundPlane: " << obj.groundPlane.x << "," << obj.groundPlane.y << "," << obj.groundPlane.z << "," << obj.groundPlane.w
				<< " enableCollision: " << obj.enableCollision
				<< " groundStiffness: " << obj.groundStiffness
				<< " softmaxAlpha: " << obj.softmaxAlpha
				<< " stableCollision: " << obj.stableCollision
				<< " solverIterations: " << obj.solverIterations
				<< " solverTolerance: " << obj.solverTolerance
				<< " newmarkTheta: " << obj.newmarkTheta
				<< " debugSaveMatrices_: " << obj.debugSaveMatrices_;
		}
	};

	struct InputVariables
	{
		bool optimizeGravity;
		real3 currentGravity;
		bool optimizeYoungsModulus;
		real currentYoungsModulus;
		bool optimizePoissonRatio;
		real currentPoissonRatio;
		bool optimizeMass;
		real currentMass;
		bool optimizeMassDamping;
		real currentMassDamping;
		bool optimizeStiffnessDamping;
		real currentStiffnessDamping;
		bool optimizeInitialLinearVelocity;
		real3 currentInitialLinearVelocity;
		bool optimizeInitialAngularVelocity;
		real3 currentInitialAngularVelocity;
		bool optimizeGroundPlane;
		real4 currentGroundPlane;

		void disableAll()
		{
			optimizeGravity = false;
			optimizeYoungsModulus = false;
			optimizePoissonRatio = false;
			optimizeMass = false;
			optimizeMassDamping = false;
			optimizeStiffnessDamping = false;
			optimizeInitialLinearVelocity = false;
			optimizeInitialAngularVelocity = false;
			optimizeGroundPlane = false;
		}
		friend std::ostream& operator<<(std::ostream& os, const InputVariables& obj)
		{
			return os
				<< "optimizeGravity: " << obj.optimizeGravity
				<< " currentGravity: " << obj.currentGravity.x << "," << obj.currentGravity.y << "," << obj.currentGravity.z
				<< " optimizeYoungsModulus: " << obj.optimizeYoungsModulus
				<< " currentYoungsModulus: " << obj.currentYoungsModulus
				<< " optimizePoissonRatio: " << obj.optimizePoissonRatio
				<< " currentPoissonRatio: " << obj.currentPoissonRatio
				<< " optimizeMass: " << obj.optimizeMass
				<< " currentMass: " << obj.currentMass
				<< " optimizeMassDamping: " << obj.optimizeMassDamping
				<< " currentMassDamping: " << obj.currentMassDamping
				<< " optimizeStiffnessDamping: " << obj.optimizeStiffnessDamping
				<< " currentStiffnessDamping: " << obj.currentStiffnessDamping
				<< " optimizeInitialLinearVelocity: " << obj.optimizeInitialLinearVelocity
				<< " currentInitialLinearVelocity: " << obj.currentInitialLinearVelocity.x << "," << obj.currentInitialLinearVelocity.y << "," << obj.currentInitialLinearVelocity.z
				<< " optimizeInitialAngularVelocity: " << obj.optimizeInitialAngularVelocity
				<< " currentInitialAngularVelocity: " << obj.currentInitialAngularVelocity.x << "," << obj.currentInitialAngularVelocity.y << "," << obj.currentInitialAngularVelocity.z
				<< " optimizeGroundPlane: " << obj.optimizeGroundPlane
				<< " currentGroundPlane: " << obj.currentGroundPlane.x << "," << obj.currentGroundPlane.y << "," << obj.currentGroundPlane.z << "," << obj.currentGroundPlane.w;
		}
	};

	struct RpropSettings
	{
		std::string epsilon;
		std::string initialStepsize;

		friend std::ostream& operator<<(std::ostream& os, const RpropSettings& obj)
		{
			return os
				<< "epsilon: " << obj.epsilon
				<< " initialStepsize: " << obj.initialStepsize;
		}
	};
	struct GradientDescentSettings
	{
		std::string epsilon;
		std::string initialStepsize;
		std::string maxStepsize;
		std::string minStepsize;

		friend std::ostream& operator<<(std::ostream& os, const GradientDescentSettings& obj)
		{
			return os
				<< "epsilon: " << obj.epsilon
				<< " initialStepsize: " << obj.initialStepsize
				<< " maxStepsize: " << obj.maxStepsize
				<< " minStepsize: " << obj.minStepsize;
		}
	};
	struct LbfgsSettings
	{
		std::string epsilon;
		int past;
		std::string delta;
		enum LineSearchAlgorithm
		{
			Armijo,
			Wolfe,
			StrongWolfe
		} lineSearchAlg;
		int linesearchMaxTrials;
		std::string linesearchMinStep;
		std::string linesearchMaxStep;
		std::string linesearchTol;

		friend std::ostream& operator<<(std::ostream& os, const LbfgsSettings& obj)
		{
			return os
				<< "epsilon: " << obj.epsilon
				<< " past: " << obj.past
				<< " delta: " << obj.delta
				<< " lineSearchAlg: " << obj.lineSearchAlg
				<< " linesearchMaxTrials: " << obj.linesearchMaxTrials
				<< " linesearchMinStep: " << obj.linesearchMinStep
				<< " linesearchMaxStep: " << obj.linesearchMaxStep
				<< " linesearchTol: " << obj.linesearchTol;
		}
	};
	enum class Optimizer
	{
		RPROP,
		GD,
		LBFGS
	};
	struct AdjointSettings
	{
		int numIterations;
		bool memorySaving;
		bool normalizeUnits;
		RpropSettings rpropSettings;
		GradientDescentSettings gdSettings;
		LbfgsSettings lbfgsSettings;
		Optimizer optimizer;

		InputVariables variables;

		friend std::ostream& operator<<(std::ostream& os, const AdjointSettings& obj)
		{
			return os
				<< "numIterations: " << obj.numIterations
				<< " memorySaving: " << obj.memorySaving
				<< " normalizeUnits: " << obj.normalizeUnits
				<< " rpropSettings: " << obj.rpropSettings
				<< " gdSettings: " << obj.gdSettings
				<< " lbfgsSettings: " << obj.lbfgsSettings
				<< " optimizer: " << (obj.optimizer==Optimizer::RPROP ? "Rprop" : obj.optimizer==Optimizer::GD ? "GD" : "LBFGS")
				<< " variables: " << obj.variables;
		}
	};

	struct CostFunctionPartialObservationsSettings
	{
		std::string timestepWeights;
		int numCameras;
		real radius;
		real centerHeight;
		int resolution;
		real noise;
		bool gpuPreprocess;
		bool gpuEvaluate;
		real maxSdf;

		friend std::ostream& operator<<(std::ostream& os, const CostFunctionPartialObservationsSettings& obj)
		{
			return os
				<< "timestepWeights: " << obj.timestepWeights
				<< " numCameras: " << obj.numCameras
				<< " radius: " << obj.radius
				<< " centerHeight: " << obj.centerHeight
				<< " resolution: " << obj.resolution
				<< " noise: " << obj.noise
				<< " gpuPreprocess: " << obj.gpuPreprocess
				<< " gpuEvaluate: " << obj.gpuEvaluate
				<< " maxSdf: " << obj.maxSdf;
		}
	};

	struct CostFunctionActiveDisplacementsSettings
	{
		std::string timestepWeights;
		real displacementWeight;
		real velocityWeight;
		real noise;

		friend std::ostream& operator<<(std::ostream& os, const CostFunctionActiveDisplacementsSettings& obj)
		{
			return os
				<< "timestepWeights: " << obj.timestepWeights
				<< " displacementWeight: " << obj.displacementWeight
				<< " velocityWeight: " << obj.velocityWeight
				<< " noise: " << obj.noise;
		}
	};

	struct SyntheticInput
	{
		InputTorusSettings inputTorusSettings;
		InputSdfSettings inputSdfSettings;
		InputType inputType;
		SimulationSettings simulationSettings;
		AdjointSettings adjointSettings;
		CostFunctionPartialObservationsSettings costFunctionPartialObservationsSettings;
		CostFunctionActiveDisplacementsSettings costFunctionActiveDisplacementsSettings;

		friend std::ostream& operator<<(std::ostream& os, const SyntheticInput& obj)
		{
			return os
				<< "inputTorusSettings: " << obj.inputTorusSettings
				<< " inputSdfSettings: " << obj.inputSdfSettings
				<< " inputType: " << (obj.inputType==InputType::TORUS ? "Torus" : "Sdf")
				<< " simulationSettings: " << obj.simulationSettings
				<< " adjointSettings: " << obj.adjointSettings
				<< " costFunctionPartialObservationsSettings: " << obj.costFunctionPartialObservationsSettings
				<< " costFunctionActiveDisplacementsSettings: " << obj.costFunctionActiveDisplacementsSettings;
		}
	};

	typedef std::shared_ptr<const void> InputConfigPointer;
	struct RecordedInput
	{
		std::string scanFile;
		float framerate;
		int numTotalFrames;
		int voxelResolution;
		std::string groundTruthMeshPath;
		InputSdfSettings inputSdfSettings;
		SimulationSettings simulationSettings;
		AdjointSettings adjointSettings;
		bool sscGpuEvaluate;
		real sscMaxSdf;
		int costIntermediateSteps;
		int costNumSteps;

		InputConfigPointer inputConfig_;

		friend std::ostream& operator<<(std::ostream& os, const RecordedInput& obj)
		{
			return os
				<< "scanFile: " << obj.scanFile
				<< " numTotalFrames: " << obj.numTotalFrames
				<< " voxelResolution: " << obj.voxelResolution
				<< " inputSdfSettings: " << obj.inputSdfSettings
				<< " simulationSettings: " << obj.simulationSettings
				<< " adjointSettings: " << obj.adjointSettings
				<< " sscGpuEvaluate: " << obj.sscGpuEvaluate
				<< " sscMaxSdf: " << obj.sscMaxSdf
				<< " costIntermediateSteps: " << obj.costIntermediateSteps
				<< " costNumSteps: " << obj.costNumSteps;
		}
	};

	SyntheticInput loadSyntheticInput(
		const std::string& filename);

	RecordedInput loadRecordedInput(
		const std::string& filename);

	typedef std::shared_ptr<void> SimulationPointer;
	typedef std::shared_ptr<void> ResultsPointer;
	typedef std::shared_ptr<void> CostFunctionPointer;

	SimulationPointer createSimulationFromTorus(
		const InputTorusSettings& inputTorusSettings);

	SimulationPointer createSimulationFromSdf(
		const InputSdfSettings& inputSdfSettings, 
		const std::string& folder);

	SimulationPointer createSimulationFromRecordedInput(
		const RecordedInput& input);

	void simulationSetSettings(
		SimulationPointer simulation,
		const SimulationSettings& settings);

	void simulationReset(SimulationPointer simulation);



	ResultsPointer createSimulationResults(SimulationPointer simulation);

	void simulationSolveForward(
		SimulationPointer simulation,
		ResultsPointer results);

	CostFunctionPointer createDirectDisplacementCostFunction(
		const CostFunctionActiveDisplacementsSettings& settings,
		ResultsPointer results);

	CostFunctionPointer createPartialObservationsCostFunction(
		const CostFunctionPartialObservationsSettings& settings,
		ResultsPointer results);

	std::tuple<ResultsPointer, CostFunctionPointer> 
	createResultsAndCostFunctionFromRecordedInput(
		const RecordedInput& input,
		SimulationPointer simulation);

	Eigen::MatrixXf partialObservationsGetDepthImage(
		CostFunctionPointer costFun, int frame, int camera);

	std::vector<glm::vec3> partialObservationsGetObservations(
		CostFunctionPointer costFun, int frame, int camera);

	typedef std::function<void(
		const SimulationSettings& var, 
		const SimulationSettings& grad, 
		real cost)> Callback;

	void solveAdjoint(
		ResultsPointer results,
		const AdjointSettings& settings,
		CostFunctionPointer costFunction,
		Callback callback);

	real evaluateGradient(
		ResultsPointer results,
		const InputVariables& variables,
		CostFunctionPointer costFunction,
		SimulationSettings& grad);

	real evaluateGradientFiniteDifferences(
		ResultsPointer results,
		const InputVariables& variables,
		CostFunctionPointer costFunction,
		real finiteDifferencesDelta,
		SimulationSettings& grad);


	void visualizationSetInput(SimulationPointer sim);
	void visualizationLoadHighResInput(const std::string& filename);
	void visualizationSetState(SimulationPointer sim);
	void visualizationExportMCMesh(const std::string& outputPath);
	void visualizationExportHighResMesh(
		const std::string& outputPath,
		bool includeNormals, bool includeOriginalPositions);
}