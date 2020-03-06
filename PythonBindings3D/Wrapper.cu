#include "Wrapper.h"

#include <Commons3D.h>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <direct.h>
#include <random>
#include <string.h>

#include <SoftBodyGrid3D.h>
#include <AdjointSolver.h>
#include <BackgroundWorker2.h>
#include <CostFunctions.h>
#include "tinyformat.h"

#include <cuMat/src/Errors.h>
#include "cuPrintf.cuh"
#include "InputConfig.h"
#include "GroundTruthToSdf.h"
#include "InputDataLoader.h"
#include "VolumeVisualization.h"

namespace ar3d_wrapper
{
	//--------------------------------------
	// HELPER
	//--------------------------------------

	template<typename T>
	std::shared_ptr<T> cast(std::shared_ptr<void> ptr)
	{
		return std::static_pointer_cast<T>(ptr);
	}
	template<typename T>
	std::shared_ptr<const T> cast(std::shared_ptr<const void> ptr)
	{
		return std::static_pointer_cast<const T>(ptr);
	}

	ar3d::SoftBodySimulation3D::InputTorusSettings w2s(const InputTorusSettings& w)
	{
		ar3d::SoftBodySimulation3D::InputTorusSettings s = {};
		memcpy(&s, &w, sizeof(InputTorusSettings));
		return s;
	}
	ar3d::SoftBodySimulation3D::InputSdfSettings w2s(const InputSdfSettings& w)
	{
		ar3d::SoftBodySimulation3D::InputSdfSettings s = {};
		s.enableDirichlet = w.enableDirichlet;
		s.centerDirichlet = w.centerDirichlet;
		s.halfsizeDirichlet = w.halfsizeDirichlet;
		s.sampleIntegrals = w.sampleIntegrals;
		s.diffusionDistance = w.diffusionDistance;
		s.zeroCutoff = w.zeroCutoff;
		s.file = w.file;
		s.voxelResolution = w.voxelResolution;
		s.offset = w.offset;
		s.size = w.size;
		s.filledCells = w.filledCells;
		//memcpy(&s, &w, sizeof(InputSdfSettings));
		//s.file = std::string(w.file);
		return s;
	}
	ar3d::SoftBodySimulation3D::Settings w2s(const SimulationSettings& w)
	{
		ar3d::SoftBodySimulation3D::Settings s = {};
		memcpy(&s, &w, sizeof(SimulationSettings));
		return s;
	}
	ar3d::AdjointSolver::InputVariables w2s(const InputVariables& w)
	{
		ar3d::AdjointSolver::InputVariables s = {};
		memcpy(&s, &w, sizeof(InputVariables));
		return s;
	}
	ar3d::AdjointSolver::Settings w2s(const AdjointSettings& w)
	{
		ar3d::AdjointSolver::Settings s = {};
		s.numIterations_ = w.numIterations;
		s.memorySaving_ = w.memorySaving;
		s.normalizeUnits_ = w.normalizeUnits;

		s.optimizer_ =
			w.optimizer == Optimizer::RPROP ? ar3d::AdjointSolver::Settings::RPROP
			: w.optimizer == Optimizer::LBFGS ? ar3d::AdjointSolver::Settings::LBFGS
			: ar3d::AdjointSolver::Settings::GRADIENT_DESCENT;
		s.rpropSettings_.epsilon_ = w.rpropSettings.epsilon;
		s.rpropSettings_.initialStepsize_ = w.rpropSettings.initialStepsize;
		s.gradientDescentSettings_.epsilon_ = w.gdSettings.epsilon;
		s.gradientDescentSettings_.linearStepsize_ = w.gdSettings.initialStepsize;
		s.gradientDescentSettings_.maxStepsize_ = w.gdSettings.maxStepsize;
		s.gradientDescentSettings_.minStepsize_ = w.gdSettings.minStepsize;
		s.lbfgsSettings_.epsilon_ = w.lbfgsSettings.epsilon;
		s.lbfgsSettings_.past_ = w.lbfgsSettings.past;
		s.lbfgsSettings_.delta_ = w.lbfgsSettings.delta;
		s.lbfgsSettings_.lineSearchAlg_ = ar3d::AdjointSolver::Settings::LbfgsSettings::LineSearchAlgorithm(w.lbfgsSettings.lineSearchAlg);
		s.lbfgsSettings_.linesearchMaxTrials_ = w.lbfgsSettings.linesearchMaxTrials;
		s.lbfgsSettings_.linesearchMinStep_ = w.lbfgsSettings.linesearchMinStep;
		s.lbfgsSettings_.linesearchMaxStep_ = w.lbfgsSettings.linesearchMaxStep;
		s.lbfgsSettings_.linesearchTol_ = w.lbfgsSettings.linesearchTol;

		s.variables_ = w2s(w.variables);
		return s;
	}
	ar3d::CostFunctionPartialObservations::GUI w2s(const CostFunctionPartialObservationsSettings& w)
	{
		ar3d::CostFunctionPartialObservations::GUI s = {};
		s.timestepWeights_ = w.timestepWeights;
		s.numCameras_ = w.numCameras;
		s.radius_ = w.radius;
		s.centerHeight_ = w.centerHeight;
		s.resolution_ = w.resolution;
		s.noise_ = w.noise;
		s.gpuPreprocess_ = w.gpuPreprocess;
		s.gpuEvaluate_ = w.gpuEvaluate;
		s.maxSdf_ = w.maxSdf;
		s.update();
		return s;
	}
	ar3d::CostFunctionActiveDisplacements::GUI w2s(const CostFunctionActiveDisplacementsSettings& w)
	{
		ar3d::CostFunctionActiveDisplacements::GUI s = {};
		s.timestepWeights_ = w.timestepWeights;
		s.displacementWeight_ = w.displacementWeight;
		s.velocityWeight_ = w.velocityWeight;
		s.noise_ = w.noise;
		return s;
	}

	InputTorusSettings s2w(const ar3d::SoftBodySimulation3D::InputTorusSettings& s)
	{
		InputTorusSettings w = {};
		memcpy(&w, &s, sizeof(InputTorusSettings));
		return w;
	}
	InputSdfSettings s2w(const ar3d::SoftBodySimulation3D::InputSdfSettings& s)
	{
		InputSdfSettings w = {};
		memcpy(&w, &s, sizeof(InputSdfSettings));
		w.file = s.file;
		return w;
	}
	SimulationSettings s2w(const ar3d::SoftBodySimulation3D::Settings& s)
	{
		SimulationSettings w = {};
		memcpy(&w, &s, sizeof(SimulationSettings));
		return w;
	}
	InputVariables s2w(const ar3d::AdjointSolver::InputVariables& s)
	{
		InputVariables w = {};
		memcpy(&w, &s, sizeof(InputVariables));
		return w;
	}
	AdjointSettings s2w(const ar3d::AdjointSolver::Settings& s)
	{
		AdjointSettings w = {};
		w.numIterations = s.numIterations_;
		w.memorySaving = s.memorySaving_;
		w.normalizeUnits = s.normalizeUnits_;

		w.rpropSettings.epsilon = s.rpropSettings_.epsilon_;
		w.rpropSettings.initialStepsize = s.rpropSettings_.initialStepsize_;
		w.gdSettings.epsilon = s.gradientDescentSettings_.epsilon_;
		w.gdSettings.initialStepsize = s.gradientDescentSettings_.linearStepsize_;
		w.gdSettings.maxStepsize = s.gradientDescentSettings_.maxStepsize_;
		w.gdSettings.minStepsize = s.gradientDescentSettings_.minStepsize_;
		w.lbfgsSettings.epsilon = s.lbfgsSettings_.epsilon_;
		w.lbfgsSettings.past = s.lbfgsSettings_.past_;
		w.lbfgsSettings.delta = s.lbfgsSettings_.delta_;
		w.lbfgsSettings.lineSearchAlg = LbfgsSettings::LineSearchAlgorithm(s.lbfgsSettings_.lineSearchAlg_);
		w.lbfgsSettings.linesearchMaxTrials = s.lbfgsSettings_.linesearchMaxTrials_;
		w.lbfgsSettings.linesearchMinStep = s.lbfgsSettings_.linesearchMinStep_;
		w.lbfgsSettings.linesearchMaxStep = s.lbfgsSettings_.linesearchMaxStep_;
		w.lbfgsSettings.linesearchTol = s.lbfgsSettings_.linesearchTol_;
		w.optimizer = s.optimizer_ ==
			ar3d::AdjointSolver::Settings::RPROP ? Optimizer::RPROP
			: s.optimizer_ == ar3d::AdjointSolver::Settings::GRADIENT_DESCENT
			? Optimizer::GD : Optimizer::LBFGS;

		w.variables = s2w(s.variables_);
		return w;
	}
	CostFunctionPartialObservationsSettings s2w(const ar3d::CostFunctionPartialObservations::GUI& s)
	{
		CostFunctionPartialObservationsSettings w = {};
		w.timestepWeights = s.timestepWeights_;
		w.numCameras = s.numCameras_;
		w.radius = s.radius_;
		w.centerHeight = s.centerHeight_;
		w.resolution = s.resolution_;
		w.noise = s.noise_;
		w.gpuPreprocess = s.gpuPreprocess_;
		w.gpuEvaluate = s.gpuEvaluate_;
		w.maxSdf = s.maxSdf_;
		return w;
	}
	CostFunctionActiveDisplacementsSettings s2w(const ar3d::CostFunctionActiveDisplacements::GUI& s)
	{
		CostFunctionActiveDisplacementsSettings w = {};
		w.timestepWeights = s.timestepWeights_;
		w.displacementWeight = s.displacementWeight_;
		w.velocityWeight = s.velocityWeight_;
		w.noise = s.noise_;
		return w;
	}


	//--------------------------------------
	// Wrapper implementations
	//--------------------------------------

	void initialize()
	{
		CUMAT_SAFE_CALL(cudaPrintfInit());
	}

	void cleanup()
	{
		cudaPrintfEnd();
	}

	SyntheticInput loadSyntheticInput(
		const std::string& filename)
	{
		cinder::DataSourceRef source = cinder::loadFile(filename);
		if (source == nullptr)
		{
			throw std::exception(("Unable to open file " + filename).c_str());
		}
		cinder::JsonTree root = cinder::JsonTree(source);

		SyntheticInput input;

		ar3d::SoftBodySimulation3D::InputSdfSettingsGui inputSdfSettingsGui;
		inputSdfSettingsGui.load(root.getChild("InputSdfSettings"));
		input.inputSdfSettings = s2w(inputSdfSettingsGui.getSettings());

		ar3d::SoftBodySimulation3D::InputTorusSettingsGui inputTorusSettingsGui;
		inputTorusSettingsGui.load(root.getChild("InputTorusSettings"));
		input.inputTorusSettings = s2w(inputTorusSettingsGui.getSettings());

		std::string inputCaseStr = root.getValueForKey<std::string>("InputCase");
		if (inputCaseStr == "Torus")
			input.inputType = InputType::TORUS;
		else
			input.inputType = InputType::SDF;

		ar3d::SoftBodySimulation3D::SettingsGui simulationSettingsGui;
		simulationSettingsGui.load(root.getChild("SimulationSettings"));
		input.simulationSettings = s2w(simulationSettingsGui.getSettings());

		ar3d::AdjointSolver::GUI adjointSettingsGui;
		adjointSettingsGui.load(root.getChild("AdjointSettings"));
		input.adjointSettings = s2w(adjointSettingsGui.getSettings());

		ar3d::CostFunctionPartialObservations::GUI costFunctionPartialObservations;
		costFunctionPartialObservations.load(root.getChild("CostFunctionPartialObservations"));
		input.costFunctionPartialObservationsSettings = s2w(costFunctionPartialObservations);

		ar3d::CostFunctionActiveDisplacements::GUI costFunctionActiveDisplacements;
		costFunctionActiveDisplacements.load(root.getChild("CostFunctionActiveDisplacements"));
		input.costFunctionActiveDisplacementsSettings = s2w(costFunctionActiveDisplacements);

		return input;
	}

	RecordedInput loadRecordedInput(const std::string& filename)
	{
		cinder::DataSourceRef source = cinder::loadFile(filename);
		if (source == nullptr)
		{
			throw std::exception(("Unable to open file " + filename).c_str());
		}
		cinder::JsonTree root = cinder::JsonTree(source);

		RecordedInput input = {};

		auto inspectScan = root.getChild("InspectScan");
		input.scanFile = inspectScan.getValueForKey("ScanFile");
		ar3d::InputConfigPtr inputConfig = ar3d::InputConfig::loadFromJson(input.scanFile);
		input.inputConfig_ = inputConfig;
		input.numTotalFrames = inputConfig->duration;
		input.framerate = inputConfig->framerate;
		input.groundTruthMeshPath = inputConfig->groundTruthMeshPath;
		input.voxelResolution = inspectScan.getChild("Reconstruction")
			.getValueForKey<int>("Resolution");

		auto reconstruction = root.getChild("Reconstruction");

		ar3d::SoftBodySimulation3D::InputSdfSettingsGui inputSdfSettingsGui;
		inputSdfSettingsGui.load(reconstruction.getChild("Input"), true);
		input.inputSdfSettings = s2w(inputSdfSettingsGui.getSettings());

		ar3d::SoftBodySimulation3D::SettingsGui simulationSettingsGui;
		simulationSettingsGui.load(reconstruction.getChild("Elasticity"));
		input.simulationSettings = s2w(simulationSettingsGui.getSettings());

		ar3d::AdjointSolver::GUI adjointSettingsGui;
		adjointSettingsGui.load(reconstruction.getChild("Adjoint"), true);
		input.adjointSettings = s2w(adjointSettingsGui.getSettings());

		input.sscGpuEvaluate = true;
		input.sscMaxSdf = 1;
		if (reconstruction.hasChild("CostIntermediateSteps"))
			input.costIntermediateSteps = reconstruction.getValueForKey<int>("CostIntermediateSteps");
		if (reconstruction.hasChild("CostNumSteps"))
			input.costNumSteps = reconstruction.getValueForKey<int>("CostNumSteps");
		
		return input;
	}

	SimulationPointer createSimulationFromTorus(
		const InputTorusSettings& inputTorusSettings)
	{
		ar3d::SoftBodySimulation3D::InputTorusSettings torusSettings = w2s(inputTorusSettings);
		ar3d::SoftBodyGrid3D::Input input = ar3d::SoftBodyGrid3D::createTorus(torusSettings);
		return std::make_shared<ar3d::SoftBodyGrid3D>(input);
	}

	SimulationPointer createSimulationFromSdf(
		const InputSdfSettings& inputSdfSettings, 
		const std::string& folder)
	{
		std::string currentDir = std::experimental::filesystem::current_path().generic_string();
		std::string targetDir = std::experimental::filesystem::canonical(folder).generic_string();
		std::cout << "currentDir=" << currentDir << std::endl;
		std::cout << "targetDir=" << targetDir << std::endl;
		std::experimental::filesystem::current_path(targetDir);
		std::cout << "chdir successfull" << std::endl;

		ar3d::SoftBodySimulation3D::InputSdfSettings sdfSettings = w2s(inputSdfSettings);
		std::cout << "settings=" << sdfSettings << std::endl;
		static ar3d::SoftBodyGrid3D::Input input;
		input = ar3d::SoftBodyGrid3D::createFromFile(sdfSettings);
		std::cout << "Input loaded" << std::endl;

		auto result = std::make_shared<ar3d::SoftBodyGrid3D>(input);
		std::cout << "Simulation created" << std::endl;

		std::experimental::filesystem::current_path(currentDir);
		std::cout << "Done" << std::endl;

		/*
		result = nullptr;
		std::cout << "Simulation deleted" << std::endl;
		input = ar3d::SoftBodyGrid3D::Input();
		std::cout << "Input deleted" << std::endl;
		sdfSettings = {};
		std::cout << "Settings deleted" << std::endl;
		*/

		return result;
	}

	SimulationPointer createSimulationFromRecordedInput(const RecordedInput& input)
	{
		std::cout << "Load input model and convert to SDF with a voxel resolution of "
			<< input.voxelResolution << std::endl;
		ar3d::InputConfigPtr inputConfig = cast<const ar3d::InputConfig>(input.inputConfig_);
		ar3d::WorldGridRealDataPtr referenceSdf
			= ar3d::groundTruthToSdf(inputConfig, 0, input.voxelResolution);

		std::cout << "Create simulation" << std::endl;
		ar3d::SoftBodySimulation3D::InputSdfSettings inputSettings = w2s(input.inputSdfSettings);

		ar3d::SoftBodyGrid3D::Input gridInput = ar3d::SoftBodyGrid3D::createFromSdf(
			inputSettings, referenceSdf);
		auto sim = std::make_shared<ar3d::SoftBodyGrid3D>(gridInput);
		sim->setSettings(w2s(input.simulationSettings));
		return sim;
	}

	void simulationSetSettings(
		SimulationPointer simulation, 
		const SimulationSettings& settings)
	{
		auto s = w2s(settings);
		s.validate();
		cast<ar3d::SoftBodyGrid3D>(simulation)->setSettings(s);
	}

	void simulationReset(SimulationPointer simulation)
	{
		cast<ar3d::SoftBodyGrid3D>(simulation)->reset();
	}

	ResultsPointer createSimulationResults(SimulationPointer simulation)
	{
		auto simResults = std::make_shared<ar3d::SimulationResults3D>();
		simResults->states_.clear();
		simResults->input_ = cast<ar3d::SoftBodyGrid3D>(simulation)->getInput();
		simResults->settings_ = cast<ar3d::SoftBodyGrid3D>(simulation)->getSettings();
		return simResults;
	}

	void simulationSolveForward(SimulationPointer simulation, ResultsPointer results)
	{
		ar3d::BackgroundWorker2 worker;
		cast<ar3d::SoftBodyGrid3D>(simulation)->solve(true, &worker, true);
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		if (results)
			cast<ar3d::SimulationResults3D>(results)->states_.push_back(cast<ar3d::SoftBodyGrid3D>(simulation)->getState().deepClone());
	}

	CostFunctionPointer createDirectDisplacementCostFunction(const CostFunctionActiveDisplacementsSettings& settings,
		ResultsPointer results)
	{
		auto s = w2s(settings);
		auto cf = std::make_shared<ar3d::CostFunctionActiveDisplacements>(
			cast<ar3d::SimulationResults3D>(results), &s);
		return cf;
	}

	CostFunctionPointer createPartialObservationsCostFunction(const CostFunctionPartialObservationsSettings& settings,
		ResultsPointer results)
	{
		auto s = w2s(settings);
		auto cf = std::make_shared<ar3d::CostFunctionPartialObservations>(
			cast<ar3d::SimulationResults3D>(results), &s);
		std::cout << "SSC, gpuPreprocess=" << cf->observations_.gpuPreprocess_
			<< ", gpuEvaluate=" << cf->observations_.gpuEvaluate_ << std::endl;
		ar3d::BackgroundWorker2 worker;
		cf->preprocess(&worker);
		return cf;
	}

	std::tuple<ResultsPointer, CostFunctionPointer> createResultsAndCostFunctionFromRecordedInput(
		const RecordedInput& input, SimulationPointer simulation)
	{
		//cast data
		ar3d::SoftBodySimulation3D::Settings simulationSettings = w2s(input.simulationSettings);
		ar3d::AdjointSolver::Settings adjointSettings = w2s(input.adjointSettings);
		ar3d::InputConfigPtr inputConfig = cast<const ar3d::InputConfig>(input.inputConfig_);
		std::shared_ptr<ar3d::SoftBodyGrid3D> sim = cast<ar3d::SoftBodyGrid3D>(simulation);

		//create result datastructure
		ar3d::SimulationResults3DPtr results = std::make_shared<ar3d::SimulationResults3D>();
		ar3d::real timestep = 1.0 / (inputConfig->framerate * (input.costIntermediateSteps + 1));
		std::cout << "IntermediateSteps=" << input.costIntermediateSteps <<
			" -> timestep=" << timestep << std::endl;
		results->input_ = sim->getInput();
		results->settings_ = sim->getSettings();
		results->settings_.timestep_ = timestep;

		//create data loader
		auto dataLoader = std::make_unique<ar3d::InputDataLoader>(inputConfig);
		ar3d::BackgroundWorker2 worker;

		//create cost function
		int numSteps = input.costNumSteps == 0
			? inputConfig->duration - 1
			: input.costNumSteps;
		std::cout << "Load " << numSteps << " observations and create cost function" << std::endl;
		std::vector<ar3d::real> timestepWeights;
		ar3d::CostFunctionPartialObservations::Observations observations;
		observations.maxSdf_ = input.sscMaxSdf;
		observations.gpuEvaluate_ = input.sscGpuEvaluate;
		observations.noise_ = 0; //not needed because we already have the observations
		observations.numCameras_ = inputConfig->cameras.size();
		observations.cameras_.resize(inputConfig->cameras.size());
		for (size_t i = 0; i < inputConfig->cameras.size(); ++i) {
			observations.cameras_[i] = inputConfig->cameras[i].camera;
		}
		for (int i = 1; i <= numSteps; ++i) {
			//add in-between frames without weight
			for (int j = 0; j < input.costIntermediateSteps; ++j) {
				timestepWeights.push_back(0);
				observations.observations_.emplace_back();
			}
			//load camera images and copy them to the GPU
			std::cout << " " << i << std::flush;
			timestepWeights.push_back(1);
			ar3d::CostFunctionPartialObservations::Observation observation;
			observation.resize(inputConfig->cameras.size());
			const auto dataFrame = dataLoader->loadFrame(i, &worker);
			for (size_t j = 0; j < inputConfig->cameras.size(); ++j) {
				Eigen::Matrix<ar3d::real, Eigen::Dynamic, Eigen::Dynamic> host = dataFrame->cameraImages[j].depthMatrix.cast<ar3d::real>().matrix();
				observation[j] = ar3d::CostFunctionPartialObservations::Image::fromEigen(host);
			}
			observations.observations_.push_back(observation);
		}
		std::cout << std::endl;
		assert(timestepWeights.size() == observations.observations_.size());
		assert(timestepWeights.size() > 0);
		ar3d::CostFunctionPtr costFunction = std::make_shared<ar3d::CostFunctionPartialObservations>(timestepWeights, observations);

		std::cout << "Done" << std::endl;
		return { results, costFunction };
	}

	Eigen::MatrixXf partialObservationsGetDepthImage(CostFunctionPointer costFun, int frame, int camera)
	{
		auto c = cast<ar3d::CostFunctionPartialObservations>(costFun);
		return c->getDepthImage(frame, camera);
	}

	std::vector<glm::vec3> partialObservationsGetObservations(CostFunctionPointer costFun, int frame, int camera)
	{
		auto c = cast<ar3d::CostFunctionPartialObservations>(costFun);
		return c->getObservations(frame, camera);
	}

	void solveAdjoint(ResultsPointer results, const AdjointSettings& settings, CostFunctionPointer costFunction,
		Callback callback)
	{
		ar3d::AdjointSolver solver(
			cast<ar3d::SimulationResults3D>(results), 
			w2s(settings), 
			cast<ar3d::ICostFunction>(costFunction));
		ar3d::BackgroundWorker2 worker;
		ar3d::AdjointSolver::Callback_t c = [callback](const ar3d::SoftBodySimulation3D::Settings& var, const ar3d::SoftBodySimulation3D::Settings& gradient, ar3d::real cost)
		{
			callback(s2w(var), s2w(gradient), cost);
		};
		solver.solve(c, &worker);
	}

	real evaluateGradient(ResultsPointer results, const InputVariables& variables, CostFunctionPointer costFunction,
		SimulationSettings& grad)
	{
		ar3d::BackgroundWorker2 worker;
		const ar3d::SoftBodyGrid3D::Input& input = cast<ar3d::SimulationResults3D>(results)->input_;
		const ar3d::SoftBodySimulation3D::Settings& settings = cast<ar3d::SimulationResults3D>(results)->settings_;
		const ar3d::AdjointSolver::InputVariables vars = w2s(variables);
		ar3d::CostFunctionPtr cf = cast<ar3d::ICostFunction>(costFunction);
		ar3d::AdjointSolver::AdjointVariables adjointVariablesOut = { 0 };

		real cost = ar3d::AdjointSolver::computeGradient(
			input, settings, vars, cf, adjointVariablesOut, 
			true, &worker, nullptr);

		grad.dampingAlpha = adjointVariablesOut.adjMassDamping_;
		grad.dampingBeta = adjointVariablesOut.adjStiffnessDamping_;
		grad.gravity = make_double4(adjointVariablesOut.adjGravity_.x, adjointVariablesOut.adjGravity_.y, adjointVariablesOut.adjGravity_.z, 0);
		grad.initialAngularVelocity = make_double4(adjointVariablesOut.adjInitialAngularVelocity.x, adjointVariablesOut.adjInitialAngularVelocity.y, adjointVariablesOut.adjInitialAngularVelocity.z, 0);
		grad.initialLinearVelocity = make_double4(adjointVariablesOut.adjInitialLinearVelocity.x, adjointVariablesOut.adjInitialLinearVelocity.y, adjointVariablesOut.adjInitialLinearVelocity.z, 0);
		grad.mass = adjointVariablesOut.adjMass_;
		grad.groundPlane = adjointVariablesOut.adjGroundPlane_;
		grad.poissonsRatio = adjointVariablesOut.adjPoissonRatio_;
		grad.youngsModulus = adjointVariablesOut.adjYoungsModulus_;

		return cost;
	}

	real evaluateGradientFiniteDifferences(ResultsPointer results, const InputVariables& variables,
		CostFunctionPointer costFunction, real finiteDifferencesDelta, SimulationSettings& grad)
	{
		ar3d::BackgroundWorker2 worker;
		const ar3d::SoftBodyGrid3D::Input& input = cast<ar3d::SimulationResults3D>(results)->input_;
		const ar3d::SoftBodySimulation3D::Settings& settings = cast<ar3d::SimulationResults3D>(results)->settings_;
		const ar3d::AdjointSolver::InputVariables vars = w2s(variables);
		ar3d::CostFunctionPtr cf = cast<ar3d::ICostFunction>(costFunction);
		ar3d::AdjointSolver::AdjointVariables adjointVariablesOut = { 0 };

		real cost = ar3d::AdjointSolver::computeGradientFiniteDifferences(
			input, settings, vars, cf, adjointVariablesOut,
			finiteDifferencesDelta, &worker, nullptr);

		grad.dampingAlpha = adjointVariablesOut.adjMassDamping_;
		grad.dampingBeta = adjointVariablesOut.adjStiffnessDamping_;
		grad.gravity = make_double4(adjointVariablesOut.adjGravity_.x, adjointVariablesOut.adjGravity_.y, adjointVariablesOut.adjGravity_.z, 0);
		grad.initialAngularVelocity = make_double4(adjointVariablesOut.adjInitialAngularVelocity.x, adjointVariablesOut.adjInitialAngularVelocity.y, adjointVariablesOut.adjInitialAngularVelocity.z, 0);
		grad.initialLinearVelocity = make_double4(adjointVariablesOut.adjInitialLinearVelocity.x, adjointVariablesOut.adjInitialLinearVelocity.y, adjointVariablesOut.adjInitialLinearVelocity.z, 0);
		grad.mass = adjointVariablesOut.adjMass_;
		grad.groundPlane = adjointVariablesOut.adjGroundPlane_;
		grad.poissonsRatio = adjointVariablesOut.adjPoissonRatio_;
		grad.youngsModulus = adjointVariablesOut.adjYoungsModulus_;

		return cost;
	}

	static ar3d::VolumeVisualization vis(nullptr, nullptr, nullptr);

	void visualizationSetInput(SimulationPointer simulation)
	{
		std::shared_ptr<ar3d::SoftBodyGrid3D> sim = cast<ar3d::SoftBodyGrid3D>(simulation);
		vis.setInput(sim->getInput());
	}

	void visualizationLoadHighResInput(const std::string& filename)
	{
		vis.setHighResInputMesh(filename);
	}

	void visualizationSetState(SimulationPointer simulation)
	{
		std::shared_ptr<ar3d::SoftBodyGrid3D> sim = cast<ar3d::SoftBodyGrid3D>(simulation);
		vis.update(sim->getState());
	}

	void visualizationExportMCMesh(const std::string& outputPath)
	{
		vis.saveMCMesh(outputPath);
	}

	void visualizationExportHighResMesh(const std::string& outputPath, 
		bool includeNormals, bool includeOriginalPositions)
	{
		if (vis.hasHighResMesh())
			vis.saveHighResultMesh(outputPath, includeNormals, includeOriginalPositions);
		else
			std::cout << "no high-resolution mesh loaded" << std::endl;
	}
}
