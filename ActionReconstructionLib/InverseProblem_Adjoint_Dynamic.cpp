#include "InverseProblem_Adjoint_Dynamic.h"

#include <sstream>
#include <fstream>
#include <LBFGS.h>
#include <tinyformat.h>
#include <cinder/app/App.h>
#include <stack>

#include "GradientDescent.h"
#include "Integration.h"
#include "InverseProblem_Adjoint_InitialConfiguration.h"
#include "AdjointUtilities.h"
#include "InverseProblem_Adjoint_YoungsModulus.h"

#define CHECK_WORKER if (worker->isInterrupted()) return output;

ar::InverseProblem_Adjoint_Dynamic::InverseProblem_Adjoint_Dynamic()
	: numIterations(10)
    , optimizeMass(false)
	, initialMass(2)
    , optimizeMassDamping(false)
	, initialMassDamping(0.1)
    , optimizeStiffnessDamping(false)
	, initialStiffnessDamping(0.1)
    , optimizeInitialPosition(false)
	, volumePrior(0.1)
	, optimizeGroundPosition(false)
	, initialGroundPlaneHeight(0.1)
	, initialGroundPlaneAngle(0.05)
	, optimizeYoungsModulus(false)
	, initialYoungsModulus(200)
	, optimizePoissonRatio(false)
	, initialPoissonRatio(0.45)
	, timestepWeights("")
	, costPositionWeighting(1)
	, costVelocityWeighting(0.2)
	, costUseMeanAmplitude(false)
	, gridBypassAdvection(false)
    , gridUsePartialObservation(false)
{}

ar::InverseProblemOutput ar::InverseProblem_Adjoint_Dynamic::solveGrid(int deformedTimestep, BackgroundWorker * worker, IntermediateResultCallback_t callback)
{
	InverseProblemOutput output = {};

	// Create simulation
	worker->setStatus("Adjoint - Dynamic: create simulation");
	SoftBodyGrid2D simulation;
	simulation.setGridResolution(input->gridResolution_);
	simulation.setSDF(input->gridReferenceSdf_);
	simulation.setExplicitDiffusion(true);
	simulation.setHardDirichletBoundaries(false);
    simulation.setAdvectionMode(SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD);
	simulation.resetBoundaries();
	for (const auto& b : input->gridDirichletBoundaries_)
		simulation.addDirichletBoundary(b.first.first, b.first.second, b.second);
	for (const auto& b : input->gridNeumannBoundaries_)
		simulation.addNeumannBoundary(b.first.first, b.first.second, b.second);
	CHECK_WORKER;

	// Set parameters with everything that can't be reconstructed here
	simulation.setGravity(input->settings_.gravity_);
	simulation.setMass(input->settings_.mass_);
	simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
	simulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
	simulation.setRotationCorrection(input->settings_.rotationCorrection_);
	simulation.setTimestep(input->settings_.timestep_);
	simulation.setEnableCollision(input->settings_.enableCollision_);
	simulation.setGroundPlane(input->settings_.groundPlaneHeight_, input->settings_.groundPlaneAngle_);
	simulation.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
	simulation.setCollisionVelocityDamping(0);
	simulation.setGroundStiffness(input->settings_.groundStiffness_);
	simulation.setCollisionSoftmaxAlpha(input->settings_.softmaxAlpha_);
	CHECK_WORKER;

	//read weights -> create cost function definition
	GridCostFunctionDefinition costFunction = gridParseWeights(timestepWeights, simulation);
	costFunction.volumePrior = volumePrior;
	costFunction.referenceVolume = gridComputeReferenceVolume(costFunction, simulation);

	//setup input
	std::vector<real> inputParameters;
	if (optimizeMass) inputParameters.push_back(initialMass);
	if (optimizeMassDamping) inputParameters.push_back(initialMassDamping);
	if (optimizeStiffnessDamping) inputParameters.push_back(initialStiffnessDamping);
	if (optimizeGroundPosition)
	{
		inputParameters.push_back(initialGroundPlaneHeight);
		inputParameters.push_back(initialGroundPlaneAngle);
	}
	if (optimizeYoungsModulus) inputParameters.push_back(initialYoungsModulus);
	if (optimizePoissonRatio) inputParameters.push_back(initialPoissonRatio);
	if (optimizeInitialPosition)
	{
		//find first reference with input data
		for (const auto& keyframe : costFunction.keyframes)
		{
			if (keyframe.weight > 0) {
				inputParameters.insert(inputParameters.end(), keyframe.sdf.data(), keyframe.sdf.data() + keyframe.sdf.size());
				break;
			}
		}
	}
	if (inputParameters.empty())
	{
		CI_LOG_E("You have to select at least one parameter to optimize for");
		return output;
	}
	VectorX value = Eigen::Map<const VectorX>(inputParameters.data(), inputParameters.size());

#if 0
	//Gradient Descent
	real finalCost = 0;
	const auto gradient = [&costFunction, &simulation, this, worker, &finalCost](const VectorX& x)
	{
		GradientInput input;
		int i = 0;
		if (optimizeMass) { input.mass = x[i]; i++; }
		if (optimizeMassDamping) { input.dampingMass = x[i]; i++; }
		if (optimizeStiffnessDamping) { input.dampingStiffness = x[i]; i++; }
		if (optimizeGroundPosition)
		{
			input.groundHeight = x[i]; i++;
			input.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { input.youngsModulus = x[i]; i++; }
		if (optimizePoissonRatio) { input.poissonRatio = x[i]; i++; }
		if (optimizeInitialPosition) { input.initialConfiguration = static_cast<GridUtils2D::grid_t>(Eigen::Map<const GridUtils2D::grid_t>(&x[i], simulation.getGridResolution(), simulation.getGridResolution())); }
		cinder::app::console() << "Input: mass=" << input.mass.value_or(-1) << ", mass-damping=" << input.dampingMass.value_or(-1) << ", stiffness-damping=" << input.dampingStiffness.value_or(-1) << std::endl;

		GradientOutput output = gradientGrid(input, costFunction, simulation, worker, false);
		finalCost = output.finalCost;

		cinder::app::console() << "Gradient Grid: \n "
			<< " Input mass=" << input.mass.value_or(NAN) << ", massDamping=" << input.dampingMass.value_or(NAN) << ", stiffnessDamping=" << input.dampingStiffness.value_or(NAN) << ", groundHeight=" << input.groundHeight.value_or(NAN) << ", groundAngle=" << input.groundAngle.value_or(NAN) << ", YoungsModulus=" << input.youngsModulus.value_or(NAN) << ", PoissonRatio=" << input.poissonRatio.value_or(NAN) << "\n"
			<< " cost=" << output.finalCost << "\n"
			<< " Gradient mass=" << output.massGradient.value_or(NAN) << " massDamping=" << output.dampingMassGradient.value_or(NAN) << ", stiffnessDamping=" << output.dampingStiffnessGradient.value_or(NAN) << ", groundHeight=" << output.groundHeightGradient.value_or(NAN) << ", groundAngle=" << output.groundAngleGradient.value_or(NAN) << ", YoungsModulus=" << output.youngsModulusGradient.value_or(NAN) << ", PoissonRatio=" << output.poissonRatioGradient.value_or(NAN)
			<< std::endl;

		i = 0;
		VectorX gradient(x.size());
		if (optimizeMass) { gradient[i] = *output.massGradient; i++; }
		if (optimizeMassDamping) { gradient[i] = *output.dampingMassGradient; i++; }
		if (optimizeStiffnessDamping) { gradient[i] = *output.dampingStiffnessGradient; i++; }
		if (optimizeGroundPosition)
		{
			gradient[i] = output.groundHeightGradient.value_or(0); i++;
			gradient[i] = output.groundAngleGradient.value_or(0); i++;
		}
		if (optimizeYoungsModulus) { gradient[i] = output.youngsModulusGradient.value_or(0); i++; }
		if (optimizePoissonRatio) { gradient[i] = output.poissonRatioGradient.value_or(0); i++; }
		if (optimizeInitialPosition)
		{
			GridUtils2D::grid_t grad = output.initialConfigurationGradient.has_value()
				? std::get<1>(*output.initialConfigurationGradient)
				: GridUtils2D::grid_t::Zero(simulation.getGridResolution(), simulation.getGridResolution());
			std::copy(grad.data(), grad.data() + simulation.getGridResolution()*simulation.getGridResolution(), &gradient[i]);
		}
		cinder::app::console() << "Gradient: mass=" << output.massGradient.value_or(-1) << ", mass-damping=" << output.dampingMassGradient.value_or(-1) << ", stiffness-damping=" << output.dampingStiffnessGradient.value_or(-1) << std::endl;
		return gradient;
	};
	//specify min and max values
	VectorX minBounds = VectorX::Constant(value.size(), -1e+20);
	VectorX maxBounds = VectorX::Constant(value.size(), +1e+20);
	{
		int i = 0;
		if (optimizeMass) { minBounds[i] = 1e-5; i++; }
		if (optimizeMassDamping) { minBounds[i] = 1e-5; i++; }
		if (optimizeStiffnessDamping) { minBounds[i] = 1e-5; i++; }
		if (optimizeGroundPosition)
		{
			minBounds[i] = 1e-5; maxBounds[i] = 1 - 1e-5; i++; //position
			minBounds[i] = -1; maxBounds[i] = 1; i++; //angle
		}
		if (optimizeYoungsModulus) { minBounds[i] = 1e-5; i++; }
		if (optimizePoissonRatio) { minBounds[i] = 1e-5; maxBounds[i] = 0.5 - 1e-5; i++; }
	}
	//run optimization
	GradientDescent<VectorX> gd(value, gradient);
	gd.setEpsilon(1e-15);
	gd.setLinearStepsize(0.001);
	gd.setMaxStepsize(1e7);
    gd.setMinStepsize(1e-7);
	gd.setMinValues(minBounds);
	gd.setMaxValues(maxBounds);
	int oi;
	for (oi = 0; oi < numIterations; ++oi) {
		worker->setStatus(tfm::format("Adjoint - Dynamic Grid: optimization %d/%d", (oi + 1), numIterations));
		if (gd.step()) break;
		if (worker->isInterrupted()) return InverseProblemOutput();

		//intermediate output
		{
			value = gd.getCurrentSolution();
			InverseProblemOutput output = {};
			output.finalCost_ = finalCost;
			int i = 0;
			if (optimizeMass) { output.mass_ = value[i]; i++; }
			if (optimizeMassDamping) { output.dampingAlpha_ = value[i]; i++; }
			if (optimizeStiffnessDamping) { output.dampingBeta_ = value[i]; i++; }
			if (optimizeGroundPosition)
			{
				output.groundHeight = value[i]; i++;
				output.groundAngle = value[i]; i++;
			}
			if (optimizeYoungsModulus) { output.youngsModulus_ = value[i]; i++; }
			if (optimizePoissonRatio) { output.poissonsRatio_ = value[i]; i++; }
			if (optimizeInitialPosition) { output.initialGridSdf_ = Eigen::Map<const GridUtils2D::grid_t>(&value[i], simulation.getGridResolution(), simulation.getGridResolution()); }
			callback(output);
		}
	}
	value = gd.getCurrentSolution();
#else
	//LBFGS
	LBFGSpp::LBFGSParam<real> params;
    params.epsilon = 1e-7;
	params.max_iterations = numIterations;
	LBFGSpp::LBFGSSolver<real> lbfgs(params);
	//define gradient
	LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&costFunction, &simulation, this, worker](const VectorX& x, VectorX& gradient) -> real {

		GradientInput input;
		int i = 0;
		if (optimizeMass) { input.mass = x[i]; if (x[i] <= 0) return 1e20; i++; }
		if (optimizeMassDamping) { input.dampingMass = x[i]; if (x[i] < 0) return 1e20; i++; }
		if (optimizeStiffnessDamping) { input.dampingStiffness = x[i]; if (x[i] < 0) return 1e20; i++; }
		if (optimizeGroundPosition)
		{
			input.groundHeight = x[i]; i++;
			input.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { input.youngsModulus = x[i]; i++; }
		if (optimizePoissonRatio) { input.poissonRatio = x[i]; i++; }
		if (optimizeInitialPosition) { input.initialConfiguration = static_cast<GridUtils2D::grid_t>(Eigen::Map<const GridUtils2D::grid_t>(&x[i], simulation.getGridResolution(), simulation.getGridResolution())); }

		GradientOutput output = gradientGrid(input, costFunction, simulation, worker, false);
		cinder::app::console() << "Gradient Grid: \n "
			<< " Input mass=" << input.mass.value_or(NAN) << ", massDamping=" << input.dampingMass.value_or(NAN) << ", stiffnessDamping=" << input.dampingStiffness.value_or(NAN) << ", groundHeight=" << input.groundHeight.value_or(NAN) << ", groundAngle=" << input.groundAngle.value_or(NAN) << ", YoungsModulus=" << input.youngsModulus.value_or(NAN) << ", PoissonRatio=" << input.poissonRatio.value_or(NAN) << "\n"
			<< " cost=" << output.finalCost << "\n"
			<< " Gradient mass=" << output.massGradient.value_or(NAN) << " massDamping=" << output.dampingMassGradient.value_or(NAN) << ", stiffnessDamping=" << output.dampingStiffnessGradient.value_or(NAN) << ", groundHeight=" << output.groundHeightGradient.value_or(NAN) << ", groundAngle=" << output.groundAngleGradient.value_or(NAN) << ", YoungsModulus=" << output.youngsModulusGradient.value_or(NAN) << ", PoissonRatio=" << output.poissonRatioGradient.value_or(NAN)
			<< std::endl;

		i = 0;
		if (optimizeMass) { gradient[i] = *output.massGradient; i++; }
		if (optimizeMassDamping) { gradient[i] = *output.dampingMassGradient; i++; }
		if (optimizeStiffnessDamping) { gradient[i] = *output.dampingStiffnessGradient; i++; }
		if (optimizeGroundPosition)
		{
			gradient[i] = output.groundHeightGradient.value_or(0); i++;
			gradient[i] = output.groundAngleGradient.value_or(0); i++;
		}
		if (optimizeYoungsModulus) { gradient[i] = output.youngsModulusGradient.value_or(0); i++; }
		if (optimizePoissonRatio) { gradient[i] = output.poissonRatioGradient.value_or(0); i++; }
		if (optimizeInitialPosition)
		{
			GridUtils2D::grid_t grad = output.initialConfigurationGradient.has_value()
				? std::get<1>(*output.initialConfigurationGradient)
				: GridUtils2D::grid_t::Zero(simulation.getGridResolution(), simulation.getGridResolution());
			std::copy(grad.data(), grad.data() + simulation.getGridResolution()*simulation.getGridResolution(), &gradient[i]);
		}

		return output.finalCost;
	});
	LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, callback, &simulation, this](const VectorX& x, const VectorX& g, const real& v, int k) -> bool {
		worker->setStatus(tfm::format("Adjoint - Dynamic: optimization %d/%d, cost %f", k, numIterations, v));

		//send to callback
		InverseProblemOutput output;
		output.finalCost_ = v;
		int i = 0;
		if (optimizeMass) { output.mass_ = x[i]; i++; }
		if (optimizeMassDamping) { output.dampingAlpha_ = x[i]; i++; }
		if (optimizeStiffnessDamping) { output.dampingBeta_ = x[i]; i++; }
		if (optimizeGroundPosition)
		{
			output.groundHeight = x[i]; i++;
			output.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { output.youngsModulus_ = x[i]; i++; }
		if (optimizePoissonRatio) { output.poissonsRatio_ = x[i]; i++; }
		if (optimizeInitialPosition) { output.initialGridSdf_ = Eigen::Map<const GridUtils2D::grid_t>(&x[i], simulation.getGridResolution(), simulation.getGridResolution()); }
		callback(output);

		return !worker->isInterrupted();
	});
    LBFGSpp::LBFGSSolver<real>::ValidationFunction_t validation([this](const VectorX& x) -> bool {
        int i = 0;
        if (optimizeMass) { if (x[i] <= 0) return false; i++; } //mass must not be non-negative
        if (optimizeMassDamping) { if (x[i] < 0) return false; i++; } //mass damping must not be negative
        if (optimizeStiffnessDamping) { if (x[i] < 0) return false; i++; } //mass damping must not be negative
        return true;
    });
	//optimize
	real finalCost = 0;
	int oi = 0;
	try {
		oi = lbfgs.minimize(fun, value, finalCost, lbfgsCallback, validation);
		CI_LOG_I("Optimized for " << oi << " iterations, final cost " << finalCost);
	} catch (const std::runtime_error& error)
	{
		CI_LOG_EXCEPTION("LBFGS failed", error);
	}
#endif

	//write output
	output.finalCost_ = finalCost;
	int i = 0;
	if (optimizeMass) { output.mass_ = value[i]; i++; }
	if (optimizeMassDamping) { output.dampingAlpha_ = value[i]; i++; }
	if (optimizeStiffnessDamping) { output.dampingBeta_ = value[i]; i++; }
	if (optimizeGroundPosition)
	{
		output.groundHeight = value[i]; i++;
		output.groundAngle = value[i]; i++;
	}
	if (optimizeYoungsModulus) { output.youngsModulus_ = value[i]; i++; }
	if (optimizePoissonRatio) { output.poissonsRatio_ = value[i]; i++; }
	if (optimizeInitialPosition) { output.initialGridSdf_ = Eigen::Map<const GridUtils2D::grid_t>(&value[i], simulation.getGridResolution(), simulation.getGridResolution()); }

	return output;
}

ar::InverseProblemOutput ar::InverseProblem_Adjoint_Dynamic::solveMesh(int deformedTimestep, BackgroundWorker * worker, IntermediateResultCallback_t callback)
{
	InverseProblemOutput output = {};

	// Create simulation
	worker->setStatus("Adjoint - Dynamic: create simulation");
	SoftBodyMesh2D simulation;
	simulation.setMesh(input->meshReferencePositions_, input->meshReferenceIndices_);
	simulation.resetBoundaries();
	for (const auto& b : input->meshDirichletBoundaries_)
		simulation.addDirichletBoundary(b.first, b.second);
	for (const auto& b : input->meshNeumannBoundaries_)
		simulation.addNeumannBoundary(b.first, b.second);
	simulation.reorderNodes();
	CHECK_WORKER;

	// Set parameters with everything that can't be reconstructed here
	simulation.setGravity(input->settings_.gravity_);
	simulation.setMass(input->settings_.mass_);
	simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
	simulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
	simulation.setRotationCorrection(input->settings_.rotationCorrection_);
	simulation.setTimestep(input->settings_.timestep_);
	simulation.setEnableCollision(input->settings_.enableCollision_);
	simulation.setGroundPlane(input->settings_.groundPlaneHeight_, input->settings_.groundPlaneAngle_);
	simulation.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
	simulation.setCollisionVelocityDamping(0);
	simulation.setGroundStiffness(input->settings_.groundStiffness_);
	simulation.setCollisionSoftmaxAlpha(input->settings_.softmaxAlpha_);
	CHECK_WORKER;

	//read weights -> create cost function definition
	MeshCostFunctionDefinition costFunction = meshParseWeights(timestepWeights, simulation);
	costFunction.volumePrior = volumePrior;
	std::tie(costFunction.referenceVolume, costFunction.windingOrder) = meshComputeReferenceVolume(costFunction, simulation);

	//setup input
	std::vector<real> inputParameters;
	if (optimizeMass) inputParameters.push_back(initialMass);
	if (optimizeMassDamping) inputParameters.push_back(initialMassDamping);
	if (optimizeStiffnessDamping) inputParameters.push_back(initialStiffnessDamping);
	if (optimizeGroundPosition)
	{
		inputParameters.push_back(initialGroundPlaneHeight);
		inputParameters.push_back(initialGroundPlaneAngle);
	}
	if (optimizeYoungsModulus) inputParameters.push_back(initialYoungsModulus);
	if (optimizePoissonRatio) inputParameters.push_back(initialPoissonRatio);
	if (optimizeInitialPosition)
	{
		//find first reference with input data
		for (const auto& keyframe : costFunction.keyframes)
		{
			if (keyframe.first > 0) {
				inputParameters.insert(inputParameters.end(), keyframe.second.position.data(), keyframe.second.position.data() + keyframe.second.position.size());
				break;
			}
		}
	}
	if (inputParameters.empty())
	{
		CI_LOG_E("You have to select at least one parameter to optimize for");
		return output;
	}
	VectorX value = Eigen::Map<const VectorX>(inputParameters.data(), inputParameters.size());

#if 1
	//Gradient Descent
	real finalCost = 0;
	const auto gradient = [&costFunction, &simulation, this, worker, &finalCost](const VectorX& x)
	{
		GradientInput input;
		int i = 0;
		if (optimizeMass) { input.mass = x[i]; i++; }
		if (optimizeMassDamping) { input.dampingMass = x[i]; i++; }
		if (optimizeStiffnessDamping) { input.dampingStiffness = x[i]; i++; }
		if (optimizeGroundPosition)
		{
			input.groundHeight = x[i]; i++;
			input.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { input.youngsModulus = x[i]; i++; }
		if (optimizePoissonRatio) { input.poissonRatio = x[i]; i++; }
		if (optimizeInitialPosition) { input.initialConfiguration = static_cast<VectorX>(Eigen::Map<const VectorX>(&x[i], simulation.getNumFreeNodes() * 2)); }

		GradientOutput output = gradientMesh(input, costFunction, simulation, worker, false);
		finalCost = output.finalCost;

		cinder::app::console() << "Gradient Mesh: \n "
			<< " Input mass=" << input.mass.value_or(NAN) << ", massDamping=" << input.dampingMass.value_or(NAN) << ", stiffnessDamping=" << input.dampingStiffness.value_or(NAN) << ", groundHeight=" << input.groundHeight.value_or(NAN) << ", groundAngle=" << input.groundAngle.value_or(NAN) << ", YoungsModulus=" << input.youngsModulus.value_or(NAN) << ", PoissonRatio=" << input.poissonRatio.value_or(NAN) << "\n"
			<< " cost=" << output.finalCost << "\n"
			<< " Gradient mass=" << output.massGradient.value_or(NAN) << " massDamping=" << output.dampingMassGradient.value_or(NAN) << ", stiffnessDamping=" << output.dampingStiffnessGradient.value_or(NAN) << ", groundHeight=" << output.groundHeightGradient.value_or(NAN) << ", groundAngle=" << output.groundAngleGradient.value_or(NAN) << ", YoungsModulus=" << output.youngsModulusGradient.value_or(NAN) << ", PoissonRatio=" << output.poissonRatioGradient.value_or(NAN)
			<< std::endl;

		VectorX gradient(x.size());
		i = 0;
		if (optimizeMass) { gradient[i] = output.massGradient.value_or(0); i++; }
		if (optimizeMassDamping) { gradient[i] = output.dampingMassGradient.value_or(0); i++; }
		if (optimizeStiffnessDamping) { gradient[i] = output.dampingStiffnessGradient.value_or(0); i++; }
		if (optimizeGroundPosition)
		{
			gradient[i] = output.groundHeightGradient.value_or(0); i++;
			gradient[i] = output.groundAngleGradient.value_or(0); i++;
		}
		if (optimizeYoungsModulus) { gradient[i] = output.youngsModulusGradient.value_or(0); i++; }
		if (optimizePoissonRatio) { gradient[i] = output.poissonRatioGradient.value_or(0); i++; }
		if (optimizeInitialPosition)
		{
			VectorX gradPos = output.initialConfigurationGradient.has_value()
				? std::get<0>(*output.initialConfigurationGradient)
				: VectorX::Zero(2 * simulation.getNumFreeNodes());
			std::copy(gradPos.data(), gradPos.data() + 2 * simulation.getNumFreeNodes(), &gradient[i]);
		}
		return gradient;
	};
	//specify min and max values
	VectorX minBounds = VectorX::Constant(value.size(), -1e+20);
	VectorX maxBounds = VectorX::Constant(value.size(), +1e+20);
	{
		int i = 0;
		if (optimizeMass) { minBounds[i] = 1e-5; i++; }
		if (optimizeMassDamping) { minBounds[i] = 1e-5; i++; }
		if (optimizeStiffnessDamping) { minBounds[i] = 1e-5; i++; }
		if (optimizeGroundPosition)
		{
			minBounds[i] = 1e-5; maxBounds[i] = 1 - 1e-5; i++; //position
			minBounds[i] = -1; maxBounds[i] = 1; i++; //angle
		}
		if (optimizeYoungsModulus) { minBounds[i] = 1e-5; i++; }
		if (optimizePoissonRatio) { minBounds[i] = 1e-5; maxBounds[i] = 0.5 - 1e-5; i++; }
		if (optimizeInitialPosition) for (int j=0; j<2*simulation.getNumFreeNodes(); ++j)
		{
			minBounds[i + j] = 0;
			maxBounds[i + j] = 1;
		}
	}
	//run optimization
	GradientDescent<VectorX> gd(value, gradient);
	gd.setEpsilon(1e-15);
	gd.setLinearStepsize(0.001);
	gd.setMaxStepsize(1e7);
	gd.setMinValues(minBounds);
	gd.setMaxValues(maxBounds);
	int oi;
	for (oi = 0; oi < numIterations; ++oi) {
		worker->setStatus(tfm::format("Adjoint - Dynamic Mesh: optimization %d/%d", (oi + 1), numIterations));
		if (gd.step()) break;
		if (worker->isInterrupted()) return InverseProblemOutput();

		//intermediate output
		{
			value = gd.getCurrentSolution();
			InverseProblemOutput output = {};
			output.finalCost_ = finalCost;
			int i = 0;
			if (optimizeMass) { output.mass_ = value[i]; i++; }
			if (optimizeMassDamping) { output.dampingAlpha_ = value[i]; i++; }
			if (optimizeStiffnessDamping) { output.dampingBeta_ = value[i]; i++; }
			if (optimizeGroundPosition)
			{
				output.groundHeight = value[i]; i++;
				output.groundAngle = value[i]; i++;
			}
			if (optimizeYoungsModulus) { output.youngsModulus_ = value[i]; i++; }
			if (optimizePoissonRatio) { output.poissonsRatio_ = value[i]; i++; }
			if (optimizeInitialPosition)
			{
				output.initialMeshPositions_ = input->meshReferencePositions_; //for the dirichlet boundaries
				for (int j = 0; j < simulation.getNumFreeNodes(); ++j)
					output.initialMeshPositions_.value()[simulation.getFreeToNodeMap().at(j)] = value.segment<2>(i + 2 * j);
			}
			callback(output);
		}
	}
	value = gd.getCurrentSolution();

#else
	//LBFGS
	LBFGSpp::LBFGSParam<real> params;
	params.epsilon = 1e-7;
	params.max_iterations = numIterations;
	LBFGSpp::LBFGSSolver<real> lbfgs(params);
	//define gradient
	LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&costFunction, &simulation, this, worker](const VectorX& x, VectorX& gradient) -> real {

		GradientInput input;
		int i = 0;
		if (optimizeMass) { input.mass = x[i]; if (x[i] <= 0) return 1e20; i++; }
		if (optimizeMassDamping) { input.dampingMass = x[i]; if (x[i] < 0) return 1e20; i++; }
		if (optimizeStiffnessDamping) { input.dampingStiffness = x[i]; if (x[i] < 0) return 1e20; i++; }
		if (optimizeGroundPosition)
		{
			input.groundHeight = x[i]; i++;
			input.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { input.youngsModulus = x[i]; i++; }
		if (optimizePoissonRatio) { input.poissonRatio = x[i]; i++; }
        if (optimizeInitialPosition) { input.initialConfiguration = static_cast<VectorX>(Eigen::Map<const VectorX>(&x[i], simulation.getNumFreeNodes() * 2)); }

		GradientOutput output = gradientMesh(input, costFunction, simulation, worker, false);
        cinder::app::console() << "Gradient Mesh: \n "
			<< " Input mass=" << input.mass.value_or(NAN) << ", massDamping=" << input.dampingMass.value_or(NAN) << ", stiffnessDamping=" << input.dampingStiffness.value_or(NAN) << ", groundHeight=" << input.groundHeight.value_or(NAN) << ", groundAngle=" << input.groundAngle.value_or(NAN) << ", YoungsModulus=" << input.youngsModulus.value_or(NAN) << ", PoissonRatio=" << input.poissonRatio.value_or(NAN) << "\n"
            << " cost=" << output.finalCost << "\n"
            << " Gradient mass=" << output.massGradient.value_or(NAN) << " massDamping=" << output.dampingMassGradient.value_or(NAN) << ", stiffnessDamping=" << output.dampingStiffnessGradient.value_or(NAN) << ", groundHeight=" << output.groundHeightGradient.value_or(NAN) << ", groundAngle=" << output.groundAngleGradient.value_or(NAN) << ", YoungsModulus=" << output.youngsModulusGradient.value_or(NAN) << ", PoissonRatio=" << output.poissonRatioGradient.value_or(NAN)
            << std::endl;

		i = 0;
		if (optimizeMass) { gradient[i] = output.massGradient.value_or(0); i++; }
		if (optimizeMassDamping) { gradient[i] = output.dampingMassGradient.value_or(0); i++; }
		if (optimizeStiffnessDamping) { gradient[i] = output.dampingStiffnessGradient.value_or(0); i++; }
		if (optimizeGroundPosition)
		{
			gradient[i] = output.groundHeightGradient.value_or(0); i++;
			gradient[i] = output.groundAngleGradient.value_or(0); i++;
		}
		if (optimizeYoungsModulus) { gradient[i] = output.youngsModulusGradient.value_or(0); i++; }
		if (optimizePoissonRatio) { gradient[i] = output.poissonRatioGradient.value_or(0); i++; }
        if (optimizeInitialPosition)
        {
            VectorX gradPos = output.initialConfigurationGradient.has_value()
                ? std::get<0>(*output.initialConfigurationGradient)
                : VectorX::Zero(2 * simulation.getNumFreeNodes());
            std::copy(gradPos.data(), gradPos.data() + 2 * simulation.getNumFreeNodes(), &gradient[i]);
        }

		return output.finalCost;
	});
	LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, &simulation, callback, this](const VectorX& x, const real& v, int k) -> bool {
		worker->setStatus(tfm::format("Adjoint - Dynamic: optimization %d/%d, cost %f", k, numIterations, v));

		//send to callback
		InverseProblemOutput output = {};
		output.finalCost_ = v;
		int i = 0;
		if (optimizeMass) { output.mass_ = x[i]; i++; }
		if (optimizeMassDamping) { output.dampingAlpha_ = x[i]; i++; }
		if (optimizeStiffnessDamping) { output.dampingBeta_ = x[i]; i++; }
		if (optimizeGroundPosition)
		{
			output.groundHeight = x[i]; i++;
			output.groundAngle = x[i]; i++;
		}
		if (optimizeYoungsModulus) { output.youngsModulus_ = x[i]; i++; }
		if (optimizePoissonRatio) { output.poissonsRatio_ = x[i]; i++; }
		if (optimizeInitialPosition)
		{
			output.initialMeshPositions_ = input->meshReferencePositions_; //for the dirichlet boundaries
			for (int j = 0; j < simulation.getNumFreeNodes(); ++j)
				output.initialMeshPositions_.value()[simulation.getFreeToNodeMap().at(j)] = x.segment<2>(i + 2 * j);
		}
		callback(output);

		return !worker->isInterrupted();
	});
    LBFGSpp::LBFGSSolver<real>::ValidationFunction_t validation([this](const VectorX& x) -> bool {
        int i = 0;
        if (optimizeMass) { if (x[i] <= 0) return false; i++; } //mass must not be non-negative
        if (optimizeMassDamping) { if (x[i] < 0) return false; i++; } //mass damping must not be negative
        if (optimizeStiffnessDamping) { if (x[i] < 0) return false; i++; } //mass damping must not be negative
		if (optimizeGroundPosition) { i += 2; }
		if (optimizeYoungsModulus) { if (x[i] <= 0) return false; i++; } //young's modulus must not be negative
		if (optimizePoissonRatio) { if (x[i] <= 0 || x[i] >= 0.5) return false; i++; } //Poisson ratio must be between (0, 0.5)
        return true;
    });
	//optimize
	real finalCost = 0;
	int oi = 0;
	try {
		oi = lbfgs.minimize(fun, value, finalCost, lbfgsCallback, validation);
		CI_LOG_I("Optimized for " << oi << " iterations, final cost " << finalCost);
	}
	catch (const std::runtime_error& error)
	{
		CI_LOG_EXCEPTION("LBFGS failed", error);
	}
#endif

	//write output
	output.finalCost_ = finalCost;
	int i = 0;
	if (optimizeMass) { output.mass_ = value[i]; i++; }
	if (optimizeMassDamping) { output.dampingAlpha_ = value[i]; i++; }
	if (optimizeStiffnessDamping) { output.dampingBeta_ = value[i]; i++; }
	if (optimizeGroundPosition)
	{
		output.groundHeight = value[i]; i++;
		output.groundAngle = value[i]; i++;
	}
	if (optimizeYoungsModulus) { output.youngsModulus_ = value[i]; i++; }
	if (optimizePoissonRatio) { output.poissonsRatio_ = value[i]; i++; }
	if (optimizeInitialPosition)
	{
		output.initialMeshPositions_ = input->meshReferencePositions_; //for the dirichlet boundaries
		for (int j = 0; j < simulation.getNumFreeNodes(); ++j)
			output.initialMeshPositions_.value()[simulation.getFreeToNodeMap().at(j)] = value.segment<2>(i + 2 * j);
	}

	return output;
}

void ar::InverseProblem_Adjoint_Dynamic::setupParams(cinder::params::InterfaceGlRef params, const std::string & group)
{
	params->addParam("InverseProblem_Adjoint_Dynamic_NumIterations", &numIterations)
		.group(group).label("Num Iterations").min(1);

    params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeMass", &optimizeMass)
		.group(group).label("Optimize Mass")
		.accessors(
			[params, this](bool v) {optimizeMass = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_InitialMass", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizeMass; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialMass", &initialMass)
		.group(group).label("Initial Mass").min(0).step(0.001).visible(optimizeMass);

    params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeMassDamping", &optimizeMassDamping)
        .group(group).label("Optimize Mass Damping")
		.accessors(
			[params, this](bool v) {optimizeMassDamping = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_InitialMassDamping", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizeMassDamping; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialMassDamping", &initialMassDamping)
		.group(group).label("Initial Mass Damping").min(0).step(0.001).visible(optimizeMassDamping);

    params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeStiffnessDamping", &optimizeStiffnessDamping)
        .group(group).label("Optimize Stiffness Damping")
		.accessors(
			[params, this](bool v) {optimizeStiffnessDamping = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_InitialStiffnessDamping", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizeStiffnessDamping; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialStiffnessDamping", &initialStiffnessDamping)
		.group(group).label("Initial Stiffness Damping").min(0).step(0.001).visible(optimizeStiffnessDamping);

    params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeInitialPosition", &optimizeInitialPosition)
        .group(group).label("Optimize Initial Positions")
		.accessors(
			[params, this](bool v) {optimizeInitialPosition = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_VolumePrior", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizeInitialPosition; });
	params->addParam("InverseProblem_Adjoint_Dynamic_VolumePrior", &volumePrior)
		.group(group).label("Volume Prior").min(0).step(0.001).visible(optimizeInitialPosition);

	params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeGroundPosition", &optimizeGroundPosition)
		.group(group).label("Optimize Ground Position")
		.accessors(
			[params, this](bool v)
				{
					optimizeGroundPosition = v;  
					params->setOptions("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneHeight", v ? "visible=true" : "visible=false");
					params->setOptions("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneAngle", v ? "visible=true" : "visible=false");
				},
			[this]() {return optimizeGroundPosition; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneHeight", &initialGroundPlaneHeight)
		.group(group).label("Initial Ground Height").min(0).step(0.001).visible(optimizeGroundPosition);
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneAngle", &initialGroundPlaneAngle)
		.group(group).label("Initial Ground Angle").min(0).step(0.001).visible(optimizeGroundPosition);

	params->addParam("InverseProblem_Adjoint_Dynamic_OptimizeYoungsModulus", &optimizeYoungsModulus)
		.group(group).label("Optimize Youngs Modulus")
		.accessors(
			[params, this](bool v) {optimizeYoungsModulus = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_InitialYoungsModulus", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizeYoungsModulus; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialYoungsModulus", &initialYoungsModulus)
		.group(group).label("Initial Youngs Modulus").min(0).step(0.001).visible(optimizeYoungsModulus);

	params->addParam("InverseProblem_Adjoint_Dynamic_OptimizePoissonRatio", &optimizePoissonRatio)
		.group(group).label("Optimize Poisson Ratio")
		.accessors(
			[params, this](bool v) {optimizePoissonRatio = v;  params->setOptions("InverseProblem_Adjoint_Dynamic_InitialPoissonRatio", v ? "visible=true" : "visible=false"); },
			[this]() {return optimizePoissonRatio; });
	params->addParam("InverseProblem_Adjoint_Dynamic_InitialPoissonRatio", &initialPoissonRatio)
		.group(group).label("Initial PoissonRatio").min(0).max(0.499).step(0.001).visible(optimizePoissonRatio);

	params->addParam("InverseProblem_Adjoint_Dynamic_TimestepWeights", &timestepWeights)
		.group(group).label("Cost: Timestep Weights");
	params->addParam("InverseProblem_Adjoint_Dynamic_CostPositionWeighting", &costPositionWeighting)
		.group(group).label("Cost: Position Weight").min(0).step(0.001);
	params->addParam("InverseProblem_Adjoint_Dynamic_CostVelocityWeighting", &costVelocityWeighting)
		.group(group).label("Cost: Velocity Weight").min(0).step(0.001);
	params->addParam("InverseProblem_Adjoint_Dynamic_CostMeanAmplitude", &costUseMeanAmplitude)
		.group(group).label("Cost: Use Mean Amplitude");

	params->addParam("InverseProblem_Adjoint_Dynamic_GridBypassAdvection", &gridBypassAdvection)
		.group(group).label("Grid: Bypass Advection");
}

void ar::InverseProblem_Adjoint_Dynamic::setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const
{
    std::string option = visible ? "visible=true" : "visible=false";

	params->setOptions("InverseProblem_Adjoint_Dynamic_NumIterations", option);
    params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeMass", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialMass", option);
    params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeMassDamping", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialMassDamping", option);
    params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeStiffnessDamping", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialStiffnessDamping", option);
    params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeInitialPosition", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_VolumePrior", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeGroundPosition", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneHeight", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialGroundPlaneAngle", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizeYoungsModulus", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialYoungsModulus", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_OptimizePoissonRatio", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_InitialPoissonRatio", option);

	params->setOptions("InverseProblem_Adjoint_Dynamic_TimestepWeights", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_CostPositionWeighting", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_CostVelocityWeighting", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_CostMeanAmplitude", option);
	params->setOptions("InverseProblem_Adjoint_Dynamic_GridBypassAdvection", option);
}

ar::InverseProblem_Adjoint_Dynamic::MeshCostFunctionDefinition 
ar::InverseProblem_Adjoint_Dynamic::meshParseWeights(const std::string& weights, const SoftBodyMesh2D& simulation) const
{
	MeshCostFunctionDefinition def;
	def.numSteps = input->numSteps_;
	if (costUseMeanAmplitude)
	{
		def.dataTerm = MeshCostFunctionDefinition::MEAN_AMPLITUDE;
		int dof = simulation.getNumFreeNodes();
		def.meanAmplitudeReference = VectorX::Zero(dof);
		VectorX disp1 = VectorX::Zero(2 * dof);
		for (int t = 0; t<input->numSteps_; ++t)
		{
			VectorX disp2 = simulation.linearize(input->meshResultsDisplacement_[t]);
			for (int i = 0; i < dof; ++i)
				def.meanAmplitudeReference[i] += (disp2.segment<2>(2 * i) - disp1.segment<2>(2 * i)).norm();
			disp1 = disp2;
		}
		def.meanAmplitudeReference /= input->numSteps_;
		return def;
	}
	else {
		def.dataTerm = MeshCostFunctionDefinition::SQUARED_DIFFERENCE;
	}

	//always compute the keyframes as well, needed for priors
	{
		def.keyframes.resize(input->numSteps_);

		def.positionWeighting = costPositionWeighting;
		def.velocityWeighting = costVelocityWeighting;

		VectorX pos = simulation.linearize(input->meshReferencePositions_);

		if (weights.empty())
		{
			//use everything
			for (int i = 0; i < input->numSteps_; ++i)
			{
				def.keyframes[i].first = 1;
				def.keyframes[i].second.position = pos + simulation.linearize(input->meshResultsDisplacement_[i]);
				def.keyframes[i].second.velocity = simulation.linearize(input->meshResultsVelocities_[i]);;
			}
			return def;
		}

		std::stringstream s(weights);
		while (s.good())
		{
			std::string token;
			s >> token;
			int i = static_cast<int>(token.find(':'));
			int index = std::stoi(token.substr(0, i)) - 1;
			if (index >= def.keyframes.size()) continue;
			float weight = std::stof(token.substr(i + 1));
			CI_LOG_I("Timestep weighting: " << index << ":" << weight);

			def.keyframes[index].first = weight;
			def.keyframes[index].second.position = pos + simulation.linearize(input->meshResultsDisplacement_[index]);
			def.keyframes[index].second.velocity = simulation.linearize(input->meshResultsVelocities_[index]);
		}
		return def;
	}
}

ar::InverseProblem_Adjoint_Dynamic::GridCostFunctionDefinition 
ar::InverseProblem_Adjoint_Dynamic::gridParseWeights(
	const std::string& weights, const SoftBodyGrid2D& simulation) const
{
	GridCostFunctionDefinition def;
	def.numSteps = input->numSteps_;
	def.keyframes.resize(def.numSteps);
	def.dataSource = gridUsePartialObservation ? GridCostFunctionDefinition::SOURCE_OBSERVATIONS :
        (gridBypassAdvection ? GridCostFunctionDefinition::SOURCE_UXY : GridCostFunctionDefinition::SOURCE_SDF);
	CI_LOG_I("Data source: " << (int)def.dataSource);

	if (weights.empty())
	{
		//use everything
		for (int i = 0; i < input->numSteps_; ++i)
		{
			def.keyframes[i].weight = 1;
			def.keyframes[i].sdf = input->gridResultsSdf_[i];
            def.keyframes[i].u = input->gridResultsUxy_[i];
            def.keyframes[i].observations = input->gridPartialObservations_[i];
            CI_LOG_D("Input Reference SDF at timestep " << i << ":\n" << def.keyframes[i].sdf);
		}
		return def;
	}

	std::stringstream s(weights);
	while (s.good())
	{
		std::string token;
		s >> token;
		int i = static_cast<int>(token.find(':'));
		int index = std::stoi(token.substr(0, i)) - 1;
		if (index >= def.keyframes.size()) continue;
		float weight = std::stof(token.substr(i + 1));
		CI_LOG_I("Timestep weighting: " << index << ":" << weight);

		def.keyframes[index].weight = weight;
		def.keyframes[index].sdf = input->gridResultsSdf_[index];
		def.keyframes[index].u = input->gridResultsUxy_[index];
        def.keyframes[index].observations = input->gridPartialObservations_[index];
        CI_LOG_D("Input Reference SDF at timestep " << index << ":\n" << def.keyframes[index].sdf);
	}
	return def;
}

ar::InverseProblem_Adjoint_Dynamic::GradientOutput ar::InverseProblem_Adjoint_Dynamic::gradientMesh(
	const GradientInput& input, const MeshCostFunctionDefinition& costFunction, SoftBodyMesh2D simulation,
	BackgroundWorker* worker, bool onlyCost)
{
	using std::vector;

	const int numSteps = costFunction.numSteps;

	//fetch all parameters and prepare output
	if (input.mass) simulation.setMass(*input.mass);
	if (input.dampingMass) simulation.setDamping(*input.dampingMass, simulation.getDampingBeta());
	if (input.dampingStiffness) simulation.setDamping(simulation.getDampingAlpha(), *input.dampingStiffness);
	if (input.groundAngle && input.groundHeight) simulation.setGroundPlane(*input.groundHeight, *input.groundAngle);
	if (input.youngsModulus) simulation.setMaterialParameters(*input.youngsModulus, simulation.getPoissonsRatio());
	if (input.poissonRatio) simulation.setMaterialParameters(simulation.getYoungsModulus(), *input.poissonRatio);
	if (input.initialConfiguration)
	{
		VectorX value = std::get<0>(*input.initialConfiguration);
		for (int j = 0; j < simulation.getNumFreeNodes(); ++j)
			simulation.getReferencePositions()[simulation.getFreeToNodeMap().at(j)] = value.segment<2>(2 * j);
	}
	GradientOutput output = {};

	//precomputations
	MeshPrecomputedValues precomputed = meshPrecomputeValues(simulation);
	int dof = simulation.getNumFreeNodes();
	CHECK_WORKER

	//storage for forward and adjoint variables
	vector<MeshForwardResult> forwardResults(numSteps + 1);
	vector<MeshForwardStorage> forwardStorage(numSteps + 1);
	vector<MeshBackwardResult> backwardResults(numSteps + 1);
	MeshParamAdjoint paramAdj = {};
	if (input.initialConfiguration.has_value()) {
		paramAdj.adjInitialPosition = VectorX::Zero(2 * dof);
	}

    //forward iteration
    forwardResults[0].u = VectorX::Zero(2 * dof); //Boundary condition at t=0
    forwardResults[0].uDot = VectorX::Zero(2 * dof);
    for (int i=0; i<numSteps; ++i)
    {
        auto[state, storage] = meshForwardStep(
            forwardResults[i], simulation, precomputed);
		forwardResults[i + 1] = state;
		forwardStorage[i + 1] = storage;
        CHECK_WORKER
    }

    //evalute cost
    output.finalCost = 0;
	if (costFunction.dataTerm == MeshCostFunctionDefinition::SQUARED_DIFFERENCE) {
		const VectorX positions = simulation.linearize(simulation.getReferencePositions());
		for (int i = 0; i < numSteps; ++i)
		{
			//cost on position
			if (costFunction.keyframes[i].second.position.size() != 0)
			{
				output.finalCost += costFunction.keyframes[i].first * costFunction.positionWeighting
					* (positions + forwardResults[i + 1].u - costFunction.keyframes[i].second.position).squaredNorm();
				backwardResults[i + 1].adjUTmp = costFunction.keyframes[i].first * costFunction.positionWeighting
					* (positions + forwardResults[i + 1].u - costFunction.keyframes[i].second.position);
				//CI_LOG_I("u:\n" << forwardResults[i + 1].u.transpose());
				//CI_LOG_I("forward pos:\n" << (positions + forwardResults[i + 1].u).transpose());
				//CI_LOG_I("reference pos:\n" << costFunction.keyframes[i].second.position.transpose());
				//CI_LOG_I("-> adj-u:\n" << backwardResults[i + 1].adjUTmp.transpose());
			}
			else
				backwardResults[i + 1].adjUTmp = VectorX::Zero(2 * dof);

			//cost on velocities
			if (costFunction.keyframes[i].second.velocity.size() != 0)
			{
				output.finalCost += costFunction.keyframes[i].first * costFunction.velocityWeighting
					* (forwardResults[i + 1].uDot - costFunction.keyframes[i].second.velocity).squaredNorm();
				backwardResults[i + 1].adjUDotTmp = costFunction.keyframes[i].first * costFunction.velocityWeighting
					* (forwardResults[i + 1].uDot - costFunction.keyframes[i].second.velocity);
			}
			else
				backwardResults[i + 1].adjUDotTmp = VectorX::Zero(2 * dof);
			//backwardState[i + 1].adjUDotTmp = VectorX::Zero(2 * dof);

			CHECK_WORKER
		}
	}
	else //costFunction.dataTerm == MeshCostFunctionDefinition::MEAN_AMPLITUDE
	{
		//current mean amplitude
		VectorX meanAmplitude = VectorX::Zero(dof);
		VectorX disp1 = VectorX::Zero(2 * dof);
		for (int t = 0; t<numSteps; ++t)
		{
			VectorX disp2 = forwardResults[t+1].u;
			for (int i = 0; i < dof; ++i)
				meanAmplitude[i] += (disp2.segment<2>(2 * i) - disp1.segment<2>(2 * i)).norm();
			disp1 = disp2;
		}
		meanAmplitude /= numSteps;
		output.finalCost += 0.5 * (meanAmplitude - costFunction.meanAmplitudeReference).squaredNorm();

		//partial derivatives
		for (int t=1; t<=numSteps; ++t)
		{
			backwardResults[t].adjUTmp = VectorX(2 * dof);
			backwardResults[t].adjUDotTmp = VectorX::Zero(2 * dof);
			for (int i=0; i<dof; ++i)
			{
				backwardResults[t].adjUTmp.segment<2>(2 * i) =
					(meanAmplitude[i] - costFunction.meanAmplitudeReference[i]) *
					((forwardResults[t].u.segment<2>(2 * i) - forwardResults[t - 1].u.segment<2>(2 * i)) / (forwardResults[t].u.segment<2>(2 * i) - forwardResults[t - 1].u.segment<2>(2 * i)).norm() +
					(t < numSteps ? (forwardResults[t].u.segment<2>(2 * i) - forwardResults[t+1].u.segment<2>(2 * i)) / (forwardResults[t + 1].u.segment<2>(2 * i) - forwardResults[t].u.segment<2>(2 * i)).norm() : Vector2(0, 0)));
			}
		}
	}

    //TODO: priors on the initial state?
    backwardResults[0].adjUTmp = VectorX::Zero(2 * dof);
    backwardResults[0].adjUDotTmp = VectorX::Zero(2 * dof);

	//prior on the initial configuration
	if (paramAdj.adjInitialPosition.size() > 0)
	{
		auto prior = meshComputeVolumePriorDerivative(simulation.linearize(simulation.getReferencePositions()), costFunction, simulation);
		output.finalCost += costFunction.volumePrior * prior.first;
		paramAdj.adjInitialPosition += costFunction.volumePrior * prior.second;
	}

    //initialize results
	//This is not strictly needed, because they are overwritten either way
    for (int i = 0; i <= numSteps; ++i) {
        backwardResults[i].adjU = VectorX::Zero(2 * dof);
        backwardResults[i].adjUDot = VectorX::Zero(2 * dof);
    }

    //For validating against a numerical gradient, only compute the cost
    if (onlyCost) return output;

    //backward / adjoint iteration
    for (int i = numSteps; i>0; --i)
    {
        meshAdjointStep(
            forwardResults[i - 1],
            forwardResults[i],
            backwardResults[i],
            backwardResults[i - 1],
			paramAdj,
            simulation, precomputed, forwardStorage[i]
        );
        CHECK_WORKER
    }

    //evaluate gradient 
	//TODO: Refactoring, this is the same as in gradientGrid

    //mass
    if (input.mass)
    {
        //real gradient = 0;
        ////mass is only a scalar factor, but here we want the actual mass distribution
        //VectorX distrMvec = precomputed.Mvec / simulation.getMass();
        //real a = simulation.getDampingAlpha() + 1 / (newmarkTheta * simulation.getTimestep());
        //real b = 1 / newmarkTheta;
        //for (int i = 1; i<=numSteps; ++i)
        //{
        //    VectorX dEdm = - a * distrMvec.cwiseProduct(forwardResults[i].u)
        //                   + a * distrMvec.cwiseProduct(forwardResults[i-1].u)
        //                   + b * distrMvec.cwiseProduct(forwardResults[i-1].uDot);
        //    gradient += backwardResults[i].adjU.dot(dEdm);
        //}
        //output.massGradient = gradient;
		output.massGradient = paramAdj.adjMass;
        CHECK_WORKER
    }

    //damping mass
    if (input.dampingMass)
    {
        //real gradient = 0;
        //const VectorX& Mvec = precomputed.Mvec;
        //for (int i = 1; i<=numSteps; ++i)
        //{
        //    VectorX dEdm = - Mvec.cwiseProduct(forwardResults[i].u)
        //                   + Mvec.cwiseProduct(forwardResults[i-1].u);
        //    gradient += backwardResults[i].adjU.dot(dEdm);
        //}
        //output.dampingMassGradient = gradient;
		output.dampingMassGradient = paramAdj.adjMassDamping;
        CHECK_WORKER
    }

    //damping stiffness
    if (input.dampingStiffness)
    {
        //real gradient = 0;
        //for (int i = 1; i <= numSteps; ++i)
        //{
        //    const MatrixX& K = precomputed.K.size()>0 ? precomputed.K : forwardStorage[i].K;
        //    VectorX dEdm = - K * forwardResults[i].u
        //                   + K * forwardResults[i-1].u;
        //    gradient += backwardResults[i].adjU.dot(dEdm);
        //}
        //output.dampingStiffnessGradient = gradient;
		output.dampingStiffnessGradient = paramAdj.adjStiffnessDamping;
        CHECK_WORKER
    }

	//ground position
	if (input.groundHeight)
	{
		output.groundHeightGradient = paramAdj.adjGroundPlaneHeight;
		output.groundAngleGradient = paramAdj.adjGroundPlaneAngle;
	}

    //initial position
    if (input.initialConfiguration)
    {
        output.initialConfigurationGradient = meshParameterDerivativeInitialPosition(
            forwardResults, forwardStorage, backwardResults, paramAdj,
            precomputed, costFunction, simulation);
    }

	//youngs modulus
	if (input.youngsModulus)
	{
		output.youngsModulusGradient = paramAdj.adjYoungsModulus;
	}
	//Poisson ratio
	if (input.poissonRatio)
	{
		output.poissonRatioGradient = paramAdj.adjPoissonRatio;
	}

    return output;
}

ar::InverseProblem_Adjoint_Dynamic::GradientOutput ar::InverseProblem_Adjoint_Dynamic::gradientMesh_Numerical(
	const GradientInput& input, const MeshCostFunctionDefinition& costFunction, SoftBodyMesh2D simulation,
	BackgroundWorker* worker, bool onlyCost)
{
	//reference cost
	real cost = gradientMesh(input, costFunction, simulation, worker, true).finalCost;

	//numerical differentiation
	real epsilon = 1e-7;
	VectorX initialPositions = std::get<0>(*input.initialConfiguration);
	VectorX gradient(initialPositions.size());
	for (int j = 0; j < initialPositions.size(); ++j)
	{
		VectorX positions = initialPositions;
		positions[j] += epsilon;
		GradientInput input2 = input;
		input2.initialConfiguration = positions;
		real cost2 = gradientMesh(input2, costFunction, simulation, worker, true).finalCost;
		gradient[j] = (cost2 - cost) / epsilon;
	}

	GradientOutput output = {};
	output.finalCost = cost;
	output.initialConfigurationGradient = gradient;
	return output;
}

ar::InverseProblem_Adjoint_Dynamic::GradientOutput ar::InverseProblem_Adjoint_Dynamic::gradientGrid(
	const GradientInput& input, const GridCostFunctionDefinition& costFunction, SoftBodyGrid2D simulation,
	BackgroundWorker* worker, bool onlyCost)
{
	using std::vector;

	const int numSteps = costFunction.numSteps;

	//fetch all parameters and prepare output
	if (input.mass) simulation.setMass(*input.mass);
	if (input.dampingMass) simulation.setDamping(*input.dampingMass, simulation.getDampingBeta());
	if (input.dampingStiffness) simulation.setDamping(simulation.getDampingAlpha(), *input.dampingStiffness);
	if (input.groundAngle && input.groundHeight) simulation.setGroundPlane(*input.groundHeight, *input.groundAngle);
	if (input.youngsModulus) simulation.setMaterialParameters(*input.youngsModulus, simulation.getPoissonsRatio());
	if (input.poissonRatio) simulation.setMaterialParameters(simulation.getYoungsModulus(), *input.poissonRatio);
	if (input.initialConfiguration) simulation.setSDF(std::get<1>(*input.initialConfiguration));
    CI_LOG_D("Settings: " << simulation.getSettings() << " " << simulation.getGridSettings());
	GradientOutput output = {};

	//precomputations
	GridPrecomputedValues precomputed = gridPrecomputeValues(simulation);
	int dof = simulation.getDoF();
	int resolution = simulation.getGridResolution();
	CHECK_WORKER

	//storage for forward and adjoint variables
	vector<GridForwardResult> forwardState(numSteps + 1);
	vector<GridForwardStorage> forwardStorage(numSteps + 1);
	vector<GridBackwardResult> backwardState(numSteps + 1);
	GridParamAdjoint paramAdj = {};
	if (input.initialConfiguration)
		paramAdj.adjInitialSdf = GridUtils2D::grid_t::Zero(simulation.getGridResolution(), simulation.getGridResolution());

	//forward iteration
    forwardState[0].u = VectorX::Zero(2 * dof); //Boundary condition at t=0
    forwardState[0].uDot = VectorX::Zero(2 * dof);
	forwardState[0].sdf = simulation.getSdfReference();
    for (int i=0; i<numSteps; ++i)
    {
        auto[state, storage] = gridForwardStep(
            forwardState[i], simulation, precomputed, 
            costFunction.dataSource == GridCostFunctionDefinition::SOURCE_OBSERVATIONS ? &costFunction.keyframes[i].observations : nullptr);
		forwardState[i + 1] = std::move(state);
		forwardStorage[i + 1] = std::move(storage);
        CHECK_WORKER
    }

    //DEBUG
    //if ((forwardStorage[1].uGridXDiffused - costFunction.keyframes[0].uX).matrix().squaredNorm() > 1e-10)
    //{
    //    CI_LOG_D("forwardStorage[1].uGridXDiffused :\n" << forwardStorage[1].uGridXDiffused);
    //    CI_LOG_D("costFunction.keyframes[0].uX :\n" << costFunction.keyframes[0].uX);
    //}
    //if ((forwardStorage[1].uGridYDiffused - costFunction.keyframes[0].uY).matrix().squaredNorm() > 1e-10)
    //{
    //    CI_LOG_D("forwardStorage[1].uGridYDiffused :\n" << forwardStorage[1].uGridYDiffused);
    //    CI_LOG_D("costFunction.keyframes[0].uY :\n" << costFunction.keyframes[0].uY);
    //}
    //if ((forwardState[1].sdf - costFunction.keyframes[0].sdf).matrix().squaredNorm() > 1e-10)
    //{
    //    CI_LOG_D("forwardState[1].sdf :\n" << forwardState[1].sdf);
    //    CI_LOG_D("costFunction.keyframes[0].sdf :\n" << costFunction.keyframes[0].sdf);
    //}

	//evalute cost
	output.finalCost = 0;
	for (int i=0; i<numSteps; ++i)
	{
		backwardState[i + 1].adjU = VectorX::Zero(2 * dof);
		backwardState[i + 1].adjUDot = VectorX::Zero(2 * dof);
		backwardState[i + 1].adjUTmp = VectorX::Zero(2 * dof);
		backwardState[i + 1].adjUDotTmp = VectorX::Zero(2 * dof);
		backwardState[i + 1].adjSdfTmp = GridUtils2D::grid_t::Zero(resolution, resolution);
		if (costFunction.keyframes[i].weight > 0)
		{
			if (costFunction.dataSource == GridCostFunctionDefinition::SOURCE_SDF) 
			{
				const auto& outputSdf = forwardState[i + 1].sdf;
				const auto& referenceSdf = costFunction.keyframes[i].sdf;
				//CI_LOG_D("Forward result SDF at timestep " << i << ":\n" << outputSdf);
				//CI_LOG_D("Forward result uX at timestep " << i << ":\n" << forwardStorage[i + 1].uGridXDiffused);
				//CI_LOG_D("Forward result uY at timestep " << i << ":\n" << forwardStorage[i + 1].uGridYDiffused);
				backwardState[i + 1].adjSdfTmp +=
					(2 * ((2 / M_PI)*outputSdf.atan() - (2 / M_PI)*referenceSdf.atan())) /
					(M_PI * (1 + outputSdf.square()));
				output.finalCost += ((2 / M_PI)*outputSdf.atan() - (2 / M_PI)*referenceSdf.atan()).matrix().squaredNorm() / 2;
			}
			else if (costFunction.dataSource == GridCostFunctionDefinition::SOURCE_UXY)
			{
				backwardState[i + 1].adjUTmp += forwardState[i + 1].u - costFunction.keyframes[i].u;
				output.finalCost += 0.5 * (forwardState[i + 1].u - costFunction.keyframes[i].u).matrix().squaredNorm();
			} else if (costFunction.dataSource == GridCostFunctionDefinition::SOURCE_OBSERVATIONS)
			{
                //Note: not +=, because the backward step checks if adjObservationSdfValues is !empty to decide the partial observation mode
			    backwardState[i + 1].adjObservationSdfValues = forwardState[i+1].observationSdfValues;
                output.finalCost += 0.5 * (forwardState[i+1].observationSdfValues).squaredNorm();
			}
		}
	}
	backwardState[0].adjU = VectorX::Zero(2 * dof);
	backwardState[0].adjUDot = VectorX::Zero(2 * dof);
	backwardState[0].adjUTmp = VectorX::Zero(2 * dof);
	backwardState[0].adjUDotTmp = VectorX::Zero(2 * dof);
	backwardState[0].adjSdfTmp = GridUtils2D::grid_t::Zero(resolution, resolution);

	//priors
	if (input.initialConfiguration)
	{
		auto ret = gridComputeVolumePriorDerivative(simulation.getSdfReference(), costFunction, simulation);
		output.finalCost += costFunction.volumePrior * ret.first;
		paramAdj.adjInitialSdf += costFunction.volumePrior * ret.second;
	}

	//For validating against a numerical gradient, only compute the cost
	if (onlyCost) return output;

	//backward / adjoint iteration
	for (int i=numSteps; i>0; --i)
	{
		const GridForwardResult& prev = forwardState[i - 1];
		const GridForwardResult& current = forwardState[i];
		const GridForwardStorage& storage = forwardStorage[i];
		GridBackwardResult& adjCurrent = backwardState[i];
		GridBackwardResult& adjPrev = backwardState[i - 1];
		gridAdjointStep(prev, current, adjCurrent, adjPrev, 
			paramAdj, simulation, precomputed, storage);
	}

	//evaluate gradient 
	//TODO: Refactoring, this is the same as in gradientMesh

	//mass
	if (input.mass)
	{
		//real gradient = 0;
		////mass is only a scalar factor, but here we want the actual mass distribution
		//VectorX distrMvec = precomputed.Mvec / simulation.getMass();
		//real a = simulation.getDampingAlpha() + 1 / (newmarkTheta * simulation.getTimestep());
		//real b = 1 / newmarkTheta;
		//for (int i = 1; i <= numSteps; ++i)
		//{
		//	VectorX dEdm = -a * distrMvec.cwiseProduct(forwardState[i].u)
		//		+ a * distrMvec.cwiseProduct(forwardState[i - 1].u)
		//		+ b * distrMvec.cwiseProduct(forwardState[i - 1].uDot);
		//	gradient += backwardState[i].adjU.dot(dEdm);
		//}
		//output.massGradient = gradient;
		output.massGradient = paramAdj.adjMass;
		CHECK_WORKER
	}

	//damping mass
	if (input.dampingMass)
	{
		//real gradient = 0;
		//const VectorX& Mvec = precomputed.Mvec;
		//for (int i = 1; i <= numSteps; ++i)
		//{
		//	VectorX dEdm = -Mvec.cwiseProduct(forwardState[i].u)
		//		+ Mvec.cwiseProduct(forwardState[i - 1].u);
		//	gradient += backwardState[i].adjU.dot(dEdm);
		//}
		//output.dampingMassGradient = gradient;
		output.dampingMassGradient = paramAdj.adjMassDamping;
		CHECK_WORKER
	}

	//damping stiffness
	if (input.dampingStiffness)
	{
		//real gradient = 0;
		//for (int i = 1; i <= numSteps; ++i)
		//{
		//	const MatrixX& K = precomputed.K.size()>0 ? precomputed.K : forwardStorage[i].K;
		//	VectorX dEdm = -K * forwardState[i].u
		//		+ K * forwardState[i - 1].u;
		//	gradient += backwardState[i].adjU.dot(dEdm);
		//}
		//output.dampingStiffnessGradient = gradient;
		output.dampingStiffnessGradient = paramAdj.adjStiffnessDamping;
		CHECK_WORKER
	}

	//initial position
	if (input.initialConfiguration)
	{
		output.initialConfigurationGradient = paramAdj.adjInitialSdf;
	}

	//ground position
	if (input.groundHeight)
	{
		output.groundHeightGradient = paramAdj.adjGroundPlaneHeight;
		output.groundAngleGradient = paramAdj.adjGroundPlaneAngle;
	}

	//youngs modulus
	if (input.youngsModulus)
	{
		output.youngsModulusGradient = paramAdj.adjYoungsModulus;
	}
	//Poisson ratio
	if (input.poissonRatio)
	{
		output.poissonRatioGradient = paramAdj.adjPoissonRatio;
	}

	return output;
}

ar::InverseProblem_Adjoint_Dynamic::MeshPrecomputedValues 
ar::InverseProblem_Adjoint_Dynamic::meshPrecomputeValues(
    const SoftBodyMesh2D& simulation)
{
    assert(simulation.isReadyForSimulation());
    //That's the only supported time integration scheme
    assert(simulation.getTimeIntegrator() == TimeIntegrator::Integrator::Newmark1);

    int n = simulation.getNumFreeNodes();
    MeshPrecomputedValues values = {};

    //material matrix
    values.matC = SoftBodySimulation::computeMaterialMatrix(
        simulation.getMaterialMu(), simulation.getMaterialLambda());

    //mass vector
    VectorX Mvec = VectorX::Zero(2 * n);
	simulation.assembleMassMatrix(Mvec);
    values.Mvec = Mvec;

    if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //with corotational correct on, we can only precompute the mass
        //all other matrices change every frame
        //-> nothing else to do
    }
    else
    {
        //no rotation correct, stiffness matrix K and load vector f are constant,
        //and so are the integrator matrices A, B, C, d.

        MatrixX K = MatrixX::Zero(2 * n, 2 * n);
        VectorX f = VectorX::Zero(2 * n);
        simulation.assembleStiffnessMatrix(values.matC, K, &f, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
        simulation.assembleForceVector(f);
        values.K = K;

        MatrixX D = TimeIntegrator::rayleighDamping(
            simulation.getDampingAlpha(), Mvec, 
            simulation.getDampingBeta(), K);

        real deltaT = simulation.getTimestep();
        values.A = TimeIntegrator_Newmark1::getMatrixPartA(K, D, Mvec, f, deltaT, newmarkTheta);
        values.B = TimeIntegrator_Newmark1::getMatrixPartB(K, D, Mvec, f, deltaT, newmarkTheta);
        values.C = TimeIntegrator_Newmark1::getMatrixPartC(K, D, Mvec, f, deltaT, newmarkTheta);
        values.d = TimeIntegrator_Newmark1::getMatrixPartD(K, D, Mvec, f, deltaT, newmarkTheta);
        values.ALu = values.A.partialPivLu();
    }
    return values;
}

std::tuple<ar::InverseProblem_Adjoint_Dynamic::MeshForwardResult, ar::InverseProblem_Adjoint_Dynamic::MeshForwardStorage> 
ar::InverseProblem_Adjoint_Dynamic::meshForwardStep(const MeshForwardResult& currentState,
    const SoftBodyMesh2D& simulation, const MeshPrecomputedValues& precomputed)
{
    assert(simulation.isReadyForSimulation());
    int n = simulation.getNumFreeNodes();
    MeshForwardStorage storage = {};

	//collision -> create forces
	VectorX boundaryForces = VectorX::Zero(2 * n);
	if (simulation.isEnableCollision())
	{
		assert(simulation.getCollisionResolution() == SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		for (int i = 0; i<n; ++i)
		{
			Vector2 force = meshImplicitCollisionForces(
				simulation.getReferencePositions()[simulation.getFreeToNodeMap().at(i)], currentState.u.segment<2>(2 * i), currentState.uDot.segment<2>(2 * i),
				simulation.getGroundPlaneHeight(), simulation.getGroundPlaneAngle(),
				simulation.getGroundStiffness(), simulation.getCollisionSoftmaxAlpha(), simulation.getTimestep(), newmarkTheta);
			boundaryForces.segment<2>(2 * i) = force;
		}
		//CI_LOG_I("force: " << boundaryForces.transpose());
	}

    //compute next displacements
    VectorX nextU;
    if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //corotational correction, recompute all matrices

        MatrixX K = MatrixX::Zero(2 * n, 2 * n);
        VectorX f = VectorX::Zero(2 * n);
        simulation.assembleForceVector(f, &boundaryForces);

		//MatrixX K2 = K;
		//VectorX f2 = f;
        simulation.assembleStiffnessMatrix(precomputed.matC, K, &f, simulation.delinearize(currentState.u));

		//for (int e = 0; e < simulation.getNumElements(); ++e) {
		//	Matrix6 Ke = Matrix6::Zero();
		//	Vector6 Fe = Vector6::Zero();
		//	int a = simulation.getTriangles()[e][0];
		//	int b = simulation.getTriangles()[e][1];
		//	int c = simulation.getTriangles()[e][2];
		//	Vector2 posA = simulation.getReferencePositions()[a];
		//	Vector2 posB = simulation.getReferencePositions()[b];
		//	Vector2 posC = simulation.getReferencePositions()[c];
		//	Vector6 posE; posE << posA, posB, posC;
		//	Vector2 ua = simulation.getNodeToFreeMap().at(a) >= 0 ? currentState.u.segment<2>(2 * simulation.getNodeToFreeMap().at(a)) : Vector2(0, 0);
		//	Vector2 ub = simulation.getNodeToFreeMap().at(b) >= 0 ? currentState.u.segment<2>(2 * simulation.getNodeToFreeMap().at(b)) : Vector2(0, 0);
		//	Vector2 uc = simulation.getNodeToFreeMap().at(c) >= 0 ? currentState.u.segment<2>(2 * simulation.getNodeToFreeMap().at(c)) : Vector2(0, 0);
		//	Vector6 ue; ue << ua, ub, uc;
		//	meshComputeElementMatrix(precomputed.matC, Ke, Fe, posE, ue, SoftBodySimulation::RotationCorrection::Corotation);
		//	simulation.placeIntoMatrix(e, &Ke, &K, &Fe, &f);
		//}
		//assert(K2 == K);//hard comparison, the computation should be exactly equal
		//assert(f2 == f); 
        storage.K = K;

        MatrixX D = TimeIntegrator::rayleighDamping(
            simulation.getDampingAlpha(), precomputed.Mvec,
            simulation.getDampingBeta(), K);

        real deltaT = simulation.getTimestep();
        MatrixX A = TimeIntegrator_Newmark1::getMatrixPartA(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
        MatrixX B = TimeIntegrator_Newmark1::getMatrixPartB(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
        MatrixX C = TimeIntegrator_Newmark1::getMatrixPartC(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
        VectorX d = TimeIntegrator_Newmark1::getMatrixPartD(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
        
        VectorX rhs = B * currentState.u + C * currentState.uDot + d;
#if 0
        nextU = currentU + deltaT * currentUDot; //initial guess for the next position
        //nextU = VectorX::Zero(currentU.size());
        Eigen::Index iters = 10 * n;
        real tol_error = 1e-10;
        storage.cgStorage = ConjugateGradientForward(A, rhs, nextU, iters, tol_error);
        CI_LOG_D("CG result: iterations=" << iters << ", error=" << tol_error);
#else
        Eigen::PartialPivLU<MatrixX> Alu = A.partialPivLu();
        nextU = Alu.solve(rhs);
        storage.ALu = Alu;
        storage.rhs = rhs;
#endif

        //storage needed for adjoint step
        storage.A = A;
        storage.B = B;
        storage.C = C;
    } 
    else
    {
        //no correction, use precomputed matrices
		//TODO: include boundary forces also here
#if 0
        VectorX rhs = precomputed.B * currentU + precomputed.C * currentUDot + precomputed.d;
        //nextU = currentU + simulation.getTimestep() * currentUDot; //initial guess for the next position
        nextU = VectorX::Zero(2 * n);
        Eigen::Index iters = 10 * n;
        real tol_error = 1e-20;
        storage.cgStorage = ConjugateGradientForward(precomputed.A, rhs, nextU, iters, tol_error);
        CI_LOG_D("CG result: iterations=" << iters << ", error=" << tol_error);
        WARN("CG result: iterations=" << iters << ", error=" << tol_error);

        VectorX nextURef = precomputed.ALu.solve(rhs);
        INFO("u (LU):\n" << nextURef.transpose());
        INFO("u (CG):\n" << nextU.transpose());
        REQUIRE(nextURef.isApprox(nextU));
#else
        VectorX rhs = precomputed.B * currentState.u + precomputed.C * currentState.uDot + precomputed.d;
        nextU = precomputed.ALu.solve(rhs);
#endif
    }

    //velocity update
    VectorX nextUDot = (1 / (newmarkTheta*simulation.getTimestep())) * (nextU - currentState.u)
                     - ((1 - newmarkTheta) / newmarkTheta) * currentState.uDot;
    //VectorX nextUDot = TimeIntegrator_Newmark1::computeUdot(nextU, currentU, currentUDot, simulation.getTimestep(), newmarkTheta);

    return { MeshForwardResult{nextU, nextUDot}, storage };
}

void ar::InverseProblem_Adjoint_Dynamic::meshAdjointStep(
    const MeshForwardResult& prevResult,
    const MeshForwardResult& currentResult,
    MeshBackwardResult& currentAdj,
    MeshBackwardResult& prevAdj,
	MeshParamAdjoint& paramAdj,
    const SoftBodyMesh2D& simulation, const MeshPrecomputedValues& precomputed,
    const MeshForwardStorage& forwardStorage)
{
    assert(simulation.isReadyForSimulation());
    real deltaT = simulation.getTimestep();
	int n = simulation.getNumFreeNodes();
    bool computeAdjInitPos = paramAdj.adjInitialPosition.size() != 0;

	//allocate storage for the adjoint of the body forces.
	//Filled by the adjoint of the displacements / rotation correction
	//Used for the adjoint of the collision
	VectorX adjf = VectorX::Zero(2 * n);

    //adjoint: velocity update
	/*
	  VectorX nextUDot = (1 / (newmarkTheta*simulation.getTimestep())) * (nextU - currentState.u)
                     - ((1 - newmarkTheta) / newmarkTheta) * currentState.uDot;
	 */
    currentAdj.adjUDot = currentAdj.adjUDotTmp;
    currentAdj.adjUTmp += (1 / (newmarkTheta * deltaT)) * currentAdj.adjUDot;
    prevAdj.adjUDotTmp += -((1 - newmarkTheta) / newmarkTheta) * currentAdj.adjUDot;
    prevAdj.adjUTmp += (-1 / (newmarkTheta * deltaT)) * currentAdj.adjUDot;

    //adjoint: displacements
    if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //corotational correction, compute adjoint through the matrices

        // A * nextU = rhs:
        // adjNextU -> adjA, adjRhs
        VectorX adjRhs = VectorX::Zero(2 * n);
        MatrixX adjA = MatrixX::Zero(2 * n, 2 * n);
#if 0
		VectorX initialGuess = currentResult.u + deltaT * currentUDot;
        ConjugateGradientAdjoint(
            forwardStorage.A, //not transposed because A is symmetric
			initialGuess,
            adjRhs, adjNextU, adjA,
            forwardStorage.cgStorage);
        adjCurrentUPre += adjNextU;
		adjCurrentUPost += adjNextU;
		currentAdj.adjUDot += deltaT * adjNextU;
		adjNextU.setZero();
#else
		currentAdj.adjU = forwardStorage.ALu.solve(currentAdj.adjUTmp); //normally, A has to be transposed, but it is symmetric
		adjRhs += currentAdj.adjU;
		adjA += - currentAdj.adjU * currentResult.u.transpose();
#endif

		//CI_LOG_I("adjA.diag():\n" << adjA.diagonal().eval().transpose());

        // rhs = B currentResult.u + C currentUDot + d: 
        // adjRhs -> adjB, adjC, adjd, currentAdj.adjU, currentAdj.adjUDot
		MatrixX adjB = adjRhs * prevResult.u.transpose();
		prevAdj.adjUTmp += forwardStorage.B.transpose() * adjRhs;
		MatrixX adjC = adjRhs * prevResult.uDot.transpose();
		prevAdj.adjUDotTmp += forwardStorage.C.transpose() * adjRhs;
		VectorX adjd = adjRhs;

		//CI_LOG_I("adjB.diag():\n" << adjB.diagonal().eval().transpose());
		//CI_LOG_I("adjC.diag():\n" << adjB.diagonal().eval().transpose());

        // A, B, C, d = f(M, D, K, f):
        // adjA, adjB, adjC, adjd -> adjD, adjK, adjf
        MatrixX adjD = MatrixX::Zero(2 * n, 2 * n);
        MatrixX adjK = MatrixX::Zero(2 * n, 2 * n);
		MatrixX adjM = MatrixX::Zero(2 * n, 2 * n);
		adjf += deltaT * adjd;
		adjM += (1 / newmarkTheta) * adjC;
		adjM += (1 / (newmarkTheta * deltaT)) * adjB;
		adjD += adjB;
		adjK += (1 - newmarkTheta) * deltaT * adjB;
		adjM += (1 / (newmarkTheta * deltaT)) * adjA;
		adjD += adjA;
		adjK += newmarkTheta * deltaT * adjA;

		//CI_LOG_I("adjM.diag():\n" << adjM.diagonal().eval().transpose());
		//CI_LOG_I("adjD.diag():\n" << adjD.diagonal().eval().transpose());

        // D = alpha_1 * M + alpha_2 * K
        // adjD -> adjK, adjAlpha_1, adjAlpha2
		adjM += simulation.getDampingAlpha() * adjD;
        adjK += simulation.getDampingBeta() * adjD;
		paramAdj.adjMassDamping += precomputed.Mvec.dot(adjD.diagonal());
		paramAdj.adjStiffnessDamping += forwardStorage.K.cwiseProduct(adjD).sum();

		//CI_LOG_I("adjM.diag():\n" << adjM.diagonal().eval().transpose());

		// M = m * M_0
		// adjM -> adjm
		VectorX M_0 = precomputed.Mvec / simulation.getMass();
		paramAdj.adjMass += M_0.dot(adjM.diagonal());

        //loop over all elements
        for (int e = simulation.getNumElements() - 1; e >= 0; --e) {
            Vector6 adjFe = Vector6::Zero();
            Matrix6 adjKe = Matrix6::Zero();
            Vector6 adjMe = Vector6::Zero();

            const int a = simulation.getTriangles()[e][0];
            const int b = simulation.getTriangles()[e][1];
            const int c = simulation.getTriangles()[e][2];
            const int ia = simulation.getNodeToFreeMap()[a];
            const int ib = simulation.getNodeToFreeMap()[b];
            const int ic = simulation.getNodeToFreeMap()[c];
            const Vector2& posA = simulation.getReferencePositions()[a];
            const Vector2& posB = simulation.getReferencePositions()[b];
            const Vector2& posC = simulation.getReferencePositions()[c];

            //adjoint: place into the matrix
            for (int i = 2; i >= 0; --i) {
                int nodeI = simulation.getTriangles()[e][i];
                bool dirichletI = simulation.getNodeStates()[nodeI] == SoftBodyMesh2D::DIRICHLET;

                if (!dirichletI) {
                    adjFe.segment<2>(2 * i) += adjf.segment<2>(simulation.getNodeToFreeMap()[nodeI] * 2);
                    adjMe.segment<2>(2 * i) += adjM.diagonal().segment<2>(simulation.getNodeToFreeMap()[nodeI] * 2);
                }

                for (int j = 2; j >= 0; --j) {
                    int nodeJ = simulation.getTriangles()[e][j];
                    bool dirichletJ = simulation.getNodeStates()[nodeJ] == SoftBodyMesh2D::DIRICHLET;
                    if (i == j && dirichletI) {
                        //ignore it, v_i is zero by definition of V
                    }
                    else if (i == j && !dirichletI) {
                        //regular, free nodes
                        adjKe.block<2, 2>(i * 2, i * 2) += adjK.block<2, 2>(simulation.getNodeToFreeMap()[nodeI] * 2, simulation.getNodeToFreeMap()[nodeI] * 2);
                    }
                    else if (dirichletI && dirichletJ) {
                        //ignore it, v_i is zero by def of V
                    }
                    else if (dirichletI && !dirichletJ) {
                        //ignore it, v_i is zero by def of V
                    }
                    else if (!dirichletI && dirichletJ) {
                        //add it to the force
                        adjKe.block<2, 2>(i * 2, j * 2) += adjf.segment<2>(simulation.getNodeToFreeMap()[nodeI] * 2) * simulation.getBoundaries()[nodeJ].transpose();
                    }
                    else {
                        //regular, free nodes
                        adjKe.block<2, 2>(i * 2, j * 2) += adjK.block<2, 2>(simulation.getNodeToFreeMap()[nodeI] * 2, simulation.getNodeToFreeMap()[nodeJ] * 2);
                    }
                }
            }

			//adjoint: rotation correction
			Vector6 xe; xe << posA, posB, posC;
			Vector6 ue = Vector6::Zero();
			if (ia >= 0) ue.segment<2>(0) = prevResult.u.segment<2>(2 * ia);
			if (ib >= 0) ue.segment<2>(2) = prevResult.u.segment<2>(2 * ib);
			if (ic >= 0) ue.segment<2>(4) = prevResult.u.segment<2>(2 * ic);
			Vector6 adjU = Vector6::Zero();
            Vector6 adjInitialPos = Vector6::Zero();
			//cinder::app::console() << e << " ";
			meshComputeElementMatrixAdjoint(
                precomputed.matC, simulation,
                xe, ue, SoftBodySimulation::RotationCorrection::Corotation,
                adjKe, adjFe, adjU,
                computeAdjInitPos ? &adjInitialPos : nullptr, &paramAdj.adjYoungsModulus, &paramAdj.adjPoissonRatio);
			if (ia >= 0) prevAdj.adjUTmp.segment<2>(2 * ia) += adjU.segment<2>(0);
			if (ib >= 0) prevAdj.adjUTmp.segment<2>(2 * ib) += adjU.segment<2>(2);
			if (ic >= 0) prevAdj.adjUTmp.segment<2>(2 * ic) += adjU.segment<2>(4);
			/*
            if (computeAdjInitPos)
            { //also compute adjoint for the initial position
                //adjInitialPos were computed already in meshComputeElementMatrixAdjoint,
                //correction based on the mass is still missing
                auto areaDeriv = InverseProblem_Adjoint_InitialConfiguration::meshTriangleAreaDerivative({ posA, posB, posC });
                int windingOrder = InverseProblem_Adjoint_InitialConfiguration::meshTriangleWindingOrder({ posA, posB, posC });
                if (ia > 0) paramAdj.adjInitialPosition.segment<2>(2 * ia) += 
                    adjInitialPos.segment<2>(0) + simulation.getMass() * windingOrder * Vector2(areaDeriv[0], areaDeriv[1]);
                if (ib > 0) paramAdj.adjInitialPosition.segment<2>(2 * ib) +=
                    adjInitialPos.segment<2>(2) + simulation.getMass() * windingOrder * Vector2(areaDeriv[2], areaDeriv[3]);
                if (ic > 0) paramAdj.adjInitialPosition.segment<2>(2 * ic) +=
                    adjInitialPos.segment<2>(4) + simulation.getMass() * windingOrder * Vector2(areaDeriv[4], areaDeriv[5]);
            }
			*/
        }
    }
    else
    {
        //no correction, simple
		//TODO: reconstruct force adjoint
#if 0
        VectorX adjRhsRef = precomputed.ALu.solve(adjNextU);

        int n = simulation.getNumFreeNodes();
        VectorX adjRhs = VectorX::Zero(2 * n);
        MatrixX adjA = MatrixX::Zero(2 * n, 2 * n);
        //VectorX initialGuess = currentResult.u + deltaT * currentUDot;
        VectorX initialGuess = VectorX::Zero(2 * n);
        VectorX adjNextUCopy = adjNextU;
        ConjugateGradientAdjoint(
            precomputed.A.transpose(), //not transposed because A is symmetric
            initialGuess,
            adjRhs, adjNextUCopy, adjA,
            forwardStorage.cgStorage);

        VectorX adjRhs2 = initialGuess;
        Eigen::Index iter = forwardStorage.cgStorage.iterations;
        real tol_error = 1e-10;
        ConjugateGradientForward(precomputed.A, adjNextU, adjRhs2, iter, tol_error);

        INFO("adjRhs (LU):\n" << adjRhsRef.transpose());
        INFO("adjRhs (adjCG):\n" << adjRhs.transpose());
        INFO("adjRhs (CG):\n" << adjRhs2.transpose());
        CHECK(adjRhsRef.isApprox(adjRhs, 1e-4));

        //currentAdj.adjU += adjNextUCopy;
        //currentAdj.adjUDot += deltaT * adjNextUCopy;
        //adjNextU.setZero();
        currentAdj.adjU += precomputed.B.transpose() * adjRhs;
        currentAdj.adjUDot += precomputed.C.transpose() * adjRhs;
#else
		/*
		 nextU = precomputed.ALu.solve(rhs);    // A * nextU = rhs
		 */
        currentAdj.adjU = precomputed.ALu.solve(currentAdj.adjUTmp);
        prevAdj.adjUDotTmp += precomputed.C.transpose() * currentAdj.adjU;
        prevAdj.adjUTmp += precomputed.B.transpose() * currentAdj.adjU;
#endif
    }

	//Adjoint: collision
	if (simulation.isEnableCollision())
	{
		assert(simulation.getCollisionResolution() == SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		for (int i = 0; i<n; ++i)
		{
			Vector2 adjForce = adjf.segment<2>(2 * i);
			Vector2 adjRefPos(0, 0), adjDisp(0, 0), adjVel(0, 0);
			meshImplicitCollisionForcesAdjoint(
				simulation.getReferencePositions()[simulation.getFreeToNodeMap().at(i)], prevResult.u.segment<2>(2 * i), prevResult.uDot.segment<2>(2 * i),
				simulation.getGroundPlaneHeight(), simulation.getGroundPlaneAngle(),
				simulation.getGroundStiffness(), simulation.getCollisionSoftmaxAlpha(), simulation.getTimestep(), newmarkTheta,
				adjForce,
				adjRefPos, adjDisp, adjVel, paramAdj.adjGroundPlaneHeight, paramAdj.adjGroundPlaneAngle);
			prevAdj.adjUTmp.segment<2>(2 * i) += adjDisp;
			prevAdj.adjUDotTmp.segment<2>(2 * i) += adjVel;
			if (computeAdjInitPos)
			{
				paramAdj.adjInitialPosition.segment<2>(2 * i) += adjRefPos;
			}
		}
	}
}

//std::stack<ar::Matrix2> TestFStack;
//std::stack<ar::Vector6> TestCurrentUeStack;
//std::stack<ar::Vector6> TestPosEStack;

void ar::InverseProblem_Adjoint_Dynamic::meshComputeElementMatrix(
    const Matrix3 & C, Matrix6 & Ke, Vector6 & Fe, 
    const Vector6 & posE, const Vector6 & currentUe,
    SoftBodySimulation::RotationCorrection rotationCorrection)
{
    Vector2 posA = posE.segment<2>(0);
    Vector2 posB = posE.segment<2>(2);
    Vector2 posC = posE.segment<2>(4);

    real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
    real area = abs(signedArea);

    //Stiffness
    Matrix36 Be;
    Be << (posB.y() - posC.y()), 0, (posC.y() - posA.y()), 0, (posA.y() - posB.y()), 0,
        0, (posC.x() - posB.x()), 0, (posA.x() - posC.x()), 0, (posB.x() - posA.x()),
        (posC.x() - posB.x()), (posB.y() - posC.y()), (posA.x() - posC.x()), (posC.y() - posA.y()), (posB.x() - posA.x()), (posA.y() - posB.y());
    Be *= (1 / signedArea);
    Ke = area * Be.transpose() * C * Be;

    //rotation correction
    if (rotationCorrection == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //compute jacobian of the displacements
        Vector2 defA = currentUe.segment<2>(0) + posA;
        Vector2 defB = currentUe.segment<2>(2) + posB;
        Vector2 defC = currentUe.segment<2>(4) + posC;
        Matrix2 Ddef, Dref;
        Ddef << defA.x() - defC.x(), defB.x() - defC.x(),
            defA.y() - defC.y(), defB.y() - defC.y();
        Dref << posA.x() - posC.x(), posB.x() - posC.x(),
            posA.y() - posC.y(), posB.y() - posC.y();
        Matrix2 F = Ddef * Dref.inverse();
		////Test
		//TestPosEStack.push(posE);
		//TestCurrentUeStack.push(currentUe);
		//TestFStack.push(F);
        //compute polar decomposition
        Matrix2 R = abs(F.determinant()) < 1e-15 
            ? Matrix2::Identity() 
            : SoftBodySimulation::polarDecomposition(F);
		//cinder::app::console() << "(forward) F={{" << F(0, 0) << "," << F(0, 1) << "},{" << F(1, 0) << "," << F(1, 1) << "}}, R={{" << R(0, 0) << "," << R(0, 1) << "},{" << R(1, 0) << "," << R(1, 1) << "}}" << std::endl;
        //augment Ke
        Matrix6 Re;
        Re << R, Matrix2::Zero(), Matrix2::Zero(),
            Matrix2::Zero(), R, Matrix2::Zero(),
            Matrix2::Zero(), Matrix2::Zero(), R;
        Vector6 xe; xe << posA, posB, posC;
        Fe -= Re * Ke * (Re.transpose() * xe - xe);
        Ke = Re * Ke * Re.transpose();
    }
}

void ar::InverseProblem_Adjoint_Dynamic::meshComputeElementMatrixAdjoint(
    const Matrix3 & C, const SoftBodyMesh2D& simulation,
    const Vector6 & posE, const Vector6 & currentUe, 
    SoftBodySimulation::RotationCorrection rotationCorrection,
    const Matrix6 & adjKe, const Vector6& adjFe, 
    Vector6 & adjCurrentUe, 
	Vector6* adjInitialPosition, real* adjYoungsModulus, real* adjPoissonRatio)
{
    bool computeAdjInitPos = adjInitialPosition != nullptr;
    assert(EIGEN_IMPLIES(computeAdjInitPos, rotationCorrection == SoftBodySimulation::RotationCorrection::Corotation));
    //adjoint of the initial position is only computed in corotation mode.

    Vector2 posA = posE.segment<2>(0);
    Vector2 posB = posE.segment<2>(2);
    Vector2 posC = posE.segment<2>(4);

    real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
    real area = abs(signedArea);

    //Stiffness
    Matrix36 Be;
    Be << (posB.y() - posC.y()), 0, (posC.y() - posA.y()), 0, (posA.y() - posB.y()), 0,
        0, (posC.x() - posB.x()), 0, (posA.x() - posC.x()), 0, (posB.x() - posA.x()),
        (posC.x() - posB.x()), (posB.y() - posC.y()), (posA.x() - posC.x()), (posC.y() - posA.y()), (posB.x() - posA.x()), (posA.y() - posB.y());
    Be *= (1 / signedArea);
    Matrix6 KeRef = area * Be.transpose() * C * Be;

	Matrix6 adjKeRef = adjKe;

    if (rotationCorrection == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //FORWARD (again)
        Vector2 defA = currentUe.segment<2>(0) + posA;
        Vector2 defB = currentUe.segment<2>(2) + posB;
        Vector2 defC = currentUe.segment<2>(4) + posC;
        Matrix2 Ddef, Dref;
        Ddef << defA.x() - defC.x(), defB.x() - defC.x(),
            defA.y() - defC.y(), defB.y() - defC.y();
        Dref << posA.x() - posC.x(), posB.x() - posC.x(),
            posA.y() - posC.y(), posB.y() - posC.y();
        Matrix2 F = Ddef * Dref.inverse();
		////Test
		//Vector6 posE2 = TestPosEStack.top(); TestPosEStack.pop();
		//Vector6 currentUe2 = TestCurrentUeStack.top(); TestCurrentUeStack.pop();
		//Matrix2 F2 = TestFStack.top(); TestFStack.pop();
		//assert(posE.isApprox(posE2));
		//assert(currentUe.isApprox(currentUe2));
		//assert(F.isApprox(F2));
        //compute polar decomposition
        Matrix2 R = abs(F.determinant()) < 1e-15
            ? Matrix2::Identity()
            : SoftBodySimulation::polarDecomposition(F);
		//cinder::app::console() << "(adjoint) F={{" << F(0, 0) << "," << F(0, 1) << "},{" << F(1, 0) << "," << F(1, 1) << "}}, R={{" << R(0, 0) << "," << R(0, 1) << "},{" << R(1, 0) << "," << R(1, 1) << "}}" << std::endl;
        //augment Ke
        Matrix6 Re;
        Re << R, Matrix2::Zero(), Matrix2::Zero(),
            Matrix2::Zero(), R, Matrix2::Zero(),
            Matrix2::Zero(), Matrix2::Zero(), R;
        Vector6 xe; xe << posA, posB, posC;

        //ADJOINT

        Matrix6 adjRe = Matrix6::Zero();
		adjKeRef.setZero();

		//Fe = Fe - Re * KeRef * (Re.transpose() * xe - xe);
		adjRe -= xe * adjFe.transpose() * Re * KeRef
		       + adjFe * xe.transpose() * Re * KeRef.transpose()
			   - adjFe * xe.transpose() * KeRef;
		adjKeRef -= Re.transpose() * adjFe * (Re.transpose() * xe - xe).transpose();

		//Ke = Re * KeRef * Re.transpose();
		adjRe += (adjKe.transpose() * Re * KeRef.transpose()) + (adjKe * Re * KeRef);
		adjKeRef += Re.transpose() * adjKe * Re;

        Matrix2 adjR = adjRe.block<2, 2>(0, 0) + adjRe.block<2, 2>(2, 2) + adjRe.block<2, 2>(4, 4);
        //adjRe.setZero();

        Matrix2 adjF = abs(F.determinant()) < 1e-15
            ? Matrix2::Zero()
            : AdjointUtilities::polarDecompositionAdjoint(F, adjR);
        //adjR.setZero();

        Matrix2 adjDdef = adjF * Dref.inverse().transpose();
        //adjF.setZero();

        adjCurrentUe.segment<2>(0) += Vector2(adjDdef(0, 0), adjDdef(1, 0));
        adjCurrentUe.segment<2>(2) += Vector2(adjDdef(0, 1), adjDdef(1, 1));
        adjCurrentUe.segment<2>(4) -= Vector2(adjDdef(0, 0) + adjDdef(0, 1), adjDdef(1, 0) + adjDdef(1, 1));
        //adjDdef.setZero();

        if (computeAdjInitPos)
        {
            //adjoint parts for the initial position
            Vector6 adjFeStiffness = adjFe;
            *adjInitialPosition += -adjFe.transpose() * Re * KeRef * (Re.transpose() - Matrix6::Identity()); //adjXe
            //adjF
            Matrix2 adjDref = -Dref.transpose().inverse() * Ddef.transpose() * adjF * Dref.transpose().inverse();
            //adjDref
            adjInitialPosition->segment<2>(0) += Vector2(adjDref(0, 0), adjDref(1, 0));
            adjInitialPosition->segment<2>(2) += Vector2(adjDref(0, 1), adjDref(1, 1));
            adjInitialPosition->segment<2>(4) -= Vector2(adjDref(0, 0) + adjDref(0, 1), adjDref(1, 0) + adjDref(1, 1));
            //adjKeStiffness
            typedef InverseProblem_Adjoint_InitialConfiguration IC;
            IC::triangle_t triangle {posA, posB, posC};
            int windingOrder = IC::meshTriangleWindingOrder(triangle);
            for (int i = 0; i < 6; ++i)
                adjInitialPosition->coeffRef(i) += IC::meshStiffnessMatrixDerivative(triangle, i, C, windingOrder).cwiseProduct(adjKeRef).sum();
            //adjFeStiffness
            auto areaDeriv = IC::meshTriangleAreaDerivative(triangle);
            Vector6 gravity3; gravity3 << simulation.getGravity(), simulation.getGravity(), simulation.getGravity();
            real gravityDotFe = gravity3.dot(adjFeStiffness);
            for (int i = 0; i < 6; ++i)
                adjInitialPosition->coeffRef(i) += areaDeriv[i] * windingOrder * gravityDotFe;
        }
    }

	//derivatives for Young's Modulus and Poisson ratio
	//Computed from adjKeRef
	if (adjYoungsModulus)
	{
		real muDyoung, lambdaDyoung;
		AdjointUtilities::computeMaterialParameters_D_YoungsModulus(
			simulation.getYoungsModulus(), simulation.getPoissonsRatio(), muDyoung, lambdaDyoung);
		Matrix3 CDyoung = SoftBodySimulation::computeMaterialMatrix(muDyoung, lambdaDyoung);
		Matrix6 KeRef_Dyoung = area * Be.transpose() * CDyoung * Be;
		*adjYoungsModulus += (adjKeRef.cwiseProduct(KeRef_Dyoung)).sum();
	}
	if (adjPoissonRatio)
	{
		real muDpoisson, lambdaDpoisson;
		AdjointUtilities::computeMaterialParameters_D_PoissonRatio(
			simulation.getYoungsModulus(), simulation.getPoissonsRatio(), muDpoisson, lambdaDpoisson);
		Matrix3 CDpoisson = SoftBodySimulation::computeMaterialMatrix(muDpoisson, lambdaDpoisson);
		Matrix6 KeRef_Dpoisson = area * Be.transpose() * CDpoisson * Be;
		*adjPoissonRatio += (adjKeRef.cwiseProduct(KeRef_Dpoisson)).sum();
	}
}

ar::Vector2 ar::InverseProblem_Adjoint_Dynamic::meshImplicitCollisionForces(
	const Vector2& refPos, const Vector2& disp,	const Vector2& vel, 
	real groundHeight, real groundAngle, 
	real groundStiffness, real softminAlpha, real timestep,	real newmarkTheta)
{
	//This is a copy of SoftBodyMesh2D::solve, collision resolution
	Vector2 pos = refPos + disp;
	auto[dist, normal] = SoftBodySimulation::groundCollision(pos, groundHeight, groundAngle);
	//force magnitude
	real fCurrent = -groundStiffness * ar::utils::softmin(dist, softminAlpha); //current timestep
	real distDt = SoftBodySimulation::groundCollisionDt(vel, groundHeight, groundAngle);
	real fDt = -groundStiffness * (ar::utils::softminDx(dist, softminAlpha) * distDt); //time derivative
	real fNext = fCurrent + timestep * fDt; //next timestep
	real f = newmarkTheta * fNext + (1 - newmarkTheta) * fCurrent; //average force magnitude
	//final force:
	return f * normal;
}

void ar::InverseProblem_Adjoint_Dynamic::meshImplicitCollisionForcesAdjoint(
	const Vector2& refPos, const Vector2& disp,	const Vector2& vel, 
	real groundHeight, real groundAngle, 
	real groundStiffness, real softminAlpha, real timestep,	real newmarkTheta, 
	const Vector2& adjForce, 
	Vector2& adjRefPos, Vector2& adjDisp, Vector2& adjVel,
	real& adjGroundHeight, real& adjGroundAngle)
{
	//forward code again:
	Vector2 pos = refPos + disp;																			// 1
	auto[dist, normal] = SoftBodySimulation::groundCollision(pos, groundHeight, groundAngle);				// 2
	real fCurrent = -groundStiffness * ar::utils::softmin(dist, softminAlpha); //current timestep			// 3
	real distDt = SoftBodySimulation::groundCollisionDt(vel, groundHeight, groundAngle);					// 4
	real fDt = -groundStiffness * (ar::utils::softminDx(dist, softminAlpha) * distDt); //time derivative	// 5
	real fNext = fCurrent + timestep * fDt; //next timestep													// 6
	real f = newmarkTheta * fNext + (1 - newmarkTheta) * fCurrent; //average force magnitude				// 7
	//return f * normal;																					// 8

	//adjoint
	real adjF = 0, adjFNext = 0, adjFCurrent = 0, adjFDt = 0, adjDist = 0, adjDistDt = 0;
	Vector2 adjN(0, 0), adjPos(0, 0);
	adjF += normal.dot(adjForce);																			// adj 8
	adjN += f * adjForce;
	adjFNext += newmarkTheta * adjF;																		// adj 7
	adjFCurrent += (1 - newmarkTheta) * adjF;
	adjFCurrent += adjFNext;																				// adj 6
	adjFDt += timestep * adjFNext;
	AdjointUtilities::softminDxAdjoint(dist, softminAlpha, -groundStiffness * distDt * adjFDt, adjDist);	// adj 5
	adjDistDt += -groundStiffness * ar::utils::softminDx(dist, softminAlpha) * adjFDt;
	AdjointUtilities::groundCollisionDtAdjoint(vel, groundHeight, groundAngle, adjDistDt, adjVel, adjGroundHeight, adjGroundAngle); // adj 4
	AdjointUtilities::softminAdjoint(dist, softminAlpha, -groundStiffness * adjFCurrent, adjDist);			// adj 3
	AdjointUtilities::groundCollisionAdjoint(pos, groundHeight, groundAngle, adjDist, adjN, adjPos, adjGroundHeight, adjGroundAngle); // adj 2
	adjRefPos += adjPos;																					// adj 1
	adjDisp += adjPos;
}

std::pair<ar::real, Eigen::VectorXi> ar::InverseProblem_Adjoint_Dynamic::meshComputeReferenceVolume(
	const MeshCostFunctionDefinition& costFunction, const SoftBodyMesh2D& simulation)
{
	typedef InverseProblem_Adjoint_InitialConfiguration IC;
	int n = simulation.getNumElements();
	real weight = 0;
	real volume = 0;
	Eigen::VectorXi windingOrder = Eigen::VectorXi::Zero(n);
	bool windingOrderComputed = false;

	for (const auto& e : costFunction.keyframes)
	{
		if (e.first == 0) continue; //no weight here
		//loop over all triangles
		for (int i=0; i<n; ++i)
		{
			IC::triangle_t tri;
			for (int j=0; j<3; ++j)
			{
				int j2 = simulation.getTriangles().at(i)[j];
				if (simulation.getNodeToFreeMap().at(j2) >= 0)
					tri[j] = e.second.position.segment<2>(2 * simulation.getNodeToFreeMap().at(j2));
				else
					tri[j] = simulation.getReferencePositions()[j2];
			}
			if (!windingOrderComputed)
				windingOrder[i] = IC::meshTriangleWindingOrder(tri);
			volume += e.first * std::abs(IC::meshTriangleArea(tri));
		}
		weight += e.first;
		windingOrderComputed = true;
	}
	volume /= weight;
	CI_LOG_I("Reference volume: " << volume);

	return std::make_pair(volume, windingOrder);
}

std::pair<ar::real, ar::VectorX> ar::InverseProblem_Adjoint_Dynamic::meshComputeVolumePriorDerivative(const VectorX& currentInitialPosition,
	const MeshCostFunctionDefinition& costFunction, const SoftBodyMesh2D& simulation)
{
	//currentInitialPosition and the returned gradient are only the coordinates of the free nodes, Dirichlet-nodes not included
	typedef InverseProblem_Adjoint_InitialConfiguration IC;
	int n = simulation.getNumFreeNodes();
	VectorX gradient = VectorX::Zero(2 * n);
	real cost = 0;

	//1. Compute current volume
	real volume = 0;
	for (int i = 0; i<simulation.getNumElements(); ++i)
	{
		IC::triangle_t tri;
		for (int j = 0; j<3; ++j)
		{
			int j2 = simulation.getTriangles().at(i)[j];
			if (simulation.getNodeToFreeMap().at(j2) >= 0)
				tri[j] = currentInitialPosition.segment<2>(2 * simulation.getNodeToFreeMap().at(j2));
			else
				tri[j] = simulation.getReferencePositions()[j2];
		}
		volume += std::abs(IC::meshTriangleArea(tri));
	}
	CI_LOG_I("Current volume: " << volume);
	cost += square(volume - costFunction.referenceVolume) / 2;

	//2. Compute gradients
	for (int i = 0; i < simulation.getNumElements(); ++i)
	{
		IC::triangle_t tri;
		for (int j = 0; j < 3; ++j)
		{
			int j2 = simulation.getTriangles().at(i)[j];
			if (simulation.getNodeToFreeMap().at(j2) >= 0)
				tri[j] = currentInitialPosition.segment<2>(2 * simulation.getNodeToFreeMap().at(j2));
			else
				tri[j] = simulation.getReferencePositions()[j];
		}
		std::array<real, 6> deriv = IC::meshTriangleAreaDerivative(tri);
		for (int j=0; j<3; ++j)
		{
			int j2 = simulation.getTriangles().at(i)[j];
			int i2 = simulation.getNodeToFreeMap().at(j2);
			if (i2 < 0) continue;
			gradient[2 * i2 + 0] += costFunction.windingOrder[i] * (volume - costFunction.referenceVolume) * deriv[2 * j + 0];
			gradient[2 * i2 + 1] += costFunction.windingOrder[i] * (volume - costFunction.referenceVolume) * deriv[2 * j + 1];
		}
	}

	return std::make_pair(cost, gradient);
}

ar::VectorX ar::InverseProblem_Adjoint_Dynamic::meshParameterDerivativeInitialPosition(
    const std::vector<MeshForwardResult>& forwardResults, 
	const std::vector<MeshForwardStorage>& forwardStorages,
    const std::vector<MeshBackwardResult>& adjointResults,
	const MeshParamAdjoint& adjParam,
	const MeshPrecomputedValues& precomputed,
    const MeshCostFunctionDefinition& costFunction, 
	const SoftBodyMesh2D& simulation)
{
    const int dof = simulation.getNumFreeNodes();
    const int numSteps = static_cast<int>(forwardResults.size()) - 1;
    typedef InverseProblem_Adjoint_InitialConfiguration IC;
    VectorX yF = VectorX::Zero(2 * dof);

    if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
    {
        //gradient with respect to the initial position is already computed in the main adjoint code
        for (int t=1; t<=numSteps; ++t)
        {
            yF += adjParam.adjInitialPosition;
        }
    } else
    {
        //compute the derivatives for the initial position seperately

        //get winding order
        IC::triangle_t referenceTriangle = {
            simulation.getReferencePositions()[simulation.getTriangles()[0].x()],
            simulation.getReferencePositions()[simulation.getTriangles()[0].y()],
            simulation.getReferencePositions()[simulation.getTriangles()[0].z()]
        };
        int windingOrder = IC::meshTriangleWindingOrder(referenceTriangle);

        for (int e = 0; e < simulation.getNumElements(); ++e)
        {
            const Eigen::Vector3i& tri = simulation.getTriangles()[e];
            for (int t = 1; t <= numSteps; ++t) {
                Vector6 ue = Vector6::Zero();
				Vector6 uePrev = Vector6::Zero();
				Vector6 ueDotPrev = Vector6::Zero();
                for (int i2 = 0; i2 < 3; ++i2)
					if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET) {
						ue.segment<2>(2 * i2) = forwardResults[t].u.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]);
						uePrev.segment<2>(2 * i2) = forwardResults[t - 1].u.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]);
						ueDotPrev.segment<2>(2 * i2) = forwardResults[t - 1].uDot.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]);
					}
				IC::triangle_t triangle{
					simulation.getReferencePositions()[tri[0]],
					simulation.getReferencePositions()[tri[1]],
					simulation.getReferencePositions()[tri[2]]
				};
				auto areaDpos = IC::meshTriangleAreaDerivative(triangle);
                for (int i = 0; i < 3; ++i)
                {
                    if (simulation.getNodeStates()[tri[i]] == SoftBodyMesh2D::DIRICHLET) continue;
                    int col = simulation.getNodeToFreeMap()[tri[i]];
					for (int coord = 0; coord < 2; ++coord)
					{
						//current parameter: vertex 'col' with coordinate 'coord'
						//F(:,j) = -dK/dp*u + df/dp
						int j = 2 * col + coord;
						// dKe/dp
						Matrix6 KeDpos = IC::meshStiffnessMatrixDerivative(triangle, 2 * i + coord, precomputed.matC, windingOrder);
						// dFe/dp
						Vector6 feDpos = IC::meshForceDerivative(triangle, 2 * i + coord, simulation.getGravity(), windingOrder);
						// dMe/dp
						real mass = simulation.getMass() * windingOrder / 3;
						//Matrix6 MeDpos = Matrix6::Zero();
						//for (int k = 0; k < 6; ++k) MeDpos(k, k) = mass * areaDpos[k];
						Matrix6 MeDpos = mass * areaDpos[j] * Matrix6::Identity();
						//Newmark integration
						Matrix6 ADpos 
							= (simulation.getDampingAlpha() + (1 / (newmarkTheta*simulation.getTimestep()))) * MeDpos
							+ (simulation.getDampingBeta() + newmarkTheta * simulation.getTimestep()) * KeDpos;
						Vector6 bDpos
							= ((simulation.getDampingAlpha() + (1 / (newmarkTheta*simulation.getTimestep()))) * MeDpos + (simulation.getDampingBeta() + (1 - newmarkTheta) * simulation.getTimestep()) * KeDpos) * uePrev
							+ ((1 / newmarkTheta) * MeDpos) * ueDotPrev
							+ simulation.getTimestep() * feDpos;
                        // dKe/dp * u
                        Vector6 y = ADpos * ue;
                        //F(:,j) -= dK/dp*u per triangle
                        //F(:,j) += Df/dp per triangle
                        // -dE/dp per triangle
                        Vector6 Fe = -(y - bDpos);

                        for (int i2 = 0; i2 < 3; ++i2)
                            if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
                                yF[j] += Fe.segment<2>(2 * i2).dot(adjointResults[t].adjU.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]));
                    }
                }
            }
        }
    }

    //prior gradient r
    VectorX r = VectorX::Zero(2 * dof);
    const VectorX positions = simulation.linearize(simulation.getReferencePositions());
    for (int i = 0; i < numSteps; ++i)
    {
        //cost on position
        if (costFunction.keyframes.at(i).second.position.size() != 0)
        {
            r += costFunction.keyframes[i].first * costFunction.positionWeighting
                * (positions + forwardResults[i + 1].u - costFunction.keyframes[i].second.position);
        }
    }

    //final gradient
    VectorX gradient = yF + r;
    return gradient;
}

ar::InverseProblem_Adjoint_Dynamic::GridPrecomputedValues ar::InverseProblem_Adjoint_Dynamic::gridPrecomputeValues(
	const SoftBodyGrid2D& simulation)
{
	//That's the only supported time integration scheme
	assert(simulation.getTimeIntegrator() == TimeIntegrator::Integrator::Newmark1);
	assert(simulation.isExplicitDiffusion());
    assert(simulation.getAdvectionMode() == SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD);

	GridPrecomputedValues values = {};
	int n = simulation.getDoF();

	//material matrix
	values.matC = SoftBodySimulation::computeMaterialMatrix(
		simulation.getMaterialMu(), simulation.getMaterialLambda());

	//mass vector
	VectorX Mvec = VectorX::Zero(2 * n);
	simulation.assembleMassMatrix(Mvec, simulation.getPosToIndex());
	values.Mvec = Mvec;

	if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
	{
		//with corotational correct on, we can only precompute the mass
		//all other matrices change every frame
		//-> nothing else to do
	}
	else
	{
		//no rotation correct, stiffness matrix K and load vector f are constant,
		//and so are the integrator matrices A, B, C, d.

		MatrixX K = MatrixX::Zero(2 * n, 2 * n);
		VectorX f = VectorX::Zero(2 * n);
		simulation.assembleStiffnessMatrix(
			values.matC, simulation.getMaterialMu(), simulation.getMaterialLambda(),
			K, &f, simulation.getPosToIndex(),
			VectorX::Zero(2 * n));
		VectorX collisionForces = VectorX::Zero(2 * n);
		simulation.assembleForceVector(f, simulation.getPosToIndex(), collisionForces);
		values.K = K;

		MatrixX D = TimeIntegrator::rayleighDamping(
			simulation.getDampingAlpha(), Mvec,
			simulation.getDampingBeta(), K);

		real deltaT = simulation.getTimestep();
		values.A = TimeIntegrator_Newmark1::getMatrixPartA(K, D, Mvec, f, deltaT, newmarkTheta);
		values.B = TimeIntegrator_Newmark1::getMatrixPartB(K, D, Mvec, f, deltaT, newmarkTheta);
		values.C = TimeIntegrator_Newmark1::getMatrixPartC(K, D, Mvec, f, deltaT, newmarkTheta);
		values.d = TimeIntegrator_Newmark1::getMatrixPartD(K, D, Mvec, f, deltaT, newmarkTheta);
		values.ALu = values.A.partialPivLu();
	}
	return values;
}

std::pair<ar::InverseProblem_Adjoint_Dynamic::GridForwardResult, ar::InverseProblem_Adjoint_Dynamic::GridForwardStorage>
ar::InverseProblem_Adjoint_Dynamic::gridForwardStep(
	const GridForwardResult& currentResult,
	const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed,
    const Matrix2X* partialObservationPoints)
{
	//FORWARD
	const int resolution = simulation.getGridResolution();
	const real h = 1.0 / (resolution - 1);
	const Vector2 size(h, h); //size of each cell

	//collect degrees of freedom in the stifness solve
	const Eigen::MatrixXi& posToIndex = simulation.getPosToIndex();
	const SoftBodyGrid2D::indexToPos_t& indexToPos = simulation.getIndexToPos();
	const int dof = simulation.getDoF();

	GridForwardResult results = {};
	GridForwardStorage storage = {};

	//compute next displacements
	VectorX nextU;
	if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
	{
		//collision forces
		VectorX collisionForces = VectorX::Zero(2 * dof);
		if (simulation.isEnableCollision())
		{
			collisionForces = simulation.resolveCollisions(posToIndex, dof, currentResult.u, currentResult.uDot);
		}

		//corotational correction, recompute all matrices
		MatrixX K = MatrixX::Zero(2 * dof, 2 * dof);
		VectorX f = VectorX::Zero(2 * dof);
		simulation.assembleForceVector(f, posToIndex, collisionForces);
		simulation.assembleStiffnessMatrix(
			precomputed.matC, simulation.getMaterialMu(), simulation.getMaterialLambda(),
			K, &f, posToIndex, currentResult.u);
		storage.K = K;

		MatrixX D = TimeIntegrator::rayleighDamping(
			simulation.getDampingAlpha(), precomputed.Mvec,
			simulation.getDampingBeta(), K);

		real deltaT = simulation.getTimestep();
		storage.A = TimeIntegrator_Newmark1::getMatrixPartA(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
		storage.B = TimeIntegrator_Newmark1::getMatrixPartB(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
		storage.C = TimeIntegrator_Newmark1::getMatrixPartC(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);
		VectorX d = TimeIntegrator_Newmark1::getMatrixPartD(K, D, precomputed.Mvec, f, deltaT, newmarkTheta);

		VectorX rhs = storage.B * currentResult.u + storage.C * currentResult.uDot + d;
		Eigen::PartialPivLU<MatrixX> Alu = storage.A.partialPivLu();
		nextU = Alu.solve(rhs);
		storage.ALu = Alu;
		storage.rhs = rhs;
	}
	else
	{
		//no correction, use precomputed matrices
		VectorX rhs = precomputed.B * currentResult.u + precomputed.C * currentResult.uDot + precomputed.d;
		nextU = precomputed.ALu.solve(rhs);
	}
	results.u = nextU;

	//velocity update
	VectorX nextUDot = (1 / (newmarkTheta*simulation.getTimestep())) * (nextU - currentResult.u)
		- ((1 - newmarkTheta) / newmarkTheta) * currentResult.uDot;
	results.uDot = nextUDot;

	//map back to a grid
	SoftBodyGrid2D::grid_t uGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
	SoftBodyGrid2D::grid_t uGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
	for (int i = 0; i < dof; ++i) {
		Eigen::Vector2i p = indexToPos.at(i);
		uGridX(p.x(), p.y()) = nextU[2 * i];
		uGridY(p.x(), p.y()) = nextU[2 * i + 1];
	}

	//perform diffusion step
	storage.validCells = simulation.computeValidCells(uGridX, uGridY, posToIndex);
	storage.uGridXDiffused = GridUtils2D::fillGridDiffusion(uGridX, storage.validCells);
	storage.uGridYDiffused = GridUtils2D::fillGridDiffusion(uGridY, storage.validCells);

    if (partialObservationPoints)
    {
        //find SDF values at the partial observation points
        int num = partialObservationPoints->cols();
        results.observationSdfValues.resize(num);
        storage.observationCellWeights.resize(2, num);
        storage.observationCells.resize(2, num);
#pragma omp parallel for
        for (int i=0; i<num; ++i)
        {
            bool found = false;
            Vector2 p = partialObservationPoints->col(i);
            for (int x=0; x<resolution-1 /*&& !found*/; ++x) for (int y=0; y<resolution-1 /*&& !found*/; ++y)
            {
                //check if this point p lies within the current cell
				const real h = 1.0 / (simulation.getGridResolution());
                std::array<Vector2, 4> corners;
                corners[0] = Vector2(x * h + storage.uGridXDiffused(x, y), y * h + storage.uGridYDiffused(x, y));
                corners[1] = Vector2((x+1) * h + storage.uGridXDiffused(x+1, y), y * h + storage.uGridYDiffused(x+1, y));
                corners[2] = Vector2(x * h + storage.uGridXDiffused(x, y+1), (y+1) * h + storage.uGridYDiffused(x, y+1));
                corners[3] = Vector2((x+1) * h + storage.uGridXDiffused(x+1, y+1), (y+1) * h + storage.uGridYDiffused(x+1, y+1));
                Vector2 ab = utils::bilinearInterpolateInv(corners, p);

				////Test
				//Vector2 p2 = utils::bilinearInterpolate(corners, ab.x(), ab.y());
				//auto[ab1, ab2] = utils::bilinearInterpolateInvAnalytic(corners, p);
				//bool insideNum = ab.x() >= 0 && ab.y() >= 0 && ab.x() <= 1 && ab.y() <= 1;
				//bool insideAna1 = ab1.x() >= 0 && ab1.y() >= 0 && ab1.x() <= 1 && ab1.y() <= 1;
				//bool insideAna2 = ab2.x() >= 0 && ab2.y() >= 0 && ab2.x() <= 1 && ab2.y() <= 1;
				//if (!p.isApprox(p2, 1e-5) && insideNum)
				//	CI_LOG_E("inverse interpolation did not converge");
				//if (insideAna1 && insideAna2)
				//	CI_LOG_E("two solutions for the same position in the same cell!");
				//if (insideAna1)
				//{
				//	if (!insideNum)
				//		CI_LOG_E("analytic solution is inside, but numerical one is not");
				//	//else if (!ab.isApprox(ab1))
				//		//CI_LOG_E("analytic and numeric solution don't match");
				//}
				//if (insideAna2)
				//{
				//	if (!insideNum)
				//		CI_LOG_E("analytic solution is inside, but numerical one is not");
				//	//else if (!ab.isApprox(ab2))
				//		//CI_LOG_E("analytic and numeric solution don't match");
				//}

                if (ab.x()<0 || ab.y()<0 || ab.x()>1 || ab.y()>1) continue; //outside
				//Test
				Vector2 p2 = utils::bilinearInterpolate(corners, ab.x(), ab.y());
				if (!p.isApprox(p2, 1e-5))
					CI_LOG_E("inverse interpolation did not converge, x="<<x<<", y="<<y<<", pTruth=("<<p.x()<<","<<p.y()<<"), pComputed=("<<p2.x()<<","<<p2.y()<<"), alpha="<<ab.x()<<", beta="<<ab.y());
                //we found the point, compute the SDF value
                std::array<real, 4> values = {
                    simulation.getSdfReference()(x, y),
                    simulation.getSdfReference()(x+1, y),
                    simulation.getSdfReference()(x, y+1),
                    simulation.getSdfReference()(x+1, y+1) };
                real value = utils::bilinearInterpolate(values, ab.x(), ab.y());
				//test
				if (found)
				{
					CI_LOG_E("observed point " << p.transpose() << " was already found in cell " << storage.observationCells.col(i).transpose()
						<< " (alpha=" << storage.observationCellWeights(0, i) << ", beta=" << storage.observationCellWeights(1, i) << ", phi=" << results.observationSdfValues[i] << ")"
						<< ", now we are in cell " << x << " " << y << " with alpha=" << ab.x() << ", beta=" << ab.y() << ", phi=" << value << " -> use minimal value");
					if (value > results.observationSdfValues[i]) continue;
				}
				//OBSERVATION: Duplicate cells are only found if bilinearInterpolateInv of one of those cells did not converge
				//-> Check if bilinearInterpolateInv has converged and only if so use that cell
				// Then I can safely stop the search and don't miss other cells that are better
				// (If other cells also find ab-values that are inside, bilinearInterpolateInv did not converge there;
				//  there can only exist one solution!)
				//-> For the 3D-case, this allows to switch the loop: Parallel over the grid cells,
				//  optimized search (Octree?) over the observations

                //write to results and break
                found = true;
                results.observationSdfValues[i] = value;
                storage.observationCells.col(i) = Eigen::Vector2i(x, y);
                storage.observationCellWeights.col(i) = ab;
				CI_LOG_D("observed point " << p.transpose() << " found in cell " << x << " " << y);
            }
            if (!found)
            {
                CI_LOG_E("observed point " << p.transpose() << " not found!");
				results.observationSdfValues[i] = 0;
				storage.observationCells.col(i) = Eigen::Vector2i(-1, -1);
				storage.observationCellWeights.col(i) = Vector2(0.5, 0.5);
            }
        }
    } 
    else 
    {
	    //advect levelset
	    storage.sdfPreReconstruction = GridUtils2D::advectGridDirectForward(
		    simulation.getSdfReference(), storage.uGridXDiffused, storage.uGridYDiffused, -1 / h, &storage.advectionWeights);

        //reconstruct SDF
        if (SoftBodyGrid2D::recoverSdf) {
            results.sdf = GridUtils2D::recoverSDFSussmann(storage.sdfPreReconstruction, 0.01, 20);
        } else
        {
            results.sdf = storage.sdfPreReconstruction;
        }
    }

	return { results, storage };
}

void ar::InverseProblem_Adjoint_Dynamic::gridAdjointStep(
	const GridForwardResult& prevResult,
	const GridForwardResult& currentResult,
	GridBackwardResult& currentAdj,
	GridBackwardResult& prevAdj,
	GridParamAdjoint& paramAdj,
	const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed,
	const GridForwardStorage& forwardStorage)
{
	//parameters
	const int resolution = simulation.getGridResolution();
	const real h = 1.0 / (resolution - 1);
	const Vector2 size(h, h);
	//collect degrees of freedom in the stifness solve
	const Eigen::MatrixXi& posToIndex = simulation.getPosToIndex();
	const SoftBodyGrid2D::indexToPos_t& indexToPos = simulation.getIndexToPos();
	const int dof = simulation.getDoF();

    SoftBodyGrid2D::grid_t adjUGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
	SoftBodyGrid2D::grid_t adjUGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
    bool usePartialObservations = currentAdj.adjObservationSdfValues.size() > 0;
	if (!currentAdj.adjSdfTmp.isZero(0) && !usePartialObservations) {
        //optimization: we only need to compute the adjoint of the advection, if we have a gradient here

        //Adjoint: reconstruct SDF
        SoftBodyGrid2D::grid_t adjInputSdf1;
        if (SoftBodyGrid2D::recoverSdf) {
            GridUtils2D::RecoverSDFSussmannAdjointStorage recoveryStorage;
            GridUtils2D::recoverSDFSussmann(forwardStorage.sdfPreReconstruction, 0.01, 20, &recoveryStorage);
            adjInputSdf1 = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
            GridUtils2D::recoverSDFSussmannAdjoint(
                forwardStorage.sdfPreReconstruction, currentResult.sdf,
                currentAdj.adjSdfTmp, adjInputSdf1, recoveryStorage);
        } else
        {
            adjInputSdf1 = currentAdj.adjSdfTmp;
        }

		//Adjoint: advect levelset
		SoftBodyGrid2D::grid_t adjInputSdf = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
		GridUtils2D::advectGridDirectForwardAdjoint(adjInputSdf, adjUGridX, adjUGridY,
			adjInputSdf1, forwardStorage.uGridXDiffused, forwardStorage.uGridXDiffused,
			simulation.getSdfReference(), forwardStorage.sdfPreReconstruction, -1 / h, forwardStorage.advectionWeights);
		if (paramAdj.adjInitialSdf.size() > 0) paramAdj.adjInitialSdf += adjInputSdf;

	}
    else if (usePartialObservations)
    {
        //partial observations
        int num = currentAdj.adjObservationSdfValues.size();
        assert(currentResult.observationSdfValues.size() == num);
        assert(forwardStorage.observationCells.cols() == num);
        assert(forwardStorage.observationCellWeights.cols() == num);
        //for each observation, compute the adjoint of the displacements
        for (int i=0; i<num; ++i)
        {
            int x = forwardStorage.observationCells(0, i);
            int y = forwardStorage.observationCells(1, i);
			if (x < 0 || y < 0) continue;
            real fx = forwardStorage.observationCellWeights(0, i);
            real fy = forwardStorage.observationCellWeights(1, i);
            //adjoint of the interpolation for the sdf value
            std::array<real, 4> values = {
                    simulation.getSdfReference()(x, y),
                    simulation.getSdfReference()(x+1, y),
                    simulation.getSdfReference()(x, y+1),
                    simulation.getSdfReference()(x+1, y+1) };
            real adjFx = 0, adjFy = 0;
            AdjointUtilities::bilinearInterpolateAdjoint2(values, fx, fy, currentAdj.adjObservationSdfValues[i], adjFx, adjFy);
            //adjoint of the inverse interpolation for the corner positions
            std::array<Vector2, 4> adjCorners;
            for (int j=0; j<4; ++j) adjCorners[j].setZero();
			const real h = 1.0 / (simulation.getGridResolution());
			std::array<Vector2, 4> corners;
			corners[0] = Vector2(x * h + forwardStorage.uGridXDiffused(x, y), y * h + forwardStorage.uGridYDiffused(x, y));
			corners[1] = Vector2((x + 1) * h + forwardStorage.uGridXDiffused(x + 1, y), y * h + forwardStorage.uGridYDiffused(x + 1, y));
			corners[2] = Vector2(x * h + forwardStorage.uGridXDiffused(x, y + 1), (y + 1) * h + forwardStorage.uGridYDiffused(x, y + 1));
			corners[3] = Vector2((x + 1) * h + forwardStorage.uGridXDiffused(x + 1, y + 1), (y + 1) * h + forwardStorage.uGridYDiffused(x + 1, y + 1));
            Vector2 p = utils::bilinearInterpolate(corners, fx, fy);
            AdjointUtilities::bilinearInterpolateInvAdjoint(corners, p, Vector2(fx, fy), Vector2(adjFx, adjFy), adjCorners);
            //write gradient to the grid
            adjUGridX(x, y) += adjCorners[0].x();       adjUGridY(x, y) += adjCorners[0].y();
            adjUGridX(x+1, y) += adjCorners[1].x();     adjUGridY(x+1, y) += adjCorners[1].y();
            adjUGridX(x, y+1) += adjCorners[2].x();     adjUGridY(x, y+1) += adjCorners[2].y();
            adjUGridX(x+1, y+1) += adjCorners[3].x();   adjUGridY(x+1, y+1) += adjCorners[3].y();
        }
    }

    //Adjoint: perform diffusion step
	//note that the adjoint is both input and output, it is modified in-place
	GridUtils2D::fillGridDiffusionAdjoint(adjUGridX, adjUGridX, forwardStorage.uGridXDiffused, forwardStorage.validCells);
	GridUtils2D::fillGridDiffusionAdjoint(adjUGridY, adjUGridY, forwardStorage.uGridXDiffused, forwardStorage.validCells);

	//Adjoint: map back to a grid
	for (int i = 0; i < dof; ++i) {
		const Eigen::Vector2i p = indexToPos.at(i);
		currentAdj.adjUTmp[2 * i] += adjUGridX(p.x(), p.y());
		currentAdj.adjUTmp[2 * i + 1] += adjUGridY(p.x(), p.y());
	}

	// Now the rest is (almost) a copy of meshAdjointStep
	real deltaT = simulation.getTimestep();

	// adjoint: velocity update
	currentAdj.adjUDot = currentAdj.adjUDotTmp;
	currentAdj.adjUTmp += (1 / (newmarkTheta * deltaT)) * currentAdj.adjUDot;
	prevAdj.adjUDotTmp -= ((1 - newmarkTheta) / newmarkTheta) * currentAdj.adjUDot;
	prevAdj.adjUTmp += (-1 / (newmarkTheta * deltaT)) * currentAdj.adjUDot;

	//adjoint: displacements
	if (simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::Corotation)
	{
		using std::array;

		//corotational correction, compute adjoint through the matrices
		int n = simulation.getDoF();

		// A * nextU = rhs:
		// adjNextU -> adjA, adjRhs
		VectorX adjRhs = VectorX::Zero(2 * n);
		MatrixX adjA = MatrixX::Zero(2 * n, 2 * n);

        currentAdj.adjU = forwardStorage.ALu.solve(currentAdj.adjUTmp); //normally, A has to be transposed, but it is symmetric
        adjRhs += currentAdj.adjU;
        adjA += -currentAdj.adjU * currentResult.u.transpose();

		// rhs = B currentU + C currentUDot + d: 
		// adjRhs -> adjB, adjC, adjd, adjCurrentU, adjCurrentUDot
		MatrixX adjB = adjRhs * prevResult.u.transpose();
		prevAdj.adjUTmp += forwardStorage.B.transpose() * adjRhs;
		MatrixX adjC = adjRhs * prevResult.uDot.transpose();
		prevAdj.adjUDotTmp += forwardStorage.C.transpose() * adjRhs;
		VectorX adjd = adjRhs;

		// A, B, C, d = f(M, D, K, f):
		// adjA, adjB, adjC, adjd -> adjD, adjK, adjf
		MatrixX adjD = MatrixX::Zero(2 * n, 2 * n);
		MatrixX adjK = MatrixX::Zero(2 * n, 2 * n);
		MatrixX adjM = MatrixX::Zero(2 * n, 2 * n);
		VectorX adjf = VectorX::Zero(2 * n);
		adjf += deltaT * adjd;
		adjM += (1 / newmarkTheta) * adjC;
		adjM += (1 / (newmarkTheta * deltaT)) * adjB;
		adjD += adjB;
		adjK += (1 - newmarkTheta) * deltaT * adjB;
		adjM += (1 / (newmarkTheta * deltaT)) * adjA;
		adjD += adjA;
		adjK += newmarkTheta * deltaT * adjA;

		// D = alpha_1 * M + alpha_2 * K
		// adjD -> adjK
		adjM += simulation.getDampingAlpha() * adjD;
		adjK += simulation.getDampingBeta() * adjD;
		paramAdj.adjMassDamping += precomputed.Mvec.dot(adjD.diagonal());
		paramAdj.adjStiffnessDamping += forwardStorage.K.cwiseProduct(adjD).sum();

		// M = m * M_0
		// adjM -> adjm
		VectorX M_0 = precomputed.Mvec / simulation.getMass();
		paramAdj.adjMass += M_0.dot(adjM.diagonal());

		//loop over all elements
		for (int xy = (resolution - 1)*(resolution - 1) - 1; xy >= 0; --xy) {
			int x = xy / (resolution - 1);
			int y = xy % (resolution - 1);
			Matrix8 adjKe = Matrix8::Zero();
			Vector8 adjFe = Vector8::Zero();

			//assemble SDF array
			array<real, 4> sdfs = { 
				simulation.getSdfReference()(x, y),
				simulation.getSdfReference()(x + 1,y), 
				simulation.getSdfReference()(x, y + 1),
				simulation.getSdfReference()(x + 1, y + 1) };
			if (utils::outside(sdfs[0]) && utils::outside(sdfs[1]) && utils::outside(sdfs[2]) && utils::outside(sdfs[3])) {
				//completely outside
				continue;
			}

			//adjoint combine Ke, Fe into full matrix
			int mapping[4] = {
				posToIndex(x, y),
				posToIndex(x + 1, y),
				posToIndex(x, y + 1),
				posToIndex(x + 1, y + 1)
			};
			bool dirichlet[4] = {
				simulation.getGridDirichlet()(x, y),
				simulation.getGridDirichlet()(x + 1, y),
				simulation.getGridDirichlet()(x, y + 1),
				simulation.getGridDirichlet()(x + 1, y + 1)
			};
			for (int i = 0; i < 4; ++i) {
				if (simulation.isHardDirichletBoundaries() && dirichlet[i])
				{
					continue;
				}
				for (int j = 0; j < 4; ++j) {
					if (simulation.isHardDirichletBoundaries() && dirichlet[j]) continue;
					adjKe.block<2, 2>(2 * i, 2 * j) += adjK.block<2, 2>(2 * mapping[i], 2 * mapping[j]);
				}
				adjFe.segment<2>(2 * i) += adjf.segment<2>(2 * mapping[i]);
			}

			const auto getOrZero = [&posToIndex](int x, int y, const VectorX& data) {return posToIndex(x, y) >= 0 ? data.segment<2>(2 * posToIndex(x, y)) : Vector2(0, 0); };
			Vector8 currentUe = (Vector8() <<
				getOrZero(x, y, prevResult.u),
				getOrZero(x+1, y, prevResult.u),
				getOrZero(x, y+1, prevResult.u),
				getOrZero(x+1, y+1, prevResult.u)).finished();
			Vector8 currentUeDot = (Vector8() <<
				getOrZero(x, y, prevResult.uDot),
				getOrZero(x + 1, y, prevResult.uDot),
				getOrZero(x, y + 1, prevResult.uDot),
				getOrZero(x + 1, y + 1, prevResult.uDot)).finished();

			Vector8 adjCurrentUe = Vector8::Zero();
			Vector8 adjCurrentUeDot = Vector8::Zero();
			std::array<real, 4> adjSdf = { real(0), real(0), real(0), real(0) };

			gridComputeElementMatrixAdjoint(
				x, y, sdfs, currentUe, currentUeDot,
				simulation.getRotationCorrection(),
				adjKe, adjFe,
				adjCurrentUe, adjCurrentUeDot, adjSdf,
				paramAdj,
				simulation, precomputed
			);

			if (posToIndex(x, y) >= 0 && !dirichlet[0]) {
				prevAdj.adjUTmp.segment<2>(2 * posToIndex(x, y)) += adjCurrentUe.segment<2>(0);
				prevAdj.adjUDotTmp.segment<2>(2 * posToIndex(x, y)) += adjCurrentUeDot.segment<2>(0);
			}
			if (posToIndex(x + 1, y) >= 0 && !dirichlet[1]) {
				prevAdj.adjUTmp.segment<2>(2 * posToIndex(x + 1, y)) += adjCurrentUe.segment<2>(2);
				prevAdj.adjUDotTmp.segment<2>(2 * posToIndex(x + 1, y)) += adjCurrentUeDot.segment<2>(2);
			}
			if (posToIndex(x, y + 1) >= 0 && !dirichlet[2]) {
				prevAdj.adjUTmp.segment<2>(2 * posToIndex(x, y + 1)) += adjCurrentUe.segment<2>(4);
				prevAdj.adjUDotTmp.segment<2>(2 * posToIndex(x, y + 1)) += adjCurrentUeDot.segment<2>(4);
			}
			if (posToIndex(x + 1, y + 1) >= 0 && !dirichlet[3]) {
				prevAdj.adjUTmp.segment<2>(2 * posToIndex(x + 1, y + 1)) += adjCurrentUe.segment<2>(6);
				prevAdj.adjUDotTmp.segment<2>(2 * posToIndex(x + 1, y + 1)) += adjCurrentUeDot.segment<2>(6);
			}
			if (paramAdj.adjInitialSdf.size()>0)
			{
				paramAdj.adjInitialSdf(x, y) += adjSdf[0];
				paramAdj.adjInitialSdf(x+1, y) += adjSdf[1];
				paramAdj.adjInitialSdf(x, y+1) += adjSdf[2];
				paramAdj.adjInitialSdf(x+1, y+1) += adjSdf[3];
			}
		}

	}
	else
	{
		//DEPRECATED: no parameters are reconstructed
		currentAdj.adjU = precomputed.ALu.solve(currentAdj.adjUTmp);
		prevAdj.adjUDotTmp += precomputed.C.transpose() * currentAdj.adjU;
		prevAdj.adjUTmp += precomputed.B.transpose() * currentAdj.adjU;
	}

}

void ar::InverseProblem_Adjoint_Dynamic::gridComputeElementMatrixAdjoint(int x, int y, const std::array<real, 4>& sdfs,
	const Vector8& currentUe, const Vector8& currentUeDot, SoftBodySimulation::RotationCorrection rotationCorrection,
	const Matrix8& adjKe, const Vector8& adjFe, Vector8& adjCurrentUe, Vector8& adjCurrentUeDot, std::array<real, 4>& adjSdf,
	GridParamAdjoint& paramAdj,
	const SoftBodyGrid2D& simulation, const GridPrecomputedValues& precomputed)
{
	//TODO: fill adjSdf

	using std::array;
	using namespace ar::utils;

	const Vector2 size = simulation.getCellSize();
	const Eigen::MatrixXi& posToIndex = simulation.getPosToIndex();

	//precompute integration weights
	const Integration2D<real>::IntegrationWeights volumeWeights
		= Integration2D<real>::getIntegrateQuadWeights(size, sdfs);
	const Integration2D<real>::IntegrationWeights boundaryWeights
		= Integration2D<real>::getIntegrateQuadBoundaryWeights(size, sdfs);

	//FORWARD

    //stiffness matrix
	array<Matrix8, 4> Kex = {
		simulation.getB1().transpose() * precomputed.matC * simulation.getB1(),
		simulation.getB2().transpose() * precomputed.matC * simulation.getB2(),
		simulation.getB3().transpose() * precomputed.matC * simulation.getB3(),
		simulation.getB4().transpose() * precomputed.matC * simulation.getB4(),
	};
	Matrix8 KeRef = Integration2D<real>::integrateQuad(volumeWeights, Kex);
    //rotation correction
	array<Matrix2, 4> Fx = {
		currentUe.segment<2>(0) * RowVector2(-1 / size.x(), -1 / size.y()),
		currentUe.segment<2>(2) * RowVector2(1 / size.x(), -1 / size.y()),
		currentUe.segment<2>(4) * RowVector2(-1 / size.x(), 1 / size.y()),
		currentUe.segment<2>(6) * RowVector2(1 / size.x(), 1 / size.y())
	};
	Matrix2 F = 0.5 * (Fx[0] + Fx[1] + Fx[2] + Fx[3] + Matrix2::Identity());
	Matrix2 R = abs(F.determinant()) < 1e-15 ? Matrix2::Identity() : SoftBodySimulation::polarDecomposition(F);
	Matrix8 Re;
	Re << R, Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(),
		Matrix2::Zero(), R, Matrix2::Zero(), Matrix2::Zero(),
		Matrix2::Zero(), Matrix2::Zero(), R, Matrix2::Zero(),
		Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(), R;
	Vector8 xe;
	xe << Vector2(x*size.x(), y*size.y()), Vector2((x + 1)*size.x(), y*size.y()),
		Vector2(x*size.x(), (y + 1)*size.y()), Vector2((x + 1)*size.x(), (y + 1)*size.y());

    //Dirichlet boundaries: ignored

	//ADJOINT

    //adjoint: dirichlet boundaries
    //For now ignored, assume that they don't influence
    // the gradient of the SDF values

    //adjoint: rotation correction
	Matrix8 adjRe = Matrix8::Zero();
	Matrix8 adjKeRef = Matrix8::Zero();
	Vector8 adjFeRef = Vector8::Zero();
	//Fe = Fe - Re * KeRef * (Re.transpose() * xe - xe);
	adjRe -= xe * adjFe.transpose() * Re * KeRef
		+ adjFe * xe.transpose() * Re * KeRef.transpose()
		- adjFe * xe.transpose() * KeRef;
	adjKeRef -= Re.transpose() * adjFe * (Re.transpose() * xe - xe).transpose();
	adjFeRef = adjFe;
	//Ke = Re * KeRef * Re.transpose();
	adjRe += (adjKe.transpose() * Re * KeRef.transpose()) + (adjKe * Re * KeRef);
	adjKeRef += Re.transpose() * adjKe * Re;

	Matrix2 adjR = adjRe.block<2, 2>(0, 0) + adjRe.block<2, 2>(2, 2) + adjRe.block<2, 2>(4, 4) + adjRe.block<2, 2>(6, 6);
	Matrix2 adjF = abs(F.determinant()) < 1e-15
		? Matrix2::Zero()
		: AdjointUtilities::polarDecompositionAdjoint(F, adjR);
	//Update adjoint U
	adjCurrentUe.segment<2>(0) 
		+= 0.5 * adjF * Vector2(-1 / size.x(), -1 / size.y());
	adjCurrentUe.segment<2>(2)
		+= 0.5 * adjF * Vector2(1 / size.x(), -1 / size.y());
	adjCurrentUe.segment<2>(4)
		+= 0.5 * adjF * Vector2(-1 / size.x(), 1 / size.y());
	adjCurrentUe.segment<2>(6)
		+= 0.5 * adjF * Vector2(1 / size.x(), 1 / size.y());

    //Initial SDF derivatives
	{
        auto deriv = Integration2D<real>::integrateQuadDsdf(
            size, sdfs, Kex);
        for (int i=0; i<4; ++i)
        {
            adjSdf[i] += deriv[i].cwiseProduct(adjKeRef).sum();
        }
	}

	//Young's Modulus derivatives
	{
		//The resulting gradient is negated
		real muDyoung, lambdaDyoung;
		AdjointUtilities::computeMaterialParameters_D_YoungsModulus(
			simulation.getYoungsModulus(), simulation.getPoissonsRatio(), muDyoung, lambdaDyoung);
		//Contributions from Nietsche-Dirichlet.
		Matrix8 KeDirichletDYoung = Matrix8::Zero();
		Vector8 FeDirichletDYoung = Vector8::Zero();
		simulation.computeNietscheDirichletBoundaries(x, y,
			KeDirichletDYoung, FeDirichletDYoung,
			muDyoung, lambdaDyoung, boundaryWeights);
		//paramAdj.adjYoungsModulus +=
		//	KeDirichletDYoung.cwiseProduct(adjKe).sum() +
		//	FeDirichletDYoung.dot(adjFe);

		//Contributes from the stiffness matrix
		Matrix3 CDyoung = SoftBodySimulation::computeMaterialMatrix(muDyoung, lambdaDyoung);
		array<Matrix8, 4> KexDyoung = {
			simulation.getB1().transpose() * CDyoung * simulation.getB1(),
			simulation.getB2().transpose() * CDyoung * simulation.getB2(),
			simulation.getB3().transpose() * CDyoung * simulation.getB3(),
			simulation.getB4().transpose() * CDyoung * simulation.getB4(),
		};
		Matrix8 KeRefDyoung = Integration2D<real>::integrateQuad(volumeWeights, KexDyoung);
		paramAdj.adjYoungsModulus += (adjKeRef.cwiseProduct(KeRefDyoung)).sum();
	}

	//Poisson's ratio derivatives
	{
		real muDpoisson, lambdaDpoisson;
		AdjointUtilities::computeMaterialParameters_D_PoissonRatio(
			simulation.getYoungsModulus(), simulation.getPoissonsRatio(), muDpoisson, lambdaDpoisson);
		//Contributions from Nietsche-Dirichlet.
		Matrix8 KeDirichletDPoisson = Matrix8::Zero();
		Vector8 FeDirichletDPoisson = Vector8::Zero();
		simulation.computeNietscheDirichletBoundaries(x, y,
			KeDirichletDPoisson, FeDirichletDPoisson,
			muDpoisson, lambdaDpoisson, boundaryWeights);
		//paramAdj.adjPoissonRatio +=
		//	KeDirichletDPoisson.cwiseProduct(adjKe).sum() +
		//	FeDirichletDPoisson.dot(adjFe);

		//Contributes from the stiffness matrix
		Matrix3 CDpoisson = SoftBodySimulation::computeMaterialMatrix(muDpoisson, lambdaDpoisson);
		array<Matrix8, 4> KexDpoisson = {
			simulation.getB1().transpose() * CDpoisson * simulation.getB1(),
			simulation.getB2().transpose() * CDpoisson * simulation.getB2(),
			simulation.getB3().transpose() * CDpoisson * simulation.getB3(),
			simulation.getB4().transpose() * CDpoisson * simulation.getB4(),
		};
		Matrix8 KeRefDpoisson = Integration2D<real>::integrateQuad(volumeWeights, KexDpoisson);
		paramAdj.adjPoissonRatio += (adjKeRef.cwiseProduct(KeRefDpoisson)).sum();
	}

	//Adjoint forces
	bool completelyInside = inside(sdfs[0]) && inside(sdfs[1]) && inside(sdfs[2]) && inside(sdfs[3]);
	if (simulation.isEnableCollision() && !completelyInside)
	{
		std::array<Vector2, 4> corners = {
			Vector2((xe + currentUe).segment<2>(0)), 
			Vector2((xe + currentUe).segment<2>(2)),
			Vector2((xe + currentUe).segment<2>(4)),
			Vector2((xe + currentUe).segment<2>(6)) };
		std::array<Vector2, 4> velocities = {
			Vector2(currentUeDot.segment<2>(0)),
			Vector2(currentUeDot.segment<2>(2)),
			Vector2(currentUeDot.segment<2>(4)),
			Vector2(currentUeDot.segment<2>(6)) };

		//Adjoint: Neumann boundaries
		//In: adjFeRef, Out: adjNeumann

		//Vector8 FeNeumann = Integration2D<real>::integrateQuadBoundary(size_, sdfs, FeNeumannX);
		array<Vector8, 4> adjFeNeumannX = {
			boundaryWeights[0] * adjFeRef,
			boundaryWeights[1] * adjFeRef,
			boundaryWeights[2] * adjFeRef,
			boundaryWeights[3] * adjFeRef
		};
		//FeNeumannX[0] = Phi1.transpose() * Phi1 * FsNeumann, ...
		Vector8 adjFsNeumann = Vector8::Zero();
		adjFsNeumann += (simulation.getPhi1().transpose() * simulation.getPhi1()).transpose() * adjFeNeumannX[0];
		adjFsNeumann += (simulation.getPhi2().transpose() * simulation.getPhi2()).transpose() * adjFeNeumannX[1];
		adjFsNeumann += (simulation.getPhi3().transpose() * simulation.getPhi3()).transpose() * adjFeNeumannX[2];
		adjFsNeumann += (simulation.getPhi4().transpose() * simulation.getPhi4()).transpose() * adjFeNeumannX[3];

		//collision forces
		array<Vector2, 4> adjPosition({ Vector2::Zero(), Vector2::Zero(), Vector2::Zero(), Vector2::Zero() });
		array<Vector2, 4> adjVelocity({ Vector2::Zero(), Vector2::Zero(), Vector2::Zero(), Vector2::Zero() });
		gridImplicitCollisionForcesAdjoint(
			sdfs, corners, velocities,
			simulation.getCollisionResolution() == SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT,
			simulation.getGroundPlaneHeight(), simulation.getGroundPlaneAngle(),
			simulation.getGroundStiffness(), simulation.getCollisionSoftmaxAlpha(),
			simulation.getTimestep(), TimeIntegrator_Newmark1::DefaultTheta,
			array<Vector2, 4>({ adjFsNeumann.segment<2>(0), adjFsNeumann.segment<2>(2), adjFsNeumann.segment<2>(4), adjFsNeumann.segment<2>(6) }),
			adjPosition, adjVelocity, adjSdf,
			paramAdj.adjGroundPlaneHeight, paramAdj.adjGroundPlaneAngle);

		for (int i = 0; i < 4; ++i) {
			adjCurrentUeDot.segment<2>(2 * i) += adjVelocity[i];
			adjCurrentUeDot.segment<2>(2 * i) += adjPosition[i];
		}
	}
}

void ar::InverseProblem_Adjoint_Dynamic::gridImplicitCollisionForcesAdjoint(
    const std::array<real, 4>& sdfs, const std::array<Vector2, 4>& positions, const std::array<Vector2, 4>& velocities, bool implicit, real groundHeight,
	real groundAngle, real groundStiffness, real softminAlpha, real timestep, real newmarkTheta,
	const std::array<ar::Vector2, 4>& adjForce, 
    std::array<Vector2, 4>& adjPositions, std::array<Vector2, 4>& adjVelocities, std::array<real, 4>& adjSdf,
    real& adjGroundHeight, real& adjGroundAngle)
{
	//FORWARD
	//the basis points for interpolation
	auto points1 = Integration2D<ar::real>::getIntersectionPoints(sdfs, positions);
	if (!points1.has_value()) return;
	auto points2 = Integration2D<ar::real>::getIntersectionPoints(sdfs, { Vector2(0,0), Vector2(1,0), Vector2(0,1), Vector2(1,1) }).value();
	//check them against collisions
	auto col1 = SoftBodySimulation::groundCollision(points1->first, groundHeight, groundAngle);
	auto col2 = SoftBodySimulation::groundCollision(points1->second, groundHeight, groundAngle);
	Vector2 normal = col1.second; //or col2.second, they are the same
	real dist1 = col1.first;
	real dist2 = col2.first;

	//Adjoint
	//adjoint: blend them into the forces
	//TODO: add dervivative for adjPoints2 (initial SDF)
	Vector2 adjF1 = Vector2::Zero(), adjF2 = Vector2::Zero();
	adjF1 += (1 - points2.first.x()) * (1 - points2.first.y()) * 0.5 * adjForce[0];
	adjF1 += points2.first.x() * (1 - points2.first.y()) * 0.5 * adjForce[1];
	adjF1 += (1 - points2.first.x()) * points2.first.y() * 0.5 * adjForce[2];
	adjF1 += points2.first.x() * points2.first.y() * 0.5 * adjForce[3];
	adjF2 += (1 - points2.second.x()) * (1 - points2.second.y()) * 0.5 * adjForce[0];
	adjF2 += points2.second.x() * (1 - points2.second.y()) * 0.5 * adjForce[1];
	adjF2 += (1 - points2.second.x()) * points2.second.y() * 0.5 * adjForce[2];
	adjF2 += points2.second.x() * points2.second.y() * 0.5 * adjForce[3];

	//adjoint: compute the forces for these two points
	Vector2 adjNormal = Vector2::Zero();
	real adjDist1 = 0, adjDist2 = 0;
	if (!implicit) { //CollisionResolution::SPRING_EXPLICT
		adjNormal += (-groundStiffness * ar::utils::softmin(dist1, softminAlpha)) * adjF1;
		adjNormal += (-groundStiffness * ar::utils::softmin(dist2, softminAlpha)) * adjF2;
		AdjointUtilities::softminAdjoint(dist1, softminAlpha, -groundStiffness * normal.dot(adjF1), adjDist1);
		AdjointUtilities::softminAdjoint(dist2, softminAlpha, -groundStiffness * normal.dot(adjF2), adjDist2);
	}
	else //CollisionResolution::SPRING_IMPLICIT
	{
		//forward:
		//current forces
		Vector2 f1Current = (-groundStiffness * ar::utils::softmin(dist1, softminAlpha)) * normal;
		Vector2 f2Current = (-groundStiffness * ar::utils::softmin(dist2, softminAlpha)) * normal;
		//velocity at the two points
		Vector2 vel1 = ar::utils::bilinearInterpolate(velocities, points2.first.x(), points2.first.y());
		Vector2 vel2 = ar::utils::bilinearInterpolate(velocities, points2.second.x(), points2.second.y());
		//time derivative
		Vector2 f1Dt = (-groundStiffness * ar::utils::softminDx(dist1, softminAlpha) * SoftBodySimulation::groundCollisionDt(vel1, groundHeight, groundAngle)) * normal;
		Vector2 f2Dt = (-groundStiffness * ar::utils::softminDx(dist2, softminAlpha) * SoftBodySimulation::groundCollisionDt(vel2, groundHeight, groundAngle)) * normal;

		//adjoint
		//blend them together to the final force
		Vector2 adjF1Next = newmarkTheta * adjF1;
		Vector2 adjF2Next = newmarkTheta * adjF2;
		Vector2 adjF1Current = (1 - newmarkTheta) * adjF1;
		Vector2 adjF2Current = (1 - newmarkTheta) * adjF2;
		//approximate next force
		adjF1Current += adjF1Next;
		adjF2Current += adjF2Next;
		Vector2 adjF1Dt = timestep * adjF1Next;
		Vector2 adjF2Dt = timestep * adjF2Next;
		//adjoint: time derivative
		Vector2 adjVel1 = Vector2::Zero();
		AdjointUtilities::softminDxAdjoint(dist1, softminAlpha, -groundStiffness * SoftBodySimulation::groundCollisionDt(vel1, groundHeight, groundAngle) * normal.dot(adjF1Dt), adjDist1);
		AdjointUtilities::groundCollisionDtAdjoint(vel1, groundHeight, groundAngle, -groundStiffness * ar::utils::softminDx(dist1, softminAlpha) * normal.dot(adjF1Dt), adjVel1, adjGroundHeight, adjGroundAngle);
		adjNormal += (-groundStiffness * ar::utils::softminDx(dist1, softminAlpha) * SoftBodySimulation::groundCollisionDt(vel1, groundHeight, groundAngle)) * adjF1Dt;
		Vector2 adjVel2 = Vector2::Zero();
		AdjointUtilities::softminDxAdjoint(dist2, softminAlpha, -groundStiffness * SoftBodySimulation::groundCollisionDt(vel2, groundHeight, groundAngle) * normal.dot(adjF2Dt), adjDist2);
		AdjointUtilities::groundCollisionDtAdjoint(vel2, groundHeight, groundAngle, -groundStiffness * ar::utils::softminDx(dist2, softminAlpha) * normal.dot(adjF2Dt), adjVel2, adjGroundHeight, adjGroundAngle);
		adjNormal += (-groundStiffness * ar::utils::softminDx(dist2, softminAlpha) * SoftBodySimulation::groundCollisionDt(vel2, groundHeight, groundAngle)) * adjF2Dt;
		//adjoint: velocity at the two points
		AdjointUtilities::bilinearInterpolateAdjoint1(points2.first.x(), points2.first.y(), adjVel1, adjVelocities);
		AdjointUtilities::bilinearInterpolateAdjoint1(points2.second.x(), points2.second.y(), adjVel2, adjVelocities);
		//adjoint: current forces
		adjNormal += (-groundStiffness * ar::utils::softmin(dist1, softminAlpha)) * adjF1Current;
		adjNormal += (-groundStiffness * ar::utils::softmin(dist2, softminAlpha)) * adjF2Current;
		AdjointUtilities::softminAdjoint(dist1, softminAlpha, -groundStiffness * normal.dot(adjF1Current), adjDist1);
		AdjointUtilities::softminAdjoint(dist2, softminAlpha, -groundStiffness * normal.dot(adjF2Current), adjDist2);
	}

	//adjoint: check collisions
	Vector2 adjPoints1a = Vector2::Zero(), adjPoints1b = Vector2::Zero();
	AdjointUtilities::groundCollisionAdjoint(points1->first, groundHeight, groundAngle, adjDist1, Vector2(0, 0), adjPoints1a, adjGroundHeight, adjGroundAngle);
	AdjointUtilities::groundCollisionAdjoint(points1->second, groundHeight, groundAngle, adjDist2, adjNormal, adjPoints1b, adjGroundHeight, adjGroundAngle);

	//adjoint: the two points of the current cell's boundary that cut the cell sides
	AdjointUtilities::getIntersectionPointsAdjoint(sdfs, positions, adjPoints1a, adjPoints1b, adjSdf, adjPositions);

}

ar::real ar::InverseProblem_Adjoint_Dynamic::gridComputeReferenceVolume(const GridCostFunctionDefinition& costFunction,
	const SoftBodyGrid2D& simulation)
{
	int resolution = simulation.getGridResolution();
	Vector2 size = simulation.getCellSize();
	real volume = 0;
	real weight = 0;

	std::array<real, 4> values = { real(1),real(1),real(1),real(1) };
	for (const auto& keyframe : costFunction.keyframes) {
		real w = keyframe.weight;
		if (w == 0 || keyframe.sdf.size() == 0) continue;
		for (int x = 0; x < resolution - 1; ++x) for (int y = 0; y < resolution - 1; ++y)
		{
			std::array<real, 4> sdf = {
				keyframe.sdf(x, y),
				keyframe.sdf(x+1, y),
				keyframe.sdf(x, y+1),
				keyframe.sdf(x+1, y+1)
			};
			volume += w * Integration2D<real>::integrateQuad(size, sdf, values);
		}
		weight += w;
	}
	volume /= weight;
	CI_LOG_I("Reference volume: " << volume);
	return volume;
}

std::pair<ar::real, ar::GridUtils2D::grid_t> ar::InverseProblem_Adjoint_Dynamic::gridComputeVolumePriorDerivative(
	const GridUtils2D::grid_t& currentInitialSDF, const GridCostFunctionDefinition& costFunction,
	const SoftBodyGrid2D& simulation)
{
	int resolution = simulation.getGridResolution();
	Vector2 size = simulation.getCellSize();

	real currentVolume = 0;
	GridUtils2D::grid_t gradient = GridUtils2D::grid_t::Zero(resolution, resolution);

	std::array<real, 4> values = { 1,1,1,1 };
	for (int x = 0; x < resolution - 1; ++x) for (int y = 0; y < resolution - 1; ++y)
	{
		std::array<real, 4> sdf = {
			currentInitialSDF(x, y),
			currentInitialSDF(x + 1, y),
			currentInitialSDF(x, y + 1),
			currentInitialSDF(x + 1, y + 1)
		};
		real v = Integration2D<real>::integrateQuad(size, sdf, values);
		currentVolume += v;
		std::array<real, 4> vDphi = Integration2D<real>::integrateQuadDsdf(size, sdf, values);
		gradient(x, y) += vDphi[0];
		gradient(x+1, y) += vDphi[1];
		gradient(x, y+1) += vDphi[2];
		gradient(x+1, y+1) += vDphi[3];
	}

	real cost = square(currentVolume - costFunction.referenceVolume) / 2;
	gradient *= (currentVolume - costFunction.referenceVolume);
	CI_LOG_I("current volume: " << currentVolume);
	return std::make_pair(cost, gradient);
}


void ar::InverseProblem_Adjoint_Dynamic::testPlot(int deformedTimestep, BackgroundWorker* worker)
{
	bool hasMesh = !input->meshResultsDisplacement_.empty();
	bool hasGrid = !input->gridResultsSdf_.empty();

	// Create simulation
	worker->setStatus("Test Plot - Init");
	SoftBodyMesh2D meshSimulation;
	if (hasMesh) {
		meshSimulation.setMesh(input->meshReferencePositions_, input->meshReferenceIndices_);
		meshSimulation.resetBoundaries();
		for (const auto& b : input->meshDirichletBoundaries_)
			meshSimulation.addDirichletBoundary(b.first, b.second);
		for (const auto& b : input->meshNeumannBoundaries_)
			meshSimulation.addNeumannBoundary(b.first, b.second);
		meshSimulation.reorderNodes();
		if (worker->isInterrupted()) return;
		meshSimulation.setGravity(input->settings_.gravity_);
		meshSimulation.setMass(input->settings_.mass_);
		meshSimulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
		meshSimulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
		meshSimulation.setRotationCorrection(input->settings_.rotationCorrection_);
		meshSimulation.setTimestep(input->settings_.timestep_);
		meshSimulation.setEnableCollision(input->settings_.enableCollision_);
		meshSimulation.setGroundPlane(input->settings_.groundPlaneHeight_, input->settings_.groundPlaneAngle_);
		meshSimulation.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		meshSimulation.setCollisionVelocityDamping(0);
		meshSimulation.setGroundStiffness(input->settings_.groundStiffness_);
		meshSimulation.setCollisionSoftmaxAlpha(input->settings_.softmaxAlpha_);
	}
	if (worker->isInterrupted()) return;

	SoftBodyGrid2D gridSimulation;
	if (hasGrid) {
		gridSimulation.setGridResolution(input->gridResolution_);
		gridSimulation.setSDF(input->gridReferenceSdf_);
		gridSimulation.setExplicitDiffusion(true);
		gridSimulation.setHardDirichletBoundaries(false);
		gridSimulation.setAdvectionMode(SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD);
		gridSimulation.resetBoundaries();
		for (const auto& b : input->gridDirichletBoundaries_)
			gridSimulation.addDirichletBoundary(b.first.first, b.first.second, b.second);
		for (const auto& b : input->gridNeumannBoundaries_)
			gridSimulation.addNeumannBoundary(b.first.first, b.first.second, b.second);
		gridSimulation.setGravity(input->settings_.gravity_);
		gridSimulation.setMass(input->settings_.mass_);
		gridSimulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
		gridSimulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
		gridSimulation.setRotationCorrection(input->settings_.rotationCorrection_);
		gridSimulation.setTimestep(input->settings_.timestep_);
		gridSimulation.setEnableCollision(input->settings_.enableCollision_);
		gridSimulation.setGroundPlane(input->settings_.groundPlaneHeight_, input->settings_.groundPlaneAngle_);
		gridSimulation.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		gridSimulation.setCollisionVelocityDamping(0);
		gridSimulation.setGroundStiffness(input->settings_.groundStiffness_);
		gridSimulation.setCollisionSoftmaxAlpha(input->settings_.softmaxAlpha_);
	}
	if (worker->isInterrupted()) return;

	//read weights -> create cost function definition
	MeshCostFunctionDefinition meshCostFunction = hasMesh ? meshParseWeights(timestepWeights, meshSimulation) : MeshCostFunctionDefinition();
	GridCostFunctionDefinition gridCostFunction = hasGrid ? gridParseWeights(timestepWeights, gridSimulation) : GridCostFunctionDefinition();

	int steps = 25;

	if (optimizeMass)
	{
		std::fstream file("../plots/Adjoint-Dynamic-Mass.csv", std::fstream::out | std::fstream::trunc);
		file << "Mass , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "Mass , Cost Mesh , Gradient Mesh , Cost Grid , Gradient Grid" << std::endl;
		real minValue = input->settings_.mass_ * 0.5;
		real maxValue = input->settings_.mass_ * 1.5;
		for (int i=0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Mass %d/%d", i+1, steps));
			real mass = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input;
			input.mass = mass;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << mass << " , " << meshOutput.finalCost << " , " << meshOutput.massGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.massGradient.value_or(0) << std::endl;
			cinder::app::console() << mass << " , " << meshOutput.finalCost << " , " << meshOutput.massGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.massGradient.value_or(0) << std::endl;
		}
		file.close();
	}

	if (optimizeMassDamping)
	{
		std::fstream file("../plots/Adjoint-Dynamic-MassDamping.csv", std::fstream::out | std::fstream::trunc);
		file << "MassDamping , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "MassDamping , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.dampingAlpha_ * 0.5;
		real maxValue = input->settings_.dampingAlpha_ * 1.5;
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Mass Damping %d/%d", i + 1, steps));
			real massDamping = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input;
			input.dampingMass = massDamping;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << massDamping << " , " << meshOutput.finalCost << " , " << meshOutput.dampingMassGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.dampingMassGradient.value_or(0) << std::endl;
			cinder::app::console() << massDamping << " , " << meshOutput.finalCost << " , " << meshOutput.dampingMassGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.dampingMassGradient.value_or(0) << std::endl;
		}
		file.close();
	}

	if (optimizeStiffnessDamping)
	{
		std::fstream file("../plots/Adjoint-Dynamic-StiffnessDamping.csv", std::fstream::out | std::fstream::trunc);
		file << "StiffnessDamping , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "StiffnessDamping , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.dampingBeta_ * 0.5;
		real maxValue = input->settings_.dampingBeta_ * 1.5;
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Stiffness Damping %d/%d", i + 1, steps));
			real stiffnessDamping = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input;
			input.dampingStiffness = stiffnessDamping;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << stiffnessDamping << " , " << meshOutput.finalCost << " , " << meshOutput.dampingStiffnessGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.dampingStiffnessGradient.value_or(0) << std::endl;
			cinder::app::console() << stiffnessDamping << " , " << meshOutput.finalCost << " , " << meshOutput.dampingStiffnessGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.dampingStiffnessGradient.value_or(0) << std::endl;
		}
		file.close();
	}

    if (optimizeYoungsModulus)
	{
		std::fstream file("../plots/Adjoint-Dynamic-YoungsModulus.csv", std::fstream::out | std::fstream::trunc);
		file << "Youngs Modulus , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "Youngs Modulus , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.youngsModulus_ * 0.5;
		real maxValue = input->settings_.youngsModulus_ * 1.5;
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Youngs Modulus %d/%d", i + 1, steps));
			real youngsModulus = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input;
			input.youngsModulus = youngsModulus;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << youngsModulus << " , " << meshOutput.finalCost << " , " << meshOutput.youngsModulusGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.youngsModulusGradient.value_or(0) << std::endl;
			cinder::app::console() << youngsModulus << " , " << meshOutput.finalCost << " , " << meshOutput.youngsModulusGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.youngsModulusGradient.value_or(0) << std::endl;
		}
		file.close();
	}

    if (optimizePoissonRatio)
	{
		std::fstream file("../plots/Adjoint-Dynamic-PoissonRatio.csv", std::fstream::out | std::fstream::trunc);
		file << "PoissonRatio , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "PoissonRatio , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.poissonsRatio_ * 0.5;
		real maxValue = std::min(input->settings_.poissonsRatio_ * 1.5, 0.4999);
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Poisson Ratio %d/%d", i + 1, steps));
			real poissonRatio = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input;
			input.poissonRatio = poissonRatio;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << poissonRatio << " , " << meshOutput.finalCost << " , " << meshOutput.poissonRatioGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.poissonRatioGradient.value_or(0) << std::endl;
			cinder::app::console() << poissonRatio << " , " << meshOutput.finalCost << " , " << meshOutput.poissonRatioGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.poissonRatioGradient.value_or(0) << std::endl;
		}
		file.close();
	}

    if (optimizeGroundPosition)
	{
		std::fstream file("../plots/Adjoint-Dynamic-GroundHeight.csv", std::fstream::out | std::fstream::trunc);
		file << "GroundHeight , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "GroundHeight , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.groundPlaneHeight_ - 0.1;
		real maxValue = input->settings_.groundPlaneHeight_ + 0.1;
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Ground Plane Height %d/%d", i + 1, steps));
			real height = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input2;
			input2.groundHeight = height;
            input2.groundAngle = input->settings_.groundPlaneAngle_;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input2, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input2, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << height << " , " << meshOutput.finalCost << " , " << meshOutput.groundHeightGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.groundHeightGradient.value_or(0) << std::endl;
			cinder::app::console() << height << " , " << meshOutput.finalCost << " , " << meshOutput.groundHeightGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.groundHeightGradient.value_or(0) << std::endl;
		}
		file.close();
	}

    if (optimizeStiffnessDamping)
	{
		std::fstream file("../plots/Adjoint-Dynamic-GroundAngle.csv", std::fstream::out | std::fstream::trunc);
		file << "GroundAngle , Cost Mesh , Gradient Mesh" << std::endl;
		cinder::app::console() << "GroundAngle , Cost Mesh , Gradient Mesh" << std::endl;
		real minValue = input->settings_.groundPlaneAngle_ - 0.05;
		real maxValue = input->settings_.groundPlaneAngle_ + 0.05;
		for (int i = 0; i<steps; ++i)
		{
			worker->setStatus(tfm::format("Test Plot - Ground Angle %d/%d", i + 1, steps));
			real angle = minValue + i * (maxValue - minValue) / (steps - 1);

			GradientInput input2;
            input2.groundHeight = input->settings_.groundPlaneHeight_;
			input2.groundAngle = angle;
			GradientOutput meshOutput = hasMesh ? gradientMesh(input2, meshCostFunction, meshSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;
			GradientOutput gridOutput = hasGrid ? gradientGrid(input2, gridCostFunction, gridSimulation, worker, false) : GradientOutput();
			if (worker->isInterrupted()) return;

			file << angle << " , " << meshOutput.finalCost << " , " << meshOutput.groundAngleGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.groundAngleGradient.value_or(0) << std::endl;
			cinder::app::console() << angle << " , " << meshOutput.finalCost << " , " << meshOutput.groundAngleGradient.value_or(0) << " , " << gridOutput.finalCost << " , " << gridOutput.groundAngleGradient.value_or(0) << std::endl;
		}
		file.close();
	}
}
