#include "InverseProblem_HumanSoftTissue.h"

#include <tinyformat.h>
#include <cinder/Log.h>
#include <LBFGS.h>

#include "GradientDescent.h"
#include "SoftBodyMesh2D.h"

namespace ar {

	ar::InverseProblem_HumanSoftTissue::InverseProblem_HumanSoftTissue()
		: initialSearchMin_(10)
		, initialSearchMax_(1000)
		, initialSearchSteps_(5)
		, optimizationSteps_(5)
	{
	}

	ar::InverseProblem_HumanSoftTissue::~InverseProblem_HumanSoftTissue()
	{
	}

	ar::InverseProblemOutput ar::InverseProblem_HumanSoftTissue::solveGrid(
		int deformedTimestep, ar::BackgroundWorker* worker, IntermediateResultCallback_t /*callback*/)
	{
		//TODO: use callback
		CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
		// Create simulation
		worker->setStatus("Human Soft Tissue Grid: create simulation");
		SoftBodyGrid2D simulation;
		simulation.setGridResolution(input->gridResolution_);
		simulation.setSDF(input->gridReferenceSdf_);
		simulation.setExplicitDiffusion(true);
		//simulation.setHardDirichletBoundaries(true);
		simulation.resetBoundaries();
		for (const auto& b : input->gridDirichletBoundaries_)
			simulation.addDirichletBoundary(b.first.first, b.first.second, b.second);
		for (const auto& b : input->gridNeumannBoundaries_)
			simulation.addNeumannBoundary(b.first.first, b.first.second, b.second);
		if (worker->isInterrupted()) return InverseProblemOutput();

		// Set parameters with everything that can't be reconstructed here
		simulation.setGravity(input->settings_.gravity_);
		simulation.setMass(input->settings_.mass_);
		simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
		simulation.setRotationCorrection(input->settings_.rotationCorrection_);
		if (worker->isInterrupted()) return InverseProblemOutput();

		// Initial brute force search for the Youngs Modulus
		worker->setStatus("Human Soft Tissue Grid: initial brute force");
		real bruteForceValue = initialSearchSteps_ == 1 ? 0.5*(initialSearchMin_ + initialSearchMax_) : initialSearchMin_;
		real bruteForceStep = initialSearchSteps_ == 1 ? 0 : (initialSearchMax_ - initialSearchMin_) / (initialSearchSteps_ - 1);
		real bestValue = bruteForceValue;
		real bestEnergy = std::numeric_limits<real>::max();
		for (int i = 0; i < initialSearchSteps_; ++i) {
			worker->setStatus(tfm::format("Human Soft Tissue Grid: initial brute force %d/%d", (i + 1), initialSearchSteps_));
			simulation.setMaterialParameters(bruteForceValue + i * bruteForceStep, input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return InverseProblemOutput();
			real loss = lossFunctionGrid(simulation, deformedTimestep, true);
			if (worker->isInterrupted()) return InverseProblemOutput();
			if (loss < bestEnergy) {
				bestEnergy = loss;
				bestValue = simulation.getYoungsModulus();
			}
			CI_LOG_I("Brute force: Young's Modulus " << simulation.getYoungsModulus() << " -> loss " << loss);
		}
		CI_LOG_I("Brute force: best Youngs' Modulus is " << bestValue << " with a loss of " << bestEnergy);

		// Optimization
#if 0
		GradientDescent<Vector1> gd(Vector1(bestValue), [&simulation, deformedTimestep, this, worker](const Vector1& v) {
			simulation.setMaterialParameters(v[0], input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return Vector1(0);
			real loss = lossFunctionGrid(simulation, deformedTimestep, false);
			return Vector1(loss);
		}, 1e-10);
		int oi;
		for (oi = 0; oi < optimizationSteps_; ++oi) {
			worker->setStatus(tfm::format("Human Soft Tissue Grid: optimization %d/%d", (oi + 1), optimizationSteps_));
			if (gd.step()) break;
			if (worker->isInterrupted()) return InverseProblemOutput();
		}
		real finalValue = gd.getCurrentSolution()[0];
#else
		LBFGSpp::LBFGSParam<real> params;
		params.epsilon = 1e-10;
		params.max_iterations = optimizationSteps_;
		LBFGSpp::LBFGSSolver<real> lbfgs(params);
		LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&simulation, deformedTimestep, this, worker](const VectorX& x, VectorX& gradient) -> real {
			simulation.setMaterialParameters(x[0], input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			real loss = lossFunctionGrid(simulation, deformedTimestep, true);
			real grad = lossFunctionGrid(simulation, deformedTimestep, false);
			gradient[0] = grad;
			CI_LOG_I("Evaluate cost function at x=" << x[0] << ", loss=" << loss << ", grad=" << grad);
			return loss;
		});
		LBFGSpp::LBFGSSolver<real>::CallbackFunction_t callback([worker](const VectorX& x, const VectorX& g, const real& v, int k) -> bool {
			return !worker->isInterrupted();
		});
		real finalCost = 0;
		VectorX finalValueV = VectorX::Constant(1, bestValue);
		int oi = lbfgs.minimize(fun, finalValueV, finalCost, callback);
		if (worker->isInterrupted()) return InverseProblemOutput();
		real finalValue = finalValueV[0];
#endif
		CI_LOG_I("Optimization: final Youngs' Modulus is " << finalValue << " after " << oi << " optimization steps, reference value is " << input->settings_.youngsModulus_);

		InverseProblemOutput output;
		output.youngsModulus_ = finalValue;
		output.resultGridSdf_ = simulation.getSdfSolution();
		output.finalCost_ = finalCost;
		return output;
	}

	ar::InverseProblemOutput ar::InverseProblem_HumanSoftTissue::solveMesh(
		int deformedTimestep, ar::BackgroundWorker* worker, IntermediateResultCallback_t /*callback*/)
	{
		//TODO: use callback
		CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
		// Create simulation
		worker->setStatus("Human Soft Tissue Mesh: create simulation");
		SoftBodyMesh2D simulation;
		simulation.setMesh(input->meshReferencePositions_, input->meshReferenceIndices_);
		simulation.resetBoundaries();
		for (const auto& b : input->meshDirichletBoundaries_)
			simulation.addDirichletBoundary(b.first, b.second);
		for (const auto& b : input->meshNeumannBoundaries_)
			simulation.addNeumannBoundary(b.first, b.second);
		if (worker->isInterrupted()) return InverseProblemOutput();

		// Set parameters with everything that can't be reconstructed here
		simulation.setGravity(input->settings_.gravity_);
		simulation.setMass(input->settings_.mass_);
		simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
		simulation.setRotationCorrection(input->settings_.rotationCorrection_);
		if (worker->isInterrupted()) return InverseProblemOutput();

		// Initial brute force search for the Youngs Modulus
		worker->setStatus("Human Soft Tissue Mesh: initial brute force");
		real bruteForceValue = initialSearchSteps_ == 1 ? 0.5*(initialSearchMin_ + initialSearchMax_) : initialSearchMin_;
		real bruteForceStep = initialSearchSteps_ == 1 ? 0 : (initialSearchMax_ - initialSearchMin_) / (initialSearchSteps_ - 1);
		real bestValue = bruteForceValue;
		real bestEnergy = std::numeric_limits<real>::max();
		for (int i = 0; i < initialSearchSteps_; ++i) {
			worker->setStatus(tfm::format("Human Soft Tissue Mesh: initial brute force %d/%d", (i+1), initialSearchSteps_));
			simulation.setMaterialParameters(bruteForceValue + i * bruteForceStep, input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return InverseProblemOutput();
			real loss = lossFunctionMesh(simulation, deformedTimestep, true);
			if (worker->isInterrupted()) return InverseProblemOutput();
			if (loss < bestEnergy) {
				bestEnergy = loss;
				bestValue = simulation.getYoungsModulus();
			}
		}
		CI_LOG_I("Brute force: best Youngs' Modulus is " << bestValue << " with a loss of " << bestEnergy);

		// Optimization
#if 0
		GradientDescent<Vector1> gd(Vector1(bestValue), [&simulation, deformedTimestep, this, worker](const Vector1& v) {
			simulation.setMaterialParameters(v[0], input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return Vector1(0);
			real loss = lossFunctionMesh(simulation, deformedTimestep, false);
			return Vector1(loss);
		}, 1e-10);
		int oi;
		for (oi = 0; oi < optimizationSteps_; ++oi) {
			worker->setStatus(tfm::format("Human Soft Tissue Mesh: optimization %d/%d", (oi + 1), optimizationSteps_));
			if (gd.step()) break;
			if (worker->isInterrupted()) return InverseProblemOutput();
		}
		real finalValue = gd.getCurrentSolution()[0];
#else
		LBFGSpp::LBFGSParam<real> params;
		params.epsilon = 1e-10;
		params.max_iterations = optimizationSteps_;
		LBFGSpp::LBFGSSolver<real> lbfgs(params);
		LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&simulation, deformedTimestep, this, worker](const VectorX& x, VectorX& gradient) -> real {
			simulation.setMaterialParameters(x[0], input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			real loss = lossFunctionMesh(simulation, deformedTimestep, true);
			real grad = lossFunctionMesh(simulation, deformedTimestep, false);
			gradient[0] = grad;
			return loss;
		});
		LBFGSpp::LBFGSSolver<real>::CallbackFunction_t callback([worker](const VectorX& x, const VectorX& g, const real& v, int k) -> bool {
			return !worker->isInterrupted();
		});
		real finalCost = 0;
		VectorX finalValueV = VectorX::Constant(1, bestValue);
		int oi = lbfgs.minimize(fun, finalValueV, finalCost, callback);
		if (worker->isInterrupted()) return InverseProblemOutput();
		real finalValue = finalValueV[0];
#endif
		CI_LOG_I("Optimization: final Youngs' Modulus is " << finalValue << " after " << oi << " optimization steps, reference value is " << input->settings_.youngsModulus_);

		InverseProblemOutput output;
		output.youngsModulus_ = finalValue;
		output.resultMeshDisp_ = simulation.getCurrentDisplacements();
		output.finalCost_ = finalCost;
		return output;
	}

	void InverseProblem_HumanSoftTissue::setupParams(cinder::params::InterfaceGlRef params, const std::string& group)
	{
		params->addParam("InverseProblem_HumanSoftTissue_InitialSearchMin", &initialSearchMin_)
			.group(group).label("Initial Search Min").min(0).step(0.01);
		params->addParam("InverseProblem_HumanSoftTissue_InitialSearchMax", &initialSearchMax_)
			.group(group).label("Initial Search Max").min(0).step(0.01);
		params->addParam("InverseProblem_HumanSoftTissue_InitialSearchSteps", &initialSearchSteps_)
			.group(group).label("Initial Search Steps").min(1);
		params->addParam("InverseProblem_HumanSoftTissue_OptimizationSteps", &optimizationSteps_)
			.group(group).label("Optimization Steps").min(0);
	}

	void InverseProblem_HumanSoftTissue::setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const
	{
		std::string option = visible ? "visible=true" : "visible=false";
		params->setOptions("InverseProblem_HumanSoftTissue_InitialSearchMin", option);
		params->setOptions("InverseProblem_HumanSoftTissue_InitialSearchMax", option);
		params->setOptions("InverseProblem_HumanSoftTissue_InitialSearchSteps", option);
		params->setOptions("InverseProblem_HumanSoftTissue_OptimizationSteps", option);
	}

	real InverseProblem_HumanSoftTissue::lossFunctionMesh(const SoftBodyMesh2D & simulation, int deformedTimestep, bool absolute)
	{
		real loss = 0;
		for (size_t i = 0; i < input->meshReferencePositions_.size(); ++i) {
			Vector2 posReference = input->meshReferencePositions_[i];
			Vector2 posExpected = posReference + input->meshResultsDisplacement_[deformedTimestep][i];
			Vector2 posActual = simulation.getCurrentPositions()[i];
			real amplitudeExpected = (posReference - posExpected).norm();
			real amplitudeActual = (posReference - posActual).norm();
			real l = amplitudeExpected - amplitudeActual;
			loss += absolute ? l*l : l;
		}
		return loss;
	}

	real InverseProblem_HumanSoftTissue::lossFunctionGrid(const SoftBodyGrid2D & simulation, int deformedTimestep, bool absolute)
	{
		real loss = 0;
		for (int i = 0; i < simulation.getDoF(); ++i)
		{
			real amplitudeExpected = input->gridResultsUxy_[deformedTimestep].segment<2>(2 * i).norm();
			real amplitudeActual = simulation.getUSolution().segment<2>(2 * i).norm();
			real l = amplitudeExpected - amplitudeActual;
			loss += absolute ? l * l : l;
		}
		return loss;
	}

}
