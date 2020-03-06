#include "InverseProblem_ProbabilisticElastography.h"

#include <tinyformat.h>
#include <cinder/Log.h>
#include <ctime>

#include "SoftBodyMesh2D.h"
#include "SoftBodyGrid2D.h"

namespace ar {

	InverseProblem_ProbabilisticElastography::InverseProblem_ProbabilisticElastography()
		: initialSearchMin_(10)
		, initialSearchMax_(1000)
		, initialSearchSteps_(5)
		, temperatureElasticity_(1e5)
		, temperatureSimilarity_(0.1)
		, burnIn_(100)
		, thinning_(5)
		, numSamples_(10)
		, proposalStdDev_(20)
		, rnd_(time(nullptr))
	{
	}

	InverseProblem_ProbabilisticElastography::~InverseProblem_ProbabilisticElastography()
	{
	}

	InverseProblemOutput InverseProblem_ProbabilisticElastography::solveGrid(
		int deformedTimestep, BackgroundWorker * worker, IntermediateResultCallback_t /*callback*/)
	{
		//TODO: use callback
		CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
		youngsModulusSamplesMesh_.clear();
		// Create simulation
		worker->setStatus("Probabilisitc Elastography Grid: create simulation");
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
		worker->setStatus("Probabilisitc Elastography Grid: initial brute force");
		real bruteForceValue = initialSearchSteps_ == 1 ? 0.5*(initialSearchMin_ + initialSearchMax_) : initialSearchMin_;
		real bruteForceStep = initialSearchSteps_ == 1 ? 0 : (initialSearchMax_ - initialSearchMin_) / (initialSearchSteps_ - 1);
		real bestValue = bruteForceValue;
		real bestEnergy = std::numeric_limits<real>::max();
		for (int i = 0; i < initialSearchSteps_; ++i) {
			worker->setStatus(tfm::format("Probabilisitc Elastography Grid: initial brute force %d/%d", (i + 1), initialSearchSteps_));
			simulation.setMaterialParameters(bruteForceValue + i * bruteForceStep, input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return InverseProblemOutput();
			real loss = lossFunctionGrid(simulation, deformedTimestep);
			if (worker->isInterrupted()) return InverseProblemOutput();
			if (loss < bestEnergy) {
				bestEnergy = loss;
				bestValue = simulation.getYoungsModulus();
			}
		}
		CI_LOG_I("Brute force: best Youngs' Modulus is " << bestValue << " with a loss of " << bestEnergy);

		// Metropolis Hastings Sampling
		real currentYoungsModulus = bestValue;
		real currentLogProb = computeLogProbability(simulation, currentYoungsModulus, deformedTimestep, worker);
		CI_LOG_I("Metropolis Hastings: initial value " << currentYoungsModulus << ", log-prob " << currentLogProb);
		if (worker->isInterrupted()) return InverseProblemOutput();
		//burn-in
		for (int i = 0; i < burnIn_; ++i) {
			worker->setStatus(tfm::format("Probabilisitc Elastography Grid: Metropolis Hastings Burn-in %d/%d", (i + 1), burnIn_));
			real nextYoungsModulus, nextLogProb;
			generateNextSample(simulation, deformedTimestep, worker, currentYoungsModulus, currentLogProb, nextYoungsModulus, nextLogProb);
			if (worker->isInterrupted()) return InverseProblemOutput();
			currentYoungsModulus = nextYoungsModulus;
			currentLogProb = nextLogProb;
			CI_LOG_I("Metropolis Hastings: burn-in " << (i + 1) << "/" << burnIn_ << ": " << currentYoungsModulus << ", log-prob " << currentLogProb);
		}
		//sampling
		for (int i = 0; i < numSamples_; ++i) {
			for (int j = 0; j < thinning_; ++j) {
				worker->setStatus(tfm::format("Probabilisitc Elastography Grid: Metropolis Hastings Sampling %d/%d (Thinning %d/%d)", (i + 1), numSamples_, (j + 1), thinning_));
				real nextYoungsModulus, nextLogProb;
				generateNextSample(simulation, deformedTimestep, worker, currentYoungsModulus, currentLogProb, nextYoungsModulus, nextLogProb);
				if (worker->isInterrupted()) return InverseProblemOutput();
				currentYoungsModulus = nextYoungsModulus;
				currentLogProb = nextLogProb;

			}
			CI_LOG_I("Metropolis Hastings: new sample " << (i + 1) << "/" << numSamples_ << ": " << currentYoungsModulus << ", log-prob " << currentLogProb);
			youngsModulusSamplesMesh_.push_back(currentYoungsModulus);
		}

		//compute mean and variance
		real mean = 0;
		for (size_t i = 0; i < youngsModulusSamplesMesh_.size(); ++i)
			mean += youngsModulusSamplesMesh_[i];
		mean /= youngsModulusSamplesMesh_.size();
		real variance = 0;
		for (size_t i = 0; i < youngsModulusSamplesMesh_.size(); ++i)
			variance += (youngsModulusSamplesMesh_[i] - mean) * (youngsModulusSamplesMesh_[i] - mean);
		variance /= youngsModulusSamplesMesh_.size() + 1;
		CI_LOG_I("Metropolis Hastings: mean Young's Modulus " << mean << " with variance " << variance);

		//return result
		worker->setStatus("Probabilistic Elastography Grid: Final solve with mean value");
		InverseProblemOutput output;
		output.youngsModulus_ = mean;
		output.youngsModulusStdDev_ = sqrt(variance);
		simulation.setMaterialParameters(mean, input->settings_.poissonsRatio_);
		simulation.resetSolution();
		worker->pushDisableStatusLogging();
		simulation.solveStaticSolution(worker);
		worker->popDisableStatusLogging();
		output.resultGridSdf_ = simulation.getSdfSolution();
		output.finalCost_ = currentLogProb;
		return output;
	}

	InverseProblemOutput InverseProblem_ProbabilisticElastography::solveMesh(
		int deformedTimestep, BackgroundWorker * worker, IntermediateResultCallback_t /*callback*/)
	{
		//TODO: use callback
		CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
		youngsModulusSamplesMesh_.clear();
		// Create simulation
		worker->setStatus("Probabilisitc Elastography Mesh: create simulation");
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
		worker->setStatus("Probabilisitc Elastography Mesh: initial brute force");
		real bruteForceValue = initialSearchSteps_ == 1 ? 0.5*(initialSearchMin_ + initialSearchMax_) : initialSearchMin_;
		real bruteForceStep = initialSearchSteps_ == 1 ? 0 : (initialSearchMax_ - initialSearchMin_) / (initialSearchSteps_ - 1);
		real bestValue = bruteForceValue;
		real bestEnergy = std::numeric_limits<real>::max();
		for (int i = 0; i < initialSearchSteps_; ++i) {
			worker->setStatus(tfm::format("Probabilisitc Elastography Mesh: initial brute force %d/%d", (i + 1), initialSearchSteps_));
			simulation.setMaterialParameters(bruteForceValue + i * bruteForceStep, input->settings_.poissonsRatio_);
			simulation.resetSolution();
			worker->pushDisableStatusLogging();
			simulation.solveStaticSolution(worker);
			worker->popDisableStatusLogging();
			if (worker->isInterrupted()) return InverseProblemOutput();
			real loss = lossFunctionMesh(simulation, deformedTimestep);
			if (worker->isInterrupted()) return InverseProblemOutput();
			if (loss < bestEnergy) {
				bestEnergy = loss;
				bestValue = simulation.getYoungsModulus();
			}
		}
		CI_LOG_I("Brute force: best Youngs' Modulus is " << bestValue << " with a loss of " << bestEnergy);

		// Metropolis Hastings Sampling
		real currentYoungsModulus = bestValue;
		real currentLogProb = computeLogProbability(simulation, currentYoungsModulus, deformedTimestep, worker);
		CI_LOG_I("Metropolis Hastings: initial value " << currentYoungsModulus << ", log-prob " << currentLogProb);
		if (worker->isInterrupted()) return InverseProblemOutput();
		//burn-in
		for (int i = 0; i < burnIn_; ++i) {
			worker->setStatus(tfm::format("Probabilisitc Elastography Mesh: Metropolis Hastings Burn-in %d/%d", (i + 1), burnIn_));
			real nextYoungsModulus, nextLogProb;
			generateNextSample(simulation, deformedTimestep, worker, currentYoungsModulus, currentLogProb, nextYoungsModulus, nextLogProb);
			if (worker->isInterrupted()) return InverseProblemOutput();
			currentYoungsModulus = nextYoungsModulus;
			currentLogProb = nextLogProb;
			CI_LOG_I("Metropolis Hastings: burn-in " << (i+1) << "/" << burnIn_ << ": " << currentYoungsModulus << ", log-prob " << currentLogProb);
		}
		//sampling
		for (int i = 0; i < numSamples_; ++i) {
			for (int j = 0; j < thinning_; ++j) {
				worker->setStatus(tfm::format("Probabilisitc Elastography Mesh: Metropolis Hastings Sampling %d/%d (Thinning %d/%d)", (i + 1), numSamples_, (j+1), thinning_));
				real nextYoungsModulus, nextLogProb;
				generateNextSample(simulation, deformedTimestep, worker, currentYoungsModulus, currentLogProb, nextYoungsModulus, nextLogProb);
				if (worker->isInterrupted()) return InverseProblemOutput();
				currentYoungsModulus = nextYoungsModulus;
				currentLogProb = nextLogProb;
				
			}
			CI_LOG_I("Metropolis Hastings: new sample " << (i + 1) << "/" << numSamples_ << ": " << currentYoungsModulus << ", log-prob " << currentLogProb);
			youngsModulusSamplesMesh_.push_back(currentYoungsModulus);
		}

		//compute mean and variance
		real mean = 0;
		for (size_t i = 0; i < youngsModulusSamplesMesh_.size(); ++i)
			mean += youngsModulusSamplesMesh_[i];
		mean /= youngsModulusSamplesMesh_.size();
		real variance = 0;
		for (size_t i = 0; i < youngsModulusSamplesMesh_.size(); ++i)
			variance += (youngsModulusSamplesMesh_[i] - mean) * (youngsModulusSamplesMesh_[i] - mean);
		variance /= youngsModulusSamplesMesh_.size() + 1;
		CI_LOG_I("Metropolis Hastings: mean Young's Modulus " << mean << " with variance " << variance);

		//return result
		worker->setStatus("Probabilistic Elastography Mesh: Final solve with mean value");
		InverseProblemOutput output;
		output.youngsModulus_ = mean;
		output.youngsModulusStdDev_ = sqrt(variance);
		simulation.setMaterialParameters(mean, input->settings_.poissonsRatio_);
		simulation.resetSolution();
		worker->pushDisableStatusLogging();
		simulation.solveStaticSolution(worker);
		worker->popDisableStatusLogging();
		output.resultMeshDisp_ = simulation.getCurrentDisplacements();
		output.finalCost_ = currentLogProb;
		return output;
	}

	void InverseProblem_ProbabilisticElastography::setupParams(cinder::params::InterfaceGlRef params, const std::string & group)
	{
		params->addParam("InverseProblem_ProbabilisticElastography_InitialSearchMin", &initialSearchMin_)
			.group(group).label("Initial Search Min").min(0).step(0.01);
		params->addParam("InverseProblem_ProbabilisticElastography_InitialSearchMax", &initialSearchMax_)
			.group(group).label("Initial Search Max").min(0).step(0.01);
		params->addParam("InverseProblem_ProbabilisticElastography_InitialSearchSteps", &initialSearchSteps_)
			.group(group).label("Initial Search Steps").min(1);
		params->addParam("InverseProblem_ProbabilisticElastography_TemperatureElasticity", &temperatureElasticity_)
			.group(group).label("Temperature Elasticity").min(0).step(1);
		params->addParam("InverseProblem_ProbabilisticElastography_TemperatureSimilarity", &temperatureSimilarity_)
			.group(group).label("Temperature Similarity").min(0).step(0.01);
		params->addParam("InverseProblem_ProbabilisticElastography_BurnIn", &burnIn_)
			.group(group).label("MCMC Burn In").min(0);
		params->addParam("InverseProblem_ProbabilisticElastography_Thinning", &thinning_)
			.group(group).label("MCMC Thinning").min(1);
		params->addParam("InverseProblem_ProbabilisticElastography_NumSamples", &numSamples_)
			.group(group).label("MCMC Num Samples").min(1);
		params->addParam("InverseProblem_ProbabilisticElastography_ProposalStdDev", &proposalStdDev_)
			.group(group).label("MCMC Proposal Std.Dev").min(0.001).step(0.01);
	}

	void InverseProblem_ProbabilisticElastography::setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const
	{
		std::string option = visible ? "visible=true" : "visible=false";
		params->setOptions("InverseProblem_ProbabilisticElastography_InitialSearchMin", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_InitialSearchMax", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_InitialSearchSteps", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_TemperatureElasticity", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_TemperatureSimilarity", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_BurnIn", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_Thinning", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_NumSamples", option);
		params->setOptions("InverseProblem_ProbabilisticElastography_ProposalStdDev", option);
	}

	real InverseProblem_ProbabilisticElastography::lossFunctionMesh(const SoftBodyMesh2D & simulation, int deformedTimestep)
	{
		real loss = 0;
		for (size_t i = 0; i < input->meshReferencePositions_.size(); ++i) {
			Vector2 posReference = input->meshReferencePositions_[i];
			Vector2 posExpected = posReference + input->meshResultsDisplacement_[deformedTimestep][i];
			Vector2 posActual = simulation.getCurrentPositions()[i];
			real amplitudeExpected = (posReference - posExpected).norm();
			real amplitudeActual = (posReference - posActual).norm();
			real l = amplitudeExpected - amplitudeActual;
			loss += l*l;
		}
		return loss;
	}

	real InverseProblem_ProbabilisticElastography::lossFunctionGrid(const SoftBodyGrid2D & simulation, int deformedTimestep)
	{
		real loss = 0;
		for (int i = 0; i < simulation.getDoF(); ++i)
		{
			real amplitudeExpected = input->gridResultsUxy_[deformedTimestep].segment<2>(2 * i).norm();
			real amplitudeActual = simulation.getUSolution().segment<2>(2 * i).norm();
			loss += square(amplitudeExpected - amplitudeActual);
		}
		return loss;
	}
	real InverseProblem_ProbabilisticElastography::computeLogProbability(
		SoftBodyGrid2D & simulation, real youngsModulus, int deformedTimestep, BackgroundWorker* worker)
	{
		//solve it
		simulation.setMaterialParameters(youngsModulus, input->settings_.poissonsRatio_);
		simulation.resetSolution();
		worker->pushDisableStatusLogging();
		simulation.solveStaticSolution(worker);
		worker->popDisableStatusLogging();
		if (worker->isInterrupted()) return 0;

		real prob = 0;

		//prior on elasticity
		real Eelastic = simulation.computeElasticEnergy(simulation.getUGridX(), simulation.getUGridY());
		prob -= Eelastic / temperatureElasticity_;

		//similarity
		real sdfClamp = 1;
		real Essd = 0;
		for (int x=0; x<input->gridResolution_; ++x) {
			for (int y = 0; y < input->gridResolution_; ++y) {
				real sdfExpected = input->gridResultsSdf_[deformedTimestep](x, y);
				real sdfActual = simulation.getSdfSolution()(x, y);
				sdfExpected = std::max(-sdfClamp, std::min(sdfClamp, sdfExpected));
				sdfActual = std::max(-sdfClamp, std::min(sdfClamp, sdfActual));
				Essd += (sdfExpected - sdfActual) * (sdfExpected - sdfActual);
			}
		}
		prob -= Essd / temperatureSimilarity_;

		return prob;
	}
	real InverseProblem_ProbabilisticElastography::computeLogProbability(
		SoftBodyMesh2D & simulation, real youngsModulus, int deformedTimestep, BackgroundWorker* worker)
	{
		//solve it
		simulation.setMaterialParameters(youngsModulus, input->settings_.poissonsRatio_);
		simulation.resetSolution();
		worker->pushDisableStatusLogging();
		simulation.solveStaticSolution(worker);
		worker->popDisableStatusLogging();
		if (worker->isInterrupted()) return 0;

		real prob = 0;

		//prior on elasticity
		real Eelastic = simulation.computeElasticEnergy(simulation.getCurrentDisplacements());
		prob -= Eelastic / temperatureElasticity_;

		//similarity
		real Essd = 0;
		for (size_t i = 0; i < input->meshReferencePositions_.size(); ++i) {
			Vector2 posReference = input->meshReferencePositions_[i];
			Vector2 posExpected = posReference + input->meshResultsDisplacement_[deformedTimestep][i];
			Vector2 posActual = simulation.getCurrentPositions()[i];
			Essd += (posExpected - posActual).norm();
		}
		prob -= Essd / temperatureSimilarity_;

		return prob;
	}

	void InverseProblem_ProbabilisticElastography::generateNextSample(
		SoftBodyMesh2D & simulation, int deformedTimestep, BackgroundWorker * worker, 
		real currentYoungsModulus, real currentLogProb, real & nextYoungsModulus, real & nextLogProb)
	{
		//propose new young's modulus
		std::normal_distribution<real> distr1(currentYoungsModulus, proposalStdDev_);
		real proposedValue = distr1(rnd_);

		//compute probability
		real newProb = computeLogProbability(simulation, proposedValue, deformedTimestep, worker);
		real propOfAccepting = std::min(real(1), std::exp(newProb - currentLogProb));

		//check if it is accepted
		std::uniform_real_distribution<real> distr2(0, 1);
		bool accepted = distr2(rnd_) <= propOfAccepting;

		CI_LOG_I("proposed value " << proposedValue << ", probability " << newProb << " -> accepted: " << (accepted ? "yes" : "no"));
		
		if (accepted) {
			nextYoungsModulus = proposedValue;
			nextLogProb = newProb;
		}
		else {
			nextYoungsModulus = currentYoungsModulus;
			nextLogProb = currentLogProb;
		}
	}

	void InverseProblem_ProbabilisticElastography::generateNextSample(
		SoftBodyGrid2D & simulation, int deformedTimestep, BackgroundWorker * worker,
		real currentYoungsModulus, real currentLogProb, real & nextYoungsModulus, real & nextLogProb)
	{
		//propose new young's modulus
		std::normal_distribution<real> distr1(currentYoungsModulus, proposalStdDev_);
		real proposedValue = distr1(rnd_);

		//compute probability
		real newProb = computeLogProbability(simulation, proposedValue, deformedTimestep, worker);
		real propOfAccepting = std::min(real(1), std::exp(newProb - currentLogProb));

		//check if it is accepted
		std::uniform_real_distribution<real> distr2(0, 1);
		bool accepted = distr2(rnd_) <= propOfAccepting;

		CI_LOG_I("proposed value " << proposedValue << ", probability " << newProb << " : " << propOfAccepting << " -> accepted: " << (accepted ? "yes" : "no"));

		if (accepted) {
			nextYoungsModulus = proposedValue;
			nextLogProb = newProb;
		}
		else {
			nextYoungsModulus = currentYoungsModulus;
			nextLogProb = currentLogProb;
		}
	}
}