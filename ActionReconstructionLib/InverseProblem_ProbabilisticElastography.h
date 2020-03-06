#pragma once

#include <vector>
#include <random>

#include "IInverseProblem.h"

namespace ar {

	class InverseProblem_ProbabilisticElastography :
		public IInverseProblem
	{
	private:
		real initialSearchMin_;
		real initialSearchMax_;
		int initialSearchSteps_;

		real temperatureElasticity_;
		real temperatureSimilarity_;
		int burnIn_;
		int thinning_;
		int numSamples_;
		real proposalStdDev_;

		std::vector<real> youngsModulusSamplesGrid_;
		std::vector<real> youngsModulusSamplesMesh_;
		std::default_random_engine rnd_;

	public:
		InverseProblem_ProbabilisticElastography();
		~InverseProblem_ProbabilisticElastography();

		// Inherited via IInverseProblem
		virtual InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
		virtual InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;

		// Inherited via IInverseProblem
		virtual void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) override;
		virtual void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const override;

	private:
		real lossFunctionMesh(const SoftBodyMesh2D& simulation, int deformedTimestep);
		real lossFunctionGrid(const SoftBodyGrid2D& simulation, int deformedTimestep);

		real computeLogProbability(SoftBodyGrid2D& simulation, real youngsModulus, int deformedTimestep, BackgroundWorker* worker);
		real computeLogProbability(SoftBodyMesh2D& simulation, real youngsModulus, int deformedTimestep, BackgroundWorker* worker);

		void generateNextSample(SoftBodyMesh2D& simulation, int deformedTimestep, BackgroundWorker* worker,
			real currentYoungsModulus, real currentLogProb, real& nextYoungsModulus, real& nextLogProb);
		void generateNextSample(SoftBodyGrid2D& simulation, int deformedTimestep, BackgroundWorker* worker,
			real currentYoungsModulus, real currentLogProb, real& nextYoungsModulus, real& nextLogProb);
	};

}
