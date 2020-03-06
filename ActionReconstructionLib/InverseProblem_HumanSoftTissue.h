#pragma once

#include "Commons.h"
#include "IInverseProblem.h"
#include "SoftBodyMesh2D.h"
#include "SoftBodyGrid2D.h"

namespace ar {

	class InverseProblem_HumanSoftTissue : public IInverseProblem
	{
	private:
		real initialSearchMin_;
		real initialSearchMax_;
		int initialSearchSteps_;
		int optimizationSteps_;

	public:
		InverseProblem_HumanSoftTissue();
		~InverseProblem_HumanSoftTissue();

		// Inherited via IInverseProblem
		virtual InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
		virtual InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;

		// Inherited via IInverseProblem
		virtual void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) override;
		virtual void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const override;

	private:
		real lossFunctionMesh(const SoftBodyMesh2D& simulation, int deformedTimestep, bool absolute);
		real lossFunctionGrid(const SoftBodyGrid2D& simulation, int deformedTimestep, bool absolute);
	};

}
