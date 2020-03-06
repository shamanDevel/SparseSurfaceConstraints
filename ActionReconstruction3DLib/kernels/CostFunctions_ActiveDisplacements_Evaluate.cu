#include "../CostFunctions.h"

#include <cuMat/Core>
#include <random>

namespace ar3d
{

	void CostFunctionActiveDisplacements::evaluate(int timestep, const Input& input, Output& output) const
	{
		assert(timestepWeights_[timestep] > 0);
		real weight = timestepWeights_[timestep];

		static std::default_random_engine rnd;
		std::normal_distribution<real> distr;
		size_t n = input.displacements_.size();

		if (displacementWeight_ > 0)
		{
			std::vector<real3> noise;
			if (noise_ > 0) {
				noise.resize(n);
				for (size_t i = 0; i < n; ++i)
					noise[i] = make_real3(distr(rnd) * noise_, distr(rnd) * noise_, distr(rnd) * noise_);
			}
			else {
				noise.resize(n, make_real3(0));
			}
			Vector3X noiseGpu = Vector3X(n);
			noiseGpu.copyFromHost(noise.data());

			output.cost_ += 0.5 * weight * displacementWeight_
				* static_cast<real>((input.displacements_ + noiseGpu - results_->states_[timestep].displacements_).squaredNorm());
			output.adjDisplacements_ += make_real3(weight * displacementWeight_)
				* (input.displacements_ + noiseGpu - results_->states_[timestep].displacements_);
		}
		if (velocityWeight_ > 0)
		{
			output.cost_ += 0.5 * weight * velocityWeight_
				* static_cast<real>((input.velocities_ - results_->states_[timestep].velocities_).squaredNorm());
			output.adjDisplacements_ += make_real3(weight * velocityWeight_)
				* (input.velocities_ - results_->states_[timestep].velocities_);
		}
	}

}