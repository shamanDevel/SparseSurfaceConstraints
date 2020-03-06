#pragma once

#include <memory>

#include "Commons3D.h"
#include "SoftBodyGrid3D.h"

namespace ar3d
{
	/**
	 * \brief Stores the results of the forward simulation as input to the adjoint problem
	 */
	struct SimulationResults3D
	{
		/**
		 * \brief ground truth settings
		 */
		SoftBodySimulation3D::Settings settings_;
		/**
		 * \brief Reference input
		 */
		SoftBodyGrid3D::Input input_;
		/**
		 * The computed states
		 */
		std::vector<SoftBodyGrid3D::State> states_;
	};
    typedef std::shared_ptr<SimulationResults3D> SimulationResults3DPtr;
}