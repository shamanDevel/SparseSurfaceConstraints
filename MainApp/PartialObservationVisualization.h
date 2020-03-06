#pragma once

#include <InputDataLoader.h>
#include <SoftBodyGrid3D.h>
#include <cinder/gl/gl.h>
#include <mutex>

namespace ar3d {

	class PartialObservationVisualization
	{
	public:
		void setObservation(
			const DataCamera& camera, 
			const InputDataLoader::depthMatrix_t& depth,
			const SoftBodyGrid3D::Input& input,
			const SoftBodyGrid3D::State& state);

		void draw();

	private:
		typedef std::pair<glm::vec3, glm::vec3> arrow;
		std::vector<arrow> arrows_;
		std::mutex mutex_;
	};

}