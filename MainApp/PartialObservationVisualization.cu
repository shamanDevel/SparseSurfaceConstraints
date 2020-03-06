#include "PartialObservationVisualization.h"

#include <algorithm>
#include <random>

#include <Utils.h>
#include <TrilinearInterpolation.h>

void ar3d::PartialObservationVisualization::setObservation(
	const DataCamera & camera, 
	const InputDataLoader::depthMatrix_t & depthImage,
	const SoftBodyGrid3D::Input& input,
	const SoftBodyGrid3D::State& state)
{
	std::vector<arrow> arrows;

	//Find the mapping from point to cell
	//this is copied from CostFunctions_PartialObservations_EvaluteCamera.cu
	
	WorldGridData<real>& referenceSdf = *input.referenceSdf_;
	referenceSdf.copyDeviceToHost();
	WorldGridData<real3> gridDisplacements(input.grid_);
	gridDisplacements.setDeviceMemory(state.gridDisplacements_);
	gridDisplacements.copyDeviceToHost();
	
	const Eigen::Vector3i& size = referenceSdf.getGrid()->getSize();
	real h = referenceSdf.getGrid()->getVoxelSize();
	int3 offset = make_int3(referenceSdf.getGrid()->getOffset().x(), referenceSdf.getGrid()->getOffset().y(), referenceSdf.getGrid()->getOffset().z());

	for (int k = 0; k < size.z() - 1; ++k) for (int j = 0; j < size.y() - 1; ++j) for (int i = 0; i < size.x() - 1; ++i)
	{
		//if (arrows.size() > 50) break;

		// load SDF values and displacements
		real sdfX[8] = {
			referenceSdf.getHost(i, j, k),
			referenceSdf.getHost(i + 1, j, k),
			referenceSdf.getHost(i, j + 1, k),
			referenceSdf.getHost(i + 1, j + 1, k),
			referenceSdf.getHost(i, j, k + 1),
			referenceSdf.getHost(i + 1, j, k + 1),
			referenceSdf.getHost(i, j + 1, k + 1),
			referenceSdf.getHost(i + 1, j + 1, k + 1),
		};
		bool outside = false;
		for (int n = 0; n < 8 && !outside; ++n) if (sdfX[n] > 1e10) outside = true;
		if (outside) continue;
		real3 dispX[8] = {
			gridDisplacements.getHost(i, j, k),
			gridDisplacements.getHost(i + 1, j, k),
			gridDisplacements.getHost(i, j + 1, k),
			gridDisplacements.getHost(i + 1, j + 1, k),
			gridDisplacements.getHost(i, j, k + 1),
			gridDisplacements.getHost(i + 1, j, k + 1),
			gridDisplacements.getHost(i, j + 1, k + 1),
			gridDisplacements.getHost(i + 1, j + 1, k + 1)
		};

		// advected positions
		const real3 posX[8] = {
			make_real3((i + offset.x) * h, (j + offset.y) * h, (k + offset.z) * h) + dispX[0],
			make_real3((i + offset.x + 1) * h, (j + offset.y) * h, (k + offset.z) * h) + dispX[0],
			make_real3((i + offset.x) * h, (j + offset.y + 1) * h, (k + offset.z) * h) + dispX[0],
			make_real3((i + offset.x + 1) * h, (j + offset.y + 1) * h, (k + offset.z) * h) + dispX[0],
			make_real3((i + offset.x) * h, (j + offset.y) * h, (k + offset.z + 1) * h) + dispX[0],
			make_real3((i + offset.x + 1) * h, (j + offset.y) * h, (k + offset.z + 1) * h) + dispX[0],
			make_real3((i + offset.x) * h, (j + offset.y + 1) * h, (k + offset.z + 1) * h) + dispX[0],
			make_real3((i + offset.x + 1) * h, (j + offset.y + 1) * h, (k + offset.z + 1) * h) + dispX[0]
		};

		// screen positions to check
		float2 minScreen, maxScreen;
		glm::vec4 posScreen = camera.viewProjMatrix * glm::vec4(posX[0].x, posX[0].y, posX[0].z, 1.0f);
		minScreen = maxScreen = make_float2(posScreen.x, posScreen.y) / posScreen.w;
		for (int n = 1; n < 8; ++n)
		{
			posScreen = camera.viewProjMatrix * glm::vec4(posX[n].x, posX[n].y, posX[n].z, 1.0f);
			float2 posScreenDehom = make_float2(posScreen.x, posScreen.y) / posScreen.w;
			minScreen = fminf(minScreen, posScreenDehom);
			maxScreen = fmaxf(maxScreen, posScreenDehom);
		}
		minScreen = (minScreen + make_float2(1, 1)) / 2 * make_float2(depthImage.rows(), depthImage.cols());
		maxScreen = (maxScreen + make_float2(1, 1)) / 2 * make_float2(depthImage.rows(), depthImage.cols());

		for (int screenY = max(0, int(floor(minScreen.y))); screenY <= min(int(depthImage.cols() - 1), int(ceil(maxScreen.y))); ++screenY)
			for (int screenX = max(0, int(floor(minScreen.x))); screenX <= min(int(depthImage.rows() - 1), int(ceil(maxScreen.x))); ++screenX)
			{
				//Fetch depth and compute expected position
				real depth = depthImage(screenX, screenY);
				if (depth <= 0) continue;
				glm::vec4 observedPos4 = camera.invViewProjMatrix * glm::vec4(screenX * 2 / float(depthImage.rows()) - 1, screenY * 2 / float(depthImage.cols()) - 1, depth * 2 - 1, 1.0);
				real3 observedPos = make_real3(observedPos4.x, observedPos4.y, observedPos4.z) / observedPos4.w;

				//Solve inverse trilinear interpolation
				real3 xyz = trilinearInverse(observedPos, posX);
				if (xyz.x < 0 || xyz.y < 0 || xyz.z < 0 || xyz.x>1 || xyz.y>1 || xyz.z>1) continue; //not in the current cell

				//append arrow
				glm::vec3 arrowStart(observedPos.x, observedPos.y, observedPos.z);
				glm::vec3 arrowEnd((offset.x + i + 0.5)*h, (offset.y + j + 0.5)*h, (offset.z + k + 0.5)*h);
				arrows.emplace_back(arrowStart, arrowEnd);

				////compute the SDF value at the current position and add to the cost
				//real sdf = trilinear(xyz, sdfX);

				////adjoint of the SDF interpolation
				//real3 adjXYZ = make_real3(0);
				//trilinearAdjoint(xyz, sdfX, sdf, adjXYZ);
			}
	}

	//shuffle it and show only the first 10 arrows
	std::random_shuffle(arrows.begin(), arrows.end());
	arrows.resize(50);

	//store arrows
	{
		std::lock_guard<std::mutex> guard(mutex_);
		arrows_ = arrows;
	}
}

void ar3d::PartialObservationVisualization::draw()
{
	std::lock_guard<std::mutex> guard(mutex_);
	cinder::gl::ScopedColor color(0, 0, 1);
	for (const auto& a : arrows_) {
		cinder::gl::drawCube(a.first, glm::vec3(1 / 200.0f));
		cinder::gl::drawLine(a.first, a.second);
	}
}
