#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <boost/optional.hpp>
#include <memory>

#include "DataCamera.h"

namespace ar3d {

	struct InputGroundTruth
	{
		double mass;
		double radius;
		double restitution;
		double dampingLinear;
		double dampingAngular;
		std::vector<Eigen::Vector3d> locations;
		std::vector<Eigen::Quaterniond> rotations;
	};

	/**
	 * \brief Contains the configuration from the input data (camera recordings).
	 * It also contains optional ground truth data from simulations, if available.
	 */
	struct InputConfig
	{
		struct CameraWithPath
		{
			DataCamera camera;
			std::string colorPath;
			std::string depthPath;
			bool flipY;
		};

		int width = 0;
		int height = 0;
		int framerate = 0;
		int duration = 0;
		std::vector<CameraWithPath> cameras;
		boost::optional<InputGroundTruth> groundTruth;
        bool groundTruthMeshes = false;
		std::string groundTruthMeshPath;
		std::string path;

		float viewCameraImageTranslation;
		float viewCameraNearPlane;

		// Returns the path to the camera image of the specified frame
		std::string getPathToFrame(int camera, int frame, bool depth) const;
		// Returns the path to the ground truth mesh at the specified frame
		// Returns an empty string if there is no ground truth mesh
		std::string getPathToGroundTruth(int frame) const;

		static std::shared_ptr<const InputConfig> loadFromJson(const std::string& path);
	};
	typedef std::shared_ptr<const InputConfig> InputConfigPtr;

}