#pragma once

#include <cinder/Matrix.h>

namespace ar3d {

	/**
	 * \brief Camera utility: stores the property of the camera and allows
	 * transformations from camera (screen) space to world space and back
	 */
	class DataCamera
	{
	public:
		glm::vec3 location;
		glm::mat4 viewProjMatrix;
		glm::mat4 invViewProjMatrix;

	public:
        DataCamera() = default;
		DataCamera(const glm::vec3& loc, const glm::mat4& mat);

		/**
		 * \brief Converts the screen coordinates into world coordinates
		 * \param screenPosition screen coordinates:
		 *		0 <= x <= 1
		 *		0 <= y <= 1
		 *		0 <= z (depth) <= 1 (0=Near Plane, 1=Far Plane)
		 * \return the world coordinates
		 */
		glm::vec3 getWorldCoordinates(const glm::vec3& screenPosition) const;
		/**
		 * \brief Converts the world coordinates into normalized screen coordinates
		 * \param worldPosition world coordinates
		 * \return screen coordinates
		 *		0 <= x <= 1
		 *		0 <= y <= 1
		 *		0 <= z (depth) <= 1 (0=Near Plane, 1=Far Plane)
		 */
		glm::vec3 getScreenCoordinates(const glm::vec3& worldPosition) const;
	};

}