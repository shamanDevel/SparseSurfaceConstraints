#include "DataCamera.h"

ar3d::DataCamera::DataCamera(const glm::vec3& loc, const glm::mat4& mat)
	: location(loc)
    , viewProjMatrix(mat)
    , invViewProjMatrix(glm::inverse(viewProjMatrix))
{
}

glm::vec3 ar3d::DataCamera::getWorldCoordinates(const glm::vec3& screenPosition) const
{
	glm::vec4 v = glm::vec4(screenPosition * 2.0f - glm::vec3(1.0, 1.0, 1.0), 1);
	glm::vec4 r = invViewProjMatrix * v;
    return glm::vec3(r.x, r.y, r.z) / r.w;
}

glm::vec3 ar3d::DataCamera::getScreenCoordinates(const glm::vec3& worldPosition) const
{
    glm::vec4 v(worldPosition, 1.0);
	glm::vec4 r = viewProjMatrix * v;
	return ((glm::vec3(r.x, r.y, r.z) / r.w) + glm::vec3(1.0, 1.0, 1.0)) / 2.0f;
}
