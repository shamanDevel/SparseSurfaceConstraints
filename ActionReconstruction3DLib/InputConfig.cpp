#include "InputConfig.h"

#include <cinder/Json.h>
#include <cinder/Log.h>
#include <tinyformat.h>

std::string ar3d::InputConfig::getPathToFrame(int camera, int frame, bool depth) const
{
	std::string finalPath;
	if (depth)
		finalPath = path + "/" + tinyformat::format(cameras[camera].depthPath.c_str(), frame);
	else
		finalPath = path + "/" + tinyformat::format(cameras[camera].colorPath.c_str(), frame);
	//CI_LOG_D("Path for camera " << camera << ", frame " << frame << ", depth " << depth << ": " << finalPath);
	return finalPath;
}

std::string ar3d::InputConfig::getPathToGroundTruth(int frame) const
{
	if (!groundTruthMeshes) return std::string();
	if (groundTruthMeshPath.find('%') == std::string::npos)
		return path + "/" + groundTruthMeshPath;
	else
		return path + "/" + tinyformat::format(groundTruthMeshPath.c_str(), frame);
}

ar3d::InputConfigPtr ar3d::InputConfig::loadFromJson(const std::string& path)
{
	//parse json
	std::string configPath = path + "/config.json";
	CI_LOG_I("read input configuration file " << configPath);
	cinder::JsonTree jsonTree(cinder::loadFile(configPath));

	//convert to input config
	ar3d::InputConfig* c = new ar3d::InputConfig();
	c->path = path;
	c->width = jsonTree.getValueForKey<int>("width");
	c->height = jsonTree.getValueForKey<int>("height");
	c->framerate = jsonTree.getValueForKey<int>("framerate");
	c->duration = jsonTree.getValueForKey<int>("duration");
	//cameras
	auto camList = jsonTree.getChild("cameras");
	size_t camCount = camList.getNumChildren();
	c->cameras.reserve(camCount);
	for (size_t i=0; i<camCount; ++i)
	{
		CameraWithPath cam;
		glm::vec3 location;
		for (int j = 0; j < 3; ++j) location[j] = camList.getChild(i).getChild("location").getValueAtIndex<double>(j);
		glm::mat4 matrix;
		for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) matrix[j][k] = camList.getChild(i).getChild("matrix").getValueAtIndex<double>(j*4+k);
		cam.camera = DataCamera(location, matrix);
		if (camList.getChild(i).hasChild("colorPath"))
			cam.colorPath = camList.getChild(i).getValueForKey("colorPath");
		else
			cam.colorPath = "cam" + cinder::toString(i) + "/rgb%d.png";
		if (camList.getChild(i).hasChild("depthPath"))
			cam.depthPath = camList.getChild(i).getValueForKey("depthPath");
		else
			cam.depthPath = "cam" + cinder::toString(i) + "/depth%d.png";
		if (camList.getChild(i).hasChild("flipY"))
			cam.flipY = camList.getChild(i).getValueForKey<bool>("flipY");
		else
			cam.flipY = true;
		c->cameras.push_back(cam);
	}
	//ground truth
	if (jsonTree.hasChild("groundTruth"))
	{
		InputGroundTruth gt;
		auto gtNode = jsonTree.getChild("groundTruth");
		gt.mass = gtNode.getValueForKey<double>("mass");
		gt.radius = gtNode.getValueForKey<double>("radius");
		gt.restitution = gtNode.getValueForKey<double>("restitution");
		gt.dampingLinear = gtNode.getValueForKey<double>("dampingLinear");
		gt.dampingAngular = gtNode.getValueForKey<double>("dampingAngular");
		gt.locations.resize(c->duration);
		gt.rotations.resize(c->duration);
		for (int i=0; i<c->duration; ++i)
		{
			for (int j = 0; j < 3; ++j) gt.locations[i][j] = gtNode.getChild("locations").getChild(i).getChild(j).getValue<double>();
			gt.rotations[i].x() = gtNode.getChild("rotations").getChild(i).getChild(0).getValue<double>();
			gt.rotations[i].y() = gtNode.getChild("rotations").getChild(i).getChild(1).getValue<double>();
			gt.rotations[i].z() = gtNode.getChild("rotations").getChild(i).getChild(2).getValue<double>();
			gt.rotations[i].w() = gtNode.getChild("rotations").getChild(i).getChild(3).getValue<double>();
		}
        c->groundTruth = gt;
	} else if (jsonTree.hasChild("groundTruthMeshes"))
	{
        c->groundTruthMeshes = jsonTree.getValueForKey<bool>("groundTruthMeshes");
		if (jsonTree.hasChild("groundTruthMeshPath"))
			c->groundTruthMeshPath = jsonTree.getValueForKey("groundTruthMeshPath");
		else
			c->groundTruthMeshPath = "groundTruth/frame%d.obj";
	}

	if (jsonTree.hasChild("viewCameraNearPlane"))
		c->viewCameraNearPlane = jsonTree.getValueForKey<float>("viewCameraNearPlane");
	else
		c->viewCameraNearPlane = 0.0f;
	if (jsonTree.hasChild("viewCameraImageTranslation"))
		c->viewCameraImageTranslation = jsonTree.getValueForKey<float>("viewCameraImageTranslation");
	else
		c->viewCameraImageTranslation = -1.0f;

	CI_LOG_I("input configuration successfully read");

	return ar3d::InputConfigPtr(c);
}
