#pragma once

#include <VolumeVisualization.h>
#include <MeshReconstruction.h>
#include <SoftBodyGrid3D.h>
#include <AdjointSolver.h>
#include <CostFunctions.h>

class ParamsGeneral;
class ParamsInspectScan;
class ParamsReconstruction;
class ParamsViewResult;

class ParamsGeneral
{
public:
	//Rendering
	bool printMode_;
	bool showBoundingBox_;
	ar3d::VolumeVisualizationParams volumeVisParams_;
	bool viewPoints_;

	enum class ShowCameraMode
	{
		Off,
		Outline,
		Depth,
		Color
	};
	static inline ShowCameraMode ToShowCameraMode(const std::string& str)
	{
		if (str == "Outline") return ShowCameraMode::Outline;
		else if (str == "Depth") return ShowCameraMode::Depth;
		else if (str == "Color") return ShowCameraMode::Color;
		return ShowCameraMode::Off;
	}
	static inline std::string FromShowCameraMode(const ShowCameraMode& mode)
	{
		switch (mode)
		{
		case ShowCameraMode::Off: return "Off";
		case ShowCameraMode::Outline: return "Outline";
		case ShowCameraMode::Depth: return "Depth";
		case ShowCameraMode::Color: return "Color";
		}
	}
	ShowCameraMode showCameraMode_;
	bool showGroundTruth_;

	//video
	bool exportAnimation_;
	std::string animationName_;

	ParamsGeneral();
	void addParams(const cinder::params::InterfaceGlRef& params);
	void load(const cinder::JsonTree& parent);
	void save(cinder::JsonTree& parent) const;
};

class ParamsInspectScan
{
public:
	std::string scanFile_;
	int frame_;
	ar3d::MeshReconstruction::SettingsGui meshReconstructionSettings;

private:
	ar3d::InputConfigPtr inputConfig_;
	cinder::params::InterfaceGlRef params_;

public:
	ParamsInspectScan();
	void addParams(const cinder::params::InterfaceGlRef& params);
	void load(const cinder::JsonTree& parent);
	void save(cinder::JsonTree& parent) const;
	const ar3d::InputConfigPtr& getConfig() const { return inputConfig_; }

private:
	void loadScan();
};

class ParamsReconstruction
{
public:
	std::string resolution_;
	ar3d::SoftBodySimulation3D::InputSdfSettingsGui inputSettings_;
	ar3d::SoftBodySimulation3D::SettingsGui simulationSettings_;
	ar3d::AdjointSolver::GUI adjointSettings_;
	int costIntermediateSteps_;
	int costNumSteps_;

	ParamsReconstruction();
	void addParams(const cinder::params::InterfaceGlRef& params);
	void load(const cinder::JsonTree& parent);
	void save(cinder::JsonTree& parent) const;
	void setWorldGrid(ar3d::WorldGridPtr grid); //updates the resolution with the grid from step 1
};

class ParamsViewResult
{
public:
	ar3d::SoftBodySimulation3D::SettingsGui simulationSettings_;

	ParamsViewResult();
	void addParams(const cinder::params::InterfaceGlRef& params);
	void load(const cinder::JsonTree& parent);
	void save(cinder::JsonTree& parent) const;
};