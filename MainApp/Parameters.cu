#include "Parameters.h"

#include <tinyformat.h>

ParamsGeneral::ParamsGeneral()
	: printMode_(false)
	, showBoundingBox_(true)
	, showCameraMode_(ShowCameraMode::Color)
	, showGroundTruth_(false)
	, exportAnimation_(false)
	, animationName_("")
	, viewPoints_(false)
{
}

void ParamsGeneral::addParams(const cinder::params::InterfaceGlRef & params)
{
	params->addParam("ParamsGeneral-PrintMode", &printMode_).label("Print Mode");
	params->addParam("ParamsGeneral-ShowBoundingBox", &showBoundingBox_).label("Show Bounding Box");
	volumeVisParams_.addParams(params);
	std::vector<std::string> showCameraEnums = { "Off", "Outline", "Depth", "Color" };
	params->addParam("ParamsGeneral-ShowCameraMode", showCameraEnums, reinterpret_cast<int*>(&showCameraMode_))
		.label("Show Camera Mode");
	params->addParam("ParamsGeneral-ShowGroundTruth", &showGroundTruth_).label("Show Ground Truth");
	params->addParam("ParamsInspectScan-ViewPoints", &viewPoints_).label("ViewPoints");
}

void ParamsGeneral::load(const cinder::JsonTree & parent)
{
	printMode_ = parent.getValueForKey<bool>("PrintMode");
	showBoundingBox_ = parent.getValueForKey<bool>("ShowBoundingBox");
	volumeVisParams_.load(parent.getChild("VolumeVis"));
	showCameraMode_ = ToShowCameraMode(parent.getValueForKey("ShowCameraMode"));
	showGroundTruth_ = parent.getValueForKey<bool>("ShowGroundTruth");
	if (parent.hasChild("ViewPoints")) viewPoints_ = parent.getValueForKey<bool>("ViewPoints");
}

void ParamsGeneral::save(cinder::JsonTree & parent) const
{
	parent.addChild(cinder::JsonTree("PrintMode", printMode_));
	parent.addChild(cinder::JsonTree("ShowBoundingBox", showBoundingBox_));
	cinder::JsonTree volumeVis = cinder::JsonTree::makeObject("VolumeVis");
	volumeVisParams_.save(volumeVis);
	parent.addChild(volumeVis);
	parent.addChild(cinder::JsonTree("ShowCameraMode", FromShowCameraMode(showCameraMode_)));
	parent.addChild(cinder::JsonTree("ShowGroundTruth", showGroundTruth_));
	parent.addChild(cinder::JsonTree("ViewPoints", viewPoints_));
}

ParamsInspectScan::ParamsInspectScan()
	: scanFile_("")
	, frame_(0)
{
}

void ParamsInspectScan::addParams(const cinder::params::InterfaceGlRef & params)
{
	params_ = params;
	params->addParam("ParamsInspectScan-ScanFile", &scanFile_, true).label("Scan File");
	params->addButton("ParamsInpectScan-Load", [this]() {
		cinder::fs::path initial = {};
		std::vector<std::string> extensions = { "json" };
		cinder::fs::path path = cinder::app::getOpenFilePath(initial, extensions);
		if (!path.empty() && path.filename().string()=="config.json") {
			scanFile_ = path.parent_path().string();
			CI_LOG_I("config folder specified: " << scanFile_);
			this->loadScan();
		}
	}, "label='Load Scan'");
	params->addParam("ParamsInpectScan-Frame", &frame_).label("Frame").min(0);
	meshReconstructionSettings.initParams(params, "Reconstruction");
}

void ParamsInspectScan::load(const cinder::JsonTree & parent)
{
	scanFile_ = parent.getValueForKey("ScanFile");
	frame_ = parent.getValueForKey<int>("Frame");
	loadScan();
	meshReconstructionSettings.load(parent.getChild("Reconstruction"));
}

void ParamsInspectScan::save(cinder::JsonTree & parent) const
{
	parent.addChild(cinder::JsonTree("ScanFile", scanFile_));
	parent.addChild(cinder::JsonTree("Frame", frame_));
	cinder::JsonTree rec = cinder::JsonTree::makeObject("Reconstruction");
	meshReconstructionSettings.save(rec);
	parent.addChild(rec);
}

void ParamsInspectScan::loadScan()
{
	if (scanFile_.empty())
		inputConfig_ = nullptr;
	else {
		inputConfig_ = ar3d::InputConfig::loadFromJson(scanFile_);
		frame_ = std::min(frame_, inputConfig_->duration - 1);
		params_->setOptions("ParamsInpectScan-Frame", "max=" + std::to_string(inputConfig_->duration - 1));
	}
}

ParamsReconstruction::ParamsReconstruction()
	: costIntermediateSteps_(0)
	, costNumSteps_(0)
{
}

void ParamsReconstruction::addParams(const cinder::params::InterfaceGlRef & params)
{
	params->addParam("ParamsReconstruction-Resolution", &resolution_, true);
	inputSettings_.initParams(params, "Input", true);
	simulationSettings_.initParams(params, "Elasticity");
	adjointSettings_.initParams(params, "Adjoint", true);
	params->addParam("AdjointCost-IntermediateSteps", &costIntermediateSteps_)
		.label("Cost: Intermediate Steps").group("Adjoint").min(0)
		.optionsStr("help='Specifies how many timesteps shall be simulated in between the timesteps of the observation. Use this for a higher resolution of the simulation than the observation. Default is zero, indicating that the time result of the simulation and the observation match'");
	params->addParam("AdjointCost-NumSteps", &costNumSteps_)
		.label("Cost: Num Steps").group("Adjoint").min(0)
		.optionsStr("help='Specifies how many timesteps from the observation shall be used. If the value is zero, then all timesteps are used.'");
}

void ParamsReconstruction::load(const cinder::JsonTree & parent)
{
	inputSettings_.load(parent.getChild("Input"), true);
	simulationSettings_.load(parent.getChild("Elasticity"));
	adjointSettings_.load(parent.getChild("Adjoint"), true);
	if (parent.hasChild("CostIntermediateSteps"))
		costIntermediateSteps_ = parent.getValueForKey<int>("CostIntermediateSteps");
	if (parent.hasChild("CostNumSteps"))
		costNumSteps_ = parent.getValueForKey<int>("CostNumSteps");
}

void ParamsReconstruction::save(cinder::JsonTree & parent) const
{
	cinder::JsonTree input = cinder::JsonTree::makeObject("Input");
	inputSettings_.save(input, true);
	parent.addChild(input);

	cinder::JsonTree elasticity = cinder::JsonTree::makeObject("Elasticity");
	simulationSettings_.save(elasticity);
	parent.addChild(elasticity);

	cinder::JsonTree adjoint = cinder::JsonTree::makeObject("Adjoint");
	adjointSettings_.save(adjoint, true);
	parent.addChild(adjoint);

	parent.addChild(cinder::JsonTree("CostIntermediateSteps", costIntermediateSteps_));
	parent.addChild(cinder::JsonTree("CostNumSteps", costNumSteps_));
}

void ParamsReconstruction::setWorldGrid(ar3d::WorldGridPtr grid)
{
	resolution_ = tinyformat::format("%d x %d x %d", 
		grid->getSize().x(), grid->getSize().y(), grid->getSize().z());
}

ParamsViewResult::ParamsViewResult()
{
}

void ParamsViewResult::addParams(const cinder::params::InterfaceGlRef & params)
{
	simulationSettings_.initParams(params, "Reconstructed Settings");
}

void ParamsViewResult::load(const cinder::JsonTree & parent)
{
	//don't do anything
}

void ParamsViewResult::save(cinder::JsonTree & parent) const
{
	//don't do anything
}
