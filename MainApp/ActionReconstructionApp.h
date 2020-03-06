#pragma once

#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>
#include <cinder/CameraUi.h>
#include <cinder/params/Params.h>
#include <cinder/Log.h>
#include <cinder/ObjLoader.h>

#include <Utils.h>
#include <InputConfig.h>
#include <DataCamera.h>
#include <InputDataLoader.h>
#include <BackgroundWorker2.h>
#include <MeshReconstruction.h>
#include <TransferFunctionEditor.h>
#include <VolumeVisualization.h>
#include <GraphPlot.h>

#include "Parameters.h"
#include "PartialObservationVisualization.h"

class ActionReconstructionApp : public ci::app::App {
public:
	ActionReconstructionApp();
	void setup() override;
	void keyDown(ci::app::KeyEvent event) override;
	void keyUp(cinder::app::KeyEvent event) override;
	void mouseDown(ci::app::MouseEvent event) override;
	void update() override;
	void draw() override;
	void cleanup() override;

private:

	//parameters
	ParamsGeneral paramsGeneral_;
	ParamsInspectScan paramsInspectScan_;
	ParamsReconstruction paramsReconstruction_;
	ParamsViewResult paramsViewResult_;
	ci::params::InterfaceGlRef paramsGeneralInterface_;
	ci::params::InterfaceGlRef paramsInspectScanInterface_;
	ci::params::InterfaceGlRef paramsReconstructionInterface_;
	ci::params::InterfaceGlRef paramsViewResultInterface_;

	//current state
	enum class State
	{
		InspectScan,
		Reconstruction,
		ViewResult
	};
	State currentState_;

	//phase 0: general stuff
	void setupGeneral();
	int frameCounter_ = 0;
	bool spaceBarPressed_ = false;
	std::string frameNames_;
	bool animationTakeScreenshot_ = false;
	bool animationRendered_ = false;
	bool exportWithNormals_ = true;

	//phase 1: inspect scan + reconstruction
	ar3d::InputDataLoaderPtr dataLoader_;
	ar3d::InputDataLoader::FramePtr frameData_;
	ar3d::MeshReconstructionPtr meshReconstruction_;
	void setupInspectScan();
	void updateInspectScan();
	void drawInspectScan();
	void inspectScanUseGroundTruth(); //use ground truth directly instead of kinect fusion
	void nextReconstruction(); //inspect scan -> reconstruction

	//phase 2: reconstruction
	ar3d::WorldGridRealDataPtr referenceSdf_;
	std::unique_ptr<ar3d::SoftBodyGrid3D> gridSimulation_;
	ar3d::PartialObservationVisualization observationVis_;
	bool visualizePartialObservations_;
	void setupReconstruction();
	void updateReconstruction();
	void drawReconstruction();
	void prevInspectScan(); //reconstruction -> inspect scan
	void nextViewResult(); //reconstruction -> view result
	void reconstructionForwardStep();
	void reconstructionResetPlots();
	void reconstructionReset();
	void reconstructionSolve();
	void reconstructionTestGradient();

	//phase 3: view result
	void setupViewResult();
	void updateViewResult();
	void drawViewResult();
	void prevReconstruction(); //view result -> reconstruction

	//rendering stuff
	std::vector<ci::vec3> projCamPoints_;
	ci::gl::BatchRef pointsBatch_;
	ci::gl::VboMeshRef groundTruthVboMesh_;
	ci::gl::GlslProgRef groundTruthShader_;
	ci::CameraPersp		camera_;
	ci::CameraUi		camUi_;
	ci::gl::VboMeshRef	floorVboMesh_;
	ci::gl::TextureRef	floorTexture_;

	//rendering helper
	void drawGroundTruth(int frame);
	void drawProjCameraPoints(int frame);
	void drawCameras(int frame);

	//processed data
	ar3d::BackgroundWorker2Ptr worker_;

	//visualization helpers
	std::unique_ptr<ar::TransferFunctionEditor> tfe_;
	std::unique_ptr<ar3d::VolumeVisualization> volumeVis_;
	typedef std::function<float(const ar3d::SoftBodySimulation3D::Settings&)> ParamGetter_t;
	typedef std::function<bool(const ar3d::AdjointSolver::InputVariables&)> ParamEnabled_t;
	typedef std::tuple<std::unique_ptr<GraphPlot>, ParamGetter_t, ParamEnabled_t> Plot_t;
	std::unique_ptr<GraphPlot> costPlot;
	std::vector<Plot_t> plots;

	//misc
	void drawWireCube(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
	void drawWireCube(const ar3d::real3& a, const ar3d::real3& b);

	void load();
	void save();
};