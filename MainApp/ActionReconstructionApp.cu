#include "ActionReconstructionApp.h"

#include <cmath>
#include <chrono>
#include <thread>
#include <time.h>

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
#include <BackgroundWorker.h>
#include <MeshReconstruction.h>
#include <TransferFunctionEditor.h>
#include <VolumeVisualization.h>
#include <CoordinateTransformation.h>
#include <cuPrintf.cuh>
#include <tinyformat.h>
#include <ObjectLoader.h>
#include <GroundTruthToSdf.h>

#include "resources/Resources.h"
#include "Parameters.h"

using namespace ci::app;
using namespace std;
using namespace ar;

ActionReconstructionApp::ActionReconstructionApp()
	: currentState_(State::InspectScan)
{
	worker_ = std::make_unique<ar3d::BackgroundWorker2>();
	std::time_t result = std::time(nullptr);
	ci::log::makeLogger<ci::log::LoggerFile>(tinyformat::format("%sMainApp%s.log", getAppPath().string(), std::asctime(std::localtime(&result))), true);
}

void ActionReconstructionApp::setup()
{
	CUMAT_SAFE_CALL(cudaPrintfInit());

	//parameter ui, must happen before user-camera
	paramsGeneralInterface_ = ci::params::InterfaceGl::create(getWindow(), "General", toPixels(ci::ivec2(300, 250)));
	paramsInspectScanInterface_ = ci::params::InterfaceGl::create(getWindow(), "Inspect Scan", toPixels(ci::ivec2(300, 580)));
	paramsReconstructionInterface_ = ci::params::InterfaceGl::create(getWindow(), "Reconstruction", toPixels(ci::ivec2(300, 580)));
	paramsViewResultInterface_ = ci::params::InterfaceGl::create(getWindow(), "View Result", toPixels(ci::ivec2(300, 580)));
	paramsReconstructionInterface_->show(false);
	paramsViewResultInterface_->show(false);

    //transfer function editor
    tfe_ = std::make_unique<TransferFunctionEditor>(getWindow());

    //volume visualization
    volumeVis_ = std::make_unique<ar3d::VolumeVisualization>(getWindow(), &camera_, &paramsGeneral_.volumeVisParams_);

	//user-camera
	camUi_ = ci::CameraUi(&camera_, getWindow());
    ci::vec3 newEye = camera_.getEyePoint() + camera_.getViewDirection() * (camera_.getPivotDistance() * (1 - 0.2f));
	camera_.setEyePoint(newEye);
	camera_.setPivotDistance(camera_.getPivotDistance() * 0.2f);

	//floor
	auto plane = cinder::geom::Plane().size(ci::vec2(4, 4)).subdivisions(ci::ivec2(5, 5));
	vector<ci::gl::VboMesh::Layout> bufferLayout = {
        ci::gl::VboMesh::Layout().usage(GL_DYNAMIC_DRAW).attrib(cinder::geom::Attrib::POSITION, 3),
        ci::gl::VboMesh::Layout().usage(GL_STATIC_DRAW).attrib(cinder::geom::Attrib::TEX_COORD_0, 2)
	};
	floorVboMesh_ = ci::gl::VboMesh::create(plane, bufferLayout);
	ci::gl::Texture::Format floorTextureFmt;
	floorTextureFmt.enableMipmapping(true);
	floorTextureFmt.setMinFilter(GL_LINEAR_MIPMAP_LINEAR);
	floorTexture_ = ci::gl::Texture::create(loadImage(loadResource(CHECKERBOARD_IMAGE)), floorTextureFmt);

    //ground truth shader
    groundTruthShader_ = ci::gl::GlslProg::create(ci::gl::GlslProg::Format()
        .vertex(R"GLSL(
#version 150
uniform mat4	ciModelViewProjection;
in vec4			ciPosition;
in vec4			ciColor;
out vec4		Color;

void main(void) {
    gl_Position = ciModelViewProjection * ciPosition;
    Color = ciColor;
}
)GLSL")
        .fragment(R"GLSL(
#version 150
in vec4		Color;
out vec4	oColor;

void main(void) {
    oColor = Color;
}
)GLSL"
    ));

	setupGeneral();
	setupInspectScan();
	setupReconstruction();
	setupViewResult();
}

void ActionReconstructionApp::setupGeneral()
{
	paramsGeneralInterface_->addButton("Load", [this]() {this->load(); });
	paramsGeneralInterface_->addButton("Save", [this]() {this->save(); });
	paramsGeneralInterface_->addParam("Rendering-SaveFrameNames", &frameNames_).label("Frame Names");
	paramsGeneralInterface_->addParam("Rendering-ExportWithNormals", &exportWithNormals_).group("Rendering").label("Export \\w normals + orig.pos.");
	paramsGeneral_.addParams(paramsGeneralInterface_);
	paramsGeneralInterface_->setPosition(glm::ivec2(5, 590));
}

void ActionReconstructionApp::setupInspectScan()
{
	paramsInspectScan_.addParams(paramsInspectScanInterface_);
	paramsInspectScanInterface_->addButton("InspectUseGroundTruth", [this]() {this->inspectScanUseGroundTruth(); }, "label='Use Ground Truth'");
	paramsInspectScanInterface_->addButton("Next-1-2", [this]() {this->nextReconstruction(); }, "label='Next - Reconstruction'");
	paramsInspectScanInterface_->setPosition(glm::ivec2(5, 5));
}

void ActionReconstructionApp::setupReconstruction()
{
	visualizePartialObservations_ = false;
	paramsReconstruction_.addParams(paramsReconstructionInterface_);
	paramsReconstructionInterface_->addButton("RecForwardStep", [this]() {this->reconstructionForwardStep(); }, "label='Forward Step' ");
	paramsReconstructionInterface_->addButton("RecReset", [this]() {this->reconstructionReset(); }, "label='Reset' key=r ");
	paramsReconstructionInterface_->addButton("RecSolve", [this]() {this->reconstructionSolve(); }, "label='Solve' key=Return ");
	paramsReconstructionInterface_->addButton("RecTest", [this]() {this->reconstructionTestGradient(); }, "label='Test Gradient' ");
	paramsReconstructionInterface_->addParam("RecPartObsVis", &visualizePartialObservations_).label("Visualize Observations");
	paramsReconstructionInterface_->addButton("Prev-2-1", [this]() {this->prevInspectScan(); }, "label='Prev - Inspect Scan'");
	paramsReconstructionInterface_->addButton("Next-2-3", [this]() {this->nextViewResult(); }, "label='Next - View Result'");
	paramsReconstructionInterface_->setPosition(glm::ivec2(5, 5));

	//plots
	typedef ar3d::AdjointSolver::InputVariables V;
	typedef ar3d::SoftBodySimulation3D::Settings S;
	costPlot = std::make_unique<GraphPlot>("Cost");
	costPlot->setTrueValue(0);
	//plots.emplace_back(std::make_unique<GraphPlot>("GravityX"), [](const S& var) {return var.gravity_.x; }, [](const V& var) {return var.optimizeGravity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("GravityY"), [](const S& var) {return var.gravity_.y; }, [](const V& var) {return var.optimizeGravity_; });
	//plots.emplace_back(std::make_unique<GraphPlot>("GravityZ"), [](const S& var) {return var.gravity_.z; }, [](const V& var) {return var.optimizeGravity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Young's Modulus"), [](const S& var) {return var.youngsModulus_; }, [](const V& var) {return var.optimizeYoungsModulus_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Poisson Ratio"), [](const S& var) {return var.poissonsRatio_; }, [](const V& var) {return var.optimizePoissonRatio_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Mass"), [](const S& var) {return var.mass_; }, [](const V& var) {return var.optimizeMass_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Mass Damping"), [](const S& var) {return var.dampingAlpha_; }, [](const V& var) {return var.optimizeMassDamping_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Stiffness Damping"), [](const S& var) {return var.dampingBeta_; }, [](const V& var) {return var.optimizeStiffnessDamping_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityX"), [](const S& var) {return var.initialLinearVelocity_.x; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityY"), [](const S& var) {return var.initialLinearVelocity_.y; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityZ"), [](const S& var) {return var.initialLinearVelocity_.z; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityX"), [](const S& var) {return var.initialAngularVelocity_.x; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityY"), [](const S& var) {return var.initialAngularVelocity_.y; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityZ"), [](const S& var) {return var.initialAngularVelocity_.z; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Theta"), [](const S& var) {return ar3d::CoordinateTransformation::cartesian2spherical(var.groundPlane_).y; }, [](const V& var) {return var.optimizeGroundPlane_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Phi"), [](const S& var) {return ar3d::CoordinateTransformation::cartesian2spherical(var.groundPlane_).z; }, [](const V& var) {return var.optimizeGroundPlane_; });
	plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Height"), [](const S& var) {return var.groundPlane_.w; }, [](const V& var) {return var.optimizeGroundPlane_; });
}

void ActionReconstructionApp::setupViewResult()
{
	paramsViewResult_.addParams(paramsViewResultInterface_);
	paramsViewResultInterface_->addButton("Prev-3-2", [this]() {this->prevReconstruction(); }, "label='Prev - Reconstruction'");
	paramsViewResultInterface_->setPosition(glm::ivec2(5, 5));
}

void ActionReconstructionApp::keyDown(KeyEvent event)
{
	App::keyDown(event);
	if (event.getChar() == 'f') {
		// Toggle full screen when the user presses the 'f' key.
		setFullScreen(!isFullScreen());
	}
	else if (event.getCode() == KeyEvent::KEY_ESCAPE) {
		// Exit full screen, or quit the application, when the user presses the ESC key.
		if (isFullScreen())
			setFullScreen(false);
		else
			quit();
	}
	else if (event.getChar() == 'p') {
		//Screenshot
        ci::Surface surface = copyWindowSurface();
		//construct filename
		time_t now = time(NULL);
		struct tm tstruct;
		char buf[100];
		localtime_s(&tstruct, &now);
		strftime(buf, sizeof(buf), "%d-%m-%Y_%H-%M-%S", &tstruct);
		string fileName = string("screenshot-") + string(buf) + ".png";
		//write out
        ci::writeImage(fileName, surface);
	} 
    else if (event.getChar() == 'l')
	{
	    //reload resources, shaders, ...
        volumeVis_->reloadResources();
	}
	else if (event.getCode() == KeyEvent::KEY_SPACE)
	{
		spaceBarPressed_ = true;
	}
}

void ActionReconstructionApp::keyUp(KeyEvent event)
{
	App::keyUp(event);
	if (event.getCode() == KeyEvent::KEY_SPACE && spaceBarPressed_)
	{
		spaceBarPressed_ = false;
	}
}

void ActionReconstructionApp::mouseDown( MouseEvent event )
{
	App::mouseDown(event);
}

void ActionReconstructionApp::update()
{
	//save animation from previous frame
	if (animationTakeScreenshot_ && animationRendered_ && !frameNames_.empty()) {
		//Screenshot
		cinder::Surface surface = copyWindowSurface();
		string fileName = tinyformat::format("../screenshots/%s%05d.png", frameNames_.c_str(), frameCounter_);
		writeImage(fileName, surface);
		CI_LOG_I("screenshot saved to " << fileName);
		////SDF file
		//if (gridSimulation_) {
		//	string fileName = tinyformat::format("../screenshots/%s%05d.sdf", frameNames_.c_str(), frameCounter_);
		//	volumeVis_->saveSdf(fileName);
		//}
		//Marching Cubes mesh file
		if (gridSimulation_) {
			string fileName = tinyformat::format("../screenshots/%s%05d.obj", frameNames_.c_str(), frameCounter_);
			volumeVis_->saveMCMesh(fileName);
		}
		//High Resolution mesh file
		if (gridSimulation_) {
			string fileName = tinyformat::format("../screenshots/%s_high%05d.obj", frameNames_.c_str(), frameCounter_);
			volumeVis_->saveHighResultMesh(fileName, exportWithNormals_, exportWithNormals_);
		}
		//done
	}
	if (animationTakeScreenshot_ && animationRendered_) {
		animationRendered_ = false;
		animationTakeScreenshot_ = false;
	}

	switch (currentState_)
	{
	case State::InspectScan: updateInspectScan(); break;
	case State::Reconstruction: updateReconstruction(); break;
	case State::ViewResult: updateViewResult(); break;
	}

	//update transfer function editor
	tfe_->setVisible(volumeVis_->needsTransferFunction());
	tfe_->update();
}

void ActionReconstructionApp::updateInspectScan()
{	
	if (!worker_->isDone()) return;

	int hasChanges = -1;

	static ar3d::InputConfigPtr oldConfigPtr = nullptr;
	static int oldFrame = -1;
	static ar3d::MeshReconstruction::Settings oldMeshReconstructionSettings;

	//check, if the input file has changed
	const ar3d::InputConfigPtr config = paramsInspectScan_.getConfig();
	if (!config) return;
	if (oldConfigPtr != config)
	{
		oldConfigPtr = paramsInspectScan_.getConfig();
		//create data loader
		dataLoader_ = std::make_shared<ar3d::InputDataLoader>(paramsInspectScan_.getConfig());
		hasChanges = 0;
		//reset
		oldFrame = -1;
		memset(&oldMeshReconstructionSettings, 0, sizeof(ar3d::MeshReconstruction::Settings));
		groundTruthVboMesh_ = nullptr;
		frameData_ = nullptr;
		meshReconstruction_ = nullptr;
	}

	//check if the frame has changed
	const int frame = paramsInspectScan_.frame_;
	if (oldFrame != frame)
	{
		oldFrame = paramsInspectScan_.frame_;
		groundTruthVboMesh_ = nullptr;
		hasChanges = 0;
	}

	//check if the reconstruction settings have changed
	const ar3d::MeshReconstruction::Settings settings = paramsInspectScan_.meshReconstructionSettings.getSettings();
	if (memcmp(&oldMeshReconstructionSettings, &settings, sizeof(ar3d::MeshReconstruction::Settings)) != 0) {
		oldMeshReconstructionSettings = paramsInspectScan_.meshReconstructionSettings.getSettings();
		if (hasChanges == -1) hasChanges = 0;
	}

	//load in a background thread
    if (hasChanges >= 0) {
        //declare background task
        std::function<void(ar3d::BackgroundWorker2*)> task = 
			[this, hasChanges, frame, settings, config] (ar3d::BackgroundWorker2* worker) {
            if (hasChanges <= 0) {
                //These steps only depend on the frame
                worker->setStatus("Load color and depth images");
                auto frameDataTmp = this->dataLoader_->loadFrame(frame, worker);
                if (worker->isInterrupted()) return;
                this->frameData_ = frameDataTmp;

                worker->setStatus("Initialize mesh reconstruction");
                auto meshReconstructionTmp = make_shared<ar3d::MeshReconstruction>(config, this->frameData_);
                if (worker->isInterrupted()) return;
                this->meshReconstruction_ = meshReconstructionTmp;

                worker->setStatus("Segment images");
                this->meshReconstruction_->runSegmentation(worker);
                if (worker->isInterrupted()) return;

                worker->setStatus("Find bounding box");
                this->meshReconstruction_->runBoundingBoxExtraction(worker);
                if (worker->isInterrupted()) return;
            }

            if (hasChanges <= 1) {
                //These depend on MeshReconstruction::Settings
                this->meshReconstruction_->settings = settings;
                worker->setStatus("Initialize 3D reconstruction");
                this->meshReconstruction_->runInitReconstruction(worker);
                if (worker->isInterrupted()) return;

                worker->setStatus("Optimize 3D reconstruction");
                this->meshReconstruction_->runReconstruction(worker);
                if (worker->isInterrupted()) return;

                worker->setStatus("Recover full signed distance function");
                this->meshReconstruction_->runSignedDistanceReconstruction(worker);
                if (worker->isInterrupted()) return;

				referenceSdf_ = meshReconstruction_->getSdfData();
            }

			frameCounter_ = frame;
			animationTakeScreenshot_ = true;
        };
        //start background worker
        worker_->launch(task);
        CI_LOG_I("Background worker started from section " << hasChanges);
    }
}

void ActionReconstructionApp::inspectScanUseGroundTruth()
{
	if (!worker_->isDone()) return;
	if (!paramsInspectScan_.getConfig()->groundTruthMeshes && !paramsInspectScan_.getConfig()->groundTruth)
	{
		CI_LOG_W("input dataset does not contain ground thruth data");
		return;
	}

	std::function<void(ar3d::BackgroundWorker2*)> task = [this](ar3d::BackgroundWorker2* worker) {
		worker->setStatus("Convert ground truth to SDF");
		int frame = paramsInspectScan_.frame_;
		int resolution = paramsInspectScan_.meshReconstructionSettings.getSettings().gridResolution;
		referenceSdf_ = ar3d::groundTruthToSdf(paramsInspectScan_.getConfig(), frame, resolution);
	};
	worker_->launch(task);
}

void ActionReconstructionApp::updateReconstruction()
{
	static ar3d::WorldGridRealDataPtr oldSdf = nullptr;
	static ar3d::SoftBodySimulation3D::InputSdfSettings oldInputSettings;
	static ar3d::SoftBodySimulation3D::Settings oldElasticitySettings;

	if (!worker_->isDone()) return;
	if (oldSdf != referenceSdf_ ||
		oldInputSettings != paramsReconstruction_.inputSettings_.getSettings())
	{
		//input changed
		oldSdf = referenceSdf_;
		oldInputSettings = paramsReconstruction_.inputSettings_.getSettings();
		oldElasticitySettings = paramsReconstruction_.simulationSettings_.getSettings();
		ar3d::BackgroundWorker2::task task = [this](ar3d::BackgroundWorker2* worker)
		{
			gridSimulation_.reset();
			worker->setStatus("Create Input");
			ar3d::SoftBodyGrid3D::Input input = ar3d::SoftBodyGrid3D::createFromSdf(
				this->paramsReconstruction_.inputSettings_.getSettings(), this->referenceSdf_);
			this->gridSimulation_ = std::make_unique<ar3d::SoftBodyGrid3D>(input);
			this->gridSimulation_->setSettings(paramsReconstruction_.simulationSettings_.getSettings());
			this->gridSimulation_->setRecordTimings(true);
			worker->setStatus("Prepare rendering");
			this->volumeVis_->setInput(this->gridSimulation_->getInput());
			//results->input_ = gridSimulation->getInput();
			frameCounter_ = 0; frameData_ = nullptr;
		};
		worker_->launch(task);
	}

	if (!worker_->isDone()) return;
	if (oldElasticitySettings != paramsReconstruction_.simulationSettings_.getSettings())
	{
		//elasticity settings changed
		oldElasticitySettings = paramsReconstruction_.simulationSettings_.getSettings();
		ar3d::BackgroundWorker2::task task = [this](ar3d::BackgroundWorker2* worker)
		{
			if (gridSimulation_) {
				gridSimulation_->setSettings(paramsReconstruction_.simulationSettings_.getSettings());
				//results->settings_ = simulationSettings_.getSettings();
			}
		};
		worker_->launch(task);
	}

	if (spaceBarPressed_ && !animationTakeScreenshot_) //space bar pressed and current frame saved
		reconstructionForwardStep();
}

void ActionReconstructionApp::reconstructionForwardStep()
{
	if (worker_->isDone() && !animationTakeScreenshot_)
	{
		ar3d::BackgroundWorker2::task task = [this](ar3d::BackgroundWorker2* worker)
		{
			gridSimulation_->solve(true, worker, true);
			volumeVis_->update(gridSimulation_->getState());
			
			worker->setStatus("Prepare partial observation visualization");
			frameData_ = dataLoader_->loadFrame(frameCounter_/(paramsReconstruction_.costIntermediateSteps_+1), worker);
			if (visualizePartialObservations_) {
				observationVis_.setObservation(
					paramsInspectScan_.getConfig()->cameras[0].camera,
					frameData_->cameraImages[0].depthMatrix,
					gridSimulation_->getInput(),
					gridSimulation_->getState());
			}

			frameCounter_++;
			animationTakeScreenshot_ = true;
			animationRendered_ = false;
		};
		worker_->launch(task);
	}
}

void ActionReconstructionApp::reconstructionResetPlots()
{
	costPlot->clear();
	for (const auto& e : plots)
	{
		std::get<0>(e)->clear();
	}
}

void ActionReconstructionApp::reconstructionReset()
{
	CI_LOG_I("Reset simulation");
	//wait for the current task
	if (worker_)
	{
		worker_->interrupt();
		worker_->wait();
	}
	//reset simulation
	frameCounter_ = 0; frameData_ = nullptr;
	if (gridSimulation_)
	{
		gridSimulation_->reset();
		volumeVis_->update(gridSimulation_->getState());
	}
	//reset plots
	reconstructionResetPlots();
}

void ActionReconstructionApp::reconstructionSolve()
{
	if (worker_->isDone())
	{
		//setup plots
		reconstructionResetPlots();
		costPlot->setMaxPoints(paramsReconstruction_.adjointSettings_.getSettings().numIterations_+1);
		for (const auto& e : plots)
		{
			std::get<0>(e)->setMaxPoints(paramsReconstruction_.adjointSettings_.getSettings().numIterations_+1);
			std::get<0>(e)->setTrueValue(std::get<1>(e)(paramsReconstruction_.simulationSettings_.getSettings()));
			std::get<0>(e)->addPoint(std::get<1>(e)(paramsReconstruction_.simulationSettings_.getSettings()));
		}
		//declare worker
		ar3d::BackgroundWorker2::task task = [this](ar3d::BackgroundWorker2* worker)
		{
			worker->setStatus("Solve: Prepare input and settings");
			ar3d::real timestep = 1.0 / (paramsInspectScan_.getConfig()->framerate * (paramsReconstruction_.costIntermediateSteps_ + 1));
			ar3d::SimulationResults3DPtr results = std::make_shared<ar3d::SimulationResults3D>();
			results->input_ = gridSimulation_->getInput();
			results->settings_ = gridSimulation_->getSettings();
			results->settings_.timestep_ = timestep;

			worker->setStatus("Solve: Load observations"); //for now, use all cameras
			std::vector<ar3d::real> timestepWeights;
			ar3d::CostFunctionPartialObservations::Observations observations;
			observations.gpuEvaluate_ = true;
			observations.maxSdf_ = 5;
			observations.noise_ = 0; //not needed because we already have the observations
			const auto& inputConfig = paramsInspectScan_.getConfig();
			observations.numCameras_ = inputConfig->cameras.size();
			observations.cameras_.resize(inputConfig->cameras.size());
			for (size_t i = 0; i < inputConfig->cameras.size(); ++i) {
				observations.cameras_[i] = inputConfig->cameras[i].camera;
			}
			int numSteps = paramsReconstruction_.costNumSteps_ == 0 ? inputConfig->duration-1 : paramsReconstruction_.costNumSteps_;
			for (int i = 1; i <= numSteps; ++i) {
				worker->setStatus(tinyformat::format("Solve: Load observation %d / %d", i, numSteps));
				//add in-between frames without weight
				for (int j = 0; j < paramsReconstruction_.costIntermediateSteps_; ++j) {
					timestepWeights.push_back(0);
					observations.observations_.emplace_back();
				}
				//load camera images and copy them to the GPU
				timestepWeights.push_back(1);
				ar3d::CostFunctionPartialObservations::Observation observation;
				observation.resize(inputConfig->cameras.size());
				const auto dataFrame = dataLoader_->loadFrame(i, worker);
				for (size_t j = 0; j < inputConfig->cameras.size(); ++j) {
					Eigen::Matrix<ar3d::real, Eigen::Dynamic, Eigen::Dynamic> host = dataFrame->cameraImages[j].depthMatrix.cast<ar3d::real>().matrix();
					observation[j] = ar3d::CostFunctionPartialObservations::Image::fromEigen(host);
				}
				observations.observations_.push_back(observation);
			}
			assert(timestepWeights.size() == observations.observations_.size());
			assert(timestepWeights.size() > 0);
			ar3d::CostFunctionPtr costFunction = std::make_shared<ar3d::CostFunctionPartialObservations>(timestepWeights, observations);

			worker->setStatus("Solve: Create AdjointSolver");
			ar3d::AdjointSolver::Settings adjointSettings = paramsReconstruction_.adjointSettings_.getSettings();
			adjointSettings.variables_.currentGravity_ = results->settings_.gravity_;
			adjointSettings.variables_.currentGroundPlane_ = results->settings_.groundPlane_;
			adjointSettings.variables_.currentMassDamping_ = results->settings_.dampingAlpha_;
			adjointSettings.variables_.currentMass_ = results->settings_.mass_;
			adjointSettings.variables_.currentPoissonRatio_ = results->settings_.poissonsRatio_;
			adjointSettings.variables_.currentStiffnessDamping_ = results->settings_.dampingBeta_;
			adjointSettings.variables_.currentYoungsModulus_ = results->settings_.youngsModulus_;
			ar3d::AdjointSolver solver(results, adjointSettings, costFunction);

			worker->setStatus("Solve: Solve it!");
			std::vector<ar3d::SoftBodySimulation3D::Settings> values;
			std::vector<ar3d::SoftBodySimulation3D::Settings> gradients;
			values.push_back(gridSimulation_->getSettings());
			{ar3d::SoftBodySimulation3D::Settings initialGrad; memset(&initialGrad, 0, sizeof(ar3d::SoftBodySimulation3D::Settings)); gradients.push_back(initialGrad); }
			ar3d::AdjointSolver::Callback_t callback = [this, &values, &gradients](const ar3d::SoftBodySimulation3D::Settings& var, const ar3d::SoftBodySimulation3D::Settings& grad, ar3d::real cost)
			{
				CI_LOG_I(var);
				CI_LOG_I(ar3d::CoordinateTransformation::cartesian2spherical(glm::double3(var.groundPlane_.x, var.groundPlane_.y, var.groundPlane_.z)));
				if (costPlot->getNumPoints()==0) costPlot->addPoint(cost);
				costPlot->addPoint(cost);
				for (const auto& e : plots)
					std::get<0>(e)->addPoint(std::get<1>(e)(var));
				values.push_back(var);
				gradients.push_back(grad);
			};
			solver.solve(callback, worker);

			//Done! Print steps
			std::stringstream ss;
			ss << "\nCost        ";
			for (const auto& e : plots) {
				if (std::get<2>(e)(paramsReconstruction_.adjointSettings_.getSettings().variables_))
					ss << std::get<0>(e)->getName() << " (gradient)    ";
			}
			ss << std::endl;
			for (int i = 1; i < values.size(); ++i)
			{
				ss << std::fixed << std::setw(12) << std::setprecision(7) << costPlot->getPoint(i) << "  ";
				for (const auto& e : plots) {
					if (std::get<2>(e)(paramsReconstruction_.adjointSettings_.getSettings().variables_)) {
						ss << std::fixed << std::setw(12) << std::setprecision(7) << std::get<1>(e)(values[i-1])
						   << " (" << std::fixed << std::setw(12) << std::setprecision(7) << std::get<1>(e)(gradients[i]) << ")  ";
					}
				}
				ss << std::endl;
			}
			CI_LOG_I(ss.str());
		};
		worker_->launch(task);
	}
}

void ActionReconstructionApp::reconstructionTestGradient()
{
	if (worker_->isDone())
	{
		ar3d::BackgroundWorker2::task task = [this](ar3d::BackgroundWorker2* worker)
		{
			worker->setStatus("Solve: Prepare input and settings");
			ar3d::real timestep = 1.0 / (paramsInspectScan_.getConfig()->framerate * (paramsReconstruction_.costIntermediateSteps_ + 1));
			ar3d::SimulationResults3DPtr results = std::make_shared<ar3d::SimulationResults3D>();
			results->input_ = gridSimulation_->getInput();
			results->settings_ = gridSimulation_->getSettings();
			results->settings_.timestep_ = timestep;

			worker->setStatus("Solve: Load observations"); //for now, use all cameras
			std::vector<ar3d::real> timestepWeights;
			ar3d::CostFunctionPartialObservations::Observations observations;
			observations.noise_ = 0; //not needed because we already have the observations
			const auto& inputConfig = paramsInspectScan_.getConfig();
			observations.numCameras_ = inputConfig->cameras.size();
			observations.cameras_.resize(inputConfig->cameras.size());
			for (size_t i = 0; i < inputConfig->cameras.size(); ++i) {
				observations.cameras_[i] = inputConfig->cameras[i].camera;
			}
			int numSteps = paramsReconstruction_.costNumSteps_ == 0 ? inputConfig->duration : paramsReconstruction_.costNumSteps_;
			for (int i = 0; i < numSteps; ++i) {
				worker->setStatus(tinyformat::format("Solve: Load observation %d / %d", (i + 1), numSteps));
				//load camera images and copy them to the GPU
				timestepWeights.push_back(1);
				ar3d::CostFunctionPartialObservations::Observation observation;
				observation.resize(inputConfig->cameras.size());
				const auto dataFrame = dataLoader_->loadFrame(i, worker);
				for (size_t j = 0; j < inputConfig->cameras.size(); ++j) {
					Eigen::Matrix<ar3d::real, Eigen::Dynamic, Eigen::Dynamic> host = dataFrame->cameraImages[j].depthMatrix.cast<ar3d::real>().matrix();
					observation[j] = ar3d::CostFunctionPartialObservations::Image::fromEigen(host);
				}
				observations.observations_.push_back(observation);
				//add in-between frames without weight
				if (i < numSteps - 1) {
					for (int j = 0; j < paramsReconstruction_.costIntermediateSteps_; ++j) {
						timestepWeights.push_back(0);
						observations.observations_.emplace_back();
					}
				}
			}
			assert(timestepWeights.size() == observations.observations_.size());
			assert(timestepWeights.size() > 0);
			ar3d::CostFunctionPtr costFunction = std::make_shared<ar3d::CostFunctionPartialObservations>(timestepWeights, observations);

			worker->setStatus("Solve: Create AdjointSolver");
			ar3d::AdjointSolver::Settings adjointSettings = paramsReconstruction_.adjointSettings_.getSettings();
			adjointSettings.variables_.currentGravity_ = results->settings_.gravity_;
			adjointSettings.variables_.currentGroundPlane_ = results->settings_.groundPlane_;
			adjointSettings.variables_.currentMassDamping_ = results->settings_.dampingAlpha_;
			adjointSettings.variables_.currentMass_ = results->settings_.mass_;
			adjointSettings.variables_.currentPoissonRatio_ = results->settings_.poissonsRatio_;
			adjointSettings.variables_.currentStiffnessDamping_ = results->settings_.dampingBeta_;
			adjointSettings.variables_.currentYoungsModulus_ = results->settings_.youngsModulus_;
			ar3d::AdjointSolver solver(results, adjointSettings, costFunction);

			solver.testGradient(worker);

		};
		worker_->launch(task);
	}
}

void ActionReconstructionApp::updateViewResult()
{
}

void ActionReconstructionApp::draw()
{
    using namespace ar::utils;
	if (paramsGeneral_.printMode_)
		cinder::gl::clear(cinder::Color(1, 1, 1));
	else
		cinder::gl::clear(cinder::Color(0, 0, 0));
	
    // WORLD SPACE
    ci::gl::enableDepthRead();
    ci::gl::enableDepthWrite();
	ci::gl::setMatrices(camera_);

	switch (currentState_)
	{
	case State::InspectScan: drawInspectScan(); break;
	case State::Reconstruction: drawReconstruction(); break;
	case State::ViewResult: drawViewResult(); break;
	}

    // WINDOW SPACE
    ci::gl::disableDepthRead();
    ci::gl::disableDepthWrite();
    ci::gl::setMatricesWindow(getWindowSize(), true);

    // Draw the background worker's status
    if (worker_ && !worker_->isDone()) {
        //draw waiting animation
        {
            ci::gl::ScopedModelMatrix scopedMatrix;
            ci::gl::ScopedColor scopedColor;
            ci::gl::translate(25, getWindowHeight() - 50);
            int step; double dummy; step = static_cast<int>(std::modf(getElapsedSeconds(), &dummy) * 8);
            for (int i = 0; i < 8; ++i) {
                float c = ((i + step)%8) / 7.0f;
                ci::gl::color(c, c, c);
                ci::gl::drawSolidRoundedRect(ci::Rectf(5, -2, 15, 2), 2);
                ci::gl::rotate(-2.0f * M_PI / 8.0f);
            }
        }
        //draw status
		cinder::gl::drawString(worker_->getStatus(), glm::vec2(50, getWindowHeight() - 50), paramsGeneral_.printMode_ ? cinder::ColorA(0, 0, 0) : cinder::ColorA(1, 1, 1));
    }

	// Draw the interface
	paramsGeneralInterface_->draw();
	paramsInspectScanInterface_->draw();
	paramsReconstructionInterface_->draw();
	paramsViewResultInterface_->draw();
    tfe_->draw();
}

void ActionReconstructionApp::drawGroundTruth(int frame)
{
	if (frame >= paramsInspectScan_.getConfig()->duration) return;
	if (paramsInspectScan_.getConfig()->groundTruth) {
		//ground truth is a ball
		ci::gl::ScopedGlslProg glslScope(ci::gl::getStockShader(ci::gl::ShaderDef().texture()));
		ci::gl::ScopedTextureBind texScope(floorTexture_);
		ci::gl::ScopedModelMatrix scopedMatrix;
		const ar3d::InputGroundTruth& groundTruth = *(paramsInspectScan_.getConfig()->groundTruth);
		ci::gl::translate(ar::utils::toGLM(groundTruth.locations[frame]));
		ci::gl::rotate(ar::utils::toGLM(groundTruth.rotations[frame]));
		ci::gl::drawSphere(ci::vec3(0, 0, 0), groundTruth.radius, 16);
	}
	else if (paramsInspectScan_.getConfig()->groundTruthMeshes)
	{
		static int oldFrame = -1;
		if (oldFrame != frame) {
			oldFrame = frame;
			groundTruthVboMesh_ = nullptr;
		}
		//ground truth is a mesh
		if (!groundTruthVboMesh_)
		{
			//ObjLoader loader(loadFile(path + "/groundTruth/frame" + std::to_string(frame) + ".obj"));
			//groundTruthVboMesh = ci::gl::VboMesh::create(loader);
			ci::TriMeshRef triMesh = ObjectLoader::loadCustomObj(paramsInspectScan_.getConfig()->getPathToGroundTruth(frame));
			groundTruthVboMesh_ = ci::gl::VboMesh::create(*triMesh);
		}
		ci::gl::ScopedGlslProg glslScope(groundTruthShader_);
		ci::gl::ScopedModelMatrix scopedMatrix;
		ci::gl::draw(groundTruthVboMesh_);
	}
}

void ActionReconstructionApp::drawProjCameraPoints(int frame)
{
	if (frame >= paramsInspectScan_.getConfig()->duration) return;
	static int oldFrame = -1;
	if (oldFrame != frame) {
		oldFrame = frame;
		projCamPoints_ = this->meshReconstruction_->getProjectedPoints(paramsInspectScan_.getConfig(), dataLoader_->loadFrame(frame));
	}
	ci::gl::ScopedColor col(1, 0, 1);
	//pointsBatch_->draw();
	for (const ci::vec3& p : projCamPoints_) {
		ci::gl::drawCube(p, ci::vec3(0.001, 0.001, 0.001));
	}
}

void ActionReconstructionApp::drawCameras(int frame)
{
	if (frameData_ == nullptr &&
		frame < paramsInspectScan_.getConfig()->duration && 
		dataLoader_ != nullptr &&
		(paramsGeneral_.showCameraMode_ == ParamsGeneral::ShowCameraMode::Color || paramsGeneral_.showCameraMode_ == ParamsGeneral::ShowCameraMode::Depth) &&
		worker_->isDone())
	{
		//load frame data
		ar3d::BackgroundWorker2::task task = [this, frame](ar3d::BackgroundWorker2* worker)
		{
			auto frameDataTmp = this->dataLoader_->loadFrame(frame, worker);
			if (worker->isInterrupted()) return;
			this->frameData_ = frameDataTmp;
		};
		worker_->launch(task);
	}

	for (size_t i = 0; i < paramsInspectScan_.getConfig()->cameras.size(); ++i) {
		const ar3d::DataCamera& c = paramsInspectScan_.getConfig()->cameras[i].camera;
		//optional: show current frame
		if (frameData_ != nullptr)
		{
			ci::gl::ScopedModelMatrix scopedMatrix;
			ci::gl::multModelMatrix(c.invViewProjMatrix);
			//ci::gl::translate(0, 0, -1);
			ci::gl::translate(0, 0, paramsInspectScan_.getConfig()->viewCameraImageTranslation);
			ci::gl::scale(1, -1, 1);
			if (paramsGeneral_.showCameraMode_ == ParamsGeneral::ShowCameraMode::Color) {
				ci::gl::draw(frameData_->cameraImages[i].getColorTexture(), ci::Rectf(-1, -1, 1, 1));
			}
			else if (paramsGeneral_.showCameraMode_ == ParamsGeneral::ShowCameraMode::Depth) {
				ci::gl::draw(frameData_->cameraImages[i].getDepthTexture(), ci::Rectf(-1, -1, 1, 1));
			}
		}
		//control points
		ci::gl::ScopedColor scopedColor;
		ci::gl::color(0.8, 0.2, 0.2);
		double dist = paramsInspectScan_.getConfig()->viewCameraNearPlane; // = 0.0; //Near plane
		ci::gl::drawSphere(c.location, 0.03f, 16);
		ci::gl::drawSphere(c.getWorldCoordinates(glm::vec3(0, 0, dist)), 0.02f, 16);
		ci::gl::drawSphere(c.getWorldCoordinates(glm::vec3(1, 0, dist)), 0.02f, 16);
		ci::gl::drawSphere(c.getWorldCoordinates(glm::vec3(0, 1, dist)), 0.02f, 16);
		ci::gl::drawSphere(c.getWorldCoordinates(glm::vec3(1, 1, dist)), 0.02f, 16);
		//lines
		ci::gl::color(0.8, 0.4, 0.4);
		ci::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(0, 0, dist)));
		ci::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(1, 0, dist)));
		ci::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(0, 1, dist)));
		ci::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(1, 1, dist)));
		ci::gl::drawLine(c.getWorldCoordinates(glm::vec3(0, 0, dist)), c.getWorldCoordinates(glm::vec3(1, 0, dist)));
		ci::gl::drawLine(c.getWorldCoordinates(glm::vec3(0, 0, dist)), c.getWorldCoordinates(glm::vec3(0, 1, dist)));
		ci::gl::drawLine(c.getWorldCoordinates(glm::vec3(1, 1, dist)), c.getWorldCoordinates(glm::vec3(1, 0, dist)));
		ci::gl::drawLine(c.getWorldCoordinates(glm::vec3(1, 1, dist)), c.getWorldCoordinates(glm::vec3(0, 1, dist)));
	}
}

void ActionReconstructionApp::drawInspectScan()
{
	if (paramsInspectScan_.getConfig() == nullptr) return;

	//WORLD SPACE
	//if (/*showFloor*/ true) {
	//	ci::gl::ScopedGlslProg glslScope(ci::gl::getStockShader(ci::gl::ShaderDef().texture()));
	//	ci::gl::ScopedTextureBind texScope(floorTexture_);
	//	ci::gl::draw(floorVboMesh_);
	//}

	if (paramsGeneral_.showGroundTruth_) {
		drawGroundTruth(paramsInspectScan_.frame_);
	}

	if (paramsGeneral_.showCameraMode_!=ParamsGeneral::ShowCameraMode::Off) {
		drawCameras(paramsInspectScan_.frame_);
	}

	if (paramsGeneral_.showBoundingBox_ && meshReconstruction_) {
		ci::gl::ScopedColor scopedColor;
		//draw the camera pyramids
		ci::gl::color(0.5, 0.5, 1.0);
		for (const ar::geom3d::Pyramid& p : meshReconstruction_->getMaskPyramids()) {
			for (const auto& v : p.edgeRays) {
				ci::gl::drawLine(ar::utils::toGLM(p.center), ar::utils::toGLM((v * 5 + p.center).eval()));
			}
		}
		if (meshReconstruction_->getBoundingBox().isValid()) {
			//draw the bounding box
			ci::gl::color(0.2, 0.2, 1.0);
			drawWireCube(meshReconstruction_->getBoundingBox().min, meshReconstruction_->getBoundingBox().max);
		}
	}

	if (meshReconstruction_) 
	{
		ar3d::WorldGridDoubleDataPtr sdf = referenceSdf_;//meshReconstruction_->getSdfData();
		if (sdf) {
			//visualizate the SDF
			ar3d::WorldGridPtr grid = meshReconstruction_->getWorldGrid();
			ci::gl::Texture3dRef tex = sdf->getTexture(ar3d::WorldGridData<double>::DataSource::HOST);

			//update volume vis and render
			ar3d::SoftBodyGrid3D::Input input;
			input.grid_ = grid;
			input.referenceSdf_ = sdf;
			volumeVis_->setInput(input);
			volumeVis_->setTransferFunction(tfe_->getTexture(), tfe_->getRangeMin(), tfe_->getRangeMax());
			volumeVis_->draw();
			if (animationTakeScreenshot_)
				animationRendered_ = true;
		}
	}

	if (paramsGeneral_.viewPoints_) {
		drawProjCameraPoints(paramsInspectScan_.frame_);
	}
}

void ActionReconstructionApp::drawReconstruction()
{
	if (!gridSimulation_) return;

	//WORLD SPACE
	
	//ground plane
	if (paramsReconstruction_.simulationSettings_.getSettings().enableCollision_) {
		cinder::gl::ScopedModelMatrix m;
		glm::vec3 ref(0, 1, 0);
		glm::vec3 n(
			paramsReconstruction_.simulationSettings_.getSettings().groundPlane_.x, 
			paramsReconstruction_.simulationSettings_.getSettings().groundPlane_.y,
			paramsReconstruction_.simulationSettings_.getSettings().groundPlane_.z);
		cinder::gl::rotate(acos(dot(ref, n)), cross(ref, n));
		cinder::gl::translate(0, paramsReconstruction_.simulationSettings_.getSettings().groundPlane_.w, 0);
		cinder::gl::ScopedGlslProg glslScope(cinder::gl::getStockShader(cinder::gl::ShaderDef().texture()));
		cinder::gl::ScopedTextureBind texScope(floorTexture_);
		cinder::gl::draw(floorVboMesh_);
	}

	//dirichlet boundaries
	{
		cinder::gl::ScopedColor c;
		cinder::gl::color(0, 0, 1, 1);
		if (paramsReconstruction_.inputSettings_.getSettings().enableDirichlet)
		{
			drawWireCube(paramsReconstruction_.inputSettings_.getSettings().centerDirichlet - paramsReconstruction_.inputSettings_.getSettings().halfsizeDirichlet,
				paramsReconstruction_.inputSettings_.getSettings().centerDirichlet + paramsReconstruction_.inputSettings_.getSettings().halfsizeDirichlet);
		}
	}

	//reference grid bounds
	if (paramsGeneral_.showBoundingBox_) {
		ar3d::WorldGridPtr grid = gridSimulation_->getState().advectedSDF_ != nullptr ? gridSimulation_->getState().advectedSDF_->getGrid() : gridSimulation_->getInput().grid_;
		cinder::gl::ScopedColor scopedColor;
		cinder::gl::color(0.2, 1.0, 0.2);
		Eigen::Vector3d voxelSize(grid->getVoxelSize(), grid->getVoxelSize(), grid->getVoxelSize());
		Eigen::Vector3d minCorner = (grid->getOffset()).cast<double>().array() * voxelSize.array();
		Eigen::Vector3d maxCorner = (grid->getOffset() + grid->getSize() + Eigen::Vector3i(1, 1, 1)).cast<double>().array() * voxelSize.array();
		drawWireCube(minCorner, maxCorner);
	}
	//advected grid bounds
	if (paramsGeneral_.showBoundingBox_) {
		cinder::gl::ScopedColor scopedColor;
		cinder::gl::color(1.0, 0.2, 0.2);
		const Eigen::Vector3d& minCorner = gridSimulation_->getState().advectedBoundingBox_.min;
		const Eigen::Vector3d& maxCorner = gridSimulation_->getState().advectedBoundingBox_.max;
		drawWireCube(minCorner, maxCorner);
	}

	if (paramsGeneral_.showGroundTruth_) {
		drawGroundTruth(frameCounter_);
	}

	if (paramsGeneral_.showCameraMode_ != ParamsGeneral::ShowCameraMode::Off) {
		drawCameras(frameCounter_);
	}

	//partial observation vis
	if (visualizePartialObservations_) {
		observationVis_.draw();
	}

	//main vis
	volumeVis_->setTransferFunction(tfe_->getTexture(), tfe_->getRangeMin(), tfe_->getRangeMax());
	volumeVis_->draw();
	if (animationTakeScreenshot_)
		animationRendered_ = true;

	//points
	if (paramsGeneral_.viewPoints_) {
		drawProjCameraPoints(frameCounter_/(paramsReconstruction_.costIntermediateSteps_+1));
	}

	// WINDOW SPACE
	cinder::gl::disableDepthRead();
	cinder::gl::disableDepthWrite();
	cinder::gl::setMatricesWindow(getWindowSize(), true);

	//draw plots
	{
		const int offset = 10;
		const int width = 0.2 * getWindowWidth();
		int numPlots = 1;
		for (const auto& e : plots) if (std::get<2>(e)(paramsReconstruction_.adjointSettings_.getSettings().variables_)) numPlots++;
		numPlots = std::max(4, numPlots);
		const int height = getWindowHeight() / numPlots;
		costPlot->setBoundingRect(cinder::Rectf(getWindowWidth() - width, offset, getWindowWidth() - offset, height - offset));
		costPlot->setPrintMode(paramsGeneral_.printMode_);
		costPlot->draw();
		int y = 1;
		for (const auto& e : plots)
		{
			if (!std::get<2>(e)(paramsReconstruction_.adjointSettings_.getSettings().variables_)) continue;
			std::get<0>(e)->setBoundingRect(cinder::Rectf(getWindowWidth() - width, y * height + offset, getWindowWidth() - offset, (y + 1) * height - offset));
			std::get<0>(e)->setPrintMode(paramsGeneral_.printMode_);
			std::get<0>(e)->draw();
			y++;
		}
	}
}

void ActionReconstructionApp::drawViewResult()
{
}

void ActionReconstructionApp::nextReconstruction()
{
	//check if we are ready to continue
	if (!worker_->isDone() || !meshReconstruction_) {
		CI_LOG_E("Input Reconstruction not completed yet");
		return;
	}

	//pass the reference SDF to the next stage
	auto s = paramsReconstruction_.simulationSettings_.getSettings();
	s.timestep_ = 1.0 / paramsInspectScan_.getConfig()->framerate;
	paramsReconstruction_.simulationSettings_.setSettings(s);

	//initialize forward simulation
	paramsReconstruction_.setWorldGrid(referenceSdf_->getGrid());

	//update UI
	currentState_ = State::Reconstruction;
	paramsInspectScanInterface_->show(false);
	paramsReconstructionInterface_->show(true);
}

void ActionReconstructionApp::prevInspectScan()
{
	//TODO
}

void ActionReconstructionApp::nextViewResult()
{
	//TODO
}

void ActionReconstructionApp::prevReconstruction()
{
	//TODO
}

void ActionReconstructionApp::cleanup()
{
    if (worker_) {
        worker_->interrupt();
		worker_->wait();
    }
	gridSimulation_.reset();
	dataLoader_.reset();
	frameData_.reset();
	referenceSdf_.reset();
	cudaPrintfEnd();
    App::quit();
}

void ActionReconstructionApp::drawWireCube(const Eigen::Vector3d& ea, const Eigen::Vector3d& eb)
{
    glm::vec3 a = utils::toGLM(ea);
    glm::vec3 b = utils::toGLM(eb);
    ci::gl::drawLine(a, glm::vec3(a.x, a.y, b.z));
    ci::gl::drawLine(a, glm::vec3(a.x, b.y, a.z));
    ci::gl::drawLine(a, glm::vec3(b.x, a.y, a.z));
    ci::gl::drawLine(glm::vec3(a.x, b.y, b.z), glm::vec3(a.x, a.y, b.z));
    ci::gl::drawLine(glm::vec3(b.x, a.y, b.z), glm::vec3(a.x, a.y, b.z));
    ci::gl::drawLine(glm::vec3(a.x, b.y, b.z), glm::vec3(a.x, b.y, a.z));
    ci::gl::drawLine(glm::vec3(b.x, b.y, a.z), glm::vec3(a.x, b.y, a.z));
    ci::gl::drawLine(glm::vec3(b.x, b.y, a.z), glm::vec3(b.x, a.y, a.z));
    ci::gl::drawLine(glm::vec3(b.x, a.y, b.z), glm::vec3(b.x, a.y, a.z));
    ci::gl::drawLine(b, glm::vec3(a.x, b.y, b.z));
    ci::gl::drawLine(b, glm::vec3(b.x, b.y, a.z));
    ci::gl::drawLine(b, glm::vec3(b.x, a.y, b.z));
}

void ActionReconstructionApp::drawWireCube(const ar3d::real3 & a, const ar3d::real3 & b)
{
	drawWireCube(Eigen::Vector3d(a.x, a.y, a.z), Eigen::Vector3d(b.x, b.y, b.z));
}

void ActionReconstructionApp::load()
{
	cinder::fs::path initialPath = "";
	cinder::fs::path path = getOpenFilePath(initialPath, std::vector<std::string>({ "json" }));
	if (path.empty())
	{
		CI_LOG_I("loading cancelled");
		return;
	}
	cinder::DataSourceRef source = cinder::loadFile(path);
	if (!source)
	{
		CI_LOG_E("Unable to load file " << path.string());
		return;
	}
	cinder::JsonTree root;
	try
	{
		root = cinder::JsonTree(source);
	}
	catch (const cinder::JsonTree::ExcJsonParserError& ex)
	{
		CI_LOG_E("Unable to load json file " << path.string() << ": " << ex.what());
		return;
	}

	paramsGeneral_.load(root.getChild("General"));
	paramsInspectScan_.load(root.getChild("InspectScan"));
	paramsReconstruction_.load(root.getChild("Reconstruction"));
	paramsViewResult_.load(root.getChild("ViewResult"));
}

void ActionReconstructionApp::save()
{
	cinder::fs::path initialPath = "";
	cinder::fs::path path = getSaveFilePath(initialPath, std::vector<std::string>({ "json" }));
	if (path.empty())
	{
		CI_LOG_I("saving cancelled");
		return;
	}
	path.replace_extension("json");

	cinder::JsonTree root = cinder::JsonTree::makeObject();

	{
		cinder::JsonTree child = cinder::JsonTree::makeObject("General");
		paramsGeneral_.save(child);
		root.addChild(child);
	}
	{
		cinder::JsonTree child = cinder::JsonTree::makeObject("InspectScan");
		paramsInspectScan_.save(child);
		root.addChild(child);
	}
	{
		cinder::JsonTree child = cinder::JsonTree::makeObject("Reconstruction");
		paramsReconstruction_.save(child);
		root.addChild(child);
	}
	{
		cinder::JsonTree child = cinder::JsonTree::makeObject("ViewResult");
		paramsViewResult_.save(child);
		root.addChild(child);
	}

	root.write(path);
	CI_LOG_I("Saved to " << path.string());
}

#if 1
CINDER_APP( ActionReconstructionApp, RendererGl(RendererGl::Options().msaa(4)), [&](App::Settings *settings)
{
	settings->setWindowSize(1600, 900);
} )
#endif
