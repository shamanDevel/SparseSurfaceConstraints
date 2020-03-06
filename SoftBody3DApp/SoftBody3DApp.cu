#include "SoftBody3DApp.h"
#include <Commons3D.h>

#include <cmath>
#include <chrono>
#include <thread>
#include <time.h>
#include <fstream>

#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>
#include <cinder/CameraUi.h>
#include <cinder/params/Params.h>
#include <cinder/Log.h>
#include <cinder/ObjLoader.h>
#include <cinder/Json.h>

#include "../MainApp/resources/Resources.h"

#include <SoftBodyMesh3D.h>
#include <SoftBodyGrid3D.h>

#include <Utils.h>
#include <DataCamera.h>
#include <BackgroundWorker.h>
#include <TransferFunctionEditor.h>
#include <VolumeVisualization.h>
#include <TetMeshVisualization.h>
#include <BackgroundWorker2.h>
#include <cuPrintf.cuh>
#include <AdjointSolver.h>
#include <CostFunctions.h>
#include "GraphPlot.h"
#include "CoordinateTransformation.h"
#include "tinyformat.h"

using namespace ci::app;
using namespace std;
using namespace ar3d;

SoftBody3DApp::SoftBody3DApp()
{
	worker = std::make_unique<BackgroundWorker2>();
	ci::log::makeLogger<ci::log::LoggerFile>(getAppPath().string() + "SoftBody3DApp.log", true);

	//showFloor = true;
	printMode = false;
	simulationMode_ = SimulationMode::Grid;
	inputCase_ = InputCase::Bar;
	showCamera = 2;
    showBoundingBox = true;
    showVoxelGrid = true;
	visualizedSdf = 1;

    SoftBodySimulation3D::InputBarSettings inputBarSettings;
	inputBarSettings.resolution = 2;
	inputBarSettings.center = make_real3(0, 1, 0);
	inputBarSettings.halfsize = make_real3(1.29, 0.4, 0.8);
	inputBarSettings.enableDirichlet = true;
	inputBarSettings.centerDirichlet = make_real3(-1.0, 1.0, 0.0);
	inputBarSettings.halfsizeDirichlet = make_real3(0.1, 0.5, 0.5);
    inputBarSettings.diffusionDistance = 5;
    inputBarSettings.sampleIntegrals = false;
    inputBarSettings_.setSettings(inputBarSettings);

    SoftBodySimulation3D::InputTorusSettings inputTorusSettings;
	inputTorusSettings.resolution = 5;
	inputTorusSettings.center = make_real3(0, 1.5, 0);
	inputTorusSettings.orientation = glm::normalize(glm::vec3(2, 1, -0.05));
	inputTorusSettings.innerRadius = 0.4;
	inputTorusSettings.outerRadius = 1.1;
	inputTorusSettings.enableDirichlet = true;
	inputTorusSettings.centerDirichlet = make_real3(-0.5, 2.8, 0.0);
	inputTorusSettings.halfsizeDirichlet = make_real3(0.4, 0.4, 0.4);
    inputTorusSettings.diffusionDistance = 5;
    inputTorusSettings.sampleIntegrals = false;
    inputTorusSettings_.setSettings(inputTorusSettings);

    SoftBodySimulation3D::InputSdfSettings inputSdfSettings;
    inputSdfSettings.file = "";
    inputSdfSettings.enableDirichlet = false;
    inputSdfSettings.centerDirichlet = make_real3(-0.5, 2.8, 0.0);
    inputSdfSettings.halfsizeDirichlet = make_real3(0.4, 0.4, 0.4);
    inputSdfSettings.diffusionDistance = 5;
    inputSdfSettings.sampleIntegrals = false;
    inputSdfSettings_.setSettings(inputSdfSettings);

    SoftBodySimulation3D::Settings simulationSettings;
	simulationSettings.gravity_ = make_real3(0, -10, 0);
	simulationSettings.youngsModulus_ = 2000;
	simulationSettings.poissonsRatio_ = 0.45;
	simulationSettings.mass_ = 1;
	simulationSettings.dampingAlpha_ = 0.1;
	simulationSettings.dampingBeta_ = 0.01;
	simulationSettings.enableCorotation_ = true;
	simulationSettings.timestep_ = 0.01;
	simulationSettings.groundPlane_ = make_real4(0, 1, 0, 0);
	simulationSettings.enableCollision_ = false;
	simulationSettings.groundStiffness_ = 10000;
	simulationSettings.softmaxAlpha_ = 100;
	simulationSettings.solverIterations_ = 100;
	simulationSettings.solverTolerance_ = 1e-5;
    simulationSettings_.setSettings(simulationSettings);

	costFunctionMode_ = CostFunctionMode::PartialObservations;
	results = std::make_shared<SimulationResults3D>();

	enableTimings = false;
    animationTakeScreenshot = false;
    frameCounter = 0;
}

SoftBody3DApp::~SoftBody3DApp()
{

}

void SoftBody3DApp::setup()
{
	CUMAT_SAFE_CALL(cudaPrintfInit());

	//parameter ui, must happen before user-camera
	params = cinder::params::InterfaceGl::create(getWindow(), "Parameters", toPixels(glm::ivec2(350, 800)));
	params->setOptions("", "refresh=0.05");
	params->addParam("PrintMode", &printMode).label("Print Mode");
    params->addButton("Save", [this]() {save(); });
    params->addButton("Load", [this]() {load(); });

    //transfer function editor
    tfe = std::make_unique<ar::TransferFunctionEditor>(getWindow());

    //visualization
    volumeVis = std::make_unique<ar3d::VolumeVisualization>(getWindow(), &camera, &volumeVisParams);
	tetVis = std::make_unique<TetMeshVisualization>();

    //Add Params
	std::vector<std::string> simulationModeNames = { "Tet", "Grid" };
	params->addParam("SimulationMode", simulationModeNames, 
		std::function<void(int)>([this](int v)
	{
		simulationMode_ = SimulationMode(v);
		simulationReset();
		updateInput0();
	}), std::function<int()>([this]()
	{
		return int(simulationMode_);
	})).label("Simulation Mode");
	//input
	std::vector<std::string> inputCaseNames = { "Bar", "Torus", "Sdf-File" };
	const auto updateInputVisibility = [this](int v)
	{
		inputCase_ = InputCase(v);
        inputBarSettings_.setVisible(params, inputCase_ == InputCase::Bar);
        inputTorusSettings_.setVisible(params, inputCase_ == InputCase::Torus);
        inputSdfSettings_.setVisible(params, inputCase_ == InputCase::File);
	};
	params->addParam("InputCase", inputCaseNames, 
		updateInputVisibility, std::function<int()>([this]()
	{
		return int(inputCase_);
	})).label("Input Case");
    inputBarSettings_.initParams(params, "Input");
    inputTorusSettings_.initParams(params, "Input");
    inputSdfSettings_.initParams(params, "Input");
	updateInputVisibility(int(inputCase_));
	//elasticity settings
    simulationSettings_.initParams(params, "Elasticity");
	params->addButton("Elasticity-Reset", [this]() {this->simulationReset(); }, "group=Elasticity label='Reset (r)'");
	params->addButton("Elasticity-SolveStatic", [this]() {this->simulationStaticStep(); }, "group=Elasticity label='Solve Static' ");
	params->addButton("Elasticity-SolveDynamic", [this]() {this->simulationDynamicStep(); }, "group=Elasticity label='Solve Dynamic (space)'");
	//adjoint
	params->addParam("Adjoint-Enable", &enableAdjoint_).group("Adjoint").label("Enable Adjoint");
    params->addParam("Adjoint-Timesteps", (int*)(nullptr), true)
        .accessors(std::function<void(int)>([](int) {}), std::function<int()>([this] {return results ? results->states_.size() : 0; }))
        .label("# Timesteps").group("Adjoint");
	adjointSettings_.initParams(params, "Adjoint");
	std::vector<std::string> costFunctionEnums = { "Direct", "Partial Observations" };
	params->addParam("Adjoint-CostFunction", costFunctionEnums, reinterpret_cast<int*>(&costFunctionMode_)).group("Adjoint").label("Cost Function").accessors(
	[this](int val)
	{
		costFunctionMode_ = CostFunctionMode(val);
		costFunctionActiveDisplacements_.setVisible(params, costFunctionMode_ == CostFunctionMode::DirectActive);
        costFunctionPartialObservations_.setVisible(params, costFunctionMode_ == CostFunctionMode::PartialObservations);
	}, [this]()
	{
		return int(costFunctionMode_);
	});
	costFunctionActiveDisplacements_.initParams(params, "Adjoint");
	costFunctionActiveDisplacements_.setVisible(params, costFunctionMode_ == CostFunctionMode::DirectActive);
    costFunctionPartialObservations_.initParams(params, "Adjoint");
    costFunctionPartialObservations_.setVisible(params, costFunctionMode_ == CostFunctionMode::PartialObservations);
	params->addButton("Adjoint-Solve", [this]() {this->solveAdjoint(); }, " group=Adjoint label=Solve key=Return ");
    params->addButton("Adjoint-TestGradient", [this]() {this->testGradient(); }, " group=Adjoint label='Test Gradient' ");
	params->addButton("Adjoint-Cancel", [this]() {this->cancel(); }, " group=Adjoint label=Cancel");
	params->addButton("Adjoint-ExportObservations", [this]() {this->exportObservations(); }, " group=Adjoint label='Test Export Observations' ");
	//rendering
	//std::vector<std::string> showCameraEnums = { "hide", "show wireframe", "show images" };
	//params->addParam("Rendering-VisiblityCamera", showCameraEnums, &showCamera).group("Rendering").label("Show Cameras");
 //   params->addParam("Rendering-ShowSegmentation", &showSegmentation).group("Rendering").label("Show Segmentation");
    params->addParam("Rendering-ShowBoundingBox", &showBoundingBox).group("Rendering").label("Show Bounding Box");
 //   params->addParam("Rendering-ShowVoxelGrid", &showVoxelGrid).group("Rendering").label("Show Voxel Grid");
    params->addParam("Rendering-SaveFrameNames", &frameNames).group("Rendering").label("Frame Names");
	params->addParam("Rendering-ExportWithNormals", &exportWithNormals_).group("Rendering").label("Export \\w normals + orig.pos.");
	params->addParam("Rendering-exportEverNFrames_", &exportEverNFrames_).group("Rendering").label("Export ever N frames").min(1);
	params->addButton("Rendering-Reload", [this]() {this->reloadShaders(); }, "group=Rendering label='Reload Shaders (l)'");
    //main visualization
	volumeVisParams.addParams(params);
	//Tests
	params->addParam("Test-Timings", &enableTimings).label("Enable Timings").group("Test");

	//user-camera
	camUi = cinder::CameraUi(&camera, getWindow());
	glm::vec3 newEye = camera.getEyePoint() + camera.getViewDirection() * (camera.getPivotDistance() * (1 - 0.2f));
	camera.setEyePoint(newEye);
	camera.setPivotDistance(camera.getPivotDistance() * 0.2f);

	//floor
	auto plane = cinder::geom::Plane().size(glm::vec2(10, 10)).subdivisions(glm::ivec2(5, 5));
	vector<cinder::gl::VboMesh::Layout> bufferLayout = {
		cinder::gl::VboMesh::Layout().usage(GL_DYNAMIC_DRAW).attrib(cinder::geom::Attrib::POSITION, 3),
		cinder::gl::VboMesh::Layout().usage(GL_STATIC_DRAW).attrib(cinder::geom::Attrib::TEX_COORD_0, 2)
	};
	floorVboMesh = cinder::gl::VboMesh::create(plane, bufferLayout);
	cinder::gl::Texture::Format floorTextureFmt;
	floorTextureFmt.enableMipmapping(true);
	floorTextureFmt.setMinFilter(GL_LINEAR_MIPMAP_LINEAR);
	floorTexture = cinder::gl::Texture::create(loadImage(loadResource(CHECKERBOARD_IMAGE)), floorTextureFmt);

    //plots
    typedef AdjointSolver::InputVariables V;
    typedef SoftBodySimulation3D::Settings S;
    costPlot = std::make_unique<GraphPlot>("Cost");
    costPlot->setTrueValue(0);
    plots.emplace_back(std::make_unique<GraphPlot>("GravityX"),         [](const S& var) {return var.gravity_.x; }, [](const V& var) {return var.optimizeGravity_; });
    plots.emplace_back(std::make_unique<GraphPlot>("GravityY"),         [](const S& var) {return var.gravity_.y; }, [](const V& var) {return var.optimizeGravity_; });
    plots.emplace_back(std::make_unique<GraphPlot>("GravityZ"),         [](const S& var) {return var.gravity_.z; }, [](const V& var) {return var.optimizeGravity_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Young's Modulus"),  [](const S& var) {return var.youngsModulus_; }, [](const V& var) {return var.optimizeYoungsModulus_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Poisson Ratio"),    [](const S& var) {return var.poissonsRatio_; }, [](const V& var) {return var.optimizePoissonRatio_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Mass"),             [](const S& var) {return var.mass_; }, [](const V& var) {return var.optimizeMass_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Mass Damping"),     [](const S& var) {return var.dampingAlpha_; }, [](const V& var) {return var.optimizeMassDamping_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Stiffness Damping"),[](const S& var) {return var.dampingBeta_; }, [](const V& var) {return var.optimizeStiffnessDamping_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityX"), [](const S& var) {return var.initialLinearVelocity_.x; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityY"), [](const S& var) {return var.initialLinearVelocity_.y; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("LinearVelocityZ"), [](const S& var) {return var.initialLinearVelocity_.z; }, [](const V& var) {return var.optimizeInitialLinearVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityX"), [](const S& var) {return var.initialAngularVelocity_.x; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityY"), [](const S& var) {return var.initialAngularVelocity_.y; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
	plots.emplace_back(std::make_unique<GraphPlot>("AngularVelocityZ"), [](const S& var) {return var.initialAngularVelocity_.z; }, [](const V& var) {return var.optimizeInitialAngularVelocity_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Theta"), [](const S& var) {return CoordinateTransformation::cartesian2spherical(var.groundPlane_).y; }, [](const V& var) {return var.optimizeGroundPlane_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Phi"),   [](const S& var) {return CoordinateTransformation::cartesian2spherical(var.groundPlane_).z; }, [](const V& var) {return var.optimizeGroundPlane_; });
    plots.emplace_back(std::make_unique<GraphPlot>("Ground Plane Height"), [](const S& var) {return var.groundPlane_.w; }, [](const V& var) {return var.optimizeGroundPlane_; });

	updateInput();
}

void SoftBody3DApp::keyDown(KeyEvent event)
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
		cinder::Surface surface = copyWindowSurface();
		//construct filename
		time_t now = time(NULL);
		struct tm tstruct;
		char buf[100];
		localtime_s(&tstruct, &now);
		strftime(buf, sizeof(buf), "%d-%m-%Y_%H-%M-%S", &tstruct);
		string fileName = string("../screenshots/SoftBody3DApp-") + string(buf) + ".png";
		//write out
		writeImage(fileName, surface);
        CI_LOG_I("screenshot saved to " << fileName);
	} 
    else if (event.getChar() == 'l')
	{
		reloadShaders();
	}
	else if (event.getChar() == 'r')
	{
		simulationReset();
	}
	else if (event.getCode() == KeyEvent::KEY_SPACE)
	{
		spaceBarPressed_ = true;
	}
}

void SoftBody3DApp::keyUp(KeyEvent event)
{
	App::keyUp(event);
	if (event.getCode() == KeyEvent::KEY_SPACE && spaceBarPressed_)
	{
		spaceBarPressed_ = false;
		if (enableTimings) saveStatistics();
	}
}

void SoftBody3DApp::mouseDown( MouseEvent event )
{
	App::mouseDown(event);
}

void SoftBody3DApp::update()
{
    //save animation from previous frame
    if (animationTakeScreenshot && animationRendered && !frameNames.empty() && frameCounter%exportEverNFrames_==0) {
		int counter = frameCounter / exportEverNFrames_;
        //Screenshot
        cinder::Surface surface = copyWindowSurface();
        string fileName = tinyformat::format("../screenshots/%s%05d.png", frameNames.c_str(), counter);
        writeImage(fileName, surface);
		////SDF file
		//if (gridSimulation) {
		//	string fileName = tinyformat::format("../screenshots/%s%05d.sdf", frameNames.c_str(), counter);
		//	volumeVis->saveSdf(fileName);
		//}
		//Marching Cubes mesh file
		if (gridSimulation) {
			string fileName = tinyformat::format("../screenshots/%s%05d.obj", frameNames.c_str(), counter);
			volumeVis->saveMCMesh(fileName);
		}
		//High Resolution mesh file
		if (gridSimulation) {
			string fileName = tinyformat::format("../screenshots/%s_high%05d.obj", frameNames.c_str(), counter);
			volumeVis->saveHighResultMesh(fileName, exportWithNormals_, exportWithNormals_);
		}
		//done
        CI_LOG_I("screenshot saved to " << fileName);
    }
	if (animationTakeScreenshot && animationRendered) {
		animationRendered = false;
		animationTakeScreenshot = false;
	}

	updateInput();
	updateSettings();

	//dynamic step
	if (spaceBarPressed_)
		simulationDynamicStep();

    //update transfer function editor
    tfe->setVisible(volumeVis->needsTransferFunction());
    tfe->update();
}

void SoftBody3DApp::draw()
{
    using namespace ar::utils;
	if (printMode)
		cinder::gl::clear(cinder::Color(1, 1, 1));
	else
		cinder::gl::clear(cinder::Color(0, 0, 0));
	
    // WORLD SPACE
	cinder::gl::enableDepthRead();
	cinder::gl::enableDepthWrite();
	cinder::gl::setMatrices(camera);

	if (simulationSettings_.getSettings().enableCollision_) {
        cinder::gl::ScopedModelMatrix m;
		glm::vec3 ref(0, 1, 0);
		glm::vec3 n(simulationSettings_.getSettings().groundPlane_.x, simulationSettings_.getSettings().groundPlane_.y, simulationSettings_.getSettings().groundPlane_.z);
		cinder::gl::rotate(acos(dot(ref, n)), cross(ref, n));
		cinder::gl::translate(0, simulationSettings_.getSettings().groundPlane_.w, 0);
        //cinder::gl::rotate(
        //    glm::lookAt(
        //        glm::vec3(0,0,0),
        //        glm::vec3(simulationSettings_.groundPlane_.x, simulationSettings_.groundPlane_.y, simulationSettings_.groundPlane_.z),
        //        glm::vec3(1,0,0)));
		cinder::gl::ScopedGlslProg glslScope(cinder::gl::getStockShader(cinder::gl::ShaderDef().texture()));
		cinder::gl::ScopedTextureBind texScope(floorTexture);
		cinder::gl::draw(floorVboMesh);
	}

	//dirichlet boundaries
	{
		cinder::gl::ScopedColor c;
		cinder::gl::color(0, 0, 1, 1);
		if (inputCase_ == InputCase::Bar && inputBarSettings_.getSettings().enableDirichlet)
		{
			drawWireCube(inputBarSettings_.getSettings().centerDirichlet - inputBarSettings_.getSettings().halfsizeDirichlet, inputBarSettings_.getSettings().centerDirichlet + inputBarSettings_.getSettings().halfsizeDirichlet);
		}
		else if (inputCase_ == InputCase::Torus && inputTorusSettings_.getSettings().enableDirichlet)
		{
			drawWireCube(inputTorusSettings_.getSettings().centerDirichlet - inputTorusSettings_.getSettings().halfsizeDirichlet, inputTorusSettings_.getSettings().centerDirichlet + inputTorusSettings_.getSettings().halfsizeDirichlet);
		}
        else if (inputCase_ == InputCase::File && inputSdfSettings_.getSettings().enableDirichlet)
        {
            drawWireCube(inputSdfSettings_.getSettings().centerDirichlet - inputSdfSettings_.getSettings().halfsizeDirichlet, inputSdfSettings_.getSettings().centerDirichlet + inputSdfSettings_.getSettings().halfsizeDirichlet);
        }
	}

	//tet-simulation
	if (tetSimulation)
	{
		tetVis->draw();
	}

	if (gridSimulation)
	{
		if (showBoundingBox) { //reference grid bounds
			ar3d::WorldGridPtr grid = gridSimulation->getState().advectedSDF_ != nullptr ? gridSimulation->getState().advectedSDF_->getGrid() : gridSimulation->getInput().grid_;
			cinder::gl::ScopedColor scopedColor;
			cinder::gl::color(0.2, 1.0, 0.2);
			Eigen::Vector3d voxelSize(grid->getVoxelSize(), grid->getVoxelSize(), grid->getVoxelSize());
			Eigen::Vector3d minCorner = (grid->getOffset()).cast<double>().array() * voxelSize.array();
			Eigen::Vector3d maxCorner = (grid->getOffset() + grid->getSize() + Eigen::Vector3i(1, 1, 1)).cast<double>().array() * voxelSize.array();
			drawWireCube(make_real3(minCorner.x(), minCorner.y(), minCorner.z()), make_real3(maxCorner.x(), maxCorner.y(), maxCorner.z()));
		}
        if (showBoundingBox) { //advected grid bounds
			cinder::gl::ScopedColor scopedColor;
			cinder::gl::color(1.0, 0.2, 0.2);
			const Eigen::Vector3d& minCorner = gridSimulation->getState().advectedBoundingBox_.min;
			const Eigen::Vector3d& maxCorner = gridSimulation->getState().advectedBoundingBox_.max;
			drawWireCube(make_real3(minCorner.x(), minCorner.y(), minCorner.z()), make_real3(maxCorner.x(), maxCorner.y(), maxCorner.z()));
		}
		//main vis
		volumeVis->setTransferFunction(tfe->getTexture(), tfe->getRangeMin(), tfe->getRangeMax());
		volatile bool animationTakeScreenshotCopy = animationTakeScreenshot;
		volumeVis->draw();
		if (animationTakeScreenshotCopy)
			animationRendered = true;
	}

    //const functions
    if (costFunctionMode_ == CostFunctionMode::PartialObservations && gridSimulation && enableAdjoint_)
        costFunctionPartialObservations_.draw();

  //  if (visualizedSdf>0 && meshReconstruction) {
		////determine the SDF to visualize
  //      WorldGridDoubleDataPtr sdf;
		//if (visualizedSdf == 1)
		//	sdf = meshReconstruction->getSdfData();
		//else
		//{
		//	int cam = visualizedSdf - 2;
		//	if (meshReconstruction->getProjectedSdfPerCamera().size() > cam)
		//		sdf = meshReconstruction->getProjectedSdfPerCamera()[cam];
		//}
		//if (sdf) {
		//	//visualizate the SDF
		//	WorldGridPtr grid = meshReconstruction->getWorldGrid();
  //          gl::Texture3dRef tex = sdf->getTexture();

  //          //update volume vis and render
  //          volumeVis->setGrid(grid, tex);
  //          volumeVis->setTransferFunction(tfe->getTexture(), tfe->getRangeMin(), tfe->getRangeMax());
  //          volumeVis->draw();
		//}
  //  }

    // WINDOW SPACE
	cinder::gl::disableDepthRead();
	cinder::gl::disableDepthWrite();
	cinder::gl::setMatricesWindow(getWindowSize(), true);

    if (enableAdjoint_)
    {
        const int offset = 10;
        const int width = 0.2 * getWindowWidth();
        int numPlots = 1;
        for (const auto& e : plots) if (std::get<2>(e)(adjointSettings_.getSettings().variables_)) numPlots++;
        numPlots = std::max(4, numPlots);
        const int height = getWindowHeight() / numPlots;
        costPlot->setBoundingRect(cinder::Rectf(getWindowWidth() - width, offset, getWindowWidth() - offset, height - offset));
        costPlot->setPrintMode(printMode);
        costPlot->draw();
        int y = 1;
        for (const auto& e : plots)
        {
            if (!std::get<2>(e)(adjointSettings_.getSettings().variables_)) continue;
            std::get<0>(e)->setBoundingRect(cinder::Rectf(getWindowWidth() - width, y * height + offset, getWindowWidth() - offset, (y + 1) * height - offset));
            std::get<0>(e)->setPrintMode(printMode);
            std::get<0>(e)->draw();
            y++;
        }
    }

    // Draw the background worker's status
    if (worker && !worker->isDone()) {
        //draw waiting animation
        {
			cinder::gl::ScopedModelMatrix scopedMatrix;
			cinder::gl::ScopedColor scopedColor;
			cinder::gl::translate(25, getWindowHeight() - 50);
            int step; double dummy; step = static_cast<int>(std::modf(getElapsedSeconds(), &dummy) * 8);
            for (int i = 0; i < 8; ++i) {
                float c = ((i + step)%8) / 7.0f;
				cinder::gl::color(c, c, c);
				cinder::gl::drawSolidRoundedRect(cinder::Rectf(5, -2, 15, 2), 2);
				cinder::gl::rotate(-2.0f * M_PI / 8.0f);
            }
        }
        //draw status
	    cinder::gl::drawString(worker->getStatus(), glm::vec2(50, getWindowHeight() - 50), printMode ? cinder::ColorA(0,0,0) : cinder::ColorA(1,1,1));
    }

	// Draw the interface
	params->draw();
    tfe->draw();
}

void SoftBody3DApp::quit()
{
    App::quit();
}

void SoftBody3DApp::cleanup()
{
	if (worker) {
		worker->interrupt();
		worker.reset();
	}
	volumeVis.reset();
	tetVis.reset();
	gridSimulation.reset();
	tetSimulation.reset();

	cudaPrintfEnd();
}

void SoftBody3DApp::resetPlots()
{
    costPlot->clear();
    for (const auto& e : plots)
    {
        std::get<0>(e)->clear();
    }
}

void SoftBody3DApp::simulationReset()
{
	CI_LOG_I("Reset simulation");
	if (worker)
	{
		worker->interrupt();
		worker->wait();
	}
    frameCounter = 0;
	results->states_.clear();
	if (tetSimulation) {
		tetSimulation->reset();
		tetSimulation->setRecordTimings(enableTimings);
		tetVis->update(tetSimulation->getState());
	}
	if (gridSimulation)
	{
		gridSimulation->reset();
		gridSimulation->setRecordTimings(enableTimings);
		volumeVis->update(gridSimulation->getState());
	}
    resetPlots();
}

void SoftBody3DApp::simulationStaticStep()
{
	if (worker->isDone())
	{
		CI_LOG_I("Simulation: solve for static solution");
		BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
		{
			if (tetSimulation) { //Mesh3D
				tetSimulation->solve(false, worker);
				tetVis->update(tetSimulation->getState());
			}
            if (gridSimulation) { //Grid3D
				gridSimulation->solve(false, worker);
				volumeVis->update(gridSimulation->getState());
			}
		};
		worker->launch(task);
	}
}

void SoftBody3DApp::simulationDynamicStep()
{
	if (worker->isDone() && !animationTakeScreenshot)
	{
		CI_LOG_I("Simulation: solve for dynamic solution");
		BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
		{
			if (tetSimulation) { //Mesh3D
				tetSimulation->solve(true, worker);
				tetVis->update(tetSimulation->getState());
			}
            if (gridSimulation) { //Grid3D
				gridSimulation->solve(true, worker, enableAdjoint_ || volumeVisParams.mode_ != VolumeVisualizationParams::Mode::MCSurface);
				volumeVis->update(gridSimulation->getState());
                if (enableAdjoint_)
				    results->states_.push_back(gridSimulation->getState().deepClone());
			}
            frameCounter++;
            animationTakeScreenshot = true;
			animationRendered = false;
		};
		worker->launch(task);
	}
}

void SoftBody3DApp::solveAdjoint()
{
	if (!enableAdjoint_)
	{
		CI_LOG_W("Adjoint not enabled");
		return;
	}
	if (results->states_.empty())
	{
		CI_LOG_W("No timesteps recorded");
		return;
	}
	if (worker->isDone())
	{
		CI_LOG_I("Simulation: solve adjoint problem");
        resetPlots();
        costPlot->setMaxPoints(adjointSettings_.getSettings().numIterations_);
        for (const auto& e : plots)
        {
            std::get<0>(e)->setMaxPoints(adjointSettings_.getSettings().numIterations_);
            std::get<0>(e)->setTrueValue(std::get<1>(e)(results->settings_));
        }
		BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
		{
			//create cost function
			CostFunctionPtr costFunction;
			switch (costFunctionMode_)
			{
			case CostFunctionMode::DirectActive:
				costFunction = std::make_shared<CostFunctionActiveDisplacements>(results, &costFunctionActiveDisplacements_);
				break;
            case CostFunctionMode::PartialObservations:
                costFunction = std::make_shared<CostFunctionPartialObservations>(results, &costFunctionPartialObservations_);
                break;
			default: throw std::exception("Unknown cost function");
			}
            costFunction->preprocess(worker);
            if (worker->isInterrupted()) return;
			//create adjoint solver
			AdjointSolver solver(results, adjointSettings_.getSettings(), costFunction);
			//solve
            AdjointSolver::Callback_t callback = [this](const SoftBodySimulation3D::Settings& var, const SoftBodySimulation3D::Settings& gradient, ar3d::real cost)
            {
                CI_LOG_I(var);
                CI_LOG_I(CoordinateTransformation::cartesian2spherical(glm::double3(var.groundPlane_.x, var.groundPlane_.y, var.groundPlane_.z)));
                costPlot->addPoint(cost);
                for (const auto& e : plots)
                    std::get<0>(e)->addPoint(std::get<1>(e)(var));
            };
			solver.solve(callback, worker);
            //Print steps
			std::stringstream ss;
            ss << "Cost    ";
            for (const auto& e : plots) {
                if (std::get<2>(e)(adjointSettings_.getSettings().variables_))
                    ss << std::get<0>(e)->getName() << "    ";
            }
            ss << std::endl;
            for (int i=0; i<costPlot->getNumPoints(); ++i)
            {
                ss << std::fixed << std::setw(12) << std::setprecision(7) << costPlot->getPoint(i) << "  ";
                for (const auto& e : plots) {
                    if (std::get<2>(e)(adjointSettings_.getSettings().variables_))
                        ss << std::fixed << std::setw(12) << std::setprecision(7) << std::get<0>(e)->getPoint(i) << "  ";
                }
                ss << std::endl;
            }
			CI_LOG_I(ss.str());
		};
		worker->launch(task);
	}
}

void SoftBody3DApp::testGradient()
{
    if (!enableAdjoint_)
    {
        CI_LOG_W("Adjoint not enabled");
        return;
    }
    if (results->states_.empty())
    {
        CI_LOG_W("No timesteps recorded");
        return;
    }
    if (worker->isDone())
    {
        CI_LOG_I("Simulation: solve adjoint problem");
        BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
        {
            //create cost function
            CostFunctionPtr costFunction;
            switch (costFunctionMode_)
            {
            case CostFunctionMode::DirectActive:
                costFunction = std::make_shared<CostFunctionActiveDisplacements>(results, &costFunctionActiveDisplacements_);
                break;
            case CostFunctionMode::PartialObservations:
                costFunction = std::make_shared<CostFunctionPartialObservations>(results, &costFunctionPartialObservations_);
                break;
            default: throw std::exception("Unknown cost function");
            }
            costFunction->preprocess(worker);
            if (worker->isInterrupted()) return;
            //create adjoint solver
            AdjointSolver solver(results, adjointSettings_.getSettings(), costFunction);
            //solve
            solver.testGradient(worker);
        };
        worker->launch(task);
    }
}

void SoftBody3DApp::exportObservations()
{
	if (!enableAdjoint_)
	{
		CI_LOG_W("Adjoint not enabled");
		return;
	}
	if (results->states_.empty())
	{
		CI_LOG_W("No timesteps recorded");
		return;
	}
	if (worker->isDone())
	{
		CI_LOG_I("Simulation: solve adjoint problem");
		BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
		{
			//create cost function
			std::shared_ptr< CostFunctionPartialObservations> costFunction
				= std::make_shared<CostFunctionPartialObservations>(results, &costFunctionPartialObservations_);
			costFunction->preprocess(worker);
			if (worker->isInterrupted()) return;
			//export observations
			costFunction->exportObservations(frameNames, exportEverNFrames_, worker);
			
		};
		worker->launch(task);
	}
}

void SoftBody3DApp::cancel()
{
	worker->interrupt();
    worker->wait();
}

void SoftBody3DApp::reloadShaders()
{
	//reload resources, shaders, ...
	volumeVis->reloadResources();
	tetVis->reloadShaders();
}

void SoftBody3DApp::drawWireCube(const real3& ea, const real3& eb)
{
    glm::vec3 a(ea.x, ea.y, ea.z);
    glm::vec3 b(eb.x, eb.y, eb.z);
    cinder::gl::drawLine(a, glm::vec3(a.x, a.y, b.z));
    cinder::gl::drawLine(a, glm::vec3(a.x, b.y, a.z));
    cinder::gl::drawLine(a, glm::vec3(b.x, a.y, a.z));
    cinder::gl::drawLine(glm::vec3(a.x, b.y, b.z), glm::vec3(a.x, a.y, b.z));
    cinder::gl::drawLine(glm::vec3(b.x, a.y, b.z), glm::vec3(a.x, a.y, b.z));
    cinder::gl::drawLine(glm::vec3(a.x, b.y, b.z), glm::vec3(a.x, b.y, a.z));
    cinder::gl::drawLine(glm::vec3(b.x, b.y, a.z), glm::vec3(a.x, b.y, a.z));
    cinder::gl::drawLine(glm::vec3(b.x, b.y, a.z), glm::vec3(b.x, a.y, a.z));
    cinder::gl::drawLine(glm::vec3(b.x, a.y, b.z), glm::vec3(b.x, a.y, a.z));
    cinder::gl::drawLine(b, glm::vec3(a.x, b.y, b.z));
    cinder::gl::drawLine(b, glm::vec3(b.x, b.y, a.z));
    cinder::gl::drawLine(b, glm::vec3(b.x, a.y, b.z));
}

cinder::TriMeshRef SoftBody3DApp::loadCustomObj(cinder::DataSourceRef dataSource)
{
    std::shared_ptr<cinder::IStreamCinder> stream(dataSource->createStream());
	cinder::TriMeshRef mesh = cinder::TriMesh::create(cinder::TriMesh::Format().positions().normals().colors(4));

    size_t lineNumber = 0;
    while (!stream->isEof()) {
        lineNumber++;
        string line = stream->readLine(), tag;
        if (line.empty() || line[0] == '#')
            continue;

        while (line.back() == '\\' && !stream->isEof()) {
            auto next = stream->readLine();
            line = line.substr(0, line.size() - 1) + next;
        }

        stringstream ss(line);
        ss >> tag;
        if (tag == "v") { // vertex
            glm::vec3 v;
            ss >> v.x >> v.y >> v.z;
            mesh->appendPosition(v);
        }
        else if (tag == "vn") { // vertex normals
            glm::vec3 v;
            ss >> v.x >> v.y >> v.z;
            mesh->appendNormal(normalize(v));
        }
        else if (tag == "vc") { // vertex colors
	        cinder::ColorAf v;
            ss >> v.r >> v.g >> v.b >> v.a;
            mesh->appendColorRgba(v);
        }
        else if (tag == "f") { // face
            uint32_t a, b, c;
            ss >> a >> b >> c;
            mesh->appendTriangle(a-1, b-1, c-1);
        }
    }

    return mesh;
}

SoftBodySimulation3D::InputBarSettings oldBarSettings;
SoftBodySimulation3D::InputTorusSettings oldTorusSettings;
int oldInputCase = -1;
void SoftBody3DApp::updateInput()
{
	if (int(inputCase_) != oldInputCase ||
		memcmp(&oldBarSettings, &inputBarSettings_.getSettings(), sizeof(SoftBodySimulation3D::InputBarSettings)) != 0 ||
		memcmp(&oldTorusSettings, &inputTorusSettings_.getSettings(), sizeof(SoftBodySimulation3D::InputTorusSettings)) != 0 ||
        inputSdfSettings_.hasChanged())
	{
		updateInput0();
	}
}

void SoftBody3DApp::updateInput0()
{
	CI_LOG_I("Input Bar Settings changed");
	if (worker->isDone())
	{
		oldInputCase = int(inputCase_);
		oldBarSettings = inputBarSettings_.getSettings();
		oldTorusSettings = inputTorusSettings_.getSettings();
		BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
		{
			//tetSimulation.reset();
			//gridSimulation.reset();
			results->states_.clear();
			if (simulationMode_ == SimulationMode::Tet) {
				gridSimulation.reset();
				worker->setStatus("Create Tet mesh");
				tetSimulation = std::make_unique<SoftBodyMesh3D>(SoftBodyMesh3D::createBar(inputBarSettings_.getSettings()));
				tetSimulation->setSettings(simulationSettings_.getSettings());
				tetSimulation->setRecordTimings(enableTimings);
				worker->setStatus("Prepare rendering");
				tetVis->setInput(tetSimulation->getInput());
			}
			else if (simulationMode_ == SimulationMode::Grid)
			{
				tetSimulation.reset();
				worker->setStatus("Create grid");
				SoftBodyGrid3D::Input input;
				std::string highResFile = "";
                if (inputCase_ == InputCase::Bar)
                    input = SoftBodyGrid3D::createBar(inputBarSettings_.getSettings());
                else if (inputCase_ == InputCase::Torus)
                    input = SoftBodyGrid3D::createTorus(inputTorusSettings_.getSettings());
                else if (inputCase_ == InputCase::File)
                {
                    if (!inputSdfSettings_.isAvailable()) return;
                    input = SoftBodyGrid3D::createFromFile(inputSdfSettings_.getSettings());
					highResFile = inputSdfSettings_.getSettings().file + ".obj";
					if (!cinder::fs::exists(cinder::fs::path(highResFile)))
						highResFile = "";
                }
                else
                    return;
				gridSimulation = std::make_unique<SoftBodyGrid3D>(input);
				gridSimulation->setSettings(simulationSettings_.getSettings());
				gridSimulation->setRecordTimings(enableTimings);
				worker->setStatus("Prepare rendering");
				volumeVis->setInput(gridSimulation->getInput());
				volumeVis->setHighResInputMesh(highResFile);
				results->input_ = gridSimulation->getInput();
			}
            frameCounter = 0;
		};
		worker->launch(task);
        //BackgroundWorker2 w;
        //task(&w);
	}
}

void SoftBody3DApp::updateSettings()
{
	static SoftBodySimulation3D::Settings oldSettings;
	if (memcmp(&oldSettings, &simulationSettings_, sizeof(SoftBodySimulation3D::Settings)) != 0)
	{
		CI_LOG_I("Simulation Settings changed");
		if (worker->isDone())
		{
			oldSettings = simulationSettings_.getSettings();
			BackgroundWorker2::task task = [this](BackgroundWorker2* worker)
			{
				if (tetSimulation)
					tetSimulation->setSettings(simulationSettings_.getSettings());
				if (gridSimulation) {
					gridSimulation->setSettings(simulationSettings_.getSettings());
					results->settings_ = simulationSettings_.getSettings();
				}
			};
			worker->launch(task);
		}
	}
}


void SoftBody3DApp::saveStatistics()
{
	assert(enableTimings);

	//construct filename
	time_t now = time(NULL);
	struct tm tstruct;
	char buf[100];
	localtime_s(&tstruct, &now);
	strftime(buf, sizeof(buf), "%d-%m-%Y_%H-%M-%S", &tstruct);

	if (tetSimulation)
	{
		assert(tetSimulation->isRecordTimings());
		try
		{
			string fileName = string("../plots/SoftBody3DApp-StatisticsTet-") + string(buf) + ".txt";
			std::ofstream o(fileName, std::ofstream::trunc);
			o << tetSimulation->getSettings() << std::endl;
			o << tetSimulation->getStatistics() << std::endl;
			o.close();
			CI_LOG_I("Statistics written to " << fileName);
		} catch (std::exception& ex)
		{
			CI_LOG_EXCEPTION("Unable to write file", ex);
		}
	}

	if (gridSimulation)
	{
		assert(gridSimulation->isRecordTimings());
		try
		{
			string fileName = string("../plots/SoftBody3DApp-StatisticsGrid-") + string(buf) + ".txt";
			std::ofstream o(fileName, std::ofstream::trunc);
			o << gridSimulation->getSettings() << std::endl;
			o << gridSimulation->getStatistics() << std::endl;
			o.close();
			CI_LOG_I("Statistics written to " << fileName);
		}
		catch (std::exception& ex)
		{
			CI_LOG_EXCEPTION("Unable to write file", ex);
		}
	}
}

void SoftBody3DApp::load()
{
    cancel();

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
    } catch (const cinder::JsonTree::ExcJsonParserError& ex)
    {
        CI_LOG_E("Unable to load json file " << path.string() << ": " << ex.what());
        return;
    }

    //Input
    simulationMode_ = ToSimulationMode(root.getValueForKey("SimulationMode"));
    inputCase_ = ToInputCase(root.getValueForKey("InputCase"));
    inputBarSettings_.load(root.getChild("InputBarSettings"));
    inputTorusSettings_.load(root.getChild("InputTorusSettings"));
    if (root.hasChild("InputSdfSettings"))
        inputSdfSettings_.load(root.getChild("InputSdfSettings"));
    inputBarSettings_.setVisible(params, inputCase_ == InputCase::Bar);
    inputTorusSettings_.setVisible(params, inputCase_ == InputCase::Torus);
    inputSdfSettings_.setVisible(params, inputCase_ == InputCase::File);

    //Simulation
    simulationSettings_.load(root.getChild("SimulationSettings"));
    enableTimings = root.getValueForKey<bool>("EnableTimings");

    //Adjoint
    enableAdjoint_ = root.getValueForKey<bool>("AdjointEnable");
    adjointSettings_.load(root.getChild("AdjointSettings"));
    costFunctionMode_ = ToCostFunctionMode(root.getValueForKey("CostFunctionMode"));
    costFunctionActiveDisplacements_.load(root.getChild("CostFunctionActiveDisplacements"));
    costFunctionPartialObservations_.load(root.getChild("CostFunctionPartialObservations"));

    //Rendering
    printMode = root.getValueForKey<bool>("PrintMode");
    volumeVisParams.load(root.getChild("Rendering"));

    CI_LOG_I("done loading " << path.string());
}

void SoftBody3DApp::save()
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

    //Input
    root.addChild(cinder::JsonTree("SimulationMode", FromSimulationMode(simulationMode_)));
    root.addChild(cinder::JsonTree("InputCase", FromInputCase(inputCase_)));
    cinder::JsonTree inputBarSettings = cinder::JsonTree::makeObject("InputBarSettings");
    inputBarSettings_.save(inputBarSettings);
    root.addChild(inputBarSettings);
    cinder::JsonTree inputTorusSettings = cinder::JsonTree::makeObject("InputTorusSettings");
    inputTorusSettings_.save(inputTorusSettings);
    root.addChild(inputTorusSettings);
    cinder::JsonTree inputSdfSettings = cinder::JsonTree::makeObject("InputSdfSettings");
    inputSdfSettings_.save(inputSdfSettings);
    root.addChild(inputSdfSettings);
    
    //Simulation
    cinder::JsonTree simulationSettings = cinder::JsonTree::makeObject("SimulationSettings");
    simulationSettings_.save(simulationSettings);
    root.addChild(simulationSettings);
    root.addChild(cinder::JsonTree("EnableTimings", enableTimings));
    root.addChild(cinder::JsonTree("NumSteps", results ? results->states_.size() : 0));

    //Adjoint
    root.addChild(cinder::JsonTree("AdjointEnable", enableAdjoint_));
    cinder::JsonTree adjointSettings = cinder::JsonTree::makeObject("AdjointSettings");
    adjointSettings_.save(adjointSettings);
    root.addChild(adjointSettings);
    root.addChild(cinder::JsonTree("CostFunctionMode", FromCostFunctionMode(costFunctionMode_)));
    cinder::JsonTree costFunctionActiveDisplacements = cinder::JsonTree::makeObject("CostFunctionActiveDisplacements");
    costFunctionActiveDisplacements_.save(costFunctionActiveDisplacements);
    root.addChild(costFunctionActiveDisplacements);
    cinder::JsonTree costFunctionPartialObservations = cinder::JsonTree::makeObject("CostFunctionPartialObservations");
    costFunctionPartialObservations_.save(costFunctionPartialObservations);
    root.addChild(costFunctionPartialObservations);

    //Rendering
    root.addChild(cinder::JsonTree("PrintMode", printMode));
    cinder::JsonTree rendering = cinder::JsonTree::makeObject("Rendering");
    volumeVisParams.save(rendering);
    root.addChild(rendering);

    root.write(path);
    CI_LOG_I("Saved to " << path.string());
}

#if 1
CINDER_APP( SoftBody3DApp, RendererGl(RendererGl::Options().msaa(8)), [&](App::Settings *settings)
{
	settings->setWindowSize(1600, 900);
} )
#endif
