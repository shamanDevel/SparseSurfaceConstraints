#include <cmath>
#include <chrono>
#include <thread>
#include <ctime>
#include <fstream>
#include <chrono>

#include <Eigen/Dense>

#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>
#include <cinder/CameraUi.h>
#include <cinder/params/Params.h>
#include <cinder/Log.h>
#include <cinder/ObjLoader.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unique_ptr.hpp>

#include "../resources/Resources.h"
#include <Utils.h>
#include <BackgroundWorker.h>
#include <TransferFunctionEditor.h>
#include <Integration.h>
#include <GridUtils.h>
#include <TimeIntegrator.h>
#include <SoftBodyGrid2D.h>
#include <SoftBodyMesh2D.h>
#include <SoftBody2DResults.h>
#include <GridVisualization.h>

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace ar;
using namespace Eigen;

class SoftBodyCutFEM2DApp : public App {
public:
	SoftBodyCutFEM2DApp();
	void setup() override;
	void keyDown(KeyEvent event) override;
    void keyUp(KeyEvent event) override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;

private:

    typedef double real;

	//AntTweakBar settings
	params::InterfaceGlRef	params;
    bool printMode;
    int gridResolution;
    bool fitObjectToGrid;
    enum class Scene
    {
        SCENE_BAR,
        SCENE_TORUS
    };
    Scene scene;
    double torusOuterRadius;
    double torusInnerRadius;
    Vector2f rectCenter;
    Vector2f rectHalfSize;
	real groundPlaneHeight;
	real groundPlaneAngle;
    enum class ComputationMode
    {
        COMPUTATION_MODE_GRID = 1,
        COMPUTATION_MODE_MESH = 2,
        COMPUTATION_MODE_BOTH = 3
    };
    int computationMode;
    bool showGrid;
    enum class GridVisualizationMode
    {
        GRID_VISUALIZATION_U,
        GRID_VISUALIZATION_SOLUTION,
		GRID_VISUALIZATION_BOUNDARY,
        _GRID_VISUALIZATION_COUNT
    };
    GridVisualizationMode showGridSolution;
    bool spaceBarPressed = false;
    TimeIntegrator::Integrator timeIntegratorType;
    TimeIntegrator::DenseLinearSolver denseLinearSolverType;
    TimeIntegrator::SparseLinearSolver sparseLinearSolverType;
    bool useSparseMatrices;
    int sparseSolverIterations;
    real sparseSolverTolerance;
    SoftBodySimulation::RotationCorrection rotationCorrectionMode;

    //soft body properties
    enum Boundary
    {
        FREE = 0,
        NEUMANN = 1,
        DIRICHLET = 2
    };
    Vector2f gravity;
    Vector2f neumannForce;
    float youngsModulus;
    float poissonsRatio;
    float mass;
    float dampingAlpha;
    float dampingBeta;
    float timestep;
	bool enableDirichletBoundaries;
	bool enableCollision;
	SoftBodySimulation::CollisionResolution collisionResolutionMode;
	float collisionVelocityDamping;
    real groundStiffness_;
    real softmaxAlpha_;
    int step;

    //grid
    SoftBodyGrid2D gridSolver;
    //true: diffusion of displacements into empty cells is done as a post-processing step (matrix contains only cells containing the object)
    //false: diffusion is included in the matrix, matrix spans the whole grid
    bool gridExplicitDiffusion;
    bool gridHardDirichletBoundaries;
    SoftBodyGrid2D::AdvectionMode gridAdvectionMode;
    double gridElapsedSeconds = 0;

    //triangles
    SoftBodyMesh2D meshSolver;
    double meshElapsedSeconds = 0;

	//output saving
	std::unique_ptr<SoftBody2DResults> results;

    //processing
    ar::BackgroundWorkerPtr worker;

    //visualization helpers
	GridVisualization visualization;

private:
    //Resets the simulation:
    //- Creates the input grid and configuration
    //- Resets the frame counter
    void reset();
    void invalidateRendering();
    void performStep(float stepsize); //stepsize=0 -> static solution

    void initGridWithTorus(); //initializes the SDF
    void initGridWithRect();

    void initTriMeshWithTorus(); //creates the vertices and inidices of the tri mesh
    void initTriMeshWithGrid();

	void saveResults();
};

SoftBodyCutFEM2DApp::SoftBodyCutFEM2DApp()
{
	//initial config
    printMode = false;
    gridResolution = 18;
    fitObjectToGrid = false;
    scene = Scene::SCENE_BAR;
    torusOuterRadius = 0.1;
    torusInnerRadius = 0.03;
    rectCenter << 0.5, 0.74;
    rectHalfSize << 0.3, 0.06;

	groundPlaneHeight = 0.58;
	groundPlaneAngle = 0;

    computationMode = (int)ComputationMode::COMPUTATION_MODE_GRID;
    showGrid = true;
    showGridSolution = GridVisualizationMode::GRID_VISUALIZATION_BOUNDARY;
    timeIntegratorType = TimeIntegrator::Integrator::Newmark1;
    denseLinearSolverType = TimeIntegrator::DenseLinearSolver::PartialPivLU;
    sparseLinearSolverType = TimeIntegrator::SparseLinearSolver::BiCGSTAB;
    useSparseMatrices = false;
    sparseSolverIterations = 100;
    sparseSolverTolerance = 1e-5;

    rotationCorrectionMode = SoftBodySimulation::RotationCorrection::Corotation;

    gridExplicitDiffusion = true;
    gridHardDirichletBoundaries = false;
    gridAdvectionMode = SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD;

    gravity = Vector2f(0, -10);
    neumannForce = Vector2f(0, 0);
    youngsModulus = 200;
    poissonsRatio = 0.45;
    mass = 1.0;
    dampingAlpha = 0.001;
    dampingBeta = 0.001;
    timestep = 0.01;
	enableDirichletBoundaries = false;
	enableCollision = true;
	collisionResolutionMode = SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT;
	collisionVelocityDamping = 0.5;
    groundStiffness_ = 1000;
    softmaxAlpha_ = 100;

    step = 0;
}

void SoftBodyCutFEM2DApp::setup()
{
	//parameter ui, must happen before user-camera
	params = params::InterfaceGl::create(getWindow(), "Parameters", toPixels(ivec2(300, 600)));
    params->setOptions("", "refresh=0.05");
    params->addParam("PrintMode", &printMode).label("Print Mode");

    vector<string> computationModeEnums(4);
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_GRID] = "grid";
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_MESH] = "mesh";
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_BOTH] = "overlay / both";
    params->addParam("Mode", computationModeEnums, (int*)&computationMode, "label='Mode'");

    params->addParam("InputResolution", 
        std::function<void(int)>([this](int newValue) {this->gridResolution = newValue; this->reset(); }), 
        std::function<int()>([this]() {return this->gridResolution; })
    ).min(4).group("Input").label("Grid resolution").keyIncr("PGUP").keyDecr("PGDOWN");
    params->addParam("FitObjectToGrid",
        std::function<void(bool)>([this](bool newValue) {this->fitObjectToGrid = newValue; this->reset(); }),
        std::function<bool()>([this]() {return this->fitObjectToGrid; })
    ).group("Input").label("Fit Object To Grid");
    vector<string> sceneEnums = { "Bar", "Torus" };
    params->addParam("Scene", sceneEnums, (int*)&scene).group("Input").label("Scene")
        .accessors(std::function<void(int)>([this](int v)
    {
        scene = (Scene)v;
        reset();
        if (scene == Scene::SCENE_TORUS) {
            params->setOptions("InputTorusOuterRadius", "visible=true");
            params->setOptions("InputTorusInnerRadius", "visible=true");
            params->setOptions("InputRectCenterX", "visible=false");
            params->setOptions("InputRectHalfSizeX", "visible=false");
            params->setOptions("InputRectCenterY", "visible=false");
            params->setOptions("InputRectHalfSizeY", "visible=false");
        } else {
            params->setOptions("InputTorusOuterRadius", "visible=false");
            params->setOptions("InputTorusInnerRadius", "visible=false");
            params->setOptions("InputRectCenterX", "visible=true");
            params->setOptions("InputRectHalfSizeX", "visible=true");
            params->setOptions("InputRectCenterY", "visible=true");
            params->setOptions("InputRectHalfSizeY", "visible=true");
        }
    }), std::function<int()>([this]() {return (int)scene; }));
    params->addParam("InputTorusOuterRadius",
        std::function<void(double)>([this](double newValue) {this->torusOuterRadius = newValue; this->reset(); }),
        std::function<double()>([this]() {return this->torusOuterRadius; })
    ).min(0.01).max(0.5).step(0.01).group("Input").label("Torus outer radius").visible(false);
    params->addParam("InputTorusInnerRadius",
        std::function<void(double)>([this](double newValue) {this->torusInnerRadius = newValue; this->reset(); }),
        std::function<double()>([this]() {return this->torusInnerRadius; })
    ).min(0.01).max(0.5).step(0.01).group("Input").label("Torus inner radius").visible(false);
    params->addParam("InputRectCenterX",
        std::function<void(float)>([this](float newValue) {this->rectCenter.x() = newValue; this->reset(); }),
        std::function<float()>([this]() {return this->rectCenter.x(); })
    ).step(0.01).group("Input").label("Rect center X").visible(true);
    params->addParam("InputRectCenterY",
        std::function<void(float)>([this](float newValue) {this->rectCenter.y() = newValue; this->reset(); }),
        std::function<float()>([this]() {return this->rectCenter.y(); })
    ).step(0.01).group("Input").label("Rect center Y").visible(true);
    params->addParam("InputRectHalfSizeX",
        std::function<void(float)>([this](float newValue) {this->rectHalfSize.x() = newValue; this->reset(); }),
        std::function<float()>([this]() {return this->rectHalfSize.x(); })
    ).step(0.01).group("Input").label("Rect half size X").visible(true);
    params->addParam("InputRectHalfSizeY",
        std::function<void(float)>([this](float newValue) {this->rectHalfSize.y() = newValue; this->reset(); }),
        std::function<float()>([this]() {return this->rectHalfSize.y(); })
    ).step(0.01).group("Input").label("Rect half size Y").visible(true);
	params->addParam("InputGroundPlaneHeight", &groundPlaneHeight).step(0.001).group("Input").label("Ground Height");
	params->addParam("InputGroundPlaneAngle", &groundPlaneAngle).step(0.001).group("Input").label("Ground Angle");
    params->addButton("InputReset", std::function<void()>([this]() {this->reset(); }), "label='Reset' group=Input key=r");

    params->addParam("SoftBodyGravity", &gravity.y()).step(0.001).group("Soft Body").label("Gravity");
    params->addParam("SoftBodyNeumannForce", &neumannForce.y()).step(0.001).group("Soft Body").label("Neumann Force")
        .accessors(std::function<void(float)>([this](float v)
    {
        neumannForce.y() = v;
        reset();
    }), std::function<float()>([this]() {
        return (float)neumannForce.y();
    }));
    params->addParam("SoftBodyYoungsModulus", &youngsModulus).min(0).step(0.01).group("Soft Body").label("Young's modulus");
    params->addParam("SoftBodyPoissonsRatio", &poissonsRatio).min(0.0001).max(0.4999).step(0.01).group("Soft Body").label("Poisson's ratio");
    params->addParam("SoftBodyMass", &mass).min(0.0001).step(0.01).group("Soft Body").label("Mass");
    params->addParam("SoftBodyDampingAlpha", &dampingAlpha).min(0).step(0.001).group("Soft Body").label("Damping on mass");
    params->addParam("SoftBodyDampingBeta", &dampingBeta).min(0).step(0.001).group("Soft Body").label("Damping on stiffness");
    vector<string> timeIntegratorTypeNames = { "Newmark 1", "Newmark 2", "Central Differences", "Linear Accelleration", "Newmark 3", "HHT-alpha" };
    params->addParam("SoftBodyTimeIntegration", timeIntegratorTypeNames, (int*)&timeIntegratorType).group("Soft Body").label("Time Integrator")
        .accessors(std::function<void(int)>([this](int v)
    {
        timeIntegratorType = (TimeIntegrator::Integrator)v;
        reset();
    }), std::function<int()>([this]() {return (int)timeIntegratorType; }));
#if SOFT_BODY_SUPPORT_SPARSE_MATRICES==1
    params->addParam("SoftBodyUseSparseMatrices", std::function<void(bool)>([this](bool v)
    {
        useSparseMatrices = v;
        if (v)
        {
            params->setOptions("SoftBodyDenseLinearSolver", "visible=false");
            params->setOptions("SoftBodySparseLinearSolver", "visible=true");
            params->setOptions("SoftBodySparseSolverIterations", "visible=true");
            params->setOptions("SoftBodySparseSolverTolerance", "visible=true");
        } else
        {
            params->setOptions("SoftBodyDenseLinearSolver", "visible=true");
            params->setOptions("SoftBodySparseLinearSolver", "visible=false");
            params->setOptions("SoftBodySparseSolverIterations", "visible=false");
            params->setOptions("SoftBodySparseSolverTolerance", "visible=false");
        }
    }), std::function<bool()>([this]()
    {
        return useSparseMatrices;
    })).group("Soft Body").label("Sparse matrices");
    vector<string> denseLinearSolverTypeNames = { "PartialPivLU", "FullPivLU", "HouseholderQR", "ColPivHousholderQR", "FullPivHouseholderQR", "CompleteOrthogonalDecomposition", "LLT", "LDLT" };
    params->addParam("SoftBodyDenseLinearSolver", denseLinearSolverTypeNames, (int*)&denseLinearSolverType).group("Soft Body").label("Dense Linear Solver");
    vector<string> sparseLinearSolverTypeNames = { "Conjugate Gradient", "BiCGSTAB ", "Sparse-LU" };
    params->addParam("SoftBodySparseLinearSolver", sparseLinearSolverTypeNames, (int*)&sparseLinearSolverType).group("Soft Body").label("Sparse Linear Solver");
    params->addParam("SoftBodySparseSolverIterations", &sparseSolverIterations).group("Soft Body").label("Sparese Solver iterations").min(0);
    params->addParam("SoftBodySparseSolverTolerance", &sparseSolverTolerance).group("Soft Body").label("Sparese Solver tolerance").min(0).step(0.00001);
#else
    vector<string> denseLinearSolverTypeNames = { "PartialPivLU", "FullPivLU", "HouseholderQR", "ColPivHousholderQR", "FullPivHouseholderQR", "CompleteOrthogonalDecomposition", "LLT", "LDLT" };
    params->addParam("SoftBodyDenseLinearSolver", denseLinearSolverTypeNames, (int*)&denseLinearSolverType).group("Soft Body").label("Dense Linear Solver");
#endif
    params->addParam("SoftBodyTimeStep", &timestep).min(0.001).step(0.001).group("Soft Body").label("Time step");
    params->addParam("SoftBodyGridExplicitDiffusion", &gridExplicitDiffusion, "group='Soft Body' label='Grid Displacement Diffusion' true='explicit (post-process)' false='implicit (matrix)'");
    params->addParam("SoftBodyGridHardDirichletBoundaries", &gridHardDirichletBoundaries).group("Soft Body").label("Grid Hard Dirichlet Boundaries");
    vector<string> softBodyAdvectionNames;
    for (int i = 0; i < static_cast<int>(SoftBodyGrid2D::AdvectionMode::_COUNT_); ++i)
        softBodyAdvectionNames.push_back(SoftBodyGrid2D::advectionModeName(static_cast<SoftBodyGrid2D::AdvectionMode>(i)));
    params->addParam("SoftBodyAdvectionMode", softBodyAdvectionNames, (int*)&gridAdvectionMode).group("Soft Body").label("Grid Advection");

    vector<string> rotationCorrectionNames = { "None", "Corotation"};
    params->addParam("SoftBodyRotationCorrection", rotationCorrectionNames, (int*)&rotationCorrectionMode).group("Soft Body").label("Rotation correction");

	params->addParam("SoftBodyDirichlet", 
		std::function<void(bool)>([this](bool newValue) {this->enableDirichletBoundaries = newValue; this->reset(); }),
		std::function<bool()>([this]() {return this->enableDirichletBoundaries; }))
	.group("Soft Body").label("Enable Dirichlet Boundaries");
	params->addParam("SoftBodyCollision", &enableCollision).group("Soft Body").label("Enable Collision");
	vector<string> collisionResolutionNames(&SoftBodySimulation::CollisionResolutionNames[0], &SoftBodySimulation::CollisionResolutionNames[0]+size_t(SoftBodySimulation::CollisionResolution::_COUNT_));
	params->addParam("SoftBodyCollisionResolution", collisionResolutionNames, (int*)&collisionResolutionMode).group("Soft Body").label("Collision Resolution");
	params->addParam("SoftBodyCollisionVelDamping", &collisionVelocityDamping).group("Soft Body").label("Col. Vel. Damping").min(0).max(1).step(0.001);
    params->addParam("SoftBodyCollisionGroundStiffness", &groundStiffness_).group("Soft Body").label("Col. Ground Stiffness").min(0).step(0.001);
    params->addParam("SoftBodyCollisionSoftmaxAlpha", &softmaxAlpha_).group("Soft Body").label("Col. Softmax-Alpha").min(1).max(1000).step(0.001);

    params->addButton("SoftBodyStep", std::function<void()>([this]() {this->performStep(timestep); }), "group='Soft Body' label='Single step'");
    params->addButton("SoftBodyStatic", std::function<void()>([this]() {this->performStep(0); }), "group='Soft Body' label='Static solution' key=RETURN");

    params->addParam("RenderingShowGrid", &showGrid).group("Rendering").label("Show grid");
    vector<string> gridVisualizationModeEnums((int)GridVisualizationMode::_GRID_VISUALIZATION_COUNT);
    gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_U] = "u";
    gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_SOLUTION] = "solution";
	gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_BOUNDARY] = "boundary";
    params->addParam("RenderingShowGridSolutionMode", gridVisualizationModeEnums, (int*)&showGridSolution, "label='Grid solution' group=Rendering");

    params->addParam("TimingMesh", &meshElapsedSeconds, true).group("Timings").label("Mesh simulation (sec)").precision(3);
    params->addParam("TimingGrid", &gridElapsedSeconds, true).group("Timings").label("Grid simulation (sec)").precision(3);

	params->addButton("SaveResults", std::function<void()>([this]() {this->saveResults(); }), "label='Save Results'");

	visualization.setup();

    //initialize grid
    reset();
}

void SoftBodyCutFEM2DApp::keyDown(KeyEvent event)
{
	App::keyDown(event);
    if (event.isHandled()) return;
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
		Surface surface = copyWindowSurface();
        //if (printMode) {
        //    //only save the grid
        //    int windowWidth = getWindow()->getWidth();
        //    int windowHeight = getWindow()->getHeight();
        //    int gridBoundary = 50;
        //    int gridSize = std::min(windowWidth, windowHeight) - 2 * gridBoundary;
        //    int gridOffsetX = windowWidth / 2;
        //    int gridOffsetY = windowHeight / 2;
        //    Surface surface2(gridSize + 10, gridSize + 10, false);
        //    surface2.copyFrom(surface, Area(gridOffsetX - 5, gridOffsetY - 5, gridOffsetX + gridSize + 10, gridOffsetY + gridSize + 10));
        //    surface = surface2;
        //}
		//construct filename
		time_t now = time(NULL);
		struct tm tstruct;
		char buf[100];
		tstruct = *localtime(&now);
		strftime(buf, sizeof(buf), "%d-%m-%Y_%H-%M-%S", &tstruct);
		string fileName = string("../screenshots/SoftBodyFEM2DApp-") + string(buf) + ".png";
		//write out
		writeImage(fileName, surface);
        CI_LOG_I("Screenshot saved to " << fileName);
	}
    else if (event.getCode() == KeyEvent::KEY_SPACE)
    {
        spaceBarPressed = true;
    }
}

void SoftBodyCutFEM2DApp::keyUp(KeyEvent event)
{
    App::keyUp(event);
    if (event.isHandled()) return;
    if (event.getCode() == KeyEvent::KEY_SPACE)
    {
        spaceBarPressed = false;
    }
}

void SoftBodyCutFEM2DApp::mouseDown( MouseEvent event )
{
	App::mouseDown(event);
}

void SoftBodyCutFEM2DApp::update()
{
    //perform time stepping
    if (spaceBarPressed) {
        performStep(timestep);
    }

    //update transfer function editor
	visualization.update();
}

void SoftBodyCutFEM2DApp::draw()
{
    using namespace ar::utils;
    if (printMode)
        gl::clear(Color(1, 1, 1));
    else
        gl::clear(Color(0, 0, 0));
	
	
    // WINDOW SPACE
    gl::disableDepthRead();
    gl::disableDepthWrite();
    gl::setMatricesWindow(getWindowSize(), true);

    //grid bounds
    int windowWidth = getWindow()->getWidth();
    int windowHeight = getWindow()->getHeight();
    int gridBoundary = 50;
    int gridSize = std::min(windowWidth, windowHeight) - 2 * gridBoundary;
    int gridOffsetX = windowWidth / 2;
    int gridOffsetY = windowHeight / 2;

    Color colors[3];
    colors[FREE] = Color(1, 0, 0);
    colors[NEUMANN] = Color(1, 1, 0);
    colors[DIRICHLET] = Color(0, 1, 0.2);

    // Draw sdf
	if (computationMode & (int)ComputationMode::COMPUTATION_MODE_GRID)
	{
		visualization.gridDrawSdf();
		if (showGrid)
			visualization.gridDrawGridLines();
		if (showGridSolution == GridVisualizationMode::GRID_VISUALIZATION_BOUNDARY)
			visualization.gridDrawObjectBoundary();
		if (!gridSolver.hasSolution())
			visualization.gridDrawBoundaryConditions(gridSolver.getGridDirichlet(), gridSolver.getGridNeumannX(), gridSolver.getGridNeumannY());
		if (gridSolver.hasSolution() && showGridSolution == GridVisualizationMode::GRID_VISUALIZATION_U) {
			gl::ScopedColor c;
			if (printMode)
				gl::color(0, 0, 0.5);
			else
				gl::color(1, 1, 1);
			visualization.gridDrawDisplacements();
		}
	}

    // Draw mesh
    if (computationMode & (int)ComputationMode::COMPUTATION_MODE_MESH) {
		visualization.meshDraw();
    }

	// Draw ground
	if (enableCollision)
	{
		visualization.drawGround(groundPlaneHeight, groundPlaneAngle);
	}

    // Draw the background worker's status
    if (worker && !worker->isDone()) {
        //draw waiting animation
        {
            gl::ScopedModelMatrix scopedMatrix;
            gl::ScopedColor scopedColor;
            gl::translate(25, getWindowHeight() - 50);
            int step; double dummy; step = static_cast<int>(std::modf(getElapsedSeconds(), &dummy) * 8);
            for (int i = 0; i < 8; ++i) {
                float c = ((i + step)%8) / 7.0f;
                gl::color(c, c, c);
                gl::drawSolidRoundedRect(Rectf(5, -2, 15, 2), 2);
                gl::rotate(-2.0f * M_PI / 8.0f);
            }
        }
        //draw status
        gl::ScopedColor scopedColor;
        if (printMode)
            gl::color(0, 0, 0);
        else
            gl::color(1, 1, 1);
        gl::drawString(worker->getStatus(), vec2(50, getWindowHeight() - 50));
    }

	// Draw the interface
	visualization.setTfeVisible(!printMode);
	params->draw();
	visualization.drawTransferFunctionEditor();
}

void SoftBodyCutFEM2DApp::reset()
{
    CI_LOG_I("reset");
    worker = nullptr;
    step = 0;

	//reset solution
	results = std::make_unique<SoftBody2DResults>();

    //create and initialize grid
    if (scene == Scene::SCENE_BAR)
        initGridWithRect();
    else if (scene == Scene::SCENE_TORUS)
        initGridWithTorus();

    //create and initialize tri mesh
    if (scene == Scene::SCENE_BAR)
        initTriMeshWithGrid();
    else if (scene == Scene::SCENE_TORUS)
        initTriMeshWithTorus();

    //invalidate texture and mesh
    invalidateRendering();
}

void SoftBodyCutFEM2DApp::invalidateRendering()
{
	if (gridSolver.hasSolution())
	{
		visualization.setGrid(gridSolver.getSdfReference(), gridSolver.getSdfSolution(), gridSolver.getUGridX(), gridSolver.getUGridY());
	} else
	{
		GridUtils2D::grid_t z = GridUtils2D::grid_t::Zero(gridSolver.getGridResolution(), gridSolver.getGridResolution());
		visualization.setGrid(gridSolver.getSdfReference(), gridSolver.getSdfReference(), z, z);
	}

	auto& pos = meshSolver.hasSolution() ? meshSolver.getCurrentPositions() : meshSolver.getReferencePositions();
	visualization.setMesh(pos, meshSolver.getTriangles(), meshSolver.getNodeStates());
}

void SoftBodyCutFEM2DApp::performStep(float stepsize)
{
	if (stepsize == 0 && !enableDirichletBoundaries)
	{
		CI_LOG_E("Can't solve for static solution if dirichlet boundaries are disabled");
		return;
	}

    if (worker != nullptr && !worker->isDone()) {
        //still running
        return;
    }

    //declare background task
    std::function<void(BackgroundWorker*)> task = [this, stepsize](BackgroundWorker* worker) {
        //pass arguments
        meshSolver.setGravity(gravity.cast<real>());
        meshSolver.setMaterialParameters(youngsModulus, poissonsRatio);
        meshSolver.setMass(mass);
        meshSolver.setDamping(dampingAlpha, dampingBeta);
        meshSolver.setDenseLinearSolver(denseLinearSolverType);
        meshSolver.setSparseLinearSolver(sparseLinearSolverType);
        meshSolver.setTimeIntegrator(timeIntegratorType);
        meshSolver.setUseSparseMatrices(useSparseMatrices);
        meshSolver.setSparseSolveIterations(sparseSolverIterations);
        meshSolver.setSparseSolveTolerance(sparseSolverTolerance);
        meshSolver.setRotationCorrection(rotationCorrectionMode);
        meshSolver.setTimestep(stepsize==0 ? 1 : stepsize);
		meshSolver.setGroundPlane(groundPlaneHeight, groundPlaneAngle);
		meshSolver.setEnableCollision(enableCollision);
		meshSolver.setCollisionResolution(collisionResolutionMode);
		meshSolver.setCollisionVelocityDamping(collisionVelocityDamping);
        meshSolver.setGroundStiffness(groundStiffness_);
        meshSolver.setCollisionSoftmaxAlpha(softmaxAlpha_);

        gridSolver.setGravity(gravity.cast<real>());
        gridSolver.setMaterialParameters(youngsModulus, poissonsRatio);
        gridSolver.setMass(mass);
        gridSolver.setDamping(dampingAlpha, dampingBeta);
        gridSolver.setDenseLinearSolver(denseLinearSolverType);
        gridSolver.setSparseLinearSolver(sparseLinearSolverType);
        gridSolver.setTimeIntegrator(timeIntegratorType);
        gridSolver.setUseSparseMatrices(useSparseMatrices);
        gridSolver.setSparseSolveIterations(sparseSolverIterations);
        gridSolver.setSparseSolveTolerance(sparseSolverTolerance);
        gridSolver.setExplicitDiffusion(gridExplicitDiffusion);
        gridSolver.setHardDirichletBoundaries(gridHardDirichletBoundaries);
        gridSolver.setAdvectionMode(gridAdvectionMode);
        gridSolver.setRotationCorrection(rotationCorrectionMode);
        gridSolver.setTimestep(stepsize == 0 ? 1 : stepsize);
		gridSolver.setGroundPlane(groundPlaneHeight, groundPlaneAngle);
		gridSolver.setEnableCollision(enableCollision);
		gridSolver.setCollisionResolution(collisionResolutionMode);
		gridSolver.setCollisionVelocityDamping(collisionVelocityDamping);
        gridSolver.setGroundStiffness(groundStiffness_);
        gridSolver.setCollisionSoftmaxAlpha(softmaxAlpha_);

		results->settings_ = gridSolver.getSettings();

        //solve it
        if (int(computationMode) & int(ComputationMode::COMPUTATION_MODE_MESH)) {
            auto start1 = std::chrono::steady_clock::now();
            if (stepsize == 0) {
                meshSolver.solveStaticSolution(worker);
                if (worker->isInterrupted()) return;
				invalidateRendering();
            }
            else {
                meshSolver.solveDynamicSolution(worker);
                if (worker->isInterrupted()) return;
				invalidateRendering();
            }
            auto duration1 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start1);
            meshElapsedSeconds = duration1.count() / 1000.0;
            results->meshResultsDisplacement_.push_back(meshSolver.getCurrentDisplacements());
        }

        if (int(computationMode) & int(ComputationMode::COMPUTATION_MODE_GRID)) {
            auto start2 = std::chrono::steady_clock::now();
            if (stepsize == 0) {
                gridSolver.solveStaticSolution(worker);
                if (worker->isInterrupted()) return;
				invalidateRendering();
            }
            else {
                gridSolver.solveDynamicSolution(worker);
                if (worker->isInterrupted()) return;
				invalidateRendering();
            }
            auto duration2 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start2);
            gridElapsedSeconds = duration2.count() / 1000.0;
            results->gridResultsSdf_.push_back(gridSolver.getSdfSolution());
            results->gridResultsUxy_.push_back(gridSolver.getUSolution());
        }

		results->numSteps_++;
    };

    //start background worker
    //worker = make_shared<BackgroundWorker>(task);
	BackgroundWorker w;
	task(&w);

    CI_LOG_I("Background worker started");
}

void SoftBodyCutFEM2DApp::initGridWithTorus()
{
    gridSolver = SoftBodyGrid2D::CreateTorus(
        torusOuterRadius, torusInnerRadius, gridResolution,
		enableDirichletBoundaries, Vector2(0, 0), neumannForce.cast<real>());
    results->initGridReference(gridSolver);
}

void SoftBodyCutFEM2DApp::initGridWithRect()
{
    gridSolver = SoftBodyGrid2D::CreateBar(
        rectCenter.cast<real>(), rectHalfSize.cast<real>(), gridResolution,
        fitObjectToGrid, enableDirichletBoundaries, Vector2(0, 0), neumannForce.cast<real>());
    results->initGridReference(gridSolver);
}

void SoftBodyCutFEM2DApp::initTriMeshWithTorus()
{
    meshSolver = SoftBodyMesh2D::CreateTorus(
        torusOuterRadius, torusInnerRadius, 
        gridResolution, enableDirichletBoundaries, Vector2(0, 0), neumannForce.cast<real>().eval());
    results->initMeshReference(meshSolver);
}

void SoftBodyCutFEM2DApp::initTriMeshWithGrid()
{
    meshSolver = SoftBodyMesh2D::CreateBar(
        rectCenter.cast<real>(), rectHalfSize.cast<real>(),
        gridResolution, fitObjectToGrid,
		enableDirichletBoundaries, Vector2(0, 0), neumannForce.cast<real>());
    results->initMeshReference(meshSolver);
}

void SoftBodyCutFEM2DApp::saveResults()
{
	//get save path
	fs::path path = getSaveFilePath(fs::path("../saves/"), std::vector<std::string>({ ".dat" }));
	if (path.empty()) {
		CI_LOG_W("Saving cancelled by the user");
		return;
	}
	std::string pathS = path.string();
	CI_LOG_I("Save results to " << pathS);

	//save it
	std::ofstream ofs(pathS, std::ofstream::binary | std::ofstream::trunc);
	boost::archive::binary_oarchive oa(ofs);
	oa << results;
	CI_LOG_I("Results saved");
}

#if 1
CINDER_APP( SoftBodyCutFEM2DApp, RendererGl, [&](App::Settings *settings)
{
	settings->setWindowSize(1280, 720);
} )
#endif
