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

#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unique_ptr.hpp>

#include <tinyformat.h>

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

#include <IInverseProblem.h>
#include <InverseProblem_HumanSoftTissue.h>
#include <InverseProblem_ProbabilisticElastography.h>
#include <InverseProblem_Adjoint_YoungsModulus.h>
#include <InverseProblem_Adjoint_InitialConfiguration.h>
#include <InverseProblem_Adjoint_Dynamic.h>
#include "PartialObservations.h"
#include <GridVisualization.h>

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace ar;
using namespace Eigen;

class InverseSoftBodyApp : public App {
public:
	InverseSoftBodyApp();
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
 
    enum class ComputationMode
    {
        COMPUTATION_MODE_GRID = 1,
        COMPUTATION_MODE_MESH = 2,
        COMPUTATION_MODE_BOTH = 3
    };
    int computationMode;

    //input - loading
    std::unique_ptr<SoftBody2DResults> inputResults;
    int inputTimestep;

    //input - generate
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
    bool spaceBarPressed = false;
    TimeIntegrator::Integrator timeIntegratorType;
    TimeIntegrator::DenseLinearSolver denseLinearSolverType;
    TimeIntegrator::SparseLinearSolver sparseLinearSolverType;
    bool useSparseMatrices;
    int sparseSolverIterations;
    real sparseSolverTolerance;
    SoftBodySimulation::RotationCorrection rotationCorrectionMode;
    Vector2f gravity;
    Vector2f neumannForce;
    float youngsModulus;
    float poissonsRatio;
    float mass;
    float dampingAlpha;
    float dampingBeta;
    float timestep;
    bool gridExplicitDiffusion;
    bool gridHardDirichletBoundaries;
    SoftBodyGrid2D::AdvectionMode gridAdvectionMode;
    bool solveForwardStatic;
    SoftBodyMesh2D meshSolver;
    SoftBodyGrid2D gridSolver;

	bool enableDirichletBoundaries_;
	bool enableCollision_;
	real groundPlaneHeight;
	real groundPlaneAngle;
	real groundStiffness_;
	real collisionSoftminAlpha_;

	//reconstruction
	enum class ReconstructionMode
	{
		HUMAN_SOFT_TISSUE, //human soft tissue algorithm
		PROBABILISTIC_ELASTOGRAPHY,
        ADJOINT_YOUNGS_MODULUS,
        ADJOINT_INITIAL_POSITIONS,
		ADJOINT_DYNAMIC
	};
	ReconstructionMode reconstructionMode;
	std::map<ReconstructionMode, std::unique_ptr<IInverseProblem>> reconstructionAlgorithms;
	InverseProblemOutput resultMesh;
	InverseProblemOutput resultGrid;
	std::optional<GridUtils2D::grid_t> cleanedSdf;
    bool usePartialObservations;
    PartialObservations partialObservations;

    //rendering
	bool showGrid;
	enum class GridVisualizationMode
	{
		GRID_VISUALIZATION_U,
		GRID_VISUALIZATION_SOLUTION,
		GRID_VISUALIZATION_BOUNDARY,
		_GRID_VISUALIZATION_COUNT
	};
	GridVisualizationMode showGridSolution;
	bool showReconstructedSolution = false;
	std::mutex invalidateRenderingMutex;
	std::condition_variable invalidateRenderingConditionVariable;
	//visualization helpers
	GridVisualization visualization;

    //processing
    ar::BackgroundWorkerPtr worker;
	double meshElapsedSeconds = 0;
	double gridElapsedSeconds = 0;


	int reconstructSdfIterations = 5;

private:
    void invalidateRendering(bool wait);
    void drawSimulation();
	void loadSimulation();

    void resetSimulation();
    void runSimulation();

	void resetReconstruction();
	void performReconstruction();

	void reconstructSDF(int mode, int iterations);
    void testPlot();
};

InverseSoftBodyApp::InverseSoftBodyApp()
{
	//initial config
    printMode = false;
    computationMode = (int)ComputationMode::COMPUTATION_MODE_GRID;
    gridResolution = 22;
    fitObjectToGrid = false;
    scene = Scene::SCENE_BAR;
    torusOuterRadius = 0.1;
    torusInnerRadius = 0.03;
    rectCenter << 0.5, 0.7;
    rectHalfSize << 0.24, 0.06;
    showGrid = true;
    timeIntegratorType = TimeIntegrator::Integrator::Newmark1;
    denseLinearSolverType = TimeIntegrator::DenseLinearSolver::PartialPivLU;
    sparseLinearSolverType = TimeIntegrator::SparseLinearSolver::BiCGSTAB;
    useSparseMatrices = false;
    sparseSolverIterations = 100;
    sparseSolverTolerance = 1e-5;
    rotationCorrectionMode = SoftBodySimulation::RotationCorrection::Corotation;
    inputTimestep = 0;

    gridExplicitDiffusion = true;
    gridHardDirichletBoundaries = false;
    gridAdvectionMode = SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD;
    gravity = Vector2f(0, -30);
    neumannForce = Vector2f(0, 0);
    youngsModulus = 200;
    poissonsRatio = 0.45;
    mass = 1.0;
    dampingAlpha = 0.01;
    dampingBeta = 0.01;
    timestep = 0.01;
	solveForwardStatic = false;

	enableDirichletBoundaries_ = true;
	enableCollision_ = false;
	groundPlaneHeight = 0.6;
	groundPlaneAngle = 0;
	groundStiffness_ = 1000;
	collisionSoftminAlpha_ = 100;

	//create reconstruction algorithms
	reconstructionAlgorithms.emplace(ReconstructionMode::HUMAN_SOFT_TISSUE, std::make_unique<InverseProblem_HumanSoftTissue>());
	reconstructionAlgorithms.emplace(ReconstructionMode::PROBABILISTIC_ELASTOGRAPHY, std::make_unique<InverseProblem_ProbabilisticElastography>());
    reconstructionAlgorithms.emplace(ReconstructionMode::ADJOINT_YOUNGS_MODULUS, std::make_unique<InverseProblem_Adjoint_YoungsModulus>());
    reconstructionAlgorithms.emplace(ReconstructionMode::ADJOINT_INITIAL_POSITIONS, std::make_unique<InverseProblem_Adjoint_InitialConfiguration>());
	reconstructionAlgorithms.emplace(ReconstructionMode::ADJOINT_DYNAMIC, std::make_unique<InverseProblem_Adjoint_Dynamic>());
	reconstructionMode = ReconstructionMode::ADJOINT_DYNAMIC;
    usePartialObservations = true;
	showGridSolution = GridVisualizationMode::GRID_VISUALIZATION_BOUNDARY;

	resetSimulation();
}

void InverseSoftBodyApp::setup()
{
	//ANT TWEAK BAR
	//parameter ui, must happen before user-camera
	params = params::InterfaceGl::create(getWindow(), "Parameters", ivec2(350, 800));
    params->setOptions("", "refresh=0.05");
    params->addParam("PrintMode", &printMode).label("Print Mode");

	//input
    params->addButton("LoadSimulation", std::function<void()>([this]() {this->loadSimulation(); }), "label='Load Simulation'");
    params->addParam("InputStep", std::function<void(int)>([this](int v)
    {
        inputTimestep = v;
		showReconstructedSolution = false;
        invalidateRendering(false);
    }), std::function<int()>([this]()
    {
        return inputTimestep;
    })).label("Input Step").min(0);
    vector<string> computationModeEnums(4);
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_GRID] = "grid";
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_MESH] = "mesh";
    computationModeEnums[(int)ComputationMode::COMPUTATION_MODE_BOTH] = "overlay / both";
    params->addParam("Mode", computationModeEnums, (int*)&computationMode, "label='Mode'");

	//Soft Body properties - Generate Inputs
    params->addParam("InputResolution",
        std::function<void(int)>([this](int newValue) {this->gridResolution = newValue; this->resetSimulation(); }),
        std::function<int()>([this]() {return this->gridResolution; })
    ).min(4).group("Generate Input").label("Grid resolution").keyIncr("PGUP").keyDecr("PGDOWN");
    params->addParam("FitObjectToGrid",
        std::function<void(bool)>([this](bool newValue) {this->fitObjectToGrid = newValue; this->resetSimulation(); }),
        std::function<bool()>([this]() {return this->fitObjectToGrid; })
    ).group("Generate Input").label("Fit Object To Grid");
    vector<string> sceneEnums = { "Bar", "Torus" };
    params->addParam("Scene", sceneEnums, (int*)&scene).group("Generate Input").label("Scene")
        .accessors(std::function<void(int)>([this](int v)
    {
        scene = (Scene)v;
        resetSimulation();
        if (scene == Scene::SCENE_TORUS) {
            params->setOptions("InputTorusOuterRadius", "visible=true");
            params->setOptions("InputTorusInnerRadius", "visible=true");
            params->setOptions("InputRectCenterX", "visible=false");
            params->setOptions("InputRectHalfSizeX", "visible=false");
            params->setOptions("InputRectCenterY", "visible=false");
            params->setOptions("InputRectHalfSizeY", "visible=false");
        }
        else {
            params->setOptions("InputTorusOuterRadius", "visible=false");
            params->setOptions("InputTorusInnerRadius", "visible=false");
            params->setOptions("InputRectCenterX", "visible=true");
            params->setOptions("InputRectHalfSizeX", "visible=true");
            params->setOptions("InputRectCenterY", "visible=true");
            params->setOptions("InputRectHalfSizeY", "visible=true");
        }
    }), std::function<int()>([this]() {return (int)scene; }));
    params->addParam("InputTorusOuterRadius",
        std::function<void(double)>([this](double newValue) {this->torusOuterRadius = newValue; this->resetSimulation(); }),
        std::function<double()>([this]() {return this->torusOuterRadius; })
    ).min(0.01).max(0.5).step(0.01).group("Generate Input").label("Torus outer radius").visible(false);
    params->addParam("InputTorusInnerRadius",
        std::function<void(double)>([this](double newValue) {this->torusInnerRadius = newValue; this->resetSimulation(); }),
        std::function<double()>([this]() {return this->torusInnerRadius; })
    ).min(0.01).max(0.5).step(0.01).group("Generate Input").label("Torus inner radius").visible(false);
    params->addParam("InputRectCenterX",
        std::function<void(float)>([this](float newValue) {this->rectCenter.x() = newValue; this->resetSimulation(); }),
        std::function<float()>([this]() {return this->rectCenter.x(); })
    ).step(0.01).group("Generate Input").label("Rect center X").visible(true);
    params->addParam("InputRectCenterY",
        std::function<void(float)>([this](float newValue) {this->rectCenter.y() = newValue; this->resetSimulation(); }),
        std::function<float()>([this]() {return this->rectCenter.y(); })
    ).step(0.01).group("Generate Input").label("Rect center Y").visible(true);
    params->addParam("InputRectHalfSizeX",
        std::function<void(float)>([this](float newValue) {this->rectHalfSize.x() = newValue; this->resetSimulation(); }),
        std::function<float()>([this]() {return this->rectHalfSize.x(); })
    ).step(0.01).group("Generate Input").label("Rect half size X").visible(true);
    params->addParam("InputRectHalfSizeY",
        std::function<void(float)>([this](float newValue) {this->rectHalfSize.y() = newValue; this->resetSimulation(); }),
        std::function<float()>([this]() {return this->rectHalfSize.y(); })
    ).step(0.01).group("Generate Input").label("Rect half size Y").visible(true);
    params->addButton("InputReset", std::function<void()>([this]() {this->resetSimulation(); }), "label='Reset' group='Generate Input' key=r");

    params->addParam("SoftBodyGravity", &gravity.y()).step(0.001).group("Generate Input").label("Gravity");
    params->addParam("SoftBodyNeumannForce", &neumannForce.y()).step(0.001).group("Generate Input").label("Neumann Force")
        .accessors(std::function<void(float)>([this](float v)
    {
        neumannForce.y() = v;
        resetSimulation();
    }), std::function<float()>([this]() {
        return (float)neumannForce.y();
    }));
    params->addParam("SoftBodyYoungsModulus", &youngsModulus).min(0).step(0.01).group("Generate Input").label("Young's modulus");
    params->addParam("SoftBodyPoissonsRatio", &poissonsRatio).min(0.0001).max(0.4999).step(0.01).group("Generate Input").label("Poisson's ratio");
    params->addParam("SoftBodyMass", &mass).min(0.0001).step(0.01).group("Generate Input").label("Mass");
    params->addParam("SoftBodyDampingAlpha", &dampingAlpha).min(0).step(0.001).group("Generate Input").label("Damping on mass");
    params->addParam("SoftBodyDampingBeta", &dampingBeta).min(0).step(0.001).group("Generate Input").label("Damping on stiffness");
    vector<string> timeIntegratorTypeNames = { "Newmark 1", "Newmark 2", "Central Differences", "Linear Accelleration", "Newmark 3", "HHT-alpha" };
    params->addParam("SoftBodyTimeIntegration", timeIntegratorTypeNames, (int*)&timeIntegratorType).group("Generate Input").label("Time Integrator")
        .accessors(std::function<void(int)>([this](int v)
    {
        timeIntegratorType = (TimeIntegrator::Integrator)v;
        resetSimulation();
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
        }
        else
        {
            params->setOptions("SoftBodyDenseLinearSolver", "visible=true");
            params->setOptions("SoftBodySparseLinearSolver", "visible=false");
            params->setOptions("SoftBodySparseSolverIterations", "visible=false");
            params->setOptions("SoftBodySparseSolverTolerance", "visible=false");
        }
    }), std::function<bool()>([this]()
    {
        return useSparseMatrices;
    })).group("Generate Input").label("Sparse matrices");
    vector<string> denseLinearSolverTypeNames = { "PartialPivLU", "FullPivLU", "HouseholderQR", "ColPivHousholderQR", "FullPivHouseholderQR", "CompleteOrthogonalDecomposition", "LLT", "LDLT" };
    params->addParam("SoftBodyDenseLinearSolver", denseLinearSolverTypeNames, (int*)&denseLinearSolverType).group("Generate Input").label("Dense Linear Solver");
    vector<string> sparseLinearSolverTypeNames = { "Conjugate Gradient", "BiCGSTAB ", "Sparse-LU" };
    params->addParam("SoftBodySparseLinearSolver", sparseLinearSolverTypeNames, (int*)&sparseLinearSolverType).group("Generate Input").label("Sparse Linear Solver");
    params->addParam("SoftBodySparseSolverIterations", &sparseSolverIterations).group("Generate Input").label("Sparese Solver iterations").min(0);
    params->addParam("SoftBodySparseSolverTolerance", &sparseSolverTolerance).group("Generate Input").label("Sparese Solver tolerance").min(0).step(0.00001);
#else
    vector<string> denseLinearSolverTypeNames = { "PartialPivLU", "FullPivLU", "HouseholderQR", "ColPivHousholderQR", "FullPivHouseholderQR", "CompleteOrthogonalDecomposition", "LLT", "LDLT" };
    params->addParam("SoftBodyDenseLinearSolver", denseLinearSolverTypeNames, (int*)&denseLinearSolverType).group("Generate Input").label("Dense Linear Solver");
#endif
    params->addParam("SoftBodyTimeStep", &timestep).min(0.001).step(0.001).group("Generate Input").label("Time step");
    params->addParam("SoftBodyGridExplicitDiffusion", &gridExplicitDiffusion, "group='Generate Input' label='Grid Displacement Diffusion' true='explicit (post-process)' false='implicit (matrix)'");
    params->addParam("SoftBodyGridHardDirichletBoundaries", &gridHardDirichletBoundaries).group("Generate Input").label("Grid Hard Dirichlet Boundaries");
    vector<string> softBodyAdvectionNames;
    for (int i = 0; i < static_cast<int>(SoftBodyGrid2D::AdvectionMode::_COUNT_); ++i)
        softBodyAdvectionNames.push_back(SoftBodyGrid2D::advectionModeName(static_cast<SoftBodyGrid2D::AdvectionMode>(i)));
    params->addParam("SoftBodyAdvectionMode", softBodyAdvectionNames, (int*)&gridAdvectionMode).group("Generate Input").label("Grid Advection");

    vector<string> rotationCorrectionNames = { "None", "Corotation" };
    params->addParam("SoftBodyRotationCorrection", rotationCorrectionNames, (int*)&rotationCorrectionMode).group("Generate Input").label("Rotation correction");

	params->addParam("SoftBodyDirichlet",
		std::function<void(bool)>([this](bool newValue) {this->enableDirichletBoundaries_ = newValue; this->resetSimulation(); }),
		std::function<bool()>([this]() {return this->enableDirichletBoundaries_; }))
		.group("Generate Input").label("Enable Dirichlet Boundaries");
	params->addParam("SoftBodyCollision", &enableCollision_).group("Generate Input").label("Enable Collision");
	params->addParam("InputGroundPlaneHeight", &groundPlaneHeight).min(0).max(1).step(0.001).group("Generate Input").label("Ground Height");
	params->addParam("InputGroundPlaneAngle", &groundPlaneAngle).min(-1).max(1).step(0.001).group("Generate Input").label("Ground Angle");
	params->addParam("SoftBodyCollisionGroundStiffness", &groundStiffness_).group("Generate Input").label("Ground Stiffness").min(0).step(0.001);
	params->addParam("SoftBodyCollisionSoftmaxAlpha", &collisionSoftminAlpha_).group("Generate Input").label("Collision Softmax-Alpha").min(1).max(1000).step(0.001);

	params->addParam("SoftBodyStaticDynamic", &solveForwardStatic).group("Generate Input").label("Solution Mode").optionsStr("true='Static' false='Dynamic'");

	//reconstruction
	vector<string> reconstructionModeNames = { 
	    "Human Soft Tissue", 
	    "Probabilistic Elastography",
        "Adjoint - Youngs Modulus",
        "Adjoint - Initial Positions",
		"Adjoint - Dynamic"
	};
	params->addParam("ReconstructionAlgorithm", reconstructionModeNames, (int*)&reconstructionMode)
		.group("Reconstruction").label("Algorithm")
		.accessors(std::function<void(int)>([this](int v) {
			reconstructionAlgorithms[reconstructionMode]->setParamsVisibility(params, false);
			reconstructionMode = static_cast<ReconstructionMode>(v);
			reconstructionAlgorithms[reconstructionMode]->setParamsVisibility(params, true);
			resetReconstruction();
		}), std::function<int()>([this]() {
			return static_cast<int>(reconstructionMode);
		}));
	for (auto& p : reconstructionAlgorithms) {
		p.second->setupParams(params, "Reconstruction");
		p.second->setParamsVisibility(params, false);
	}
	reconstructionAlgorithms[reconstructionMode]->setParamsVisibility(params, true);
    params->addParam("ReconstructionUsePartialObservation", &usePartialObservations).label("Partial Observation").group("Reconstruction");
	params->addButton("ReconstructionRun", std::function<void()>([this]() {this->performReconstruction(); }), "group='Reconstruction' label='Solve' key=RETURN");
	params->addButton("ReconstructionCancel", std::function<void()>([this]() {
		if (worker) worker->interrupt();
	}), "group='Reconstruction' label='Cancel'");

    //partial observations
    partialObservations.initParams(params);

	//rendering
    params->addParam("RenderingShowGrid", &showGrid).group("Rendering").label("Show grid");
	vector<string> gridVisualizationModeEnums((int)GridVisualizationMode::_GRID_VISUALIZATION_COUNT);
	gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_U] = "u";
	gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_SOLUTION] = "solution";
	gridVisualizationModeEnums[(int)GridVisualizationMode::GRID_VISUALIZATION_BOUNDARY] = "boundary";
	params->addParam("RenderingShowGridSolutionMode", gridVisualizationModeEnums, (int*)&showGridSolution, "label='Grid solution' group=Rendering");
	params->addParam("RenderingShowSolution", std::function<void(bool)>([this](bool v)
	{
		showReconstructedSolution = v;
		invalidateRendering(false);
	}), std::function<bool()>([this]()
	{
		return showReconstructedSolution;
	})).label("Show Reconstruction");

	params->addParam("RecontructSdfIterations", &reconstructSdfIterations).label("Reconstuct Sdf - Iterations").min(0).group("Test");
	params->addButton("ReconstructSdfViscosity", std::function<void()>([this]() {this->reconstructSDF(1, this->reconstructSdfIterations); }), "label='Reconstruct SDF - Viscosity' group=Test ");
	params->addButton("ReconstructSdfUpwind", std::function<void()>([this]() {this->reconstructSDF(2, this->reconstructSdfIterations); }), "label='Reconstruct SDF - Upwind' group=Test ");
	params->addButton("ReconstructSdfSussmann", std::function<void()>([this]() {this->reconstructSDF(3, this->reconstructSdfIterations); }), "label='Reconstruct SDF - Sussmann' group=Test ");
    params->addButton("ReconstructSdfFastMarching", std::function<void()>([this]() {this->reconstructSDF(4, this->reconstructSdfIterations); }), "label='Reconstruct SDF - Fast Marching' group=Test ");
    params->addButton("TestPlot", std::function<void()>([this]() {this->testPlot(); }), "label='Test Plot' group=Test ");
	params->setOptions("Test", "opened=false");

	visualization.setup();
}

void InverseSoftBodyApp::keyDown(KeyEvent event)
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
		string fileName = string("../screenshots/InverseSoftBodyApp-") + string(buf) + ".png";
		//write out
		writeImage(fileName, surface);
        CI_LOG_I("Screenshot saved to " << fileName);
	}
    else if (event.getCode() == KeyEvent::KEY_SPACE)
    {
        spaceBarPressed = true;
    }
}

void InverseSoftBodyApp::keyUp(KeyEvent event)
{
    App::keyUp(event);
    if (event.isHandled()) return;
    if (event.getCode() == KeyEvent::KEY_SPACE)
    {
        spaceBarPressed = false;
    }
}

void InverseSoftBodyApp::mouseDown( MouseEvent event )
{
	App::mouseDown(event);
}

void InverseSoftBodyApp::update()
{
    //perform time stepping
    if (spaceBarPressed) {
		runSimulation();
    }

	//update transfer function editor
	visualization.update();
}

void InverseSoftBodyApp::drawSimulation()
{
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

	// Draw partial observations
	if (computationMode & (int)ComputationMode::COMPUTATION_MODE_GRID
		&& usePartialObservations) {
		gl::ScopedMatrices m;
		visualization.applyTransformation();
		gl::ScopedColor c;
		partialObservations.draw();
	}

	// Draw mesh
	if (computationMode & (int)ComputationMode::COMPUTATION_MODE_MESH) {
		visualization.meshDraw();
	}

	// Draw ground
	if (enableCollision_)
	{
		visualization.drawGround(groundPlaneHeight, groundPlaneAngle);
	}
}

void InverseSoftBodyApp::draw()
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

    //draw the simulation
    if (inputResults != nullptr)
    {
        drawSimulation();
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

	//draw result
	{
		static Font font("Arial", 20);
		static ColorA color(1, 1, 1, 1);
		gl::ScopedColor scopedColor;
		if (printMode)
			gl::color(0, 0, 0);
		else
			gl::color(1, 1, 1);
		int gridSize = std::min(getWindowWidth(), getWindowHeight()) - 100;
		int x = (getWindowWidth() + gridSize) / 2 + 50;
		int y = 50;
		int stepY = 15;

		gl::drawString("Results Mesh", vec2(x, y), color, font); y += stepY;
		gl::drawString("<Prop>: <Computed> - <Truth> (<StdDev>)", vec2(x, y), color, font); y += stepY;
		if (resultMesh.youngsModulus_.has_value()) {
			gl::drawString(
				tfm::format("Young's Modulus: %.2f - %.2f (%.2f)", resultMesh.youngsModulus_.value(), inputResults->settings_.youngsModulus_, resultMesh.youngsModulusStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.poissonsRatio_.has_value()) {
			gl::drawString(
				tfm::format("Poisson Ratio: %.2f - %.2f", resultMesh.poissonsRatio_.value(), inputResults->settings_.poissonsRatio_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.mass_.has_value()) {
			gl::drawString(
				tfm::format("Mass: %.2f - %.2f (%.2f)", resultMesh.mass_.value(), inputResults->settings_.mass_, resultMesh.massStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.dampingAlpha_.has_value()) {
			gl::drawString(
				tfm::format("Mass Damping: %.2f - %.2f (%.2f)", resultMesh.dampingAlpha_.value(), inputResults->settings_.dampingAlpha_, resultMesh.dampingAlphaStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.dampingBeta_.has_value()) {
			gl::drawString(
				tfm::format("Stiffness Damping: %.2f - %.2f (%.2f)", resultMesh.dampingBeta_.value(), inputResults->settings_.dampingBeta_, resultMesh.dampingBetaStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.groundHeight.has_value())
		{
			gl::drawString(
				tfm::format("Ground Height: %.3f - %.3f", resultMesh.groundHeight.value(), inputResults->settings_.groundPlaneHeight_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultMesh.groundAngle.has_value())
		{
			gl::drawString(
				tfm::format("Ground Angle: %.3f - %.3f", resultMesh.groundAngle.value(), inputResults->settings_.groundPlaneAngle_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		gl::drawString(tfm::format("Final Cost: %.3f", resultMesh.finalCost_), vec2(x, y), color, font); y += stepY;
		if (meshElapsedSeconds > 0)
			gl::drawString(tfm::format("Time: %.2f sec", meshElapsedSeconds), vec2(x, y), color, font); y += stepY;

		y += stepY;
		gl::drawString("Results Grid", vec2(x, y), color, font); y += stepY;
		gl::drawString("<Prop>: <Computed> - <Truth> (<StdDev>)", vec2(x, y), color, font); y += stepY;
		if (resultGrid.youngsModulus_.has_value()) {
			gl::drawString(
				tfm::format("Young's Modulus: %.2f - %.2f (%.2f)", resultGrid.youngsModulus_.value(), inputResults->settings_.youngsModulus_, resultGrid.youngsModulusStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.poissonsRatio_.has_value()) {
			gl::drawString(
				tfm::format("Poisson Ratio: %.2f - %.2f", resultMesh.poissonsRatio_.value(), inputResults->settings_.poissonsRatio_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.mass_.has_value()) {
			gl::drawString(
				tfm::format("Mass: %.2f - %.2f (%.2f)", resultGrid.mass_.value(), inputResults->settings_.mass_, resultGrid.massStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.dampingAlpha_.has_value()) {
			gl::drawString(
				tfm::format("Mass Damping: %.2f - %.2f (%.2f)", resultGrid.dampingAlpha_.value(), inputResults->settings_.dampingAlpha_, resultGrid.dampingAlphaStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.dampingBeta_.has_value()) {
			gl::drawString(
				tfm::format("Stiffness Damping: %.2f - %.2f (%.2f)", resultGrid.dampingBeta_.value(), inputResults->settings_.dampingBeta_, resultGrid.dampingBetaStdDev_.value_or(0))
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.groundHeight.has_value())
		{
			gl::drawString(
				tfm::format("Ground Height: %.3f - %.3f", resultGrid.groundHeight.value(), inputResults->settings_.groundPlaneHeight_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		if (resultGrid.groundAngle.has_value())
		{
			gl::drawString(
				tfm::format("Ground Angle: %.3f - %.3f", resultGrid.groundAngle.value(), inputResults->settings_.groundPlaneAngle_)
				, vec2(x, y), color, font);
			y += stepY;
		}
		gl::drawString(tfm::format("Final Cost: %.3f", resultGrid.finalCost_), vec2(x, y), color, font); y += stepY;
		if (gridElapsedSeconds > 0)
			gl::drawString(tfm::format("Time: %.2f sec", gridElapsedSeconds), vec2(x, y), color, font); y += stepY;
	}

	// Draw the interface
	visualization.setTfeVisible(!printMode);
	params->draw();
	visualization.drawTransferFunctionEditor();

	//notify waiting threads
	invalidateRenderingConditionVariable.notify_all();
}

void InverseSoftBodyApp::invalidateRendering(bool wait)
{
	//GRID
	SoftBodyGrid2D::grid_t referenceSdf = inputResults->gridReferenceSdf_;
	SoftBodyGrid2D::grid_t currentSdf;
	SoftBodyGrid2D::grid_t currentUx = GridUtils2D::grid_t::Zero(gridSolver.getGridResolution(), gridSolver.getGridResolution());
	SoftBodyGrid2D::grid_t currentUy = GridUtils2D::grid_t::Zero(gridSolver.getGridResolution(), gridSolver.getGridResolution());
	if (inputTimestep > 0) {
		if (cleanedSdf.has_value()) {
			currentSdf = cleanedSdf.value();
		}
		else if (showReconstructedSolution && resultGrid.resultGridSdf_.has_value()) {
			currentSdf = resultGrid.resultGridSdf_.value();
		}
		else if (showReconstructedSolution && resultGrid.initialGridSdf_.has_value()) {
			currentSdf = resultGrid.initialGridSdf_.value();
		}
		else {
			if (inputResults->gridResultsSdf_.size() > inputTimestep - 1) {
				currentSdf = inputResults->gridResultsSdf_[inputTimestep - 1];
				currentUx = inputResults->gridResultsUx_[inputTimestep - 1];
				currentUy = inputResults->gridResultsUy_[inputTimestep - 1];
			}
			else
				currentSdf = inputResults->gridReferenceSdf_;
		}
	}
	else
	{
		currentSdf = inputResults->gridReferenceSdf_;
	}
	if (gridSolver.hasSolution())
	{
		visualization.setGrid(gridSolver.getSdfReference(), gridSolver.getSdfSolution(), gridSolver.getUGridX(), gridSolver.getUGridY());
	}
	else
	{
		GridUtils2D::grid_t z = GridUtils2D::grid_t::Zero(gridSolver.getGridResolution(), gridSolver.getGridResolution());
		visualization.setGrid(gridSolver.getSdfReference(), gridSolver.getSdfReference(), z, z);
	}

	//MESH
	auto pos = inputResults->meshReferencePositions_;
	if (inputTimestep > 0) {
		if (showReconstructedSolution && resultMesh.resultMeshDisp_.has_value()) {
			for (size_t i = 0; i < pos.size(); ++i)
				pos[i] += resultMesh.resultMeshDisp_.value()[i];
		}
		else if (showReconstructedSolution && resultMesh.initialMeshPositions_.has_value()) {
			pos = resultMesh.initialMeshPositions_.value();
		}
		else {
			if (inputResults->meshResultsDisplacement_.size() > inputTimestep - 1)
				for (size_t i = 0; i < pos.size(); ++i)
					pos[i] += inputResults->meshResultsDisplacement_[inputTimestep - 1][i];
		}
	}
	visualization.setMesh(pos, inputResults->meshReferenceIndices_, meshSolver.getNodeStates());

	//wait for the rendering to happen
	if (wait)
	{
		std::unique_lock<std::mutex> lock(invalidateRenderingMutex);
		invalidateRenderingConditionVariable.wait_for(lock, 50ms);
	}
}

void InverseSoftBodyApp::loadSimulation()
{
	//get save path
	fs::path path = getOpenFilePath(fs::path("../saves/"));
	if (path.empty()) {
		CI_LOG_W("Saving cancelled by the user");
		return;
	}
	std::string pathS = path.string();
	CI_LOG_I("Load results from " << pathS);

	//load it
	std::ifstream ifs(pathS, std::ifstream::binary);
	boost::archive::binary_iarchive ia(ifs);
	ia >> inputResults;
	CI_LOG_I("Results loaded");
    params->setOptions("InputStep", "max=" + std::to_string(inputResults->numSteps_));
	inputTimestep = 0;

	resetReconstruction();
}

void InverseSoftBodyApp::resetSimulation()
{
	//reset solution
	inputResults = std::make_unique<SoftBody2DResults>();

	//create scene
	switch (scene)
	{
	case Scene::SCENE_BAR:
		gridSolver = SoftBodyGrid2D::CreateBar(
			rectCenter.cast<real>(), rectHalfSize.cast<real>(), gridResolution,
			fitObjectToGrid, enableDirichletBoundaries_, Vector2(0, 0), neumannForce.cast<real>());
		inputResults->initGridReference(gridSolver);
		meshSolver = SoftBodyMesh2D::CreateBar(
			rectCenter.cast<real>(), rectHalfSize.cast<real>(),
			gridResolution, fitObjectToGrid,
			enableDirichletBoundaries_, Vector2(0, 0), neumannForce.cast<real>());
		inputResults->initMeshReference(meshSolver);
		break;
	case Scene::SCENE_TORUS:
		gridSolver = SoftBodyGrid2D::CreateTorus(
			torusOuterRadius, torusInnerRadius, gridResolution,
			enableDirichletBoundaries_, Vector2(0, 0), neumannForce.cast<real>());
		inputResults->initGridReference(gridSolver);
		meshSolver = SoftBodyMesh2D::CreateTorus(
			torusOuterRadius, torusInnerRadius,
			gridResolution, enableDirichletBoundaries_, Vector2(0, 0), neumannForce.cast<real>().eval());
		inputResults->initMeshReference(meshSolver);
	}

	resetReconstruction();
	//invalidate texture and mesh
	showReconstructedSolution = false;
	inputTimestep = 0;
	invalidateRendering(false);
}

void InverseSoftBodyApp::runSimulation()
{
	if (worker != nullptr && !worker->isDone()) {
		//still running
		return;
	}

	if (solveForwardStatic && !enableDirichletBoundaries_)
	{
		CI_LOG_E("Can't solve for static solution if dirichlet boundaries are disabled");
		return;
	}

	//declare background task
	std::function<void(BackgroundWorker*)> task = [this](BackgroundWorker* worker) {
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
		meshSolver.setTimestep(timestep);
		meshSolver.setGroundPlane(groundPlaneHeight, groundPlaneAngle);
		meshSolver.setEnableCollision(enableCollision_);
		meshSolver.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		meshSolver.setCollisionVelocityDamping(0);
		meshSolver.setGroundStiffness(groundStiffness_);
		meshSolver.setCollisionSoftmaxAlpha(collisionSoftminAlpha_);
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
		gridSolver.setTimestep(timestep);
		gridSolver.setGroundPlane(groundPlaneHeight, groundPlaneAngle);
		gridSolver.setEnableCollision(enableCollision_);
		gridSolver.setCollisionResolution(SoftBodySimulation::CollisionResolution::SPRING_IMPLICIT);
		gridSolver.setCollisionVelocityDamping(0);
		gridSolver.setGroundStiffness(groundStiffness_);
		gridSolver.setCollisionSoftmaxAlpha(collisionSoftminAlpha_);

		inputResults->settings_ = gridSolver.getSettings();

		//solve it
		if (int(computationMode) & int(ComputationMode::COMPUTATION_MODE_MESH)) {
			auto start1 = std::chrono::steady_clock::now();
			if (solveForwardStatic) {
				meshSolver.solveStaticSolution(worker);
				if (worker->isInterrupted()) return;
			}
			else {
				meshSolver.solveDynamicSolution(worker);
				if (worker->isInterrupted()) return;
			}
			auto duration1 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start1);
			meshElapsedSeconds = duration1.count() / 1000.0;
			inputResults->meshResultsDisplacement_.push_back(meshSolver.getCurrentDisplacements());
			inputResults->meshResultsVelocities_.push_back(meshSolver.getCurrentVelocities());
		}

		if (int(computationMode) & int(ComputationMode::COMPUTATION_MODE_GRID)) {
			auto start2 = std::chrono::steady_clock::now();
			if (solveForwardStatic) {
				gridSolver.solveStaticSolution(worker);
				if (worker->isInterrupted()) return;
			}
			else {
				gridSolver.solveDynamicSolution(worker);
				if (worker->isInterrupted()) return;
			}
			auto duration2 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start2);
            CI_LOG_D("Reference simulation result:\n" << gridSolver.getSdfSolution());
            CI_LOG_D("Reference uX:\n" << gridSolver.getUGridX());
            CI_LOG_D("Reference uY:\n" << gridSolver.getUGridY());
            CI_LOG_D("Settings: " << gridSolver.getSettings() << " " << gridSolver.getGridSettings());
			gridElapsedSeconds = duration2.count() / 1000.0;
			inputResults->gridResultsSdf_.push_back(gridSolver.getSdfSolution());
			inputResults->gridResultsUx_.push_back(gridSolver.getUGridX());
			inputResults->gridResultsUy_.push_back(gridSolver.getUGridY());
			inputResults->gridResultsUxy_.push_back(gridSolver.getUSolution());
            partialObservations.setSdf(gridSolver.getSdfReference(), gridSolver.getSdfSolution(), gridSolver.getUGridX(), gridSolver.getUGridY());
            inputResults->gridPartialObservations_.push_back(partialObservations.getObservations());
		}

		inputResults->numSteps_++;
		inputTimestep = inputResults->numSteps_;
		showReconstructedSolution = false;
		params->setOptions("InputStep", "max=" + std::to_string(inputResults->numSteps_));
		invalidateRendering(false);
	};

	//start background worker
	worker = make_shared<BackgroundWorker>(task);
	CI_LOG_I("Background worker started");
}

void InverseSoftBodyApp::resetReconstruction()
{
	CI_LOG_D("reset");
	worker = nullptr;

	cleanedSdf.reset();
	resultMesh = InverseProblemOutput();
	resultGrid = InverseProblemOutput();
	meshElapsedSeconds = 0;
	gridElapsedSeconds = 0;
	invalidateRendering(false);
}

void InverseSoftBodyApp::performReconstruction()
{
	if (worker != nullptr && !worker->isDone()) {
		//still running
		return;
	}

	if (this->inputTimestep == 0) {
		CI_LOG_I("no timestep selected");
		return;
	}

	resetReconstruction();

	//declare background task
	std::function<void(BackgroundWorker*)> task = [this](BackgroundWorker* worker) {
		//get algorithm
		IInverseProblem* alg = reconstructionAlgorithms[reconstructionMode].get();
		alg->setInput(inputResults.get());
        if (InverseProblem_Adjoint_Dynamic* algD = dynamic_cast<InverseProblem_Adjoint_Dynamic*>(alg))
        {
            algD->setGridUsePartialObservation(usePartialObservations);
        }
		//run them
		if (computationMode & (int)ComputationMode::COMPUTATION_MODE_MESH) {
			auto start1 = std::chrono::steady_clock::now();
			resultMesh = alg->solveMesh(this->inputTimestep-1, worker, [this](const InverseProblemOutput& intermediateSolution)
			{
				showReconstructedSolution = true;
				resultMesh = intermediateSolution;
				invalidateRendering(true);
			});
			auto duration1 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start1);
			meshElapsedSeconds = duration1.count() / 1000.0;
			if (worker->isInterrupted()) return;
		}
		if (computationMode & (int)ComputationMode::COMPUTATION_MODE_GRID) {
			auto start2 = std::chrono::steady_clock::now();
			resultGrid = alg->solveGrid(this->inputTimestep-1, worker, [this](const InverseProblemOutput& intermediateSolution)
			{
				showReconstructedSolution = true;
				resultGrid = intermediateSolution;
				invalidateRendering(true);
			});
			if (worker->isInterrupted()) return;
			auto duration2 = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start2);
			gridElapsedSeconds = duration2.count() / 1000.0;
		}
		showReconstructedSolution = true;
		invalidateRendering(true);
	};

	//start background worker
	worker = make_shared<BackgroundWorker>(task);
	CI_LOG_I("Background worker started");
}

void InverseSoftBodyApp::reconstructSDF(int mode, int iterations)
{
	GridUtils2D::grid_t sdf;
	if (inputTimestep > 0) {
		if (showReconstructedSolution && resultGrid.resultGridSdf_.has_value()) {
			sdf = resultGrid.resultGridSdf_.value();
		}
		else if (showReconstructedSolution && resultGrid.initialGridSdf_.has_value()) {
			sdf = resultGrid.initialGridSdf_.value();
		}
		else {
			sdf = inputResults->gridResultsSdf_[inputTimestep - 1];
		}
	}
	else
	{
		sdf = inputResults->gridReferenceSdf_;
	}

	real viscosity = 0.01;
	if (iterations > 0) {
        if (mode == 1)
            sdf = GridUtils2D::recoverSDFViscosity(sdf, viscosity, iterations);
        else if (mode == 2)
            sdf = GridUtils2D::recoverSDFUpwind(sdf, iterations);
        else if (mode == 3)
            sdf = GridUtils2D::recoverSDFSussmann(sdf, viscosity, iterations);
        else if (mode == 4)
            sdf = GridUtils2D::recoverSDFFastMarching(sdf);
	}

	cleanedSdf = sdf;
	invalidateRendering(false);

	CI_LOG_I("SDF cleaned up");
}

void InverseSoftBodyApp::testPlot()
{
    std::function<void(BackgroundWorker*)> task = [this](BackgroundWorker* worker) {
		//get algorithm
		IInverseProblem* alg = reconstructionAlgorithms[reconstructionMode].get();
		if (InverseProblem_Adjoint_Dynamic* algD = dynamic_cast<InverseProblem_Adjoint_Dynamic*>(alg))
        {
            algD->setGridUsePartialObservation(usePartialObservations);
        }
        alg->setInput(inputResults.get());
		//run test plot
		alg->testPlot(this->inputTimestep, worker);
    };
    worker = make_shared<BackgroundWorker>(task);
    CI_LOG_I("Background worker started");
}

#if 1
CINDER_APP( InverseSoftBodyApp, RendererGl(RendererGl::Options().msaa(8)), [&](App::Settings *settings)
{
	settings->setWindowSize(1600, 900);
} )
#endif
