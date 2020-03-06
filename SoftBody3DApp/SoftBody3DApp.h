#pragma once

#include <Commons3D.h>

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
#include <GraphPlot.h>

class SoftBody3DApp : public cinder::app::App {
public:
    SoftBody3DApp();
    virtual ~SoftBody3DApp();
    void setup() override;
    void keyDown(cinder::app::KeyEvent event) override;
    void keyUp(cinder::app::KeyEvent event) override;
    void mouseDown(cinder::app::MouseEvent event) override;
    void update() override;
    void draw() override;
    void quit() override;
    void cleanup() override;

private:

    //user-camera
    cinder::CameraPersp		camera;
    cinder::CameraUi		camUi;

    //floor
    cinder::gl::VboMeshRef	floorVboMesh;
    cinder::gl::TextureRef	floorTexture;

    //AntTweakBar settings
    cinder::params::InterfaceGlRef	params;
    bool printMode;
    //input
    enum class SimulationMode
    {
        Tet, Grid
    };
    SimulationMode simulationMode_;
    static inline SimulationMode ToSimulationMode(const std::string& str) { return str == "Tet" ? SimulationMode::Tet : SimulationMode::Grid; }
    static inline std::string FromSimulationMode(const SimulationMode& mode) { return mode == SimulationMode::Tet ? "Tet" : "Grid"; }
    enum class InputCase
    {
        Bar, Torus, File
    };
    InputCase inputCase_;
    static inline InputCase ToInputCase(const std::string& str)
    {
        if (str == "Bar") return InputCase::Bar;
        else if (str == "Torus") return InputCase::Torus;
        else if (str == "File") return InputCase::File;
        else throw std::exception("Unknown input case");
    }
    static inline std::string FromInputCase(const InputCase& mode)
    {
        switch (mode)
        {
        case InputCase::Bar: return "Bar";
        case InputCase::Torus: return "Torus";
        case InputCase::File: return "File";
        default: throw std::exception("Unknown input case");
        }
    }
    ar3d::SoftBodySimulation3D::InputBarSettingsGui inputBarSettings_;
    ar3d::SoftBodySimulation3D::InputTorusSettingsGui inputTorusSettings_;
    ar3d::SoftBodySimulation3D::InputSdfSettingsGui inputSdfSettings_;
    //elasticity settings
    ar3d::SoftBodySimulation3D::SettingsGui simulationSettings_;
    bool spaceBarPressed_ = false;
    //adjoint
    bool enableAdjoint_ = true;
    ar3d::AdjointSolver::GUI adjointSettings_;
    enum class CostFunctionMode
    {
        DirectActive, PartialObservations
    };
    CostFunctionMode costFunctionMode_;
    static inline CostFunctionMode ToCostFunctionMode(const std::string& str) { return str == "DirectActive" ? CostFunctionMode::DirectActive : CostFunctionMode::PartialObservations; }
    static inline std::string FromCostFunctionMode(const CostFunctionMode& mode) { return mode == CostFunctionMode::DirectActive ? "DirectActive" : "PartialObservations"; }
    ar3d::CostFunctionActiveDisplacements::GUI costFunctionActiveDisplacements_;
    ar3d::CostFunctionPartialObservations::GUI costFunctionPartialObservations_;
    //rendering
    int showCamera; //0=off, 1=outline, 2=outline+image
    bool showBoundingBox;
    bool showVoxelGrid;
    int visualizedSdf; //0=off, 1=result, 2 - (n+1) = camera projected sdf
    bool enableTimings;
    std::string frameNames;
    bool animationTakeScreenshot = false;
	bool animationRendered = false;
	bool exportWithNormals_ = true;
	int exportEverNFrames_ = 1;

    //simulation
    ar3d::BackgroundWorker2Ptr worker;
    std::unique_ptr<ar3d::SoftBodyMesh3D> tetSimulation;
    std::unique_ptr<ar3d::SoftBodyGrid3D> gridSimulation;
    int frameCounter;

    //adjoint
    ar3d::SimulationResults3DPtr results;

    //visualization helpers
    std::unique_ptr<ar::TransferFunctionEditor> tfe;
	ar3d::VolumeVisualizationParams volumeVisParams;
    std::unique_ptr<ar3d::VolumeVisualization> volumeVis;
    std::unique_ptr<ar3d::TetMeshVisualization> tetVis;
    typedef std::function<float(const ar3d::SoftBodySimulation3D::Settings&)> ParamGetter_t;
    typedef std::function<bool(const ar3d::AdjointSolver::InputVariables&)> ParamEnabled_t;
    typedef std::tuple<std::unique_ptr<GraphPlot>, ParamGetter_t, ParamEnabled_t> Plot_t;
    std::unique_ptr<GraphPlot> costPlot;
    std::vector<Plot_t> plots;

private:

    void load();
    void save();

    void resetPlots();
    void simulationReset();
    void simulationStaticStep();
    void simulationDynamicStep();

    void solveAdjoint();
    void testGradient();
	void exportObservations();
    void cancel();

    void reloadShaders();
    void drawWireCube(const ar3d::real3& a, const ar3d::real3& b);
    cinder::TriMeshRef loadCustomObj(cinder::DataSourceRef dataSource);

    void updateInput();
    void updateInput0();
    void updateSettings();
    void saveStatistics();
};