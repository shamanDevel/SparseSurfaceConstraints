#include "Wrapper.h"

#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <cinder/GeomIo.h>
#include <cinder/TriMesh.h>
#include "ObjectLoader.h"

namespace py = pybind11;
#define VERSION_INFO "1.0"

using namespace ar3d_wrapper;

template <typename T>
std::string str(const T& t)
{
	std::stringstream s;
	s << t;
	return s.str();
}

std::shared_ptr<void> test_alloc()
{
	return std::make_shared<std::vector<int>>(1 << 12);
}

PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);


PYBIND11_MODULE(PythonBindings3D, m) {
	m.doc() = "Python Bindings for the inverse elasticity simulation";

	py::class_<std::shared_ptr<void>>(m, "NativePointer");

	m.def("test_alloc", &test_alloc, "Test");

	py::class_<real4>(m, "real4")
		.def(py::init())
		.def_readwrite("x", &real4::x)
		.def_readwrite("y", &real4::y)
		.def_readwrite("z", &real4::z)
		.def_readwrite("w", &real4::w)
		.def("__repr__", [](const real4& a)
	{
		std::stringstream s;
		s << "(" << a.x << "," << a.y << "," << a.z << "," << a.w << ")";
		return s.str();
	})
		.def("copy", [](const real4& a) {return a; });

	py::class_<glm::vec3>(m, "vec3")
		.def(py::init())
		.def_readwrite("x", &glm::vec3::x)
		.def_readwrite("y", &glm::vec3::y)
		.def_readwrite("z", &glm::vec3::z)
		.def("__repr__", [](const glm::vec3& a)
	{
		std::stringstream s;
		s << "(" << a.x << "," << a.y << "," << a.z << ")";
		return s.str();
	})
		.def("copy", [](const glm::vec3& a) {return a; });

	py::class_<glm::ivec3>(m, "ivec3")
		.def(py::init())
		.def_readwrite("x", &glm::ivec3::x)
		.def_readwrite("y", &glm::ivec3::y)
		.def_readwrite("z", &glm::ivec3::z)
		.def("__repr__", [](const glm::ivec3& a)
	{
		std::stringstream s;
		s << "(" << a.x << "," << a.y << "," << a.z << ")";
		return s.str();
	})
		.def("copy", [](const glm::ivec3& a) {return a; });

	py::class_<InputTorusSettings>(m, "InputTorusSettings")
		.def(py::init())
		.def_readwrite("enableDirichlet", &IInputSettings::enableDirichlet)
		.def_readwrite("centerDirichlet", &IInputSettings::centerDirichlet)
		.def_readwrite("halfsizeDirichlet", &IInputSettings::halfsizeDirichlet)
		.def_readwrite("sampleIntegrals", &IInputSettings::sampleIntegrals)
		.def_readwrite("diffusionDistance", &IInputSettings::diffusionDistance)
		.def_readwrite("zeroCutoff", &IInputSettings::zeroCutoff)
		.def_readwrite("resolution", &InputTorusSettings::resolution)
		.def_readwrite("center", &InputTorusSettings::center)
		.def_readwrite("orientation", &InputTorusSettings::orientation)
		.def_readwrite("innerRadius", &InputTorusSettings::innerRadius)
		.def_readwrite("outerRadius", &InputTorusSettings::outerRadius)
		.def("__repr__", [](const InputTorusSettings& a) { return str(a); })
		.def("copy", [](const InputTorusSettings& a) {return a; });

	py::class_<InputSdfSettings>(m, "InputSdfSettings")
		.def(py::init())
		.def_readwrite("enableDirichlet", &IInputSettings::enableDirichlet)
		.def_readwrite("centerDirichlet", &IInputSettings::centerDirichlet)
		.def_readwrite("halfsizeDirichlet", &IInputSettings::halfsizeDirichlet)
		.def_readwrite("sampleIntegrals", &IInputSettings::sampleIntegrals)
		.def_readwrite("diffusionDistance", &IInputSettings::diffusionDistance)
		.def_readwrite("zeroCutoff", &IInputSettings::zeroCutoff)
		.def_readonly("file", &InputSdfSettings::file)
		.def_readwrite("filledCells", &InputSdfSettings::filledCells)
		.def_readonly("voxelResolution", &InputSdfSettings::voxelResolution)
		.def_readonly("offset", &InputSdfSettings::offset)
		.def_readonly("size", &InputSdfSettings::size)
		.def("__repr__", [](const InputSdfSettings& a) { return str(a); })
		.def("copy", [](const InputSdfSettings& a) {return a; });

	py::enum_<InputType>(m, "InputType")
		.value("Torus", InputType::TORUS)
		.value("SDF", InputType::SDF);

	py::class_<SimulationSettings>(m, "SimulationSettings")
		.def(py::init())
		.def_readwrite("gravity", &SimulationSettings::gravity)
		.def_readwrite("youngsModulus", &SimulationSettings::youngsModulus)
		.def_readwrite("poissonsRatio", &SimulationSettings::poissonsRatio)
		.def_readwrite("mass", &SimulationSettings::mass)
		.def_readwrite("dampingAlpha", &SimulationSettings::dampingAlpha)
		.def_readwrite("dampingBeta", &SimulationSettings::dampingBeta)
		.def_readwrite("materialLambda", &SimulationSettings::materialLambda)
		.def_readwrite("materialMu", &SimulationSettings::materialMu)
		.def_readwrite("enableCorotation", &SimulationSettings::enableCorotation)
		.def_readwrite("timestep", &SimulationSettings::timestep)
		.def_readwrite("initialLinearVelocity", &SimulationSettings::initialLinearVelocity)
		.def_readwrite("initialAngularVelocity", &SimulationSettings::initialAngularVelocity)
		.def_readwrite("groundPlane", &SimulationSettings::groundPlane)
		.def_readwrite("enableCollision", &SimulationSettings::enableCollision)
		.def_readwrite("groundStiffness", &SimulationSettings::groundStiffness)
		.def_readwrite("softmaxAlpha", &SimulationSettings::softmaxAlpha)
		.def_readwrite("stableCollision", &SimulationSettings::stableCollision)
		.def_readwrite("solverIterations", &SimulationSettings::solverIterations)
		.def_readwrite("solverTolerance", &SimulationSettings::solverTolerance)
		.def_readwrite("newmarkTheta", &SimulationSettings::newmarkTheta)
		.def("__repr__", [](const SimulationSettings& a) { return str(a); })
		.def("copy", [](const SimulationSettings& a) {return a; });

	py::class_<InputVariables>(m, "InputVariables")
		.def(py::init())
		.def_readwrite("optimizeGravity", &InputVariables::optimizeGravity)
		.def_readwrite("currentGravity", &InputVariables::currentGravity)
		.def_readwrite("optimizeYoungsModulus", &InputVariables::optimizeYoungsModulus)
		.def_readwrite("currentYoungsModulus", &InputVariables::currentYoungsModulus)
		.def_readwrite("optimizePoissonRatio", &InputVariables::optimizePoissonRatio)
		.def_readwrite("currentPoissonRatio", &InputVariables::currentPoissonRatio)
		.def_readwrite("optimizeMass", &InputVariables::optimizeMass)
		.def_readwrite("currentMass", &InputVariables::currentMass)
		.def_readwrite("optimizeMassDamping", &InputVariables::optimizeMassDamping)
		.def_readwrite("currentMassDamping", &InputVariables::currentMassDamping)
		.def_readwrite("optimizeStiffnessDamping", &InputVariables::optimizeStiffnessDamping)
		.def_readwrite("currentStiffnessDamping", &InputVariables::currentStiffnessDamping)
		.def_readwrite("optimizeInitialLinearVelocity", &InputVariables::optimizeInitialLinearVelocity)
		.def_readwrite("currentInitialLinearVelocity", &InputVariables::currentInitialLinearVelocity)
		.def_readwrite("optimizeInitialAngularVelocity", &InputVariables::optimizeInitialAngularVelocity)
		.def_readwrite("currentInitialAngularVelocity", &InputVariables::currentInitialAngularVelocity)
		.def_readwrite("optimizeGroundPlane", &InputVariables::optimizeGroundPlane)
		.def_readwrite("currentGroundPlane", &InputVariables::currentGroundPlane)
		.def("disableAll", &InputVariables::disableAll)
		.def("__repr__", [](const InputVariables& a) { return str(a); })
		.def("copy", [](const InputVariables& a) {return a; });

	py::class_<RpropSettings>(m, "RpropSettings")
		.def(py::init())
		.def_readwrite("epsilon", &RpropSettings::epsilon)
		.def_readwrite("initialStepsize", &RpropSettings::initialStepsize)
		.def("__repr__", [](const RpropSettings& a) { return str(a); })
		.def("copy", [](const RpropSettings& a) {return a; });

	py::class_<GradientDescentSettings>(m, "GradientDescentSettings")
		.def(py::init())
		.def_readwrite("epsilon", &GradientDescentSettings::epsilon)
		.def_readwrite("initialStepsize", &GradientDescentSettings::initialStepsize)
		.def_readwrite("maxStepsize", &GradientDescentSettings::maxStepsize)
		.def_readwrite("minStepsize", &GradientDescentSettings::minStepsize)
		.def("__repr__", [](const GradientDescentSettings& a) { return str(a); })
		.def("copy", [](const GradientDescentSettings& a) {return a; });

	py::enum_<LbfgsSettings::LineSearchAlgorithm>(m, "LineSearchAlgorithm")
		.value("Armijo", LbfgsSettings::Armijo)
		.value("Wolfe", LbfgsSettings::Wolfe)
		.value("StrongWolfe", LbfgsSettings::StrongWolfe);

	py::class_<LbfgsSettings>(m, "LbfgsSettings")
		.def(py::init())
		.def_readwrite("epsilon", &LbfgsSettings::epsilon)
		.def_readwrite("past", &LbfgsSettings::past)
		.def_readwrite("delta", &LbfgsSettings::delta)
		.def_readwrite("lineSearchAlg", &LbfgsSettings::lineSearchAlg)
		.def_readwrite("linesearchMaxTrials", &LbfgsSettings::linesearchMaxTrials)
		.def_readwrite("linesearchMinStep", &LbfgsSettings::linesearchMinStep)
		.def_readwrite("linesearchMaxStep", &LbfgsSettings::linesearchMaxStep)
		.def_readwrite("linesearchTol", &LbfgsSettings::linesearchTol)
		.def("__repr__", [](const LbfgsSettings& a) { return str(a); })
		.def("copy", [](const LbfgsSettings& a) {return a; });

	py::enum_<Optimizer>(m, "Optimizer")
		.value("RPROP", Optimizer::RPROP)
		.value("GD", Optimizer::GD)
		.value("LBFGS", Optimizer::LBFGS);

	py::class_<AdjointSettings>(m, "AdjointSettings")
		.def(py::init())
		.def_readwrite("numIterations", &AdjointSettings::numIterations)
		.def_readwrite("memorySaving", &AdjointSettings::memorySaving)
		.def_readwrite("normalizeUnits", &AdjointSettings::normalizeUnits)
		.def_readwrite("rpropSettings", &AdjointSettings::rpropSettings)
		.def_readwrite("gdSettings", &AdjointSettings::gdSettings)
		.def_readwrite("lbfgsSettings", &AdjointSettings::lbfgsSettings)
		.def_readwrite("optimizer", &AdjointSettings::optimizer)
		.def_readwrite("variables", &AdjointSettings::variables)
		.def("__repr__", [](const AdjointSettings& a) { return str(a); })
		.def("copy", [](const AdjointSettings& a) {return a; });

	py::class_<CostFunctionPartialObservationsSettings>(m, "CostFunctionPartialObservationsSettings")
		.def(py::init())
		.def_readwrite("timestepWeights", &CostFunctionPartialObservationsSettings::timestepWeights)
		.def_readwrite("numCameras", &CostFunctionPartialObservationsSettings::numCameras)
		.def_readwrite("radius", &CostFunctionPartialObservationsSettings::radius)
		.def_readwrite("centerHeight", &CostFunctionPartialObservationsSettings::centerHeight)
		.def_readwrite("resolution", &CostFunctionPartialObservationsSettings::resolution)
		.def_readwrite("noise", &CostFunctionPartialObservationsSettings::noise)
		.def_readwrite("gpuPreprocess", &CostFunctionPartialObservationsSettings::gpuPreprocess)
		.def_readwrite("gpuEvaluate", &CostFunctionPartialObservationsSettings::gpuEvaluate)
		.def_readwrite("maxSdf", &CostFunctionPartialObservationsSettings::maxSdf)
		.def("__repr__", [](const CostFunctionPartialObservationsSettings& a) { return str(a); })
		.def("copy", [](const CostFunctionPartialObservationsSettings& a) {return a; });

	py::class_<CostFunctionActiveDisplacementsSettings>(m, "CostFunctionActiveDisplacementsSettings")
		.def(py::init())
		.def_readwrite("timestepWeights", &CostFunctionActiveDisplacementsSettings::timestepWeights)
		.def_readwrite("displacementWeight", &CostFunctionActiveDisplacementsSettings::displacementWeight)
		.def_readwrite("velocityWeight", &CostFunctionActiveDisplacementsSettings::velocityWeight)
		.def_readwrite("noise", &CostFunctionActiveDisplacementsSettings::noise)
		.def("__repr__", [](const CostFunctionActiveDisplacementsSettings& a) { return str(a); })
		.def("copy", [](const CostFunctionActiveDisplacementsSettings& a) {return a; });

	py::class_<SyntheticInput>(m, "SyntheticInput")
		.def(py::init())
		.def_readwrite("inputTorusSettings", &SyntheticInput::inputTorusSettings)
		.def_readwrite("inputSdfSettings", &SyntheticInput::inputSdfSettings)
		.def_readwrite("inputType", &SyntheticInput::inputType)
		.def_readwrite("simulationSettings", &SyntheticInput::simulationSettings)
		.def_readwrite("adjointSettings", &SyntheticInput::adjointSettings)
		.def_readwrite("costFunctionPartialObservationsSettings", &SyntheticInput::costFunctionPartialObservationsSettings)
		.def_readwrite("costFunctionActiveDisplacementsSettings", &SyntheticInput::costFunctionActiveDisplacementsSettings)
		.def("__repr__", [](const SyntheticInput& a) { return str(a); })
		.def("copy", [](const SyntheticInput& a) {return a; });

	py::class_<RecordedInput>(m, "RecordedInput")
		.def(py::init())
		.def_readonly("scanFile", &RecordedInput::scanFile)
		.def_readonly("framerate", &RecordedInput::framerate)
		.def_readonly("numTotalFrames", &RecordedInput::numTotalFrames)
		.def_readonly("groundTruthMeshPath", &RecordedInput::groundTruthMeshPath)
		.def_readwrite("voxelResolution", &RecordedInput::voxelResolution)
		.def_readwrite("inputSdfSettings", &RecordedInput::inputSdfSettings)
		.def_readwrite("simulationSettings", &RecordedInput::simulationSettings)
		.def_readwrite("adjointSettings", &RecordedInput::adjointSettings)
		.def_readwrite("sscGpuEvaluate", &RecordedInput::sscGpuEvaluate)
		.def_readwrite("sscMaxSdf", &RecordedInput::sscMaxSdf)
		.def_readwrite("costIntermediateSteps", &RecordedInput::costIntermediateSteps)
		.def_readwrite("costNumSteps", &RecordedInput::costNumSteps)
		.def("__repr__", [](const RecordedInput& a) { return str(a); })
		.def("copy", [](const RecordedInput& a) {return a; });


	m.def("initialize", &initialize, "Initialization, has to be called before any other calls of this wrapper");
	m.def("cleanup", &cleanup, "Cleans up any lingering GPU data");

	m.def("loadSyntheticInput", &loadSyntheticInput, 
		"Loads the configuration json for a synthetic example from the specified filename");
	m.def("loadRecordedInput", &loadRecordedInput, 
		"Loads the configuration json for a real-world setting from the specified filename");

	m.def("createSimulationFromTorus", &createSimulationFromTorus, 
		"Creates the simulation from the torus settings",
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("createSimulationFromSdf", &createSimulationFromSdf, 
		"Creates the simulation from the SDF settings. The SDF file is loaded relatively to the specified folder",
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	m.def("createSimulationFromRecordedInput", &createSimulationFromRecordedInput,
		"Creates the simulation from the recorded input loaded by 'loadRecordedInput'",
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	m.def("simulationSetSettings", &simulationSetSettings, "Updates the simulation settings");
	m.def("simulationReset", &simulationReset, "Resets the simulation");
	m.def("createSimulationResults", &createSimulationResults, "Allocates memory to store the states from the reference forward simulation");
	m.def("simulationSolveForward", &simulationSolveForward, 
		"Performs a single step in the forward simulation and places the result in the specified structure",
		py::arg("simulation"),
		py::arg("results")=ResultsPointer(nullptr),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	m.def("createDirectDisplacementCostFunction", &createDirectDisplacementCostFunction, "Creates the direct-displacement cost function");
	m.def("createPartialObservationsCostFunction", &createPartialObservationsCostFunction, 
		"Creates the partial-observations (SSC) cost function",
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("createResultsAndCostFunctionFromRecordedInput", &createResultsAndCostFunctionFromRecordedInput,
		"Creates and returns the tuple (results, SSC cost function) for the adjoint simulation from the recorded input.",
		py::arg("RecordedInput as created by 'loadRecordedInput'"),
		py::arg("simulation as created by 'createSimulationFromRecordedInput'"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	m.def("partialObservationsGetDepthImage", &partialObservationsGetDepthImage,
		"Returns the observed depth image as numpy array",
		py::arg("PartialObservationCostFunction"), py::arg("frame"), py::arg("camera"));
	m.def("partialObservationsGetObservations", &partialObservationsGetObservations,
		"Returns the observations projected into world space as a list of points",
		py::arg("PartialObservationCostFunction"), py::arg("frame"), py::arg("camera"));

	m.def("solveAdjoint", &solveAdjoint, 
		"Solves the adjoint simulation. The callback is called in every iteration and receives the current variables, gradients and cost function value",
		py::arg("results"), py::arg("settings"), py::arg("cost_function"), py::arg("callback"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("evaluateGradient", &evaluateGradient,
		"Evaluates the gradient at the current position with the adjoint method",
		py::arg("results"), py::arg("variables"), py::arg("cost_function"), py::arg("output-gradients"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("evaluateGradientFiniteDifferences", &evaluateGradientFiniteDifferences,
		"Evaluates the gradient at the current position with finite differences",
		py::arg("results"), py::arg("variables"), py::arg("cost_function"), py::arg("delta"), py::arg("output-gradients"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	m.def("visualizationSetInput", &visualizationSetInput,
		"Visualization: Sets the input configuration",
		py::arg("simulation"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("visualizationLoadHighResInput", &visualizationLoadHighResInput,
		"Visualization: Loads the high resolution reference mesh from the specified file path",
		py::arg("filename"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("visualizationSetState", &visualizationSetState,
		"Visualization: Sets the current state of the forward simulation",
		py::arg("simulation"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("visualizationExportMCMesh", &visualizationExportMCMesh,
		"Visualization: Exports the marching cubes mesh as .ply to the specified output path",
		py::arg("path"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
	m.def("visualizationExportHighResMesh", &visualizationExportHighResMesh,
		"Visualization: Exports the transformed high-resolution mesh as .ply to the specified output path.\n"
		"Optionally with normals and original positions.",
		py::arg("path"), py::arg("includeNormals"), py::arg("includeOriginalPositions"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

	auto exportPlaneMesh = [](real4 plane, float size, const std::string& path)
	{
		auto source = cinder::geom::Plane().size(glm::vec2(size, size)).subdivisions(glm::ivec2(5, 5));
		cinder::TriMeshRef mesh = cinder::TriMesh::create(source);
		glm::vec3 ref(0, 1, 0);
		glm::vec3 n(plane.x, plane.y, plane.z);
		glm::mat4 mat(1.0f);
		if (dot(ref, n) < 0.999999f) {
			auto rot = glm::rotate(acos(dot(ref, n)), cross(ref, n));
			mat *= rot;
		}
		auto trans = glm::translate(glm::vec3(0, plane.w, 0));
		mat *= trans;
		for (size_t i=0; i<mesh->getNumVertices(); ++i)
		{
			glm::vec4 p = mat * glm::vec4(mesh->getPositions<3>()[i], 1.0f);
			glm::vec3 p3 = p;
			mesh->getPositions<3>()[i] = p3;
		}
		ObjectLoader::saveMeshPly(mesh, path);
	};
	m.def("exportPlaneMesh", exportPlaneMesh,
		"exports the collision plane from the settings (a real4) as a transformed plane of given size as a .ply mesh",
		py::arg("plane"), py::arg("size"), py::arg("path"),
		py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}
