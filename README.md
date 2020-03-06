# InverseElasticity

Reference implementation of "Sparse Surface Constraints for Combining Physics-based Elasticity Simulation and Correspondence-Free Object Reconstruction"

A short tutorial on how to use the code.

See the Releases-page for datasets, binaries and benchmark results

# Projects in the solution
 - Libraries:
   - ActionReconstructionLib: 2D simulation code, no CUDA dependency
   - ActionReconstruction3DLib: 3D simulation code
 - Main Apps, single source files that plug the calls exposed by the libraries above together with a simple GUI:
   - SoftBodyFEMApp: simple 2D forward simulation (no CUDA)
   - InverseSoftBodyApp: Forward and Inverse simulation in the 2D case (no CUDA)
   - SoftBody3DApp: Forward and inverse simulation for synthetic cases, 3D
   - MainApp: Inverse simulation for inputs from external 3D scans, 3D
 - Remaining projects are benchmarks (projects ending in 'Benchmarks') or tests (ending in 'Test')


# HowTo simulate in 2D
(Based on SoftBodyFEM2DApp.cpp in the project SoftBodyFEMApp)

Assume all includes and namespaces are imported.

Initialize grid solver. There are utilities to create grids filled with a rect or a torus. For example:
```
SoftBodyGrid2D gridSolver = SoftBodyGrid2D::CreateTorus(
    torusOuterRadius, torusInnerRadius, gridResolution,
    enableDirichletBoundaries, dirichletForce, neumannForce);
```

If you want the inverse simulation, initialize the results:
```
SoftBody2DResults results;
results.initGridReference(gridSolver);
```

Perform a forward step. This is performed in the background thread of ar::BackgroundWorker.
```
//Set simulation parameters
gridSolver.set....
//Pass them to the results for a ground truth and for the values that are not reconsructed
results.settings_ = gridSolver.getSettings();
//solve it (of of the following two)
gridSolver.solveStaticSolution(worker);
gridSolver.solveDynamicSolution(worker);
//save result as observation for the reconstruction
results.gridResultsSdf_.push_back(gridSolver.getSdfSolution());
results.gridResultsUxy_.push_back(gridSolver.getUSolution());
```

Solve the inverse problem:
```
//create the solver
InverseProblem_Adjoint_Dynamic solver;
//set parameters, including starting values and reconstructed parameters
solver.setGridUsePartialObservation(true);
...
//set recorded input
solver.setInput(results);
//solve with callback
solver.solveGrid(numTimesteps, worker, [this](const InverseProblemOutput& intermediateSolution)
{
	//whatever you want
});
```

# HowTo simulate in 3D
(Based on SoftBody3DApp.cu)

Load input from an SDF file. See the other SoftBodyGrid3D::create... methods for alternatives.
```
SoftBodySimulation3D::InputSdfSettings inputSettings = {...};
SoftBodyGrid3D::Input input = SoftBodyGrid3D::createFromFile(inputSettings);
```

Create the simulation
```
SoftBodySimulation3D::Settings settings = {...}; //all simulation parameters
SoftBodyGrid3D simulation(input);
simulation.setSettings(settings);
```

If you want the inverse simulation, initialize the results:
```
SimulationResults3DPtr results = std::make_shared<ar3d::SimulationResults3D>();
results->input_ = simulation.getInput();
results->settings_ = simulation.getSettings();
```

Simulate a single forward step, performed withing a task of BackgroundWorker2.
```
simulation.solve(true, true);
//capture state to use them as observations
results->states_.push_back(simulation.getState().deepClone());
```

Create cost function:
```
ar3d::CostFunctionPtr costFunction = std::make_shared<ar3d::CostFunctionPartialObservations>(results, &costFunctionPartialObservations);
costFunction->preprocess(&worker);
```

Solve the inverse problem:
```
//specify a ton of settings: optimizer, step sizes, initial values, which parameters to optimize, ...
AdjointSolver::Settings settings;
... 
//create the solver
AdjointSolver solver(results, settings, costFunction);
//solve with callback
AdjointSolver::Callback_t callback = [&](const ar3d::SoftBodySimulation3D::Settings& var, const ar3d::SoftBodySimulation3D::Settings& gradient, ar3d::real cost)
{
	//whatever you want
};
solver.solve(callback, &worker);
```

# UI

Graphical output was done with Cinder.
For 2D, have a look at GridVisualization.h in ActionReconstructionLib
For 3D, have a look at VolumeVisulization.h in ActionReconstruction3DLib

# Contanct

If you have any questions, feel free to contact me: Sebastian Weiss, sebastian13.weiss@tum.de
