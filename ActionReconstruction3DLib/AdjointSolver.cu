#include "AdjointSolver.h"
#include "CommonKernels.h"

#include <cuMat/Core>
#include <cinder/app/AppBase.h>
#include "tinyformat.h"
#include "GradientDescent.h"
#include "RpropGradientDescent.h"
#include "LBFGS.h"
#include "Utils3D.h"
#include "CoordinateTransformation.h"
#include "CudaTimer.h"
#include "DebugUtils.h"

//For testing: set to 1 to enforce a symmetric matrix in the CG
//If 0, small unsymmetries of a few ulps are in the matrix due to the ordering of the operations
//If 1, the upper and lower triangular parts are averaged to create a numerically exact symmetric matrix
#define MAKE_NEWMARK_SYMMETRIC 0

//Specifies if the forward iteration shall be stopped if the linear solver did not converge
// 1: no convergence
// 2: NaN
#define FORWARD_BREAK_ON_DIVERGENCE 2

//Specifies if the adjoint step shall ignore the current step (no gradients added) if the linear solver did not converge
#define ADJOINT_IGNORE_DIVERGENCE 0

//Some way more verbose logging
#define ADJOINT_VERBOSE_LOGGING 0

namespace ar3d
{
    AdjointSolver::PrecomputedValues AdjointSolver::allocatePrecomputedValues(const Input& input)
    {
        PrecomputedValues p;
        p.lumpedMass_ = VectorX(input.numActiveNodes_);
        p.lumpedMass_.setZero();
        p.bodyForces_ = Vector3X(input.numActiveNodes_);
        p.bodyForces_.setZero();
		p.initialVelocity_ = Vector3X(input.numActiveNodes_);
		p.initialVelocity_.setZero();
        return p;
    }

    AdjointSolver::ForwardState AdjointSolver::allocateForwardState(const Input& input)
    {
		ForwardState s;
		s.displacements_ = Vector3X(input.numActiveNodes_);
		s.displacements_.setZero();
		s.velocities_ = Vector3X(input.numActiveNodes_);
		s.velocities_.setZero();
		return s;
    }

    AdjointSolver::ForwardStorage AdjointSolver::allocateForwardStorage(const Input& input)
    {
		ForwardStorage s;
		s.forces_ = Vector3X(input.numActiveNodes_);
		s.stiffness_ = SMatrix3x3(input.sparsityPattern_);
		s.newmarkA_ = SMatrix3x3(input.sparsityPattern_);
		s.newmarkB_ = Vector3X(input.numActiveNodes_);
		return s;
    }

    void AdjointSolver::BackwardState::reset()
    {
		adjDisplacements_.setZero();
		adjVelocities_.setZero();
        if (adjGridDisplacements_.size() > 0) adjGridDisplacements_.setZero();
    }

    AdjointSolver::BackwardState AdjointSolver::allocateBackwardState(const Input& input, int costFunctionInput)
    {
		BackwardState s;
		s.adjDisplacements_ = Vector3X(input.numActiveNodes_);
		s.adjVelocities_ = Vector3X(input.numActiveNodes_);
        if (costFunctionInput & int(ICostFunction::RequiredInput::GridDisplacements))
            s.adjGridDisplacements_ = WorldGridData<real3>::DeviceArray_t(input.grid_->getSize().x(), input.grid_->getSize().y(), input.grid_->getSize().z());
		s.reset();
		return s;
    }

    AdjointSolver::BackwardStorage AdjointSolver::allocateBackwardStorage(const Input& input)
    {
        BackwardStorage s;
        s.unaryLumpedMass_ = VectorX(input.numActiveNodes_);
        s.unaryBodyForces_ = Vector3X(input.numActiveNodes_);
        s.adjNewmarkA_ = SMatrix3x3(input.sparsityPattern_);
        s.adjNewmarkB_ = Vector3X(input.numActiveNodes_);
        s.adjStiffness_ = SMatrix3x3(input.sparsityPattern_);
        s.adjForces_ = Vector3X(input.numActiveNodes_);
        s.adjMass_ = VectorX(input.numActiveNodes_);
        s.adjForces_.setZero();
        s.adjMass_.setZero();
        return s;
    }

    AdjointSolver::AdjointVariables& AdjointSolver::AdjointVariables::operator*=(double scaling)
    {
        adjGravity_ *= scaling;
        adjYoungsModulus_ *= scaling;
        adjPoissonRatio_ *= scaling;
        adjMass_ *= scaling;
        adjMassDamping_ *= scaling;
        adjStiffnessDamping_ *= scaling;
        adjGroundPlane_ *= scaling;
		adjInitialAngularVelocity *= scaling;
		adjInitialLinearVelocity *= scaling;
        return *this;
    }

    AdjointSolver::InputVariables::InputVariables()
		 : optimizeGravity_(false)
	     , currentGravity_(make_real3(0, -10, 0))
	     , optimizeYoungsModulus_(false)
	     , currentYoungsModulus_(2000)
	     , optimizePoissonRatio_(false)
	     , currentPoissonRatio_(0.45)
	     , optimizeMass_(false)
	     , currentMass_(1)
	     , optimizeMassDamping_(false)
	     , currentMassDamping_(0.1)
	     , optimizeStiffnessDamping_(false)
	     , currentStiffnessDamping_(0.01)
	     , optimizeInitialLinearVelocity_(false)
	     , currentInitialLinearVelocity_(make_real3(0,0,0))
	     , optimizeInitialAngularVelocity_(false)
	     , currentInitialAngularVelocity_(make_real3(0, 0, 0))
	     , optimizeGroundPlane_(false)
	     , currentGroundPlane_(make_real4(0, 1, 0, 0))
	{
	}

	AdjointSolver::CostFunctionTmp AdjointSolver::allocateCostFunctionTmp(const Input & input, CostFunctionPtr costFunction)
	{
		CostFunctionTmp tmp;
		if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::ActiveDisplacements))
		{
			tmp.costOutput_.adjDisplacements_ = Vector3X(input.numActiveNodes_);
			tmp.costOutput_.adjVelocities_ = Vector3X(input.numActiveNodes_);
		}
        if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::GridDisplacements))
        {
            tmp.costOutput_.adjGridDisplacements_ = WorldGridData<real3>::DeviceArray_t(input.grid_->getSize().x(), input.grid_->getSize().y(), input.grid_->getSize().z());
        }
		return tmp;
	}

	bool AdjointSolver::performForwardStep(
		const ForwardState& prevState, ForwardState& nextStateOut, ForwardStorage& nextStorageOut,
		const Input& input, const PrecomputedValues& precomputed, const SoftBodySimulation3D::Settings& settings,
        int costFunctionRequiredInput, bool memorySaving)
	{
        //reset storage
        nextStorageOut.forces_.setZero();
        nextStorageOut.stiffness_.setZero();
        nextStorageOut.newmarkA_.setZero();
        nextStorageOut.newmarkB_.setZero();

		SoftBodyGrid3D::State s;
		s.displacements_ = prevState.displacements_;
		s.velocities_ = prevState.velocities_;

		//1. collision forces
		nextStorageOut.forces_.inplace() = precomputed.bodyForces_;
		if (settings.enableCollision_)
		{
			SoftBodyGrid3D::applyCollisionForces(input, settings, s, nextStorageOut.forces_);
		}

		//2. stiffness matrix
		SoftBodyGrid3D::computeStiffnessMatrix(input, s, settings, nextStorageOut.stiffness_, nextStorageOut.forces_);

		//3. Solve
        CI_LOG_D("Norm of PrevDisplacement: " << static_cast<real>(prevState.displacements_.norm()));
		CommonKernels::newmarkTimeIntegration(
			nextStorageOut.stiffness_, nextStorageOut.forces_, precomputed.lumpedMass_,
			prevState.displacements_, prevState.velocities_,
			settings.dampingAlpha_, settings.dampingBeta_, settings.timestep_,
			nextStorageOut.newmarkA_, nextStorageOut.newmarkB_, settings.newmarkTheta_);
		nextStateOut.displacements_.inplace() = prevState.displacements_ + make_real3(settings.timestep_) * prevState.velocities_;
		int iterations = settings.solverIterations_;
		real tolError = settings.solverTolerance_;
#if MAKE_NEWMARK_SYMMETRIC == 1
        nextStorageOut.newmarkA_ = DebugUtils::makeSymmetric(nextStorageOut.newmarkA_);
#endif
		CommonKernels::solveCG(
			nextStorageOut.newmarkA_, nextStorageOut.newmarkB_, nextStateOut.displacements_,
			iterations, tolError);
        CI_LOG_D("Norm of NextDisplacement: " << static_cast<real>(nextStateOut.displacements_.norm()));
		CommonKernels::newmarkComputeVelocity(
			prevState.displacements_, prevState.velocities_,
			nextStateOut.displacements_, nextStateOut.velocities_,
			settings.timestep_, settings.newmarkTheta_);
#if FORWARD_BREAK_ON_DIVERGENCE==2
        bool failedToConverge = std::isnan(tolError);
#else
		bool failedToConverge = iterations == settings.solverIterations_;
#endif

        //4. Post-Processing if needed
        if (costFunctionRequiredInput & int(ICostFunction::RequiredInput::GridDisplacements))
        {
            //diffuse displacements over the whole grid
            const Eigen::Vector3i& size = input.grid_->getSize();
            nextStateOut.gridDisplacements_ = WorldGridData<real3>::DeviceArray_t::Constant(size.x(), size.y(), size.z(), make_real3(0));
            SoftBodyGrid3D::State diffusionState;
            diffusionState.displacements_ = nextStateOut.displacements_;
            SoftBodyGrid3D::DiffusionRhs diffusionTmp1 = SoftBodyGrid3D::DiffusionRhs(input.numDiffusedNodes_, 1, 3);
            SoftBodyGrid3D::DiffusionRhs diffusionTmp2 = SoftBodyGrid3D::DiffusionRhs(input.numDiffusedNodes_, 1, 3);
            SoftBodyGrid3D::diffuseDisplacements(
                input, diffusionState, nextStateOut.gridDisplacements_,
                diffusionTmp1, diffusionTmp2);
        }

        return !failedToConverge;
	}

	real AdjointSolver::evaluateCostFunction(
		CostFunctionPtr costFunction, int timestep, CostFunctionTmp& tmp,
		const ForwardState& forwardState, BackwardState& backwardStateOut, const Input& input)
	{
		if (!costFunction->hasTimestep(timestep)) return 0;
		tmp.costOutput_.cost_ = 0;
		//prepare input and output
		if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::ActiveDisplacements))
		{
			tmp.costInput_.displacements_ = forwardState.displacements_;
			tmp.costInput_.velocities_ = forwardState.velocities_;
			tmp.costOutput_.adjDisplacements_.setZero();
			tmp.costOutput_.adjVelocities_.setZero();
		}
        if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::GridDisplacements))
        {
            tmp.costInput_.gridDisplacements_ = forwardState.gridDisplacements_;
            tmp.costInput_.referenceSDF_ = input.referenceSdf_;
            tmp.costOutput_.adjGridDisplacements_.setZero();
        }
		//evaluate cost function
		costFunction->evaluate(timestep, tmp.costInput_, tmp.costOutput_);
		//apply output
		if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::ActiveDisplacements))
		{
			backwardStateOut.adjDisplacements_ += tmp.costOutput_.adjDisplacements_;
			backwardStateOut.adjVelocities_    += tmp.costOutput_.adjVelocities_;
		}
        if (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::GridDisplacements))
        {
            backwardStateOut.adjGridDisplacements_ += tmp.costOutput_.adjGridDisplacements_;
        }
		return tmp.costOutput_.cost_;
	}

	void AdjointSolver::performBackwardStep(
		const ForwardState& prevState, const ForwardState& nextState,
		const BackwardState& adjNextState, BackwardState& adjPrevStateOut, 
		AdjointVariables& adjVariablesOut,
		const Input& input, const PrecomputedValues& precomputed, 
        ForwardStorage& nextStorage, BackwardStorage& adjStorage,
        const SoftBodySimulation3D::Settings& settings,
        int costFunctionRequiredInput, bool memorySaving)
	{
        if (memorySaving)
        {
            //in memory saving mode, the intermediate variables (in nextStorage) are not saved
            //and thus we recompute them here.
            //This is a copy of performForwardStep without the linear solvers

            nextStorage.forces_.setZero();
            nextStorage.stiffness_.setZero();
            nextStorage.newmarkA_.setZero();
            nextStorage.newmarkB_.setZero();

            SoftBodyGrid3D::State s;
            s.displacements_ = prevState.displacements_;
            s.velocities_ = prevState.velocities_;

            //1. collision forces
            nextStorage.forces_.inplace() = precomputed.bodyForces_;
            if (settings.enableCollision_)
            {
                SoftBodyGrid3D::applyCollisionForces(input, settings, s, nextStorage.forces_);
            }

            //2. stiffness matrix
            SoftBodyGrid3D::computeStiffnessMatrix(input, s, settings, nextStorage.stiffness_, nextStorage.forces_);

            //3. Newmark time integration / Solve is ommitted
            CommonKernels::newmarkTimeIntegration(
                nextStorage.stiffness_, nextStorage.forces_, precomputed.lumpedMass_,
                prevState.displacements_, prevState.velocities_,
                settings.dampingAlpha_, settings.dampingBeta_, settings.timestep_,
                nextStorage.newmarkA_, nextStorage.newmarkB_, settings.newmarkTheta_);
#if MAKE_NEWMARK_SYMMETRIC==1
            nextStorage.newmarkA_ = DebugUtils::makeSymmetric(nextStorage.newmarkA_);
#endif
        }

        Vector3X adjNextDisplacement = adjNextState.adjDisplacements_.deepClone();

        //adj4. Postprocessing
        if (costFunctionRequiredInput & int(ICostFunction::RequiredInput::GridDisplacements))
        {
            //adjoint of displacement diffusion over the whole grid
            adjointDiffuseDisplacements(input, adjNextState.adjGridDisplacements_, adjNextDisplacement);
        }

#if ADJOINT_VERBOSE_LOGGING==1
		std::vector<real3> adjNextDisplacementHost(adjNextDisplacement.size());
		adjNextDisplacement.copyToHost(&adjNextDisplacementHost[0]);
		cinder::app::console() << "adjNextDisplacement:\n";
		for (int i = 0; i < adjNextDisplacementHost.size(); ++i) {
			real3 v = adjNextDisplacementHost[i];
			tinyformat::format(cinder::app::console(), "  [%3d](%7.5f, %7.5f, %7.5f)\n", i, v.x, v.y, v.z);
		}
#endif

        //adj3. Solve
        CI_LOG_D("Norm of adjNextDisplacement: " << static_cast<real>(adjNextDisplacement.norm()));
        CommonKernels::adjointNewmarkComputeVelocity(
            adjNextState.adjVelocities_, adjNextDisplacement,
            adjPrevStateOut.adjVelocities_, adjPrevStateOut.adjDisplacements_,
            settings.timestep_, settings.newmarkTheta_);
        adjStorage.adjNewmarkA_.setZero();
        adjStorage.adjNewmarkB_.setZero();

        CI_LOG_D("Norm of adjNextDisplacement: " << static_cast<real>(adjNextDisplacement.norm()));
        bool converged = CommonKernels::adjointSolveCG(
            nextStorage.newmarkA_, nextStorage.newmarkB_,
            nextState.displacements_, adjNextDisplacement,
            adjStorage.adjNewmarkA_, adjStorage.adjNewmarkB_,
            settings.solverIterations_*2, settings.solverTolerance_); //adjoint solve needs longer (no good initial guess)
        CI_LOG_D("Norm of adjNewmarkA: " << static_cast<real>(adjStorage.adjNewmarkA_.norm()));
        CI_LOG_D("Norm of adjNewmarkB: " << static_cast<real>(adjStorage.adjNewmarkB_.norm()));
#if ADJOINT_IGNORE_DIVERGENCE==1
        if (!converged)
        {
            CI_LOG_E("adjoint CG not converged, force gradients to zero");
            //This may be a bit too harsh, since all gradients are lost,
            // but I don't have a better idea
            adjStorage.adjNewmarkA_.setZero();
            adjStorage.adjNewmarkB_.setZero();
        }
#endif

        adjStorage.adjStiffness_.setZero();
        adjStorage.adjMass_.setZero();
        adjStorage.adjForces_.setZero();
        DeviceScalar adjMassDamping = DeviceScalar::Zero();
        DeviceScalar adjStiffnessDamping = DeviceScalar::Zero();
        CommonKernels::adjointNewmarkTimeIntegration(
            nextStorage.stiffness_, nextStorage.forces_, precomputed.lumpedMass_,
            prevState.displacements_, prevState.velocities_,
            settings.dampingAlpha_, settings.dampingBeta_,
            adjStorage.adjNewmarkA_, adjStorage.adjNewmarkB_,
            adjStorage.adjStiffness_, adjStorage.adjForces_, adjStorage.adjMass_,
            adjPrevStateOut.adjDisplacements_, adjPrevStateOut.adjVelocities_,
            adjMassDamping, adjStiffnessDamping,
            settings.timestep_, settings.newmarkTheta_);
        adjVariablesOut.adjMassDamping_ += static_cast<real>(adjMassDamping);
        adjVariablesOut.adjStiffnessDamping_ += static_cast<real>(adjStiffnessDamping);
        CI_LOG_D("Norm of adjPrevDisplacement: " << static_cast<real>(adjPrevStateOut.adjDisplacements_.norm()));
        CI_LOG_D("Norm of adjPrevVelocities: " << static_cast<real>(adjPrevStateOut.adjVelocities_.norm()));

        //adj2. Stiffness matrix
        adjStorage.adjLambda_.setZero();
        adjStorage.adjMu_.setZero();
        adjointComputeStiffnessMatrix(
            input, prevState.displacements_, settings,
            adjStorage.adjStiffness_, adjStorage.adjForces_,
            adjPrevStateOut.adjDisplacements_,
            adjStorage.adjLambda_, adjStorage.adjMu_);
        real adjLambda = static_cast<real>(adjStorage.adjLambda_);
        real adjMu = static_cast<real>(adjStorage.adjMu_);
        adjointComputeMaterialParameters(
            settings.youngsModulus_, settings.poissonsRatio_,
            adjMu, adjLambda,
            adjVariablesOut.adjYoungsModulus_, adjVariablesOut.adjPoissonRatio_);

        //adj1. Collision Forces
        if (settings.enableCollision_) {
            adjointApplyCollisionForces(input, settings,
                prevState.displacements_, prevState.velocities_,
                adjStorage.adjForces_,
                adjPrevStateOut.adjDisplacements_, adjPrevStateOut.adjVelocities_,
                adjVariablesOut.adjGroundPlane_);
        }

        //Adjoint of body forces and mass
        adjVariablesOut.adjMass_ += static_cast<real>(
            precomputed.lumpedMass_.dot(adjStorage.adjMass_));
        real3 adjGravityTmp = static_cast<real3>(
            precomputed.bodyForces_.cwiseMul(adjStorage.adjForces_)
                .reduction<cuMat::functor::Sum<real3>, cuMat::Axis::All>(cuMat::functor::Sum<real3>(), make_real3(0, 0, 0)));
        adjVariablesOut.adjGravity_ += make_double3(-adjGravityTmp.x, -adjGravityTmp.y, -adjGravityTmp.z);
		//HACK: the gradient points into the wrong direction. Hence I added a minus here

#if 0
        //Scale gradient
        //IMPORTANT: With the fix to adjointNewmarkComputeVelocity, I don't need that anymore!
        real dispNorm = static_cast<real>(adjPrevStateOut.adjDisplacements_.norm());
        double scale = dispNorm < 1e-10 ? 1 : 1.0 / dispNorm;
        adjPrevStateOut.scale_ = scale * adjNextState.scale_;
        CI_LOG_I("Current scale: " << scale << ", total scale: " << adjPrevStateOut.scale_);
        adjPrevStateOut.adjDisplacements_ *= make_real3(static_cast<real>(scale));
        adjPrevStateOut.adjVelocities_ *= make_real3(static_cast<real>(scale));
        adjVariablesOut *= scale;
#endif
	}

	real AdjointSolver::computeGradient(
		const Input& input, const SoftBodySimulation3D::Settings& settings_,
		const InputVariables& variables, CostFunctionPtr costFunction, 
        AdjointVariables& adjointVariablesOut, bool memorySaving, 
		BackgroundWorker2* worker, Statistics* statistics)
	{
		adjointVariablesOut = { 0 };

		//update settings with the current state
		SoftBodySimulation3D::Settings settings = settings_;
		if (variables.optimizeGravity_) settings.gravity_ = variables.currentGravity_;
		if (variables.optimizeYoungsModulus_) settings.youngsModulus_ = variables.currentYoungsModulus_;
		if (variables.optimizePoissonRatio_) settings.poissonsRatio_ = variables.currentPoissonRatio_;
		if (variables.optimizeMass_) settings.mass_ = variables.currentMass_;
		if (variables.optimizeMassDamping_) settings.dampingAlpha_ = variables.currentMassDamping_;
		if (variables.optimizeStiffnessDamping_) settings.dampingBeta_ = variables.currentStiffnessDamping_;
		if (variables.optimizeInitialLinearVelocity_) settings.initialLinearVelocity_ = variables.currentInitialLinearVelocity_;
		if (variables.optimizeInitialAngularVelocity_) settings.initialAngularVelocity_ = variables.currentInitialAngularVelocity_;
		if (variables.optimizeGroundPlane_) settings.groundPlane_ = variables.currentGroundPlane_;
		settings.validate();

        if (statistics)
        {
            statistics->numActiveNodes = input.numActiveNodes_;
            statistics->numEmptyNodes = input.grid_->getSize().prod() - input.numActiveNodes_;
            statistics->numActiveElements = input.numActiveCells_;
        }
        CudaTimer timer;

		//query number of timesteps
		int numSteps = costFunction->getNumSteps();
		CI_LOG_D(numSteps << " timesteps have to be computed");

        //first check if enough memory is available
        size_t freeMemory = cuMat::Context::getFreeDeviceMemory();
        size_t totalMemory = cuMat::Context::getTotalDeviceMemory();
        size_t requiredMemory = memorySaving
            ? (numSteps + 1) * (input.numActiveNodes_ * 2 * sizeof(real3)) + (input.numActiveNodes_ * 6 * sizeof(real3)) + 2 * input.sparsityPattern_.nnz * sizeof(real3x3) + 
                (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::GridDisplacements) ? 2 * input.grid_->getSize().prod() * sizeof(real3) : 0)
            : (numSteps + 1) * (input.numActiveNodes_ * 6 * sizeof(real3) + 2 * input.sparsityPattern_.nnz * sizeof(real3x3) + 
                (costFunction->getRequiredInput() & int(ICostFunction::RequiredInput::GridDisplacements) ? 2 * input.grid_->getSize().prod() * sizeof(real3) : 0));
        if (requiredMemory > freeMemory)
        {
            CI_LOG_E("Not enough memory! Free memory: " << (freeMemory >> 20) << "MB, required memory: " << (requiredMemory >> 20) << "MB (total: " << (totalMemory >> 20) << "MB)");
            if (!memorySaving) CI_LOG_E("consider enabling memory saving mode");
            return 0;
        }
        CI_LOG_I("Enough memory available. Free memory: " << (freeMemory >> 20) << "MB, required memory: " << (requiredMemory >> 20) << "MB (total: " << (totalMemory >> 20) << "MB)");

		//allocations
		std::vector<ForwardState> forwardStates(numSteps + 1);
		std::vector<ForwardStorage> forwardStorages(memorySaving ? 1 : numSteps + 1);
		std::vector<BackwardState> backwardStates(memorySaving ? 2 : numSteps + 1);
        if (memorySaving)
        {
            for (int t = 0; t <= numSteps; ++t) forwardStates[t] = allocateForwardState(input);
            forwardStorages[0] = allocateForwardStorage(input);
            backwardStates[0] = allocateBackwardState(input, costFunction->getRequiredInput());
            backwardStates[1] = allocateBackwardState(input, costFunction->getRequiredInput());
        }
        else {
            for (int t = 0; t <= numSteps; ++t)
            {
                forwardStates[t] = allocateForwardState(input);
                if (t > 0) forwardStorages[t] = allocateForwardStorage(input);
                backwardStates[t] = allocateBackwardState(input, costFunction->getRequiredInput());
            }
        }
        BackwardStorage adjStorage = allocateBackwardStorage(input);

        //precomputations
        CI_LOG_D("precompute values");
        PrecomputedValues precomputed = allocatePrecomputedValues(input);
        {
            SoftBodySimulation3D::Settings settings2 = settings;
            settings2.gravity_ = make_real3(1, 1, 1);
            settings2.mass_ = 1;
            adjStorage.unaryLumpedMass_.setZero();
            adjStorage.unaryBodyForces_.setZero();
            SoftBodyGrid3D::computeMassMatrix(input, settings2, adjStorage.unaryLumpedMass_);
            SoftBodyGrid3D::computeBodyForces(input, settings2, adjStorage.unaryBodyForces_);
        }
        precomputed.lumpedMass_ = settings.mass_ * adjStorage.unaryLumpedMass_;
        precomputed.bodyForces_ = settings.gravity_ * adjStorage.unaryBodyForces_;
		SoftBodyGrid3D::computeInitialVelocity(input, settings, precomputed.initialVelocity_);

		//apply initial velocity
		forwardStates[0].velocities_.inplace() = precomputed.initialVelocity_;

		//forward
		CI_LOG_D("forward steps");
		for (int t=1; t<=numSteps; ++t)
		{
			if (worker && worker->isInterrupted()) return 0;
            CI_LOG_D("Timestep " << t);
            timer.start();
			bool converged = performForwardStep(
				forwardStates[t - 1], forwardStates[t], forwardStorages[memorySaving ? 0 : t],
				input, precomputed, settings, costFunction->getRequiredInput(),
                memorySaving);
            timer.stop();
            if (statistics) statistics->forwardTime.push_back(timer.duration());
#if FORWARD_BREAK_ON_DIVERGENCE>0
            if (!converged) {
                CI_LOG_E("Linear solver in the forward step did not converge, stop iteration and evaluate gradient only until timestep " << (t - 1));
				std::cout << "Linear solver in the forward step did not converge, stop iteration and evaluate gradient only until timestep " << (t - 1) << std::endl;
                numSteps = t - 1;
                break;
            }
#endif
		}

		//cost function
		real finalCost = 0;
		CostFunctionTmp costFunctionTmp = allocateCostFunctionTmp(input, costFunction);
        if (!memorySaving)
        {
            CI_LOG_D("evaluate cost function");
            for (int t = 1; t <= numSteps; ++t)
            {
				if (worker && worker->isInterrupted()) return 0;
                timer.start();
                finalCost += evaluateCostFunction(
                    costFunction, t - 1, costFunctionTmp,
                    forwardStates[t], backwardStates[t], input);
                timer.stop();
                if (statistics) statistics->costTime.push_back(timer.duration());
            }
        }

		//backward/adjoint
		CI_LOG_D("adjoint steps");
		for (int t=numSteps; t>0; --t)
		{
			if (worker && worker->isInterrupted()) return 0;
            CI_LOG_I("Timestep " << t);
            if (!memorySaving) {
                timer.start();
                performBackwardStep(
                    forwardStates[t - 1], forwardStates[t],
                    backwardStates[t], backwardStates[t - 1],
                    adjointVariablesOut,
                    input, precomputed,
                    forwardStorages[t], adjStorage,
                    settings, costFunction->getRequiredInput(), false);
                timer.stop();
                if (statistics) statistics->backwardTime.push_back(timer.duration());
            }
            else
            {
                const int idxCurrent = t % 2;
                const int idxNext = 1 - idxCurrent;

                backwardStates[idxNext].reset();
                timer.start();
                finalCost += evaluateCostFunction(
                    costFunction, t - 1, costFunctionTmp,
                    forwardStates[t], backwardStates[idxCurrent], input);
                timer.stop();
                if (statistics) statistics->costTime.push_back(timer.duration());

                timer.start();
                performBackwardStep(
                    forwardStates[t - 1], forwardStates[t],
                    backwardStates[idxCurrent], backwardStates[idxNext],
                    adjointVariablesOut,
                    input, precomputed,
                    forwardStorages[0], adjStorage,
                    settings, costFunction->getRequiredInput(), true);
                timer.stop();
                if (statistics) statistics->backwardTime.push_back(timer.duration());
            }
		}
        adjointVariablesOut *= 1.0 / backwardStates[0].scale_;

        //adjoint of precomputations (initial velocities)
		if (variables.optimizeInitialLinearVelocity_ || variables.optimizeInitialAngularVelocity_)
			adjointComputeInitialVelocity(input, 
				settings.initialLinearVelocity_, settings.initialAngularVelocity_, 
				backwardStates[0].adjVelocities_, 
				adjointVariablesOut.adjInitialLinearVelocity, adjointVariablesOut.adjInitialAngularVelocity);

		//done
		return finalCost;
	}

	real AdjointSolver::computeGradientFiniteDifferences(
		const Input& input, const SoftBodySimulation3D::Settings& settings_,
		const InputVariables& variables, CostFunctionPtr costFunction,
		AdjointVariables& adjointVariablesOut, real finiteDifferencesDelta,
		BackgroundWorker2* worker, Statistics* statistics)
	{
		adjointVariablesOut = { 0 };

		//update settings with the current state
		SoftBodySimulation3D::Settings settings = settings_;
		if (variables.optimizeGravity_) settings.gravity_ = variables.currentGravity_;
		if (variables.optimizeYoungsModulus_) settings.youngsModulus_ = variables.currentYoungsModulus_;
		if (variables.optimizePoissonRatio_) settings.poissonsRatio_ = variables.currentPoissonRatio_;
		if (variables.optimizeMass_) settings.mass_ = variables.currentMass_;
		if (variables.optimizeMassDamping_) settings.dampingAlpha_ = variables.currentMassDamping_;
		if (variables.optimizeStiffnessDamping_) settings.dampingBeta_ = variables.currentStiffnessDamping_;
		if (variables.optimizeInitialLinearVelocity_) settings.initialLinearVelocity_ = variables.currentInitialLinearVelocity_;
		if (variables.optimizeInitialAngularVelocity_) settings.initialAngularVelocity_ = variables.currentInitialAngularVelocity_;
		if (variables.optimizeGroundPlane_) settings.groundPlane_ = variables.currentGroundPlane_;
		settings.validate();

		if (statistics)
		{
			statistics->numActiveNodes = input.numActiveNodes_;
			statistics->numEmptyNodes = input.grid_->getSize().prod() - input.numActiveNodes_;
			statistics->numActiveElements = input.numActiveCells_;
		}
		CudaTimer timer;

		//query number of timesteps
		int numSteps = costFunction->getNumSteps();
		CI_LOG_D(numSteps << " timesteps have to be computed");

		//allocations
		ForwardState forwardStates[2];
    	forwardStates[0] = allocateForwardState(input);
		forwardStates[1] = allocateForwardState(input);
		ForwardStorage forwardStorage = allocateForwardStorage(input);
		PrecomputedValues precomputed = allocatePrecomputedValues(input);
		BackwardState backwardState = allocateBackwardState(input, costFunction->getRequiredInput());

		//single forward evaluation
		auto evaluate = [&](SoftBodySimulation3D::Settings settings2, Statistics* statistics2) -> real
		{
			settings2.validate();
			//precomputations + initial computations
			precomputed.lumpedMass_.setZero();
			precomputed.bodyForces_.setZero();
			precomputed.initialVelocity_.setZero();
			for (int i = 0; i < 2; ++i) {
				forwardStates[i].displacements_.setZero();
				forwardStates[i].gridDisplacements_.setZero();
				forwardStates[i].velocities_.setZero();
			}
			SoftBodyGrid3D::computeMassMatrix(input, settings2, precomputed.lumpedMass_);
			SoftBodyGrid3D::computeBodyForces(input, settings2, precomputed.bodyForces_);
			SoftBodyGrid3D::computeInitialVelocity(input, settings2, precomputed.initialVelocity_);
			forwardStates[0].velocities_.inplace() = precomputed.initialVelocity_;

			//forward + cost function
			real finalCost = 0;
			CostFunctionTmp costFunctionTmp = allocateCostFunctionTmp(input, costFunction);
			CI_LOG_D("forward steps + cost function");
			for (int t = 1; t <= numSteps; ++t)
			{
				if (worker && worker->isInterrupted()) return 0;
				CI_LOG_D("Timestep " << t);
				timer.start();
				bool converged = performForwardStep(
					forwardStates[(t-1)%2], forwardStates[t%2], forwardStorage,
					input, precomputed, settings2, costFunction->getRequiredInput(),
					true);
				timer.stop();
				if (statistics2) statistics2->forwardTime.push_back(timer.duration());
				timer.start();
				finalCost += evaluateCostFunction(
					costFunction, t - 1, costFunctionTmp,
					forwardStates[t % 2], backwardState, input);
				timer.stop();
				if (statistics2) statistics2->costTime.push_back(timer.duration());
			}
			return finalCost;
		};

		//evaluate current setting
		CI_LOG_D("main evaluation");
		real finalCost = evaluate(settings, statistics);

		//evaluate once for every parameter
#define EVALUATE_FD(optimVar, settingsVar, currentVar, adjVar)	\
	if (variables.optimVar) {									\
		CI_LOG_D("evaluate " CUMAT_STR(settingsVar));			\
		SoftBodySimulation3D::Settings settings2 = settings;	\
		settings2.settingsVar = variables.currentVar + settings_.settingsVar * finiteDifferencesDelta;		\
		real cost = evaluate(settings2, nullptr);								\
		adjointVariablesOut. adjVar = (cost - finalCost) / (settings_.settingsVar * finiteDifferencesDelta);		\
		std::cout << "Evaluate " << #optimVar <<									\
    		", x1=" << settings.settingsVar << ", x2=" << settings2.settingsVar <<	\
			", c1=" << finalCost << ", c2=" << cost <<								\
    		" -> grad=" << adjointVariablesOut.adjVar << std::endl;					\
	}
		EVALUATE_FD(optimizeGravity_, gravity_.x, currentGravity_.x, adjGravity_.x);
		EVALUATE_FD(optimizeGravity_, gravity_.y, currentGravity_.y, adjGravity_.y);
		EVALUATE_FD(optimizeGravity_, gravity_.z, currentGravity_.z, adjGravity_.z);
		EVALUATE_FD(optimizeMassDamping_, dampingAlpha_, currentMassDamping_, adjMassDamping_);
		EVALUATE_FD(optimizeStiffnessDamping_, dampingBeta_, currentStiffnessDamping_, adjStiffnessDamping_);
		EVALUATE_FD(optimizePoissonRatio_, poissonsRatio_, currentPoissonRatio_, adjPoissonRatio_);
		//EVALUATE_FD(optimizeYoungsModulus_, youngsModulus_, currentYoungsModulus_, adjYoungsModulus_);

		if (variables.optimizeYoungsModulus_)
		{
			CI_LOG_D("evaluate " CUMAT_STR(youngsModulus_));
			SoftBodySimulation3D::Settings settings2 = settings;
			settings2.youngsModulus_ = variables.currentYoungsModulus_ + settings_.youngsModulus_ *
				finiteDifferencesDelta;
			real cost = evaluate(settings2, nullptr);
			adjointVariablesOut.adjYoungsModulus_ = (cost - finalCost) / (settings_.youngsModulus_ *
				finiteDifferencesDelta);
			std::cout << "Evaluate " << "optimizeYoungsModulus_" << ", x1=" << settings.youngsModulus_ << ", x2=" <<
				settings2.youngsModulus_ << ", c1=" << finalCost << ", c2=" << cost << " -> grad=" <<
				adjointVariablesOut.adjYoungsModulus_ << std::endl;
		};

#undef EVALUATE_FD

		//done
		return finalCost;
	}

    void AdjointSolver::adjointComputeMaterialParameters(
        double k, double p, double adjMu, double adjLambda,
        double& adjYoungOut, double& adjPoissonOut)
    {
        adjYoungOut +=
            adjMu * (1 / (2.*(1 + p))) +
            adjLambda * (p / ((1 - 2 * p)*(1 + p)));
        adjPoissonOut +=
            adjMu * (-k / (2.*ar3d::utils::square(1 + p))) +
            adjLambda * (-((k*p) / ((1 - 2 * p)*ar3d::utils::square(1 + p))) + k / ((1 - 2 * p)*(1 + p)) +
                         (2 * k*p) / (ar3d::utils::square(1 - 2 * p)*(1 + p)));
    }

    AdjointSolver::Settings::Settings()
		: numIterations_(20), optimizer_(GRADIENT_DESCENT), memorySaving_(false), normalizeUnits_(true)
	{
        gradientDescentSettings_.epsilon_ = "1e-7";
        gradientDescentSettings_.linearStepsize_ = "0.001";
        gradientDescentSettings_.maxStepsize_ = "";
        gradientDescentSettings_.minStepsize_ = "";

		rpropSettings_.epsilon_ = "1e-7";
		rpropSettings_.initialStepsize_ = "0.001";

        lbfgsSettings_.epsilon_ = "1e-7";
        lbfgsSettings_.past_ = 0;
        lbfgsSettings_.delta_ = "";
        lbfgsSettings_.lineSearchAlg_ = LbfgsSettings::Wolfe;
        lbfgsSettings_.linesearchMaxTrials_ = 2;
        lbfgsSettings_.linesearchMinStep_ = "";
        lbfgsSettings_.linesearchMaxStep_ = "";
	}

	AdjointSolver::GUI::GUI()
	{
	}

	void AdjointSolver::GUI::initParams(cinder::params::InterfaceGlRef params, 
		const std::string& group, const bool noInitialValues)
	{
        params_ = params;
		static const std::string t = "visible=true";
		static const std::string f = "visible=false";

        //GENERAL PARAMETERS
		params->addParam("AdjointSolver-NumIterations", &settings_.numIterations_)
			.group(group).label("Num Iterations").min(1);
        params->addParam("AdjointSolver-MemorySaving", &settings_.memorySaving_)
            .group(group).label("Memory Saving")
            .optionsStr("help='False: save everything from the forward pass (fast, memory intense). True: only save minimal information, recompute more (slower, less memory)'");
		params->addParam("AdjointSolver-NormalizeUnits", &settings_.normalizeUnits_)
			.group(group).label("Normalize Units");
        std::vector<std::string> optimizerNames = { "Gradient Descent", "Rprop", "LBFGS" };
        params->addParam("AdjointSolver-Optimizer", optimizerNames, reinterpret_cast<int*>(&settings_.optimizer_))
            .group(group).label("Optimizer").updateFn([params, this]()
        {
            bool v = settings_.optimizer_ == Settings::GRADIENT_DESCENT;
            params->setOptions("AdjointSolver-GD-Epsilon", v ? t : f);
            params->setOptions("AdjointSolver-GD-LinearStepsize", v ? t : f);
            params->setOptions("AdjointSolver-GD-MaxStepsize", v ? t : f);
            params->setOptions("AdjointSolver-GD-MinStepsize", v ? t : f);

			v = settings_.optimizer_ == Settings::RPROP;
			params->setOptions("AdjointSolver-Rprop-Epsilon", v ? t : f);
			params->setOptions("AdjointSolver-Rprop-InitialStepsize", v ? t : f);

            v = settings_.optimizer_ == Settings::LBFGS;
            params->setOptions("AdjointSolver-LBFGS-Epsilon", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-Past", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-Delta", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-Algorithm", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-LinesearchMaxTrials", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-LinesearchMinStep", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-LinesearchMaxStep", v ? t : f);
            params->setOptions("AdjointSolver-LBFGS-LinesearchTol", v ? t : f);
        });

        //OPTIMIZED VARIABLES
		params->addParam("AdjointSolver-OptimizeGravity", &settings_.variables_.optimizeGravity_)
			.group(group).label("Optimize Gravity").accessors(
			[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeGravity_ = v;
			if (!noInitialValues) {
				params->setOptions("AdjointSolver-InitialGravityX", v ? t : f);
				params->setOptions("AdjointSolver-InitialGravityY", v ? t : f);
				params->setOptions("AdjointSolver-InitialGravityZ", v ? t : f);
			}
		}, [this]()
		{
			return settings_.variables_.optimizeGravity_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialGravityX", &settings_.variables_.currentGravity_.x)
				.group(group).label("Initial Gravity X").step(0.01f).visible(settings_.variables_.optimizeGravity_);
			params->addParam("AdjointSolver-InitialGravityY", &settings_.variables_.currentGravity_.y)
				.group(group).label("Initial Gravity Y").step(0.01f).visible(settings_.variables_.optimizeGravity_);
			params->addParam("AdjointSolver-InitialGravityZ", &settings_.variables_.currentGravity_.z)
				.group(group).label("Initial Gravity Z").step(0.01f).visible(settings_.variables_.optimizeGravity_);
		}

		params->addParam("AdjointSolver-OptimizeYoungsModulus", &settings_.variables_.optimizeYoungsModulus_)
			.group(group).label("Optimize Young's Modulus").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeYoungsModulus_ = v;
			if (!noInitialValues)
				params->setOptions("AdjointSolver-InitialYoungsModulus", v ? t : f);
		}, [this]()
		{
			return settings_.variables_.optimizeYoungsModulus_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialYoungsModulus", &settings_.variables_.currentYoungsModulus_)
				.group(group).label("Initial Young's Modulus").step(0.01f).min(0).visible(settings_.variables_.optimizeYoungsModulus_);
		}

		params->addParam("AdjointSolver-OptimizePoissonRatio", &settings_.variables_.optimizePoissonRatio_)
			.group(group).label("Optimize Poisson Ratio").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizePoissonRatio_ = v;
			if (!noInitialValues) 
				params->setOptions("AdjointSolver-InitialPoissonRatio", v ? t : f);
		}, [this]()
		{
			return settings_.variables_.optimizePoissonRatio_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialPoissonRatio", &settings_.variables_.currentPoissonRatio_)
				.group(group).label("Initial Poisson Ratio").step(0.001f).min(0.1f).max(0.49f).visible(settings_.variables_.optimizePoissonRatio_);
		}

		params->addParam("AdjointSolver-OptimizeMass", &settings_.variables_.optimizeMass_)
			.group(group).label("Optimize Mass").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeMass_ = v;
			if (!noInitialValues)
				params->setOptions("AdjointSolver-InitialMass", v ? t : f);
		}, [this]()
		{
			return settings_.variables_.optimizeMass_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialMass", &settings_.variables_.currentMass_)
				.group(group).label("Initial Mass").step(0.01f).min(0.01f).visible(settings_.variables_.optimizeMass_);
		}

		params->addParam("AdjointSolver-OptimizeMassDamping", &settings_.variables_.optimizeMassDamping_)
			.group(group).label("Optimize Mass Damping").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeMassDamping_ = v;
			if (!noInitialValues) 
				params->setOptions("AdjointSolver-InitialMassDamping", v ? t : f);
		}, [this]()
		{
			return settings_.variables_.optimizeMassDamping_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialMassDamping", &settings_.variables_.currentMassDamping_)
				.group(group).label("Initial Mass Damping").step(0.001f).min(0.0f).visible(settings_.variables_.optimizeMassDamping_);
		}

		params->addParam("AdjointSolver-OptimizeStiffnessDamping", &settings_.variables_.optimizeStiffnessDamping_)
			.group(group).label("Optimize Stiffness Damping").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeStiffnessDamping_ = v;
			if (!noInitialValues) 
				params->setOptions("AdjointSolver-InitialStiffnessDamping", v ? t : f);
		}, [this]()
		{
			return settings_.variables_.optimizeStiffnessDamping_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialStiffnessDamping", &settings_.variables_.currentStiffnessDamping_)
				.group(group).label("Initial Stiffness Damping").step(0.001f).min(0.0f).visible(settings_.variables_.optimizeStiffnessDamping_);
		}

		params->addParam("AdjointSolver-OptimizeInitialLinearVelocity", &settings_.variables_.optimizeInitialLinearVelocity_)
			.group(group).label("Optimize Initial Linear Velocity").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeInitialLinearVelocity_ = v;
			if (!noInitialValues) {
				params->setOptions("AdjointSolver-InitialLinearVelocityX", v ? t : f);
				params->setOptions("AdjointSolver-InitialLinearVelocityY", v ? t : f);
				params->setOptions("AdjointSolver-InitialLinearVelocityZ", v ? t : f);
			}
		}, [this]()
		{
			return settings_.variables_.optimizeInitialLinearVelocity_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialLinearVelocityX", &settings_.variables_.currentInitialLinearVelocity_.x)
				.group(group).label("Initial Linear Velocity X").step(0.01f).visible(settings_.variables_.optimizeInitialLinearVelocity_);
			params->addParam("AdjointSolver-InitialLinearVelocityY", &settings_.variables_.currentInitialLinearVelocity_.y)
				.group(group).label("Initial Linear Velocity Y").step(0.01f).visible(settings_.variables_.optimizeInitialLinearVelocity_);
			params->addParam("AdjointSolver-InitialLinearVelocityZ", &settings_.variables_.currentInitialLinearVelocity_.z)
				.group(group).label("Initial Linear Velocity Z").step(0.01f).visible(settings_.variables_.optimizeInitialLinearVelocity_);
		}

		params->addParam("AdjointSolver-OptimizeInitialAngularVelocity", &settings_.variables_.optimizeInitialAngularVelocity_)
			.group(group).label("Optimize Initial Angular Velocity").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeInitialAngularVelocity_ = v;
			if (!noInitialValues) {
				params->setOptions("AdjointSolver-InitialAngularVelocityX", v ? t : f);
				params->setOptions("AdjointSolver-InitialAngularVelocityY", v ? t : f);
				params->setOptions("AdjointSolver-InitialAngularVelocityZ", v ? t : f);
			}
		}, [this]()
		{
			return settings_.variables_.optimizeInitialAngularVelocity_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialAngularVelocityX", &settings_.variables_.currentInitialAngularVelocity_.x)
				.group(group).label("Initial Angular Velocity X").step(0.01f).visible(settings_.variables_.optimizeInitialAngularVelocity_);
			params->addParam("AdjointSolver-InitialAngularVelocityY", &settings_.variables_.currentInitialAngularVelocity_.y)
				.group(group).label("Initial Angular Velocity Y").step(0.01f).visible(settings_.variables_.optimizeInitialAngularVelocity_);
			params->addParam("AdjointSolver-InitialAngularVelocityZ", &settings_.variables_.currentInitialAngularVelocity_.z)
				.group(group).label("Initial Angular Velocity Z").step(0.01f).visible(settings_.variables_.optimizeInitialAngularVelocity_);
		}

		params->addParam("AdjointSolver-OptimizeGroundPlane", &settings_.variables_.optimizeGroundPlane_)
			.group(group).label("Optimize Ground Plane").accessors(
				[params, this, noInitialValues](bool v)
		{
			settings_.variables_.optimizeGroundPlane_ = v;
			if (!noInitialValues) {
				params->setOptions("AdjointSolver-InitialGroundPlaneAngle", v ? t : f);
				params->setOptions("AdjointSolver-InitialGroundPlaneHeight", v ? t : f);
			}
		}, [this]()
		{
			return settings_.variables_.optimizeGroundPlane_;
		});
		if (!noInitialValues) {
			params->addParam("AdjointSolver-InitialGroundPlaneAngle", reinterpret_cast<glm::tvec3<real, glm::highp>*>(&settings_.variables_.currentGroundPlane_.x))
				.group(group).label("Initial Ground Plane Angle").visible(settings_.variables_.optimizeGroundPlane_);
			params->addParam("AdjointSolver-InitialGroundPlaneHeight", &settings_.variables_.currentGroundPlane_.w)
				.group(group).label("Initial Ground Plane Height").step(0.01f).visible(settings_.variables_.optimizeGroundPlane_);
		}

        //OPTIMIZER SETTINGS
        params->addParam("AdjointSolver-GD-Epsilon", &settings_.gradientDescentSettings_.epsilon_)
            .group(group).label("GD: Epsilon").visible(settings_.optimizer_ == Settings::GRADIENT_DESCENT)
            .optionsStr("help='Terminates if the norm of the gradient falls below this epsilon. Leave empty for default value.'");
        params->addParam("AdjointSolver-GD-LinearStepsize", &settings_.gradientDescentSettings_.linearStepsize_)
            .group(group).label("GD: Initial").visible(settings_.optimizer_ == Settings::GRADIENT_DESCENT)
            .optionsStr("help='Initial step size. Leave empty for default value.'");
        params->addParam("AdjointSolver-GD-MaxStepsize", &settings_.gradientDescentSettings_.maxStepsize_)
            .group(group).label("GD: Max Stepsize").visible(settings_.optimizer_ == Settings::GRADIENT_DESCENT)
            .optionsStr("help='Maximal step size. If empty, no restriction is applied'");
        params->addParam("AdjointSolver-GD-MinStepsize", &settings_.gradientDescentSettings_.minStepsize_)
            .group(group).label("GD: Min Stepsize").visible(settings_.optimizer_ == Settings::GRADIENT_DESCENT)
            .optionsStr("help='Minimal step size. If empty, no restriction is applied'");

		params->addParam("AdjointSolver-Rprop-Epsilon", &settings_.rpropSettings_.epsilon_)
			.group(group).label("Rprop: Epsilon").visible(settings_.optimizer_ == Settings::RPROP)
			.optionsStr("help='Terminates if the norm of the gradient falls below this epsilon. Leave empty for default value.'");
		params->addParam("AdjointSolver-Rprop-InitialStepsize", &settings_.rpropSettings_.initialStepsize_)
			.group(group).label("Rprop: Initial").visible(settings_.optimizer_ == Settings::RPROP)
			.optionsStr("help='Initial step size. Leave empty for default value.'");

        params->addParam("AdjointSolver-LBFGS-Epsilon", &settings_.lbfgsSettings_.epsilon_)
            .group(group).label("LBFGS: Epsilon").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Terminates if the norm of the gradient falls below this epsilon. Leave empty for default value.'");
        params->addParam("AdjointSolver-LBFGS-Past", &settings_.lbfgsSettings_.past_).min(0)
            .group(group).label("LBFGS: Past Distance").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Number of steps into the past for tests if the cost function reached a plateau. Set to zero to disable.'");
        params->addParam("AdjointSolver-LBFGS-Delta", &settings_.lbfgsSettings_.delta_)
            .group(group).label("LBFGS: Past Delta").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Tolerance for plateau termination criterion'");
        std::vector<std::string> lbfgsLinesearchAlgs = { "Armijo", "Wolfe", "StrongWolfe" };
        params->addParam("AdjointSolver-LBFGS-Algorithm", lbfgsLinesearchAlgs, reinterpret_cast<int*>(&settings_.lbfgsSettings_.lineSearchAlg_))
            .group(group).label("LBFGS: LS Algorithm").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='The linesearch algorithm used to find the best step size'");
        params->addParam("AdjointSolver-LBFGS-LinesearchMaxTrials", &settings_.lbfgsSettings_.linesearchMaxTrials_)
            .group(group).label("LBFGS: LS max trials").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='The maximal number of trials in the line search'");
        params->addParam("AdjointSolver-LBFGS-LinesearchMinStep", &settings_.lbfgsSettings_.linesearchMinStep_)
            .group(group).label("LBFGS: LS min step").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Minimal step size in the line search step. Leave empty for default value.'");
        params->addParam("AdjointSolver-LBFGS-LinesearchMaxStep", &settings_.lbfgsSettings_.linesearchMaxStep_)
            .group(group).label("LBFGS: LS max step").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Maximal step size in the line search step. Leave empty for default value.'");
        params->addParam("AdjointSolver-LBFGS-LinesearchTol", &settings_.lbfgsSettings_.linesearchTol_)
            .group(group).label("LBFGS: LS tolerance").visible(settings_.optimizer_ == Settings::LBFGS)
            .optionsStr("help='Tolerance in Armijo condition. Leave empty for default value.'");

	}

    void AdjointSolver::GUI::load(const cinder::JsonTree& parent, bool noInitialValues)
    {
        settings_.numIterations_ = parent.getValueForKey<int>("NumIterations");
        settings_.optimizer_ = Settings::ToOptimizer(parent.getValueForKey("Optimizer"));
        if (parent.hasChild("MemorySaving")) settings_.memorySaving_ = parent.getValueForKey<bool>("MemorySaving");
		if (parent.hasChild("NormalizeUnits")) settings_.normalizeUnits_ = parent.getValueForKey<bool>("NormalizeUnits");

        const cinder::JsonTree& gd = parent.getChild("GradientDescent");
        settings_.gradientDescentSettings_.epsilon_ = gd.getValueForKey("Epsilon");
        settings_.gradientDescentSettings_.linearStepsize_ = gd.getValueForKey("LinearStepsize");
        settings_.gradientDescentSettings_.maxStepsize_ = gd.getValueForKey("MaxStepsize");
        settings_.gradientDescentSettings_.minStepsize_ = gd.getValueForKey("MinStepsize");

		if (parent.hasChild("Rprop")) {
			const cinder::JsonTree& gd = parent.getChild("Rprop");
			settings_.rpropSettings_.epsilon_ = gd.getValueForKey("Epsilon");
			settings_.rpropSettings_.initialStepsize_ = gd.getValueForKey("InitialStepsize");
		}

        const cinder::JsonTree& lbfgs = parent.getChild("LBFGS");
        settings_.lbfgsSettings_.epsilon_ = lbfgs.getValueForKey("Epsilon");
        settings_.lbfgsSettings_.past_ = lbfgs.getValueForKey<int>("Past");
        settings_.lbfgsSettings_.delta_ = lbfgs.getValueForKey("Delta");
        settings_.lbfgsSettings_.lineSearchAlg_ = Settings::LbfgsSettings::ToLineSearchAlg(lbfgs.getValueForKey("LineSearchAlg"));
        settings_.lbfgsSettings_.linesearchMaxTrials_ = lbfgs.getValueForKey<int>("LineSearchMaxTrials");
        settings_.lbfgsSettings_.linesearchMinStep_ = lbfgs.getValueForKey("LineSearchMinStep");
        settings_.lbfgsSettings_.linesearchMaxStep_ = lbfgs.getValueForKey("LineSearchMaxStep");
        settings_.lbfgsSettings_.linesearchTol_ = lbfgs.getValueForKey("LineSearchTol");

        const cinder::JsonTree& input = parent.getChild("InitialValues");
        settings_.variables_.optimizeGravity_ = input.getValueForKey<bool>("OptimizeGravity");
		if (!noInitialValues) {
			settings_.variables_.currentGravity_.x = input.getChild("InitialGravity").getValueAtIndex<real>(0);
			settings_.variables_.currentGravity_.y = input.getChild("InitialGravity").getValueAtIndex<real>(1);
			settings_.variables_.currentGravity_.z = input.getChild("InitialGravity").getValueAtIndex<real>(2);
		}
        settings_.variables_.optimizeYoungsModulus_ = input.getValueForKey<bool>("OptimizeYoungsModulus");
		if (!noInitialValues) settings_.variables_.currentYoungsModulus_ = input.getValueForKey<real>("InitialYoungsModulus");
        settings_.variables_.optimizePoissonRatio_ = input.getValueForKey<bool>("OptimizePoissonRatio");
		if (!noInitialValues) settings_.variables_.currentPoissonRatio_ = input.getValueForKey<real>("InitialPoissonRatio");
        settings_.variables_.optimizeMass_ = input.getValueForKey<bool>("OptimizeMass");
		if (!noInitialValues) settings_.variables_.currentMass_ = input.getValueForKey<real>("InitialMass");
        settings_.variables_.optimizeMassDamping_ = input.getValueForKey<bool>("OptimizeMassDamping");
		if (!noInitialValues)  settings_.variables_.currentMassDamping_ = input.getValueForKey<real>("InitialMassDamping");
        settings_.variables_.optimizeStiffnessDamping_ = input.getValueForKey<bool>("OptimizeStiffnessDamping");
		if (!noInitialValues) settings_.variables_.currentStiffnessDamping_ = input.getValueForKey<real>("InitialStiffnessDamping");
		if (input.hasChild("OptimizeInitialLinearVelocity")) settings_.variables_.optimizeInitialLinearVelocity_ = input.getValueForKey<bool>("OptimizeInitialLinearVelocity");
		if (!noInitialValues && input.hasChild("InitialLinearVelocity")) {
			settings_.variables_.currentInitialLinearVelocity_.x = input.getChild("InitialLinearVelocity").getValueAtIndex<real>(0);
			settings_.variables_.currentInitialLinearVelocity_.y = input.getChild("InitialLinearVelocity").getValueAtIndex<real>(1);
			settings_.variables_.currentInitialLinearVelocity_.z = input.getChild("InitialLinearVelocity").getValueAtIndex<real>(2);
		}
		if (input.hasChild("OptimizeInitialAngularVelocity")) settings_.variables_.optimizeInitialAngularVelocity_ = input.getValueForKey<bool>("OptimizeInitialAngularVelocity");
		if (!noInitialValues && input.hasChild("InitialAngularVelocity")) {
			settings_.variables_.currentInitialAngularVelocity_.x = input.getChild("InitialAngularVelocity").getValueAtIndex<real>(0);
			settings_.variables_.currentInitialAngularVelocity_.y = input.getChild("InitialAngularVelocity").getValueAtIndex<real>(1);
			settings_.variables_.currentInitialAngularVelocity_.z = input.getChild("InitialAngularVelocity").getValueAtIndex<real>(2);
		}
        settings_.variables_.optimizeGroundPlane_ = input.getValueForKey<bool>("OptimizeGroundPlane");
		if (!noInitialValues) {
			settings_.variables_.currentGroundPlane_.x = input.getChild("InitialGroundPlane").getValueAtIndex<real>(0);
			settings_.variables_.currentGroundPlane_.y = input.getChild("InitialGroundPlane").getValueAtIndex<real>(1);
			settings_.variables_.currentGroundPlane_.z = input.getChild("InitialGroundPlane").getValueAtIndex<real>(2);
			settings_.variables_.currentGroundPlane_.w = input.getChild("InitialGroundPlane").getValueAtIndex<real>(3);
		}

        if (params_) {
            static const std::string t = "visible=true";
            static const std::string f = "visible=false";

            bool v = settings_.optimizer_ == Settings::GRADIENT_DESCENT;
            params_->setOptions("AdjointSolver-GD-Epsilon", v ? t : f);
            params_->setOptions("AdjointSolver-GD-LinearStepsize", v ? t : f);
            params_->setOptions("AdjointSolver-GD-MaxStepsize", v ? t : f);
            params_->setOptions("AdjointSolver-GD-MinStepsize", v ? t : f);
            v = settings_.optimizer_ == Settings::LBFGS;
            params_->setOptions("AdjointSolver-LBFGS-Epsilon", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-Past", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-Delta", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-Algorithm", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-LinesearchMaxTrials", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-LinesearchMinStep", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-LinesearchMaxStep", v ? t : f);
            params_->setOptions("AdjointSolver-LBFGS-LinesearchTol", v ? t : f);

			if (!noInitialValues) {
				params_->setOptions("AdjointSolver-InitialGravityX", settings_.variables_.optimizeGravity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialGravityY", settings_.variables_.optimizeGravity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialGravityZ", settings_.variables_.optimizeGravity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialYoungsModulus", settings_.variables_.optimizeYoungsModulus_ ? t : f);
				params_->setOptions("AdjointSolver-InitialPoissonRatio", settings_.variables_.optimizePoissonRatio_ ? t : f);
				params_->setOptions("AdjointSolver-InitialMass", settings_.variables_.optimizeMass_ ? t : f);
				params_->setOptions("AdjointSolver-InitialMassDamping", settings_.variables_.optimizeMassDamping_ ? t : f);
				params_->setOptions("AdjointSolver-InitialStiffnessDamping", settings_.variables_.optimizeStiffnessDamping_ ? t : f);
				params_->setOptions("AdjointSolver-InitialLinearVelocityX", settings_.variables_.optimizeInitialLinearVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialLinearVelocityY", settings_.variables_.optimizeInitialLinearVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialLinearVelocityZ", settings_.variables_.optimizeInitialLinearVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialAngularVelocityX", settings_.variables_.optimizeInitialAngularVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialAngularVelocityY", settings_.variables_.optimizeInitialAngularVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialAngularVelocityZ", settings_.variables_.optimizeInitialAngularVelocity_ ? t : f);
				params_->setOptions("AdjointSolver-InitialGroundPlaneAngle", settings_.variables_.optimizeGroundPlane_ ? t : f);
				params_->setOptions("AdjointSolver-InitialGroundPlaneHeight", settings_.variables_.optimizeGroundPlane_ ? t : f);
			}
		}
    }

    void AdjointSolver::GUI::save(cinder::JsonTree& parent, bool noInitialValues) const
    {
        parent.addChild(cinder::JsonTree("NumIterations", settings_.numIterations_));
        parent.addChild(cinder::JsonTree("Optimizer", Settings::FromOptimizer(settings_.optimizer_)));
        parent.addChild(cinder::JsonTree("MemorySaving", settings_.memorySaving_));
		parent.addChild(cinder::JsonTree("NormalizeUnits", settings_.normalizeUnits_));

        cinder::JsonTree gd = cinder::JsonTree::makeObject("GradientDescent");
        gd.addChild(cinder::JsonTree("Epsilon", settings_.gradientDescentSettings_.epsilon_));
        gd.addChild(cinder::JsonTree("LinearStepsize", settings_.gradientDescentSettings_.linearStepsize_));
        gd.addChild(cinder::JsonTree("MaxStepsize", settings_.gradientDescentSettings_.maxStepsize_));
        gd.addChild(cinder::JsonTree("MinStepsize", settings_.gradientDescentSettings_.minStepsize_));
        parent.addChild(gd);

		cinder::JsonTree rprop = cinder::JsonTree::makeObject("Rprop");
		rprop.addChild(cinder::JsonTree("Epsilon", settings_.rpropSettings_.epsilon_));
		rprop.addChild(cinder::JsonTree("InitialStepsize", settings_.rpropSettings_.initialStepsize_));
		parent.addChild(rprop);

        cinder::JsonTree lbfgs = cinder::JsonTree::makeObject("LBFGS");
        lbfgs.addChild(cinder::JsonTree("Epsilon", settings_.lbfgsSettings_.epsilon_));
        lbfgs.addChild(cinder::JsonTree("Past", settings_.lbfgsSettings_.past_));
        lbfgs.addChild(cinder::JsonTree("Delta", settings_.lbfgsSettings_.delta_));
        lbfgs.addChild(cinder::JsonTree("LineSearchAlg", Settings::LbfgsSettings::FromLineSearchAlg(settings_.lbfgsSettings_.lineSearchAlg_)));
        lbfgs.addChild(cinder::JsonTree("LineSearchMaxTrials", settings_.lbfgsSettings_.linesearchMaxTrials_));
        lbfgs.addChild(cinder::JsonTree("LineSearchMinStep", settings_.lbfgsSettings_.linesearchMinStep_));
        lbfgs.addChild(cinder::JsonTree("LineSearchMaxStep", settings_.lbfgsSettings_.linesearchMaxStep_));
        lbfgs.addChild(cinder::JsonTree("LineSearchTol", settings_.lbfgsSettings_.linesearchTol_));
        parent.addChild(lbfgs);

        cinder::JsonTree input = cinder::JsonTree::makeObject("InitialValues");
        input.addChild(cinder::JsonTree("OptimizeGravity", settings_.variables_.optimizeGravity_));
		if (!noInitialValues)
			input.addChild(cinder::JsonTree::makeArray("InitialGravity")
				.addChild(cinder::JsonTree("", settings_.variables_.currentGravity_.x))
				.addChild(cinder::JsonTree("", settings_.variables_.currentGravity_.y))
				.addChild(cinder::JsonTree("", settings_.variables_.currentGravity_.z)));
        input.addChild(cinder::JsonTree("OptimizeYoungsModulus", settings_.variables_.optimizeYoungsModulus_));
		if (!noInitialValues) input.addChild(cinder::JsonTree("InitialYoungsModulus", settings_.variables_.currentYoungsModulus_));
        input.addChild(cinder::JsonTree("OptimizePoissonRatio", settings_.variables_.optimizePoissonRatio_));
		if (!noInitialValues) input.addChild(cinder::JsonTree("InitialPoissonRatio", settings_.variables_.currentPoissonRatio_));
        input.addChild(cinder::JsonTree("OptimizeMass", settings_.variables_.optimizeMass_));
		if (!noInitialValues) input.addChild(cinder::JsonTree("InitialMass", settings_.variables_.currentMass_));
        input.addChild(cinder::JsonTree("OptimizeMassDamping", settings_.variables_.optimizeMassDamping_));
		if (!noInitialValues) input.addChild(cinder::JsonTree("InitialMassDamping", settings_.variables_.currentMassDamping_));
        input.addChild(cinder::JsonTree("OptimizeStiffnessDamping", settings_.variables_.optimizeStiffnessDamping_));
		if (!noInitialValues) input.addChild(cinder::JsonTree("InitialStiffnessDamping", settings_.variables_.currentStiffnessDamping_));
		input.addChild(cinder::JsonTree("OptimizeInitialLinearVelocity", settings_.variables_.optimizeInitialLinearVelocity_));
		if (!noInitialValues)
			input.addChild(cinder::JsonTree::makeArray("InitialLinearVelocity")
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialLinearVelocity_.x))
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialLinearVelocity_.y))
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialLinearVelocity_.z)));
		input.addChild(cinder::JsonTree("OptimizeInitialAngularVelocity", settings_.variables_.optimizeInitialAngularVelocity_));
		if (!noInitialValues)
			input.addChild(cinder::JsonTree::makeArray("InitialAngularVelocity")
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialAngularVelocity_.x))
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialAngularVelocity_.y))
				.addChild(cinder::JsonTree("", settings_.variables_.currentInitialAngularVelocity_.z)));
        input.addChild(cinder::JsonTree("OptimizeGroundPlane", settings_.variables_.optimizeGroundPlane_));
		if (!noInitialValues)
			input.addChild(cinder::JsonTree::makeArray("InitialGroundPlane")
				.addChild(cinder::JsonTree("", settings_.variables_.currentGroundPlane_.x))
				.addChild(cinder::JsonTree("", settings_.variables_.currentGroundPlane_.y))
				.addChild(cinder::JsonTree("", settings_.variables_.currentGroundPlane_.z))
				.addChild(cinder::JsonTree("", settings_.variables_.currentGroundPlane_.w)));
        parent.addChild(input);
    }

    AdjointSolver::AdjointSolver(SimulationResults3DPtr reference, const Settings& settings, CostFunctionPtr costFunction)
		: reference_(reference)
		, settings_(settings)
		, costFunction_(costFunction)
	{
		reference_->input_.assertSizes();
		reference_->settings_.validate();
	}

	bool AdjointSolver::solve(const Callback_t& callback, BackgroundWorker2* worker)
	{
		//helper functions
		typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;
		const static auto packInputVariables = [](const InputVariables& var) -> Vec
		{
			std::vector<double> params;
			if (var.optimizeGravity_)
			{
				params.push_back(var.currentGravity_.x);
				params.push_back(var.currentGravity_.y);
				params.push_back(var.currentGravity_.z);
			}
			if (var.optimizeYoungsModulus_)
				params.push_back(var.currentYoungsModulus_);
			if (var.optimizePoissonRatio_)
				params.push_back(var.currentPoissonRatio_);
			if (var.optimizeMass_)
				params.push_back(var.currentMass_);
			if (var.optimizeMassDamping_)
				params.push_back(var.currentMassDamping_);
			if (var.optimizeStiffnessDamping_)
				params.push_back(var.currentStiffnessDamping_);
			if (var.optimizeInitialLinearVelocity_)
			{
				params.push_back(var.currentInitialLinearVelocity_.x);
				params.push_back(var.currentInitialLinearVelocity_.y);
				params.push_back(var.currentInitialLinearVelocity_.z);
			}
			if (var.optimizeInitialAngularVelocity_)
			{
				params.push_back(var.currentInitialAngularVelocity_.x);
				params.push_back(var.currentInitialAngularVelocity_.y);
				params.push_back(var.currentInitialAngularVelocity_.z);
			}
			if (var.optimizeGroundPlane_)
			{
				const real4 spherical = CoordinateTransformation::cartesian2spherical(var.currentGroundPlane_);
				params.push_back(spherical.y);
				params.push_back(spherical.z);
				params.push_back(var.currentGroundPlane_.w);
			}
			if (params.empty()) return Vec();
			Vec result = Eigen::Map<Vec>(params.data(), params.size());
			return result;
		};
		const static auto packMinMax = [](const InputVariables& var) -> std::pair<Vec, Vec>
		{
			std::vector<double> min, max;
			static const double BIG = 1e10;
			static const double SMALL = 1e-10;
			if (var.optimizeGravity_)
			{
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
			}
			if (var.optimizeYoungsModulus_) {
				min.push_back(1); max.push_back(BIG);
			}
			if (var.optimizePoissonRatio_) {
				min.push_back(0.01); max.push_back(0.49);
			}
			if (var.optimizeMass_) {
				min.push_back(SMALL); max.push_back(BIG);
			}
			if (var.optimizeMassDamping_) {
				min.push_back(SMALL); max.push_back(BIG);
			}
			if (var.optimizeStiffnessDamping_) {
				min.push_back(SMALL); max.push_back(BIG);
			}
			if (var.optimizeInitialLinearVelocity_)
			{
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
			}
			if (var.optimizeInitialAngularVelocity_)
			{
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
			}
			if (var.optimizeGroundPlane_)
			{
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
				min.push_back(-BIG); max.push_back(BIG);
			}
			Vec minVec = Eigen::Map<Vec>(min.data(), min.size());
			Vec maxVec = Eigen::Map<Vec>(max.data(), max.size());
			return std::make_pair(minVec, maxVec);
		};
		const static auto unpackInputVariables = [](const InputVariables& ref, const Vec& params) -> InputVariables
		{
			InputVariables var;
			int i = 0;
			if (ref.optimizeGravity_)
			{
				var.optimizeGravity_ = true;
				var.currentGravity_.x = static_cast<real>(params[i++]);
				var.currentGravity_.y = static_cast<real>(params[i++]);
				var.currentGravity_.z = static_cast<real>(params[i++]);
			}
			if (ref.optimizeYoungsModulus_) {
				var.optimizeYoungsModulus_ = true;
				var.currentYoungsModulus_ = static_cast<real>(params[i++]);
			}
			if (ref.optimizePoissonRatio_) {
				var.optimizePoissonRatio_ = true;
				var.currentPoissonRatio_ = static_cast<real>(params[i++]);
			}
			if (ref.optimizeMass_) {
				var.optimizeMass_ = true;
				var.currentMass_ = static_cast<real>(params[i++]);
			}
			if (ref.optimizeMassDamping_) {
				var.optimizeMassDamping_ = true;
				var.currentMassDamping_ = static_cast<real>(params[i++]);
			}
			if (ref.optimizeStiffnessDamping_) {
				var.optimizeStiffnessDamping_ = true;
				var.currentStiffnessDamping_ = static_cast<real>(params[i++]);
			}
			if (ref.optimizeInitialLinearVelocity_)
			{
				var.optimizeInitialLinearVelocity_ = true;
				var.currentInitialLinearVelocity_.x = static_cast<real>(params[i++]);
				var.currentInitialLinearVelocity_.y = static_cast<real>(params[i++]);
				var.currentInitialLinearVelocity_.z = static_cast<real>(params[i++]);
			}
			if (ref.optimizeInitialAngularVelocity_)
			{
				var.optimizeInitialAngularVelocity_ = true;
				var.currentInitialAngularVelocity_.x = static_cast<real>(params[i++]);
				var.currentInitialAngularVelocity_.y = static_cast<real>(params[i++]);
				var.currentInitialAngularVelocity_.z = static_cast<real>(params[i++]);
			}
			if (ref.optimizeGroundPlane_)
			{
				double theta = params[i++];
				double phi = params[i++];
				const double3 spherical = make_double3(1, theta, phi);
				const double3 cartesian = CoordinateTransformation::spherical2cartesian(spherical);
				var.optimizeGroundPlane_ = true;
				var.currentGroundPlane_.x = static_cast<real>(cartesian.x);
				var.currentGroundPlane_.y = static_cast<real>(cartesian.y);
				var.currentGroundPlane_.z = static_cast<real>(cartesian.z);
				var.currentGroundPlane_.w = static_cast<real>(params[i++]);
			}
			return var;
		};

		static const auto packGradient = [](const InputVariables& ref, const InputVariables& in, const AdjointVariables& adj) -> Vec
		{
			std::vector<double> params;
			if (ref.optimizeGravity_)
			{
				params.push_back(adj.adjGravity_.x);
				params.push_back(adj.adjGravity_.y);
				params.push_back(adj.adjGravity_.z);
			}
			if (ref.optimizeYoungsModulus_)
				params.push_back(adj.adjYoungsModulus_);
			if (ref.optimizePoissonRatio_)
				params.push_back(adj.adjPoissonRatio_);
			if (ref.optimizeMass_)
				params.push_back(adj.adjMass_);
			if (ref.optimizeMassDamping_)
				params.push_back(adj.adjMassDamping_);
			if (ref.optimizeStiffnessDamping_)
				params.push_back(adj.adjStiffnessDamping_);
			if (ref.optimizeInitialLinearVelocity_)
			{
				params.push_back(adj.adjInitialLinearVelocity.x);
				params.push_back(adj.adjInitialLinearVelocity.y);
				params.push_back(adj.adjInitialLinearVelocity.z);
			}
			if (ref.optimizeInitialAngularVelocity_)
			{
				params.push_back(adj.adjInitialAngularVelocity.x);
				params.push_back(adj.adjInitialAngularVelocity.y);
				params.push_back(adj.adjInitialAngularVelocity.z);
			}
			if (ref.optimizeGroundPlane_)
			{
				const double4 spherical = CoordinateTransformation::cartesian2spherical(make_double4(in.currentGroundPlane_.x, in.currentGroundPlane_.y, in.currentGroundPlane_.z, 0));
				const double4 adjSpherical = CoordinateTransformation::spherical2cartesianAdjoint(spherical, adj.adjGroundPlane_);
				params.push_back(adjSpherical.y);
				params.push_back(adjSpherical.z);
				params.push_back(adj.adjGroundPlane_.w);
			}
			Vec result = Eigen::Map<Vec>(params.data(), params.size());
			return result;
		};

		static const auto varToSettings = [](const InputVariables& var) -> SoftBodySimulation3D::Settings
		{
			SoftBodySimulation3D::Settings settings;
			settings.gravity_ = var.currentGravity_;
			settings.youngsModulus_ = var.currentYoungsModulus_;
			settings.poissonsRatio_ = var.currentPoissonRatio_;
			settings.mass_ = var.currentMass_;
			settings.dampingAlpha_ = var.currentMassDamping_;
			settings.dampingBeta_ = var.currentStiffnessDamping_;
			settings.initialLinearVelocity_ = var.currentInitialLinearVelocity_;
			settings.initialAngularVelocity_ = var.currentInitialAngularVelocity_;
			settings.groundPlane_ = var.currentGroundPlane_;
			return settings;
		};

		static const auto toDouble = [](const std::string& str, double def) -> double
		{
			try
			{
				double val = std::stod(str);
				return val;
			}
			catch (const std::invalid_argument& ex)
			{
				return def;
			}
		};

		//initialize statistics
		Statistics statistics;

		//Prepare initial values
		Vec initial = packInputVariables(settings_.variables_);
		Vec min, max; std::tie(min, max) = packMinMax(settings_.variables_);
		//callback(varToSettings(settings_.variables_), 0);

		//Initialize scaling
		Vec paramScaling = Vec::Ones(initial.size());
		Vec paramScalingInv = Vec::Ones(initial.size());
		if (settings_.normalizeUnits_) {
			int i = 0;
			if (settings_.variables_.optimizeGravity_)
			{
				real scale = std::max({ real(1e-8), abs(settings_.variables_.currentGravity_.x),  abs(settings_.variables_.currentGravity_.y),  abs(settings_.variables_.currentGravity_.z) });
				paramScalingInv[i] = paramScalingInv[i + 1] = paramScalingInv[i + 2] = scale;
				i += 3;
			}
			if (settings_.variables_.optimizeYoungsModulus_)
				paramScalingInv[i++] = settings_.variables_.currentYoungsModulus_;
			if (settings_.variables_.optimizePoissonRatio_)
				paramScalingInv[i++] = settings_.variables_.currentPoissonRatio_;
			if (settings_.variables_.optimizeMass_)
				paramScalingInv[i++] = settings_.variables_.currentMass_;
			if (settings_.variables_.optimizeMassDamping_)
				paramScalingInv[i++] = std::max(real(1e-5), settings_.variables_.currentMassDamping_);
			if (settings_.variables_.optimizeStiffnessDamping_)
				paramScalingInv[i++] = std::max(real(1e-5), settings_.variables_.currentStiffnessDamping_);
			if (settings_.variables_.optimizeInitialLinearVelocity_) {
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialLinearVelocity_.x));
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialLinearVelocity_.y));
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialLinearVelocity_.z));
			}
			if (settings_.variables_.optimizeInitialAngularVelocity_) {
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialAngularVelocity_.x));
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialAngularVelocity_.y));
				paramScalingInv[i++] = std::max(real(1), abs(settings_.variables_.currentInitialAngularVelocity_.z));
			}
			if (settings_.variables_.optimizeGroundPlane_)
			{
				const real4 spherical = CoordinateTransformation::cartesian2spherical(settings_.variables_.currentGroundPlane_);
				real scale = std::max({ real(1e-8), spherical.y, spherical.z });
				paramScalingInv[i] = paramScalingInv[i + 1] = scale;
				paramScalingInv[i + 2] = std::max(real(1), settings_.variables_.currentGroundPlane_.w);
				i += 3;
			}
			paramScaling = paramScalingInv.cwiseInverse();
			CI_LOG_I("parameter scaling: " << paramScalingInv.transpose());
		}

		if (settings_.optimizer_ == Settings::GRADIENT_DESCENT) {
			//Gradient descent
			real finalCost = 0;
			const auto fun = [this, &finalCost, &statistics, &paramScaling, &paramScalingInv, &worker](const Vec& xs) -> Vec
			{
				const Vec x = xs.cwiseProduct(paramScalingInv);
				CI_LOG_I("X: " << x.transpose());
				InputVariables var = unpackInputVariables(this->settings_.variables_, x);
				AdjointVariables adj = { 0 };
				if (isUseAdjoint())
					finalCost = computeGradient(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, settings_.memorySaving_, worker, &statistics);
				else
					finalCost = computeGradientFiniteDifferences(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, finiteDifferencesDelta_, worker, &statistics);
				Vec gradient = packGradient(this->settings_.variables_, var, adj);
				CI_LOG_I("GradientDescent-Step:\n cost=" << finalCost << "\n values:" << var << "\n gradient:" << adj << " (" << gradient.transpose() << ")");
				return gradient.cwiseProduct(paramScalingInv);
			};
			ar::GradientDescent<Vec> gd(initial.cwiseProduct(paramScaling).eval(), fun);
			gd.setEpsilon(toDouble(settings_.gradientDescentSettings_.epsilon_, 1e-15));
			gd.setLinearStepsize(toDouble(settings_.gradientDescentSettings_.linearStepsize_, 0.001));
			gd.setMaxStepsize(toDouble(settings_.gradientDescentSettings_.maxStepsize_, 1e20));
			gd.setMinStepsize(toDouble(settings_.gradientDescentSettings_.epsilon_, 0));
			gd.setMinValues(min.cwiseProduct(paramScaling));
			gd.setMaxValues(max.cwiseProduct(paramScaling));
			for (int oi = 0; oi < settings_.numIterations_; ++oi)
			{
				worker->setStatus(tfm::format("Adjoint: optimization %d/%d", (oi + 1), settings_.numIterations_));
				if (gd.step()) break;
				if (worker->isInterrupted()) break;
				//fetch intermediate
				this->finalCost_ = finalCost;
				this->finalVariables_ = varToSettings(unpackInputVariables(settings_.variables_, gd.getCurrentSolution().cwiseProduct(paramScalingInv)));
				auto gradient = varToSettings(unpackInputVariables(settings_.variables_, gd.getCurrentGradient().cwiseProduct(paramScaling)));
				callback(this->finalVariables_, gradient, this->finalCost_);
			}
			//fetch final output
			this->finalCost_ = finalCost;
			this->finalVariables_ = varToSettings(unpackInputVariables(settings_.variables_, gd.getCurrentSolution().cwiseProduct(paramScalingInv)));
		}
		else if (settings_.optimizer_ == Settings::RPROP) {
			//RProp (Resilient Back Propagation) Gradient descent
			real finalCost = 0;
			const auto fun = [this, &finalCost, &statistics, &paramScaling, &paramScalingInv, &worker](const Vec& xs) -> Vec
			{
				const Vec x = xs.cwiseProduct(paramScalingInv);
				CI_LOG_I("X: " << x.transpose());
				InputVariables var = unpackInputVariables(this->settings_.variables_, x);
				AdjointVariables adj = { 0 };
				if (isUseAdjoint())
					finalCost = computeGradient(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, settings_.memorySaving_, worker, &statistics);
				else
					finalCost = computeGradientFiniteDifferences(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, finiteDifferencesDelta_, worker, &statistics);
				Vec gradient = packGradient(this->settings_.variables_, var, adj);
				CI_LOG_I("Rprop-GradientDescent-Step:\n cost=" << finalCost << "\n values:" << var << "\n gradient:" << adj << " (" << gradient.transpose() << ")");
				return gradient.cwiseProduct(paramScalingInv);
			};
			ar::RpropGradientDescent<Vec> rprop(initial.cwiseProduct(paramScaling).eval(), fun);
			rprop.setEpsilon(toDouble(settings_.rpropSettings_.epsilon_, 1e-15));
			rprop.setInitialStepsize(toDouble(settings_.rpropSettings_.initialStepsize_, 0.001));
			rprop.setMinValues(min.cwiseProduct(paramScaling));
			rprop.setMaxValues(max.cwiseProduct(paramScaling));
			for (int oi = 0; oi < settings_.numIterations_; ++oi)
			{
				worker->setStatus(tfm::format("Adjoint: optimization %d/%d", (oi + 1), settings_.numIterations_));
				auto currentVariables = varToSettings(unpackInputVariables(settings_.variables_, rprop.getCurrentSolution().cwiseProduct(paramScalingInv)));
				if (rprop.step()) break;
				if (worker->isInterrupted()) break;
				//fetch intermediate
				this->finalCost_ = finalCost;
				this->finalVariables_ = varToSettings(unpackInputVariables(settings_.variables_, rprop.getCurrentSolution().cwiseProduct(paramScalingInv)));
				auto gradient = varToSettings(unpackInputVariables(settings_.variables_, rprop.getCurrentGradient().cwiseProduct(paramScaling)));
				callback(currentVariables, gradient, this->finalCost_);
			}
			//fetch final output
			this->finalCost_ = finalCost;
			this->finalVariables_ = varToSettings(unpackInputVariables(settings_.variables_, rprop.getCurrentSolution().cwiseProduct(paramScalingInv)));
		}
		else if (settings_.optimizer_ == Settings::LBFGS)
		{
			//LBFGS
			LBFGSpp::LBFGSParam<double> params;
			params.epsilon = toDouble(settings_.lbfgsSettings_.epsilon_, params.epsilon);
			params.past = settings_.lbfgsSettings_.past_;
			params.delta = toDouble(settings_.lbfgsSettings_.delta_, params.delta);
			params.linesearch = settings_.lbfgsSettings_.lineSearchAlg_ == Settings::LbfgsSettings::Armijo
				? LBFGSpp::LINE_SEARCH_ALGORITHM::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO
				: settings_.lbfgsSettings_.lineSearchAlg_ == Settings::LbfgsSettings::Wolfe
				? LBFGSpp::LINE_SEARCH_ALGORITHM::LBFGS_LINESEARCH_BACKTRACKING_WOLFE
				: LBFGSpp::LINE_SEARCH_ALGORITHM::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
			params.max_linesearch = settings_.lbfgsSettings_.linesearchMaxTrials_;
			params.min_step = toDouble(settings_.lbfgsSettings_.linesearchMinStep_, params.min_step);
			params.max_step = toDouble(settings_.lbfgsSettings_.linesearchMaxStep_, params.max_step);
			params.ftol = toDouble(settings_.lbfgsSettings_.linesearchTol_, params.ftol);
			params.max_iterations = settings_.numIterations_;
			LBFGSpp::LBFGSSolver<double> lbfgs(params);
			LBFGSpp::LBFGSSolver<double>::ObjectiveFunction_t fun = [this, &statistics, &paramScaling, &paramScalingInv, &worker](const Vec& xs, Vec& gradient) -> double
			{
				const Vec x = xs.cwiseProduct(paramScalingInv);
				InputVariables var = unpackInputVariables(this->settings_.variables_, x);
				AdjointVariables adj = { 0 };
				double finalCost;
				if (isUseAdjoint())
					finalCost = computeGradient(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, settings_.memorySaving_, worker, &statistics);
				else
					finalCost = computeGradientFiniteDifferences(
						reference_->input_, reference_->settings_,
						var, costFunction_, adj, finiteDifferencesDelta_, worker, &statistics);
				gradient = packGradient(this->settings_.variables_, var, adj).cwiseProduct(paramScalingInv);
				CI_LOG_I("LBFGS-Step:\n cost=" << finalCost << "\n values:" << var << "\n gradient:" << adj);
				return finalCost;
			};
			LBFGSpp::LBFGSSolver<double>::CallbackFunction_t lbfgsCallback = [this, worker, callback, &paramScaling, &paramScalingInv](const Vec& x, const Vec& g, const double& v, int k) -> bool
			{
				worker->setStatus(tfm::format("Adjoint: optimization %d/%d, cost %f", k + 1, settings_.numIterations_, v));
				InputVariables var = unpackInputVariables(this->settings_.variables_, x.cwiseProduct(paramScalingInv));
				auto gradient = varToSettings(unpackInputVariables(this->settings_.variables_, g.cwiseProduct(paramScaling)));
				callback(varToSettings(var), gradient, static_cast<real>(v));
				return !worker->isInterrupted();
			};
			LBFGSpp::LBFGSSolver<double>::ValidationFunction_t validation = [&min, &max, &paramScaling](const Vec& x) -> bool
			{
				return (x.array() >= min.cwiseProduct(paramScaling).array()).all() && (x.array() <= max.cwiseProduct(paramScaling).array()).all();
			};
			double finalCost = 0;
			Vec value = initial.cwiseProduct(paramScaling);
			try {
				int oi = lbfgs.minimize(fun, value, finalCost, lbfgsCallback, validation);
				CI_LOG_I("Optimized for " << oi << " iterations, final cost " << finalCost);
			}
			catch (const std::runtime_error& error)
			{
				CI_LOG_EXCEPTION("LBFGS failed", error);
				return false;
			}
			//fetch final output
			this->finalCost_ = static_cast<real>(finalCost);
			this->finalVariables_ = varToSettings(unpackInputVariables(settings_.variables_, value.cwiseProduct(paramScalingInv)));
		}

		SoftBodySimulation3D::Settings finalGradient; memset(&finalGradient, 0, sizeof(SoftBodySimulation3D::Settings));
		callback(this->finalVariables_, finalGradient, this->finalCost_);
        CI_LOG_I("Result: cost = " << finalCost_ << ", values:" << this->finalVariables_);
        CI_LOG_I(statistics);
        return true;
	}

    void AdjointSolver::testGradient(BackgroundWorker2* worker)
    {
        static const int numSteps = 15;
		static const int halfNumSteps = numSteps / 2;
        static const real frac = 0.5;
		CI_LOG_I("TEST GRADIENT");

#define GRADIENT(OPT, VAR1, VAR2, VARADJ, NAME)                                                         \
        if (settings_.variables_.OPT && !worker->isInterrupted())                                       \
        {                                                                                               \
            InputVariables variables;                                                                   \
            variables.OPT = true;                                                                       \
            const real minValue = reference_->settings_.VAR1==0 ? -halfNumSteps : reference_->settings_.VAR1 * frac;                                    \
            const real maxValue = reference_->settings_.VAR1==0 ? +halfNumSteps : 2*reference_->settings_.VAR1 - minValue;                              \
            typedef std::array<double, 3> entry;                                                        \
            std::vector<entry> entries;                                                                 \
            for (int i=0; i<numSteps && !worker->isInterrupted(); ++i)                                  \
            {                                                                                           \
                worker->setStatus(tinyformat::format("%s gradient %d/%d", NAME, i+1, numSteps));        \
                const real value = minValue + (maxValue - minValue) * i / (numSteps - 1);               \
                variables.VAR2 = value;                                                                 \
                AdjointVariables adj = { 0 };                                                           \
                const real cost = computeGradient(                                                      \
                    reference_->input_, reference_->settings_, variables, costFunction_, adj, settings_.memorySaving_);          \
                entries.push_back(entry{ value, cost, adj.VARADJ });                                    \
            }                                                                                           \
			std::stringstream ss;																		\
            ss << NAME << ":" << std::endl;                                         \
            ss << "  Value     Cost      Gradient" << std::endl;                    \
            for (int i=0; i<entries.size(); ++i)                                                        \
            {                                                                                           \
                ss                                                                  \
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][0]     \
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][1]     \
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][2]     \
                    << std::endl;                                                                       \
            }                                                                                           \
			CI_LOG_I(ss.str());																				\
        }

        GRADIENT(optimizeMass_, mass_, currentMass_, adjMass_, "Mass");
        //GRADIENT(optimizeGravity_, gravity_.x, currentGravity_.x, adjGravity_.x, "GravityX");
        GRADIENT(optimizeGravity_, gravity_.y, currentGravity_.y, adjGravity_.y, "GravityY");
        //GRADIENT(optimizeGravity_, gravity_.z, currentGravity_.z, adjGravity_.z, "GravityZ");
        GRADIENT(optimizeMassDamping_, dampingAlpha_, currentMassDamping_, adjMassDamping_, "DampingMass");
        GRADIENT(optimizeStiffnessDamping_, dampingBeta_, currentStiffnessDamping_, adjStiffnessDamping_, "DampingStiffness");
        GRADIENT(optimizeYoungsModulus_, youngsModulus_, currentYoungsModulus_, adjYoungsModulus_, "Young's Modulus");
        GRADIENT(optimizePoissonRatio_, poissonsRatio_, currentPoissonRatio_, adjPoissonRatio_, "Poisson Ratio");
		GRADIENT(optimizeInitialLinearVelocity_, initialLinearVelocity_.x, currentInitialLinearVelocity_.x, adjInitialLinearVelocity.x, "LinearVelocityX");
		GRADIENT(optimizeInitialLinearVelocity_, initialLinearVelocity_.y, currentInitialLinearVelocity_.y, adjInitialLinearVelocity.y, "LinearVelocityY");
		GRADIENT(optimizeInitialLinearVelocity_, initialLinearVelocity_.z, currentInitialLinearVelocity_.z, adjInitialLinearVelocity.z, "LinearVelocityZ");
		GRADIENT(optimizeInitialAngularVelocity_, initialAngularVelocity_.x, currentInitialAngularVelocity_.x, adjInitialAngularVelocity.x, "AngularVelocityX");
		GRADIENT(optimizeInitialAngularVelocity_, initialAngularVelocity_.y, currentInitialAngularVelocity_.y, adjInitialAngularVelocity.y, "AngularVelocityY");
		GRADIENT(optimizeInitialAngularVelocity_, initialAngularVelocity_.z, currentInitialAngularVelocity_.z, adjInitialAngularVelocity.z, "AngularVelocityZ");

#undef GRADIENT

        // young's modulus and poisson's ratio in uniform
        if (settings_.variables_.optimizeYoungsModulus_ && settings_.variables_.optimizePoissonRatio_ && !worker->isInterrupted())
        {
            InputVariables variables;
            variables.optimizePoissonRatio_ = true;
            variables.optimizeYoungsModulus_ = true;
            const real minYoungValue = reference_->settings_.youngsModulus_ * frac;
            const real maxYoungValue = 2 * reference_->settings_.youngsModulus_ - minYoungValue;
            const real minPoissonValue = reference_->settings_.poissonsRatio_ * frac;
            const real maxPoissonValue = 2 * reference_->settings_.poissonsRatio_ - minPoissonValue;
            typedef std::array<double, 5> entry;
            std::vector<entry> entries;
            for (int i = 0; i < numSteps && !worker->isInterrupted(); ++i)
            for (int j = 0; j < numSteps && !worker->isInterrupted(); ++j)
            {
                worker->setStatus(tinyformat::format("%s gradient %d/%d", "Young+Poisson", i + 1, numSteps));
                const real young = minYoungValue + (maxYoungValue - minYoungValue) * i / (numSteps - 1);
                const real poisson = minPoissonValue + (maxPoissonValue - minPoissonValue) * j / (numSteps - 1);
                variables.currentYoungsModulus_ = young;
                variables.currentPoissonRatio_ = poisson;
                AdjointVariables adj = {0};
                const real cost = computeGradient(
                    reference_->input_, reference_->settings_, variables, costFunction_, adj, settings_.memorySaving_);
                entries.push_back(entry{young, poisson, cost, adj.adjYoungsModulus_, adj.adjPoissonRatio_});
            }
			std::stringstream ss;
            ss << "Young's Modulus and Poisson's Ratio:" << std::endl;
            ss << "  YoungModulus    PoissonRatio     Cost      GradientYoung    GradientPoisson" << std::endl;
            for (int i = 0; i < entries.size(); ++i)
            {
                ss
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][0]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][1]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][2]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][3]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][4]
                    << std::endl;
            }  
			CI_LOG_I(ss.str());
        }

        // stiffness and mass damping in uniform
        if (settings_.variables_.optimizeMassDamping_ && settings_.variables_.optimizeStiffnessDamping_ && !worker->isInterrupted())
        {
            InputVariables variables;
            variables.optimizeMassDamping_ = true;
            variables.optimizeStiffnessDamping_ = true;
            const real minMassValue = reference_->settings_.dampingAlpha_ * frac;
            const real maxMassValue = 2 * reference_->settings_.dampingAlpha_ - minMassValue;
            const real minStiffnessValue = reference_->settings_.dampingBeta_ * frac;
            const real maxStiffnessValue = 2 * reference_->settings_.dampingBeta_ - minStiffnessValue;
            typedef std::array<double, 5> entry;
            std::vector<entry> entries;
            for (int i = 0; i < numSteps && !worker->isInterrupted(); ++i)
                for (int j = 0; j < numSteps && !worker->isInterrupted(); ++j)
                {
                    worker->setStatus(tinyformat::format("%s gradient %d/%d", "AllDamping", i + 1, numSteps));
                    const real mass = minMassValue + (maxMassValue - minMassValue) * i / (numSteps - 1);
                    const real stiffness = minStiffnessValue + (maxStiffnessValue - minStiffnessValue) * j / (numSteps - 1);
                    variables.currentMassDamping_ = mass;
                    variables.currentStiffnessDamping_ = stiffness;
                    AdjointVariables adj = { 0 };
                    const real cost = computeGradient(
                        reference_->input_, reference_->settings_, variables, costFunction_, adj, settings_.memorySaving_);
                    entries.push_back(entry{ mass, stiffness, cost, adj.adjMassDamping_, adj.adjStiffnessDamping_ });
                }
			std::stringstream ss;
            ss << "Mass Damping and Stiffness Damping:" << std::endl;
            ss << "  Mass-Damping    Stiffness-Damping     Cost      Gradient-Mass    Gradient-Stiffness" << std::endl;
            for (int i = 0; i < entries.size(); ++i)
            {
                ss
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][0]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][1]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][2]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][3]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][4]
                    << std::endl;
            }
			CI_LOG_I(ss.str());
        }

        //Ground Plane
        if (settings_.variables_.optimizeGroundPlane_ && !worker->isInterrupted())
        {
            InputVariables variables;
            variables.optimizeGroundPlane_ = true;
            const real referenceHeight = reference_->settings_.groundPlane_.w;
            const real referenceTheta = CoordinateTransformation::cartesian2spherical(reference_->settings_.groundPlane_).y;
            const real referencePhi = CoordinateTransformation::cartesian2spherical(reference_->settings_.groundPlane_).z;
            const real minTheta = referenceTheta - frac * M_PI * 0.5;
            const real maxTheta = referenceTheta + frac * M_PI * 0.5;
            const real minPhi = referencePhi - frac * M_PI * 0.5;
            const real maxPhi = referencePhi + frac * M_PI * 0.5;
            typedef std::array<double, 5> entry;
            std::vector<entry> entries;
            for (int i = 0; i < numSteps && !worker->isInterrupted(); ++i)
                for (int j = 0; j < numSteps && !worker->isInterrupted(); ++j)
                {
                    worker->setStatus(tinyformat::format("%s gradient %d/%d", "GroundPlaneOrientation", i + 1, numSteps));
                    const real theta = minTheta + (maxTheta - minTheta) * i / (numSteps - 1);
                    const real phi = minPhi + (maxPhi - minPhi) * j / (numSteps - 1);
                    variables.currentGroundPlane_ = CoordinateTransformation::spherical2cartesian(make_real3(1, theta, phi));
                    variables.currentGroundPlane_.w = referenceHeight;
                    AdjointVariables adj = { 0 };
                    const real cost = computeGradient(
                        reference_->input_, reference_->settings_, variables, costFunction_, adj, settings_.memorySaving_);
                    double4 adjSpherical = CoordinateTransformation::spherical2cartesianAdjoint(make_double4(1, theta, phi, 0), adj.adjGroundPlane_);
                    entries.push_back(entry{ theta, phi, cost, adjSpherical.y, adjSpherical.z });

                    CI_LOG_V(
                        << "  " << std::fixed << std::setw(12) << std::setprecision(7) << theta
                        << "  " << std::fixed << std::setw(12) << std::setprecision(7) << phi
                        << "  " << std::fixed << std::setw(12) << std::setprecision(7) << cost
                        << "  " << std::fixed << std::setw(12) << std::setprecision(7) << adjSpherical.y
                        << "  " << std::fixed << std::setw(12) << std::setprecision(7) << adjSpherical.z
                        << "  (" << CoordinateTransformation::spherical2cartesian(make_real3(1, theta, phi)).x
                        << ", "  << CoordinateTransformation::spherical2cartesian(make_real3(1, theta, phi)).y
                        << ", "  << CoordinateTransformation::spherical2cartesian(make_real3(1, theta, phi)).z << ")"
                        );
                }
			std::stringstream ss;
            ss << "Ground Plane Orientation:" << std::endl;
            ss << "  Theta    Phi     Cost      Gradient-Theta    Gradient-Phi" << std::endl;
            for (int i = 0; i < entries.size(); ++i)
            {
                ss
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][0]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][1]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][2]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][3]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][4]
                    << std::endl;
            }
			CI_LOG_I(ss.str());
        }
        if (settings_.variables_.optimizeGroundPlane_ && !worker->isInterrupted())
        {
            InputVariables variables;
            variables.optimizeGroundPlane_ = true;
            const real minValue = reference_->settings_.groundPlane_.w - 0.1;
            const real maxValue = 2 * reference_->settings_.groundPlane_.w + 0.1;
            typedef std::array<double, 3> entry;
            std::vector<entry> entries;
            for (int i = 0; i < numSteps && !worker->isInterrupted(); ++i)
            {
                worker->setStatus(tinyformat::format("%s gradient %d/%d", "Ground Height", i + 1, numSteps));
                const real value = minValue + (maxValue - minValue) * i / (numSteps - 1);
                variables.currentGroundPlane_ = reference_->settings_.groundPlane_;
                variables.currentGroundPlane_.w = value;
                AdjointVariables adj = { 0 };
                const real cost = computeGradient(reference_->input_, reference_->settings_, variables, costFunction_,
                    adj, settings_.memorySaving_);
                entries.push_back(entry{ value, cost, adj.adjGroundPlane_.w });
            }
			std::stringstream ss;
            ss << "Ground Height" << ":" << std::endl;
            ss << "  Value     Cost      Gradient" << std::endl;
            for (int i = 0; i < entries.size(); ++i)
            {
                ss << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][0]
                    << "  " << std::fixed << std::setw(12) << std::setprecision(7) << entries[i][1] << "  " << std::
                    fixed << std::setw(12) << std::setprecision(7) << entries[i][2] << std::endl;
            }
			CI_LOG_I(ss.str());
        }

		CI_LOG_I("DONE");
    }

    std::ostream& operator<<(std::ostream& os, const AdjointSolver::AdjointVariables& obj)
    {
        if (obj.adjGravity_.x != 0 || obj.adjGravity_.y != 0 || obj.adjGravity_.z != 0)
            os << " Gravity=(" << obj.adjGravity_.x << "," << obj.adjGravity_.y << "," << obj.adjGravity_.z << ")";
        if (obj.adjYoungsModulus_ != 0)
            os << " YoungsModulus=" << obj.adjYoungsModulus_;
        if (obj.adjPoissonRatio_ != 0)
            os << " PoissonRatio=" << obj.adjPoissonRatio_;
        if (obj.adjMass_ != 0)
            os << " Mass=" << obj.adjMass_;
        if (obj.adjMassDamping_ != 0)
            os << " MassDamping=" << obj.adjMassDamping_;
        if (obj.adjStiffnessDamping_ != 0)
            os << " StiffnessDamping=" << obj.adjStiffnessDamping_;
		if (obj.adjInitialLinearVelocity.x != 0 || obj.adjInitialLinearVelocity.y != 0 || obj.adjInitialLinearVelocity.z != 0)
			os << " InitialLinearVelocity=(" << obj.adjInitialLinearVelocity.x << "," << obj.adjInitialLinearVelocity.y << "," << obj.adjInitialLinearVelocity.z << ")";
		if (obj.adjInitialAngularVelocity.x != 0 || obj.adjInitialAngularVelocity.y != 0 || obj.adjInitialAngularVelocity.z != 0)
			os << " InitialAngularVelocity=(" << obj.adjInitialAngularVelocity.x << "," << obj.adjInitialAngularVelocity.y << "," << obj.adjInitialAngularVelocity.z << ")";
		if (obj.adjGroundPlane_.x != 0 || obj.adjGroundPlane_.y != 0 || obj.adjGroundPlane_.z != 0 || obj.adjGroundPlane_.w != 0)
			os << " GroundPlane=(" << obj.adjGroundPlane_.x << "," << obj.adjGroundPlane_.y << "," << obj.adjGroundPlane_.z << "," << obj.adjGroundPlane_.w << ")";

        return os;
    }

    std::ostream& operator<<(std::ostream& os, const AdjointSolver::InputVariables& obj)
    {
        if (obj.optimizeGravity_)
            os << " Gravity=(" << obj.currentGravity_.x << "," << obj.currentGravity_.y << "," << obj.currentGravity_.z << ")";
        if (obj.optimizeYoungsModulus_)
            os << " YoungsModulus=" << obj.currentYoungsModulus_;
        if (obj.optimizePoissonRatio_)
            os << " PoissonRatio=" << obj.currentPoissonRatio_;
        if (obj.optimizeMass_)
            os << " Mass=" << obj.currentMass_;
        if (obj.optimizeMassDamping_)
            os << " MassDamping=" << obj.currentMassDamping_;
        if (obj.optimizeStiffnessDamping_)
            os << " StiffnessDamping=" << obj.currentStiffnessDamping_;
		if (obj.optimizeInitialLinearVelocity_)
			os << " InitialLinearVelocity=(" << obj.currentInitialLinearVelocity_.x << "," << obj.currentInitialLinearVelocity_.y << "," << obj.currentInitialLinearVelocity_.z << ")";
		if (obj.optimizeInitialAngularVelocity_)
			os << " InitialAngularVelocity=(" << obj.currentInitialAngularVelocity_.x << "," << obj.currentInitialAngularVelocity_.y << "," << obj.currentInitialAngularVelocity_.z << ")";
        if (obj.optimizeGroundPlane_)
            os << " GroundPlane=(" << obj.currentGroundPlane_.x << "," << obj.currentGroundPlane_.y << "," << obj.currentGroundPlane_.z << "," << obj.currentGroundPlane_.w << ")";
        return os;
    }
}

