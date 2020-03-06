#include "InverseProblem_Adjoint_InitialConfiguration.h"

#include <Eigen/Dense>
#include <tinyformat.h>
#include <cinder/Log.h>
#include <numeric>
#include <LBFGS.h>
#include <cinder/app/App.h>

#include "GradientDescent.h"
#include "Integration.h"

using namespace std;
using namespace Eigen;

#define DEBUG_SAVE_MATRICES 0


namespace ar {

    InverseProblem_Adjoint_InitialConfiguration::InverseProblem_Adjoint_InitialConfiguration()
        : meshPositionPrior(0)
		, gridPriors({0.01, 0.01, 5})
		, gridHyperEpsilon(0.5)
        , numIterations(10)
    {
    }

    InverseProblemOutput InverseProblem_Adjoint_InitialConfiguration::solveGrid(int deformedTimestep,
        BackgroundWorker* worker, IntermediateResultCallback_t callback)
    {
        CI_LOG_I("Solve for Initial Configuration at timestep " << deformedTimestep);
        // Create simulation
        worker->setStatus("Adjoint - Initial Configuration Grid: create simulation");
        SoftBodyGrid2D simulation;
        simulation.setGridResolution(input->gridResolution_);
        simulation.setSDF(input->gridReferenceSdf_);
        simulation.setExplicitDiffusion(true);
        simulation.setHardDirichletBoundaries(true);
        simulation.setAdvectionMode(SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD);
        simulation.resetBoundaries();
        for (const auto& b : input->gridDirichletBoundaries_)
            simulation.addDirichletBoundary(b.first.first, b.first.second, b.second);
        for (const auto& b : input->gridNeumannBoundaries_)
            simulation.addNeumannBoundary(b.first.first, b.first.second, b.second);
        if (worker->isInterrupted()) return InverseProblemOutput();

        // Set parameters with everything that can't be reconstructed here
        simulation.setGravity(input->settings_.gravity_);
        simulation.setMass(input->settings_.mass_);
        simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
        simulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
        simulation.setRotationCorrection(SoftBodySimulation::RotationCorrection::None);
        if (worker->isInterrupted()) return InverseProblemOutput();

        GridUtils2D::grid_t targetSdf = input->gridResultsSdf_[deformedTimestep];

#if 1
		//Gradient Descent
		//define gradient of the cost function
        real finalCost = 0;
        const auto gradient = 
            [&simulation, this, worker, targetSdf, &finalCost](const VectorX& x)
        {
            real cost;
			grid_t currentSdf = GridUtils2D::delinearize(x, simulation.getGridResolution(), simulation.getGridResolution());
			auto gradientGrid = gradientGrid_AdjointMethod(targetSdf, simulation,
				currentSdf, gridPriors, cost, worker);
			VectorX gradient = GridUtils2D::linearize(gradientGrid);
            finalCost = cost;
            return gradient;
        };
		//Optimize
        VectorX start = GridUtils2D::linearize(targetSdf);
        GradientDescent<VectorX> gd(start, gradient, 1e-10, 0.5);
		gd.setMinStepsize(0.0001);
        gd.setMaxStepsize(0.4);
        int oi;
        for (oi = 0; oi < numIterations; ++oi) {
            worker->setStatus(tfm::format("Adjoint - Initial Configuration Grid: optimization %d/%d", (oi + 1), numIterations));
            if (gd.step()) break;
			CI_LOG_I("Gradient descent, step size is " << gd.getLastStepSize());
            if (worker->isInterrupted()) return InverseProblemOutput();

            InverseProblemOutput output;
            output.initialGridSdf_ = GridUtils2D::delinearize(gd.getCurrentSolution(), input->gridResolution_, input->gridResolution_);
            output.finalCost_ = finalCost;
            callback(output);
        }
        VectorX finalValueV = gd.getCurrentSolution();
#else
        //LBFGS with Hyper-Optimization over the Gravity
        LBFGSpp::LBFGSParam<real> params;
        params.epsilon = 1e-10;
        params.max_iterations = numIterations;
        LBFGSpp::LBFGSSolver<real> lbfgs(params);
        //define gradient
        LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&simulation, this, worker, targetSdf](const VectorX& x, VectorX& gradient) -> real {
            real cost;
            grid_t currentSdf = GridUtils2D::delinearize(x, simulation.getGridResolution(), simulation.getGridResolution());
            auto gradientGrid = gradientGrid_AdjointMethod(targetSdf, simulation,
                currentSdf, gridPriors, cost, worker);
            gradient = GridUtils2D::linearize(gradientGrid);
            return cost;
        });
        LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, callback, this](const VectorX& x, const real& v, int k) -> bool {
            worker->setStatus(tfm::format("Adjoint - Initial Configuration Grid: optimization %d/%d", k, numIterations));

			InverseProblemOutput output;
			output.initialGridSdf_ = GridUtils2D::delinearize(x, input->gridResolution_, input->gridResolution_);
			output.finalCost_ = v;
			callback(output);

            return !worker->isInterrupted();
        });
        //optimize
        Vector2 gravity = input->settings_.gravity_;
        real hyperStepsize = 1;
        real hyperStep = 0;
        real finalCost = 0;
        VectorX finalValueV = GridUtils2D::linearize(targetSdf);
        int totalOI = 0;
        while (true)
        {
            CI_LOG_I("Try optimization with " << (hyperStep + hyperStepsize) * 100 << "% of the final gravity");
            simulation.setGravity((hyperStep + hyperStepsize) * gravity);
            VectorX currentValue = finalValueV;
            int oi = lbfgs.minimize(fun, currentValue, finalCost, lbfgsCallback);
            totalOI += oi;
            if (worker->isInterrupted()) break;
            if (finalCost > gridHyperEpsilon)
            {
                //not converged to global minimum, try smaller step
                hyperStepsize /= 2;
                CI_LOG_I("Not converged (cost " << finalCost << "), decrease hyper-stepsize to " << hyperStepsize);
            }
            else
            {
                //converged, increase gravity
                finalValueV = currentValue;
                hyperStep += hyperStepsize;
                hyperStepsize *= 1.2;
                CI_LOG_I("Converged, take " << hyperStep * 100 << "% of the final gravity as input to the next step and increase the step size to " << hyperStepsize);
                if (hyperStep >= 1 - 1e-5)
                    break; //finished
                if (hyperStep + hyperStepsize > 1)
                    hyperStepsize = 1 - hyperStep;
            }
        }
#endif

        InverseProblemOutput output;
        output.initialGridSdf_ = GridUtils2D::delinearize(finalValueV, input->gridResolution_, input->gridResolution_);
        output.finalCost_ = finalCost;

		CI_LOG_D("Ground truth sdf:\n" << input->gridReferenceSdf_);
		CI_LOG_D("Target sdf:\n" << targetSdf);
		CI_LOG_D("Reconstructed input sdf:\n" << output.initialGridSdf_.value());
		CI_LOG_D("Final cost: " << output.finalCost_);

        return output;
    }

    InverseProblemOutput InverseProblem_Adjoint_InitialConfiguration::solveMesh(int deformedTimestep,
        BackgroundWorker* worker, IntermediateResultCallback_t callback)
    {
        CI_LOG_I("Solve for Initial Configuration at timestep " << deformedTimestep);
        // Create simulation
        worker->setStatus("Adjoint - Initial Configuration Mesh: create simulation");
        SoftBodyMesh2D simulation;
        simulation.setMesh(input->meshReferencePositions_, input->meshReferenceIndices_);
        simulation.resetBoundaries();
        for (const auto& b : input->meshDirichletBoundaries_)
            simulation.addDirichletBoundary(b.first, b.second);
        for (const auto& b : input->meshNeumannBoundaries_)
            simulation.addNeumannBoundary(b.first, b.second);
        simulation.reorderNodes();
        if (worker->isInterrupted()) return InverseProblemOutput();

        // Set parameters with everything that can't be reconstructed here
        simulation.setGravity(input->settings_.gravity_);
        simulation.setMass(input->settings_.mass_);
        simulation.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
		simulation.setMaterialParameters(input->settings_.youngsModulus_, input->settings_.poissonsRatio_);
        simulation.setRotationCorrection(SoftBodySimulation::RotationCorrection::None);
        if (worker->isInterrupted()) return InverseProblemOutput();

        
		SoftBodyMesh2D::Vector2List targetPositions = input->meshReferencePositions_;
		for (size_t i = 0; i < input->meshReferencePositions_.size(); ++i) targetPositions[i] += input->meshResultsDisplacement_[deformedTimestep][i];
#if 0
		//Gradient Descent
        //define gradient of the cost function
		real finalCost = 0;
        const auto gradient = [&simulation, &outputU, &finalCost, this, worker, deformedTimestep](const VectorX& x)
        {
            real cost;
			VectorX gradient = gradientMesh(deformedTimestep, simulation, x, outputU, cost, worker);
			finalCost = cost;
			return gradient;
        };

        //run optimization
		VectorX initialPositions = linearizePositions(input->meshReferencePositions_) + linearizePositions(input->meshResultsDisplacement_[deformedTimestep]);
        GradientDescent<VectorX> gd(initialPositions, gradient, 1e-10, 0.0001);
        int oi;
        for (oi = 0; oi < numIterations; ++oi) {
            worker->setStatus(tfm::format("Adjoint - Initial Configuration Mesh: optimization %d/%d", (oi + 1), numIterations));
            if (gd.step()) break;
            if (worker->isInterrupted()) return InverseProblemOutput();
        }
        VectorX finalValueV = gd.getCurrentSolution();
#else
		//LBFGS with Hyper-Optimization over the Gravity
		LBFGSpp::LBFGSParam<real> params;
		params.epsilon = 1e-10;
		params.max_iterations = numIterations;
		LBFGSpp::LBFGSSolver<real> lbfgs(params);
		//define gradient
		LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&targetPositions, &simulation, this, worker](const VectorX& x, VectorX& gradient) -> real {
			real cost;
			gradient = gradientMesh_AdjointMethod(targetPositions, simulation, x, meshPositionPrior, cost, worker);
			return cost;
		});
		LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, callback, this](const VectorX& x, const VectorX& g, const real& v, int k) -> bool {
			worker->setStatus(tfm::format("Adjoint - Initial Configuration Mesh: optimization %d/%d", k, numIterations));

			InverseProblemOutput output;
			output.initialMeshPositions_ = delinearizePositions(x);
			output.finalCost_ = v;
			callback(output);

			return !worker->isInterrupted();
		});
		//optimize
		Vector2 gravity = input->settings_.gravity_;
		real hyperStepsize = 1;
		real hyperStep = 0;
		real finalCost = 0;
		static const real hyperEpsilon = 1e-4;
        VectorX finalValueV = linearizePositions(input->meshReferencePositions_) + linearizePositions(input->meshResultsDisplacement_[deformedTimestep]);
		int totalOI = 0;
		while (true)
		{
			CI_LOG_I("Try optimization with " << (hyperStep + hyperStepsize) * 100 << "% of the final gravity");
			simulation.setGravity((hyperStep + hyperStepsize) * gravity);
			VectorX currentValue = finalValueV;
			int oi = lbfgs.minimize(fun, currentValue, finalCost, lbfgsCallback);
			totalOI += oi;
			if (worker->isInterrupted()) break;
			if (finalCost > hyperEpsilon)
			{
				//not converged to global minimum, try smaller step
				hyperStepsize /= 2;
				CI_LOG_I("Not converged (cost " << finalCost << "), decrease hyper-stepsize to " << hyperStepsize);
			} else
			{
				//converged, increase gravity
				finalValueV = currentValue;
				hyperStep += hyperStepsize;
				hyperStepsize *= 1.2;
				CI_LOG_I("Converged, take " << hyperStep * 100 << "% of the final gravity as input to the next step and increase the step size to " << hyperStepsize);
				if (hyperStep >= 1 - 1e-5)
					break; //finished
				if (hyperStep + hyperStepsize > 1)
					hyperStepsize = 1 - hyperStep;
			}
		}
#endif

		MatrixX positionLog(2, input->meshReferencePositions_.size() * 2);
		positionLog << finalValueV.transpose(), linearizePositions(input->meshReferencePositions_).transpose();
        CI_LOG_I("Optimization done after " << totalOI << " optimization steps.  Final positions | Reference positions:\n" << positionLog);

        InverseProblemOutput output;
        output.initialMeshPositions_ = delinearizePositions(finalValueV);
		output.finalCost_ = finalCost;

		simulation.setGravity(gravity);

        return output;
    }

    VectorX InverseProblem_Adjoint_InitialConfiguration::gradientMesh_AdjointMethod(
		const SoftBodyMesh2D::Vector2List& targetPositions, SoftBodyMesh2D& simulation,
        const VectorX& currentInitialPositions, real positionPrior, real& outputCost, BackgroundWorker* worker)
    {
        real poissonsRatio = simulation.getPoissonsRatio();
        real youngsModulus = simulation.getYoungsModulus();
		Vector2 gravity = simulation.getGravity();

        int n = static_cast<int>(currentInitialPositions.size() / 2);
		assert(n == simulation.getReferencePositions().size());
		for (int i = 0; i < n; ++i) {
			simulation.getReferencePositions()[i] = currentInitialPositions.segment<2>(2 * i);
		}

        //get winding order
        triangle_t referenceTriangle = {
			targetPositions[simulation.getTriangles()[0].x()],
			targetPositions[simulation.getTriangles()[0].y()],
			targetPositions[simulation.getTriangles()[0].z()]
        };
        int windingOrder = meshTriangleWindingOrder(referenceTriangle);

        //FORWARD

        MatrixX K = MatrixX::Zero(2 * simulation.getNumFreeNodes(), 2 * simulation.getNumFreeNodes());
        VectorX f = VectorX::Zero(2 * simulation.getNumFreeNodes());

        //assemble force vector f and stiffness matrix K
        simulation.assembleForceVector(f);
        if (worker->isInterrupted()) return VectorX::Zero(2*n);
        real materialMu, materialLambda;
        SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
        Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);
        simulation.assembleStiffnessMatrix(C, K, &f, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
        if (worker->isInterrupted()) return VectorX::Zero(2 * n);

        //solve for the current displacement
		assert(K.isApprox(K.transpose())); //Test if K is indeed symmetric
        PartialPivLU<MatrixX> Klu = K.partialPivLu();
        VectorX u = Klu.solve(f);

        // BACKWARD
		int p = simulation.getNumFreeNodes();

        //compute derivate of the cost function with respect to the output u (displacement)
        VectorX costDu(2 * p);
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
			if (simulation.getNodeToFreeMap()[i] < 0) continue;
			costDu.segment<2>(2 * simulation.getNodeToFreeMap()[i])
				= currentInitialPositions.segment<2>(2 * i) + u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
				- targetPositions[i];
        }
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(costDu, tfm::format("AdjointYoung-GradientMesh-costDu-k%f.csv", youngsModulus));
#endif

        //solve for the dual
        //In theory, I have to use K', but because I know that K is symmetric,
        //I can use K directly and not the transpose.
        VectorX lambda = Klu.solve(costDu); //rhs is called 'g'

		//compute yF, the dual of gX
		VectorX yF = VectorX::Zero(2 * p);
		for (int e = 0; e < simulation.getNumElements(); ++e)
		{
			const Vector3i& tri = simulation.getTriangles()[e];
			Vector6 ue = Vector6::Zero();
			for (int i2 = 0; i2 < 3; ++i2)
				if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
					ue.segment<2>(2 * i2) = u.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]);
			for (int i = 0; i < 3; ++i)
			{
				if (simulation.getNodeStates()[tri[i]] == SoftBodyMesh2D::DIRICHLET) continue;
				int col = simulation.getNodeToFreeMap()[tri[i]];
				for (int coord = 0; coord < 2; ++coord)
				{
					//current parameter: vertex 'col' with coordinate 'coord'
					//F(:,j) = -dK/dp*u + df/dp
					int j = 2 * col + coord;
					triangle_t triangle{
						currentInitialPositions.segment<2>(2 * tri[0]),
						currentInitialPositions.segment<2>(2 * tri[1]),
						currentInitialPositions.segment<2>(2 * tri[2])
					};
					Matrix6 KeDpos = meshStiffnessMatrixDerivative(triangle, 2 * i + coord, C, windingOrder);
					// dK/dp * u
					Vector6 y = KeDpos * ue;
					//F(:,j) -= dK/dp*u per triangle
					//F(:,j) += Df/dp per triangle
					Vector6 feDpos = meshForceDerivative(triangle, 2 * i + coord, gravity, windingOrder);
					// -dE/dp per triangle
					Vector6 Fe = -(y - feDpos);

					for (int i2 = 0; i2 < 3; ++i2)
						if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
							yF[j] += Fe.segment<2>(2 * i2).dot(lambda.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]));
				}
			}
		}

		//assemble gradient
		VectorX priorGradient = VectorX::Zero(2 * p); //TODO
		VectorX r = costDu; // dJ / dp
		VectorX gradient = yF + r + positionPrior * priorGradient;

		//map gradients back to the whole grid
		VectorX fullGradient = VectorX::Zero(2 * simulation.getNumNodes());
		for (int i = 0; i<p; ++i)
		{
			fullGradient.segment<2>(2 * simulation.getFreeToNodeMap()[i]) = gradient.segment<2>(2 * i);
		}

        //evaluate cost
        //cost on position difference
        real costMain = 0;
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0) continue;
            costMain += (currentInitialPositions.segment<2>(2*i) + u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                - targetPositions[i]).squaredNorm() / 2;
        }
        //cost by the prior
        real costPrior = 0;
        for (int tri = 0; tri<simulation.getNumElements(); ++tri)
        {
            triangle_t triangle{
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][0]),
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][1]),
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][2])
            };
            costPrior += windingOrder * meshTriangleArea(triangle) / (
                (triangle[0] - triangle[1]).squaredNorm() + (triangle[0] - triangle[2]).squaredNorm() + (triangle[2] - triangle[1]).squaredNorm()
                );
        }
        //final cost
        outputCost = costMain + positionPrior * costPrior;
        //CI_LOG_I("Optimization: cost " << outputCost << ", gradient " << gradient.transpose());

        return fullGradient;
    }

    VectorX InverseProblem_Adjoint_InitialConfiguration::gradientMesh_Forward(
        const SoftBodyMesh2D::Vector2List & targetPositions, SoftBodyMesh2D & simulation, 
        const VectorX & currentInitialPositions, real positionPrior, real & outputCost, BackgroundWorker * worker)
    {
        real poissonsRatio = simulation.getPoissonsRatio();
        real youngsModulus = simulation.getYoungsModulus();
		Vector2 gravity = simulation.getGravity();

        int n = static_cast<int>(currentInitialPositions.size() / 2);
        assert(n == simulation.getReferencePositions().size());
        for (int i = 0; i < n; ++i) {
            simulation.getReferencePositions()[i] = currentInitialPositions.segment<2>(2 * i);
        }

        //get winding order
        triangle_t referenceTriangle = {
            targetPositions[simulation.getTriangles()[0].x()],
            targetPositions[simulation.getTriangles()[0].y()],
            targetPositions[simulation.getTriangles()[0].z()]
        };
        int windingOrder = meshTriangleWindingOrder(referenceTriangle);

        //FORWARD

        MatrixX K = MatrixX::Zero(2 * simulation.getNumFreeNodes(), 2 * simulation.getNumFreeNodes());
        VectorX f = VectorX::Zero(2 * simulation.getNumFreeNodes());

        //assemble force vector F and stiffness matrix K
        simulation.assembleForceVector(f);
        if (worker->isInterrupted()) return VectorX::Zero(2 * n);
        real materialMu, materialLambda;
        SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
        Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);
        simulation.assembleStiffnessMatrix(C, K, &f, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
        if (worker->isInterrupted()) return VectorX::Zero(2 * n);

        //solve for the current displacement
        PartialPivLU<MatrixX> Klu = K.partialPivLu();
        VectorX u = Klu.solve(f);

        // DERIVATIVES
        int p = simulation.getNumFreeNodes();

        //compute derivate of the cost function with respect to the output u (displacement)
        VectorX costDu(2 * p);
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0) continue;
            costDu.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                = currentInitialPositions.segment<2>(2 * i) + u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                - targetPositions[i];
        }

        MatrixX F = MatrixX::Zero(2 * p, 2 * p);

        //current parameter: vertex 'col' with coordinate 'coord' -> j=2*col+coord
        //F(:,j) = -dK/dp*u + df/dp

        for (int e=0; e<simulation.getNumElements(); ++e)
        {
            const Vector3i& tri = simulation.getTriangles()[e];
            for (int i=0; i<3; ++i)
            {
                if (simulation.getNodeStates()[tri[i]] == SoftBodyMesh2D::DIRICHLET) continue;
                int col = simulation.getNodeToFreeMap()[tri[i]];
                for (int coord = 0; coord < 2; ++coord)
                {
                    //current parameter: vertex 'col' with coordinate 'coord'
                    //F(:,j) = -dK/dp*u + df/dp
                    int j = 2 * col + coord;
                    triangle_t triangle{
                        currentInitialPositions.segment<2>(2 * tri[0]),
                        currentInitialPositions.segment<2>(2 * tri[1]),
                        currentInitialPositions.segment<2>(2 * tri[2])
                    };
                    Matrix6 KeDpos = meshStiffnessMatrixDerivative(triangle, 2 * i + coord, C, windingOrder);
                    Vector6 ue = Vector6::Zero();
                    for (int i2 = 0; i2 < 3; ++i2)
                        if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
                            ue.segment<2>(2 * i2) = u.segment<2>(2 * simulation.getNodeToFreeMap()[tri[i2]]);
                    Vector6 y = KeDpos * ue;
                    //F(:,j) -= dK/dp*u per triangle
                    for (int i2 = 0; i2 < 3; ++i2)
                        if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
                            F.block<2, 1>(2 * simulation.getNodeToFreeMap()[tri[i2]], j) -= y.segment<2>(2 * i2);
					//F(:,j) += Df/dp per triangle
					Vector6 FeDpos = meshForceDerivative(triangle, 2 * i + coord, gravity, windingOrder);
					for (int i2 = 0; i2 < 3; ++i2)
						if (simulation.getNodeStates()[tri[i2]] != SoftBodyMesh2D::DIRICHLET)
							F.block<2, 1>(2 * simulation.getNodeToFreeMap()[tri[i2]], j) += FeDpos.segment<2>(2 * i2);
                }
            }
        }

        //assemble final gradient
        VectorX priorGradient = VectorX::Zero(2 * p); //TODO
        MatrixX X = Klu.solve(F);
        VectorX g = costDu; // dJ / du
        VectorX gX = g.transpose() * X;
		VectorX r = costDu; // dJ / dp
        VectorX gradient = gX + r + positionPrior * priorGradient;

        //map gradients back to the whole grid
        VectorX fullGradient = VectorX::Zero(2 * simulation.getNumNodes());
        for (int i=0; i<p; ++i)
        {
            fullGradient.segment<2>(2 * simulation.getFreeToNodeMap()[i]) = gradient.segment<2>(2 * i);
        }

        //evaluate cost
        //cost on position difference
        real costMain = 0;
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0) continue;
            costMain += (currentInitialPositions.segment<2>(2 * i) + u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                - targetPositions[i]).squaredNorm() / 2;
        }
        //cost by the prior
        real costPrior = 0;
        for (int tri = 0; tri<simulation.getNumElements(); ++tri)
        {
            triangle_t triangle{
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][0]),
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][1]),
                currentInitialPositions.segment<2>(2 * simulation.getTriangles()[tri][2])
            };
            costPrior += windingOrder * meshTriangleArea(triangle) / (
                (triangle[0] - triangle[1]).squaredNorm() + (triangle[0] - triangle[2]).squaredNorm() + (triangle[2] - triangle[1]).squaredNorm()
                );
        }
        //final cost
        outputCost = costMain + positionPrior * costPrior;
        //CI_LOG_I("Optimization: cost " << outputCost << ", gradient " << gradient.transpose());

        return fullGradient;
    }

    VectorX InverseProblem_Adjoint_InitialConfiguration::gradientMesh_Numerical(
        const SoftBodyMesh2D::Vector2List & targetPositions, SoftBodyMesh2D & simulation, 
        const VectorX & currentInitialPositions, real positionPrior, real & outputCost, BackgroundWorker * worker)
    {
        real poissonsRatio = simulation.getPoissonsRatio();
        real youngsModulus = simulation.getYoungsModulus();
        int n = simulation.getNumNodes();
        for (int i = 0; i < n; ++i) {
            simulation.getReferencePositions()[i] = currentInitialPositions.segment<2>(2 * i);
        }

        int p = simulation.getNumFreeNodes();
        VectorX gradient = VectorX::Zero(2 * n);

        // Reference forward problem
        real materialMu, materialLambda;
        SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
        Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);

        MatrixX K = MatrixX::Zero(2 * p, 2 * p);
        VectorX F = VectorX::Zero(2 * p);
        simulation.assembleForceVector(F);
        simulation.assembleStiffnessMatrix(C, K, &F, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
        VectorX u = K.partialPivLu().solve(F);
        if (worker->isInterrupted()) return gradient;

        real costReference = 0;
        for (int i=0; i<p; ++i)
        {
            costReference += (currentInitialPositions.segment<2>(2 * simulation.getFreeToNodeMap()[i])
                + u.segment<2>(2 * i)
                - targetPositions[simulation.getFreeToNodeMap()[i]])
                .squaredNorm() / 2;
        }

        //numerical differentiation
        real epsilon = 1e-7;
        for (int j=0; j<p; ++j)
        {
            for (int c=0; c<2; ++c)
            {
                VectorX positions = currentInitialPositions;
                positions[2 * simulation.getFreeToNodeMap()[j] + c] += epsilon;
                for (int i = 0; i < n; ++i) {
                    simulation.getReferencePositions()[i] = positions.segment<2>(2 * i);
                }
                K.setZero();
                F.setZero();
                simulation.assembleForceVector(F);
                simulation.assembleStiffnessMatrix(C, K, &F, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
                VectorX u = K.partialPivLu().solve(F);
				if (worker->isInterrupted()) return gradient;

                real cost = 0;
                for (int i = 0; i<p; ++i)
                {
                    cost += (positions.segment<2>(2 * simulation.getFreeToNodeMap()[i])
                        + u.segment<2>(2 * i)
                        - targetPositions[simulation.getFreeToNodeMap()[i]])
                        .squaredNorm() / 2;
                }
                gradient[2 * simulation.getFreeToNodeMap()[j] + c] = (cost - costReference) / epsilon;
            }
        }

        outputCost = costReference;
        return gradient;
    }

	InverseProblem_Adjoint_InitialConfiguration::grid_t InverseProblem_Adjoint_InitialConfiguration::gradientGrid_AdjointMethod(
		const grid_t & targetSdf, SoftBodyGrid2D & simulation,
		const grid_t & initialSdf, const GridPriors& priors,
		real & outputCost, BackgroundWorker * worker, bool onlyCost)
	{
		//only these modes are supported
		assert(simulation.isExplicitDiffusion());
		assert(simulation.isHardDirichletBoundaries());
		assert(simulation.getRotationCorrection() == SoftBodySimulation::RotationCorrection::None);
        assert(simulation.getAdvectionMode() == SoftBodyGrid2D::AdvectionMode::DIRECT_FORWARD);

		//get settings from the simulation
		real poissonsRatio = simulation.getPoissonsRatio();
		real youngsModulus = simulation.getYoungsModulus();
		Vector2 gravity = simulation.getGravity();
		int resolution = simulation.getGridResolution();
		Vector2 size = simulation.getCellSize();
		real h = size.x();

		// allocate output
		outputCost = 0;
		grid_t outputGradient = grid_t::Zero(resolution, resolution);

		//FORWARD

		//set initial / current SDF as the reference SDF in the simulation
		//and the degrees of freedom are computed as well
		simulation.setSDF(initialSdf);
		//collect degrees of freedom in the stifness solve
		const Eigen::MatrixXi& posToIndex = simulation.getPosToIndex();
		const SoftBodyGrid2D::indexToPos_t& indexToPos = simulation.getIndexToPos();
		const int dof = simulation.getDoF();

		MatrixX K = MatrixX::Zero(dof * 2, dof * 2);
		VectorX f = VectorX::Zero(dof * 2);
		const VectorX prevU = VectorX::Zero(dof * 2); //previous displacements for rotation correction. Not needed here

		//assemble force vector f and stiffness matrix K
		VectorX collisionForces = VectorX::Zero(2 * dof);
		simulation.assembleForceVector(f, posToIndex, collisionForces);
		if (worker->isInterrupted()) return outputGradient;
		real materialMu, materialLambda;
		SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
		Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);
		simulation.assembleStiffnessMatrix(C, materialMu, materialLambda, K, &f, posToIndex, prevU);
		if (worker->isInterrupted()) return outputGradient;

		//solve for the current displacement
		PartialPivLU<MatrixX> Klu = K.partialPivLu();
		VectorX u = Klu.solve(f);
		if (worker->isInterrupted()) return outputGradient;

		//map back to a grid
		grid_t uGridX = grid_t::Zero(resolution, resolution);
		grid_t uGridY = grid_t::Zero(resolution, resolution);
		for (int i = 0; i < dof; ++i) {
			Vector2i p = indexToPos.at(i);
			uGridX(p.x(), p.y()) = u[2 * i];
			uGridY(p.x(), p.y()) = u[2 * i + 1];
		}
		if (worker->isInterrupted()) return outputGradient;

		//perform diffusion step
		GridUtils2D::bgrid_t validCells = simulation.computeValidCells(uGridX, uGridY, posToIndex);
		uGridX = GridUtils2D::fillGridDiffusion(uGridX, validCells);
		uGridY = GridUtils2D::fillGridDiffusion(uGridY, validCells);
		if (worker->isInterrupted()) return outputGradient;

		//advect levelset
		grid_t advectionWeights;
		grid_t forwardSdf = GridUtils2D::advectGridDirectForward(initialSdf, uGridX, uGridY, -1 / h, &advectionWeights);
		if (worker->isInterrupted()) return outputGradient;

		//reconstruct SDF
        GridUtils2D::RecoverSDFSussmannAdjointStorage recoveryStorage;
        grid_t forwardSdf2 = simulation.getSdfRecoveryIterations() > 0
            ? GridUtils2D::recoverSDFSussmann(forwardSdf, real(0.01), simulation.getSdfRecoveryIterations(), &recoveryStorage)
			: forwardSdf;

		// COST
		grid_t costWeighting = advectionWeights.min(1);
		outputCost += gridCostSimilarity(forwardSdf, targetSdf, &costWeighting);
		// prior on the SDF
		real costPriorSdf = 0;
		grid_t priorSdfGrad = GridUtils2D::recoverSDFSussmannGradient(initialSdf, initialSdf, GridUtils2D::bgrid_t::Constant(resolution, resolution, true), costPriorSdf, priors.sdfEpsilon);
		outputCost += priors.sdfPrior * costPriorSdf;

        if (onlyCost)
            return outputGradient;

		// BACKWARD

		//The derivative of the cost with respect to the forward (output) SDF
		grid_t costDforwardSdf1 = gridCostSimilarityDerivative(forwardSdf2, targetSdf, &costWeighting);

		// propagate back, get the adjoint solutions

        // Adjoint: recover SDF
        grid_t adjOutputSdf = grid_t::Zero(resolution, resolution);
        GridUtils2D::recoverSDFSussmannAdjoint(forwardSdf, forwardSdf2, costDforwardSdf1, adjOutputSdf, recoveryStorage);

		//Adjoint: advect levelset
		grid_t adjInputSdf = grid_t::Zero(resolution, resolution);
		grid_t adjUGridX = grid_t::Zero(resolution, resolution);
		grid_t adjUGridY = grid_t::Zero(resolution, resolution);
		GridUtils2D::advectGridDirectForwardAdjoint(
			adjInputSdf, adjUGridX, adjUGridY, adjOutputSdf,
			uGridX, uGridY, initialSdf, forwardSdf, -1 / h, advectionWeights);
		if (worker->isInterrupted()) return outputGradient;

		//Adjoint: perform diffusion step
		GridUtils2D::fillGridDiffusionAdjoint(adjUGridX, adjUGridX, uGridX, validCells); //note that the adjoint is both input and output
		GridUtils2D::fillGridDiffusionAdjoint(adjUGridY, adjUGridY, uGridY, validCells); //it is modified in-place
		if (worker->isInterrupted()) return outputGradient;

		//Adjoint: map back to a grid
		VectorX adjU(2 * dof);
		for (int i = 0; i < dof; ++i) {
			Vector2i p = indexToPos.at(i);
			adjU[2 * i] = adjUGridX(p.x(), p.y());
			adjU[2 * i + 1] = adjUGridY(p.x(), p.y());
		}
		if (worker->isInterrupted()) return outputGradient;

		//Adjoint: solve for the current displacement
		//since K is symmetric, I can use K directly instead of K' as normally required by the Adjoint Method.
		VectorX lambdaU = Klu.solve(adjU);
		if (worker->isInterrupted()) return outputGradient;

		// DERIVATIVES OF THE UNKNOWNS
		// Matrix-Free assembly of y'F, y = (lambdaU, adjInputSdf)

		// advection / currentSdf
		grid_t gradientAdvection 
    		= GridUtils2D::advectGridDirectForwardAdjOpMult(adjInputSdf, uGridX, uGridY, -1 / h, advectionWeights);

		// elasticity / lambdaU -> compute element-wise
        grid_t gradientElasticity =
            gridElasticityAdjOpMult(lambdaU, u, initialSdf,
                gravity, C, simulation, posToIndex);

		// ASSEMBLE FINAL GRADIENT
		grid_t gradientPriors = grid_t::Zero(resolution, resolution); //(r in the notes)
		gradientPriors += priorSdfGrad.unaryExpr([&priors](real v)
		{
			return priors.sdfPrior * std::clamp(v, -priors.sdfMaxDistance, priors.sdfMaxDistance);
		});
		outputGradient = gradientAdvection + gradientElasticity + gradientPriors;

        CI_LOG_I("Optimize: cost " << outputCost);

		return outputGradient;
	}

	void InverseProblem_Adjoint_InitialConfiguration::setupParams(cinder::params::InterfaceGlRef params,
		const std::string& group)
	{
		params->addParam("InverseProblem_Adjoint_InitialConfiguration_PositionPrior", &meshPositionPrior)
			.group(group).label("Mesh Prior - Initial Positions").min(0).step(0.01);
        params->addParam("InverseProblem_Adjoint_InitialConfiguration_SDFPrior1", &gridPriors.sdfPrior)
            .group(group).label("Grid Prior - Weight on SDF").min(0).step(0.01);
		params->addParam("InverseProblem_Adjoint_InitialConfiguration_SDFPrior2", &gridPriors.sdfEpsilon)
			.group(group).label("Grid Prior - SDF Epsilon").min(0).step(0.01);
		params->addParam("InverseProblem_Adjoint_InitialConfiguration_SDFPrior3", &gridPriors.sdfMaxDistance)
			.group(group).label("Grid Prior - SDF Max distance").min(0).step(0.1);
		params->addParam("InverseProblem_Adjoint_InitialConfiguration_GridHyperEpsilon", &gridHyperEpsilon)
			.group(group).label("Grid Hyper Epsilon").min(0).step(0.01);
		params->addParam("InverseProblem_Adjoint_InitialConfiguration_NumIterations", &numIterations)
			.group(group).label("Iterations").min(1);
	}

	void InverseProblem_Adjoint_InitialConfiguration::setParamsVisibility(cinder::params::InterfaceGlRef params,
		bool visible) const
	{
		std::string option = visible ? "visible=true" : "visible=false";
		params->setOptions("InverseProblem_Adjoint_InitialConfiguration_PositionPrior", option);
        params->setOptions("InverseProblem_Adjoint_InitialConfiguration_SDFPrior1", option);
		params->setOptions("InverseProblem_Adjoint_InitialConfiguration_SDFPrior2", option);
		params->setOptions("InverseProblem_Adjoint_InitialConfiguration_SDFPrior3", option);
		params->setOptions("InverseProblem_Adjoint_InitialConfiguration_GridHyperEpsilon", option);
		params->setOptions("InverseProblem_Adjoint_InitialConfiguration_NumIterations", option);
	}

	VectorX InverseProblem_Adjoint_InitialConfiguration::linearizePositions(const SoftBodyMesh2D::Vector2List & positions)
    {
        size_t n = positions.size();
        VectorX x(n*2);
        for (size_t i=0; i<n; ++i)
        {
            x.segment<2>(2 * i) = positions[i];
        }
        return x;
    }

    SoftBodyMesh2D::Vector2List InverseProblem_Adjoint_InitialConfiguration::delinearizePositions(const VectorX & linearizedPositions)
    {
        size_t n = linearizedPositions.size() / 2;
        SoftBodyMesh2D::Vector2List l(n);
        for (size_t i=0; i<n; ++i)
        {
            l[i] = linearizedPositions.segment<2>(2 * i);
        }
        return l;
    }

    real InverseProblem_Adjoint_InitialConfiguration::meshTriangleArea(const triangle_t & triangle)
    {
        return ((triangle[1].x() - triangle[0].x())*(triangle[2].y() - triangle[0].y()) - (triangle[2].x() - triangle[0].x())*(triangle[1].y() - triangle[0].y())) / 2;
    }

	int InverseProblem_Adjoint_InitialConfiguration::meshTriangleWindingOrder(const triangle_t & triangle)
	{
		return meshTriangleArea(triangle) > 0 ? 1 : -1;
	}

    std::array<real, 6> InverseProblem_Adjoint_InitialConfiguration::meshTriangleAreaDerivative(const triangle_t & triangle)
    {
        return std::array<real, 6>({
            (triangle[1].y() - triangle[2].y()) / 2,
            (triangle[2].x() - triangle[1].x()) / 2,
            (triangle[2].y() - triangle[0].y()) / 2,
            (triangle[0].x() - triangle[2].x()) / 2,
            (triangle[0].y() - triangle[1].y()) / 2,
            (triangle[1].x() - triangle[0].x()) / 2
        });
    }

    Matrix36 InverseProblem_Adjoint_InitialConfiguration::meshDerivativeMatrix(const triangle_t & triangle)
    {
        Matrix36 Be;
        Be << (triangle[1].y() - triangle[2].y()), 0, (triangle[2].y() - triangle[0].y()), 0, (triangle[0].y() - triangle[1].y()), 0,
            0, (triangle[2].x() - triangle[1].x()), 0, (triangle[0].x() - triangle[2].x()), 0, (triangle[1].x() - triangle[0].x()),
            (triangle[2].x() - triangle[1].x()), (triangle[1].y() - triangle[2].y()), (triangle[0].x() - triangle[2].x()), (triangle[2].y() - triangle[0].y()), (triangle[1].x() - triangle[0].x()), (triangle[0].y() - triangle[1].y());
        Be *= (1 / meshTriangleArea(triangle));
        return Be;
    }

    Matrix36 InverseProblem_Adjoint_InitialConfiguration::meshDerivativeMatrixDerivative(const triangle_t & triangle, int i)
    {
        static const std::array<Matrix36, 6> Bprime = {
            (Matrix36() << 0,0,0,0,0,0, 0,0,0,1,0,-1, 0,0,1,0,-1,0).finished(), //d x1
            (Matrix36() << 0,0,-1,0,1,0, 0,0,0,0,0,0, 0,0,0,-1,0,1).finished(), //d y1
            (Matrix36() << 0,0,0,0,0,0, 0,-1,0,0,0,1, -1,0,0,0,1,0).finished(), //d x2
            (Matrix36() << 1,0,0,0,-1,0, 0,0,0,0,0,0, 0,1,0,0,0,-1).finished(), //d y2
            (Matrix36() << 0,0,0,0,0,0, 0,1,0,-1,0,0, 1,0,-1,0,0,0).finished(), //d x3
            (Matrix36() << -1,0,1,0,0,0, 0,0,0,0,0,0, 0,-1,0,1,0,0).finished()  //d y3
        };
        //return (1 / meshTriangleArea(triangle)) * (Bprime[i] - meshDerivativeMatrix(triangle) * meshTriangleAreaDerivative(triangle)[i]);
        real area = meshTriangleArea(triangle);
        return (Bprime[i] / area - meshDerivativeMatrix(triangle)*meshTriangleAreaDerivative(triangle)[i] / area);
    }

    Matrix6 InverseProblem_Adjoint_InitialConfiguration::meshStiffnessMatrix(const triangle_t & triangle, const Matrix3 & C)
    {
        Matrix36 Be = meshDerivativeMatrix(triangle);
        return abs(meshTriangleArea(triangle)) * Be.transpose() * C * Be;
    }

    Matrix6 InverseProblem_Adjoint_InitialConfiguration::meshStiffnessMatrixDerivative(const triangle_t & triangle, int i, const Matrix3& C, int windingOrder)
    {
        Matrix36 Be = meshDerivativeMatrix(triangle);
        Matrix36 BePrime = meshDerivativeMatrixDerivative(triangle, i);
        return windingOrder * (meshTriangleAreaDerivative(triangle)[i] * (Be.transpose() * C * Be)
             + meshTriangleArea(triangle) * (BePrime.transpose() * C * Be + Be.transpose() * C * BePrime));
    }

	Vector6 InverseProblem_Adjoint_InitialConfiguration::meshForce(const triangle_t & triangle, const Vector2 & gravity)
	{
		return abs(meshTriangleArea(triangle)) * (Vector6() << gravity, gravity, gravity).finished();
	}

	Vector6 InverseProblem_Adjoint_InitialConfiguration::meshForceDerivative(const triangle_t & triangle, int i, const Vector2 & gravity, int windingOrder)
	{
		return meshTriangleAreaDerivative(triangle)[i] * windingOrder * (Vector6() << gravity, gravity, gravity).finished();
	}

	real InverseProblem_Adjoint_InitialConfiguration::gridCostSimilarity(
		const grid_t & currentPhi, const grid_t & targetPhi, 
		const grid_t * weighting)
	{
		if (weighting)
			return ((*weighting) * (currentPhi.unaryExpr(&sdfTransformFun) - targetPhi.unaryExpr(&sdfTransformFun))).matrix().squaredNorm() / 2;
		else
			return (currentPhi.unaryExpr(&sdfTransformFun) - targetPhi.unaryExpr(&sdfTransformFun)).matrix().squaredNorm() / 2;
	}

	InverseProblem_Adjoint_InitialConfiguration::grid_t InverseProblem_Adjoint_InitialConfiguration::gridCostSimilarityDerivative(
		const grid_t& currentPhi, const grid_t& targetPhi,
		const grid_t* weighting)
	{
		if (weighting)
			return currentPhi.unaryExpr(&sdfTransformFunDerivative) * (*weighting) * (*weighting) * (currentPhi.unaryExpr(&sdfTransformFun) - targetPhi.unaryExpr(&sdfTransformFun));
		else
			return currentPhi.unaryExpr(&sdfTransformFunDerivative) * (currentPhi.unaryExpr(&sdfTransformFun) - targetPhi.unaryExpr(&sdfTransformFun));
	}

	std::array<Vector8, 4> InverseProblem_Adjoint_InitialConfiguration::gridGravityForceDerivative(
		const std::array<real, 4>& sdfs, const Vector2& gravity, const SoftBodyGrid2D& simulation)
	{
		Vector8 bodyForces;
		bodyForces << gravity, gravity, gravity, gravity;
		std::array<real, 4> derivatives = Integration2D<real>::integrateQuadDsdf<real>(simulation.getCellSize(), sdfs, { 1,1,1,1 });
		return {
			bodyForces * derivatives[0],
			bodyForces * derivatives[1],
			bodyForces * derivatives[2],
			bodyForces * derivatives[3]
		};
	}

	Vector8 InverseProblem_Adjoint_InitialConfiguration::gridGravityForce(const std::array<real, 4>& sdfs,
		const Vector2& gravity, const SoftBodyGrid2D& simulation)
	{
		Vector8 bodyForces;
		bodyForces << gravity, gravity, gravity, gravity;
		return bodyForces * Integration2D<real>::integrateQuad<real>(simulation.getCellSize(), sdfs, { 1,1,1,1 });
	}

	std::array<Matrix8, 4> InverseProblem_Adjoint_InitialConfiguration::gridStiffnessMatrixDerivative(
		const std::array<real, 4>& sdfs, const Matrix3& C, const SoftBodyGrid2D& simulation)
	{
		array<Matrix8, 4> Kex = {
			simulation.getB1().transpose() * C * simulation.getB1(),
			simulation.getB2().transpose() * C * simulation.getB2(),
			simulation.getB3().transpose() * C * simulation.getB3(),
			simulation.getB4().transpose() * C * simulation.getB4(),
		};
		return Integration2D<real>::integrateQuadDsdf(simulation.getCellSize(), sdfs, Kex);
	}

	Matrix8 InverseProblem_Adjoint_InitialConfiguration::gridStiffnessMatrix(const std::array<real, 4>& sdfs,
		const Matrix3& C, const SoftBodyGrid2D& simulation)
	{
		array<Matrix8, 4> Kex = {
			simulation.getB1().transpose() * C * simulation.getB1(),
			simulation.getB2().transpose() * C * simulation.getB2(),
			simulation.getB3().transpose() * C * simulation.getB3(),
			simulation.getB4().transpose() * C * simulation.getB4(),
		};
		return Integration2D<real>::integrateQuad(simulation.getCellSize(), sdfs, Kex);
	}

	InverseProblem_Adjoint_InitialConfiguration::grid_t InverseProblem_Adjoint_InitialConfiguration::gridElasticityAdjOpMult(
        const VectorX & lambdaU, const VectorX& u, const grid_t & sdf,
        const Vector2 & gravity, const Matrix3 & C, const SoftBodyGrid2D & simulation,
        const Eigen::MatrixXi & posToIndex)
    {
		////Test: not matrix-free
		//int dof = lambdaU.size() / 2;
		//MatrixX F = MatrixX::Zero(2 * dof, dof);

        int resolution = simulation.getGridResolution();
        grid_t gradientElasticity = grid_t::Zero(resolution, resolution);
        for (int xy = 0; xy < (resolution - 1)*(resolution - 1); ++xy) {
            int x = xy / (resolution - 1);
            int y = xy % (resolution - 1);

            array<real, 4> sdfs = { sdf(x, y), sdf(x + 1,y), sdf(x, y + 1), sdf(x + 1, y + 1) };
            if (utils::outside(sdfs[0]) && utils::outside(sdfs[1]) && utils::outside(sdfs[2]) && utils::outside(sdfs[3])) continue;

            std::array<Vector8, 4> FeDphi = gridGravityForceDerivative(sdfs, gravity, simulation);
            std::array<Matrix8, 4> KeDphi = gridStiffnessMatrixDerivative(sdfs, C, simulation);

            int mapping[4] = {
                posToIndex(x, y),
                posToIndex(x + 1, y),
                posToIndex(x, y + 1),
                posToIndex(x + 1, y + 1)
            };
            bool dirichlet[4] = {
                simulation.getGridDirichlet()(x, y),
                simulation.getGridDirichlet()(x + 1, y),
                simulation.getGridDirichlet()(x, y + 1),
                simulation.getGridDirichlet()(x + 1, y + 1)
            };
            Vector8 ue;
            Vector8 lambdaUE;
            for (int i = 0; i < 4; ++i) {
                ue.segment<2>(2 * i) = u.segment<2>(2 * mapping[i]);
                lambdaUE.segment<2>(2 * i) = lambdaU.segment<2>(2 * mapping[i]);
            }
            for (int i = 0; i < 4; ++i) {
                if (dirichlet[i]) continue;
                Vector8 EeDphi = -(KeDphi[i] * ue - FeDphi[i]);
                gradientElasticity(x + (i % 2), y + (i / 2)) += lambdaUE.dot(EeDphi);
				//for (int j=0; j<4; ++j)
				//{
				//	F.block<2, 1>(2 * mapping[j], mapping[i]) += EeDphi.segment<2>(2 * j);
				//}
            }
        }

		//VectorX result = lambdaU.transpose() * F;
		//for (int x=0; x<resolution; ++x) for (int y=0; y<resolution; ++y)
		//{
		//	int i = posToIndex(x, y);
		//	if (i >= 0)
		//		gradientElasticity(x, y) = result[i];
		//}

        return gradientElasticity;
    }
}
