#include "InverseProblem_Adjoint_YoungsModulus.h"

#include <Eigen/Dense>
#include <tinyformat.h>
#include <cinder/Log.h>
#include <numeric>
#include <LBFGS.h>
#include <fstream>

#include "GradientDescent.h"
#include "AdjointUtilities.h"

using namespace std;
using namespace Eigen;

#define DEBUG_SAVE_MATRICES 0


namespace ar {

    InverseProblem_Adjoint_YoungsModulus::InverseProblem_Adjoint_YoungsModulus()
        : initialYoungsModulus(50)
        , youngsModulusPrior(0.1)
        , numIterations(10)
    {
    }

    InverseProblemOutput InverseProblem_Adjoint_YoungsModulus::solveGrid(int deformedTimestep,
        BackgroundWorker* worker, IntermediateResultCallback_t callback)
    {
        CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
        // Create simulation
        worker->setStatus("Adjoint - Young's Modulus Grid: create simulation");
        SoftBodyGrid2D simulation;
        simulation.setGridResolution(input->gridResolution_);
        simulation.setSDF(input->gridReferenceSdf_);
        simulation.setExplicitDiffusion(true);
        simulation.setHardDirichletBoundaries(false);
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
        simulation.setRotationCorrection(SoftBodySimulation::RotationCorrection::None);
        if (worker->isInterrupted()) return InverseProblemOutput();

        //cost function:
        //F(youngsModulus) = 0.5*|| disp(youngsModulus) - dispRef ||^2 + priorWeight / youngsModulus

        GridUtils2D::grid_t outputSdf(input->gridResolution_, input->gridResolution_);

		//TODO: once the gradient is fixed, switch to LBFGS
#if 1
		//Gradient Descent
		//define gradient of the cost function
		real finalCost = 0;
        const auto gradient = [&simulation, &outputSdf, &finalCost, this, worker, deformedTimestep](const Vector1& youngsModulusV)
        {
            real youngsModulus = youngsModulusV.x();
            real cost;
            real gradient = gradientGrid(deformedTimestep, simulation, youngsModulus, outputSdf, cost, worker);
			finalCost = cost;
            return Vector1(gradient);
        };
		//Optimize
        GradientDescent<Vector1> gd(Vector1(initialYoungsModulus), gradient, 1e-10, 10);
		gd.setMinStepsize(0.0001);
		gd.setMaxStepsize(0.5);
        int oi;
        for (oi = 0; oi < numIterations; ++oi) {
            worker->setStatus(tfm::format("Adjoint - Young's Modulus Grid: optimization %d/%d", (oi + 1), numIterations));
            if (gd.step()) break;

            {
				InverseProblemOutput output;
				output.youngsModulus_ = gd.getCurrentSolution()[0];
				output.resultGridSdf_ = outputSdf;
				output.finalCost_ = finalCost;
				callback(output);
            }

            if (worker->isInterrupted()) return InverseProblemOutput();
        }
        real finalValue = gd.getCurrentSolution()[0];
#else
		//LBFGS
		LBFGSpp::LBFGSParam<real> params;
		params.epsilon = 1e-10;
		params.max_iterations = numIterations;
		LBFGSpp::LBFGSSolver<real> lbfgs(params);
		//define gradient
		LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&simulation, &outputSdf, this, worker, deformedTimestep](const VectorX& x, VectorX& gradient) -> real {
			real youngsModulus = x[0];
			real cost;
			real grad = gradientGrid(deformedTimestep, simulation, youngsModulus, outputSdf, cost, worker);
			gradient[0] = grad;
			return cost;
		});
		LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, callback, &outputSdf, this](const VectorX& x, const real& v, int k) -> bool {
			worker->setStatus(tfm::format("Adjoint - Young's Modulus: optimization %d/%d", k, numIterations));

			InverseProblemOutput output;
			output.youngsModulus_ = x[0];
			output.resultGridSdf_ = outputSdf;
			output.finalCost_ = v;
			callback(output);

			return !worker->isInterrupted();
		});
		//optimize
		real finalCost = 0;
		VectorX finalValueV = VectorX::Constant(1, initialYoungsModulus);
		int oi = lbfgs.minimize(fun, finalValueV, finalCost, lbfgsCallback);
		if (worker->isInterrupted()) return InverseProblemOutput();
		real finalValue = finalValueV[0];
#endif
        CI_LOG_I("Optimization: final Youngs' Modulus is " << finalValue << " after " << oi << " optimization steps, reference value is " << input->settings_.youngsModulus_);

        InverseProblemOutput output;
        output.youngsModulus_ = finalValue;
        output.resultGridSdf_ = outputSdf;
		output.finalCost_ = finalCost;
        return output;
    }

    InverseProblemOutput InverseProblem_Adjoint_YoungsModulus::solveMesh(int deformedTimestep,
        BackgroundWorker* worker, IntermediateResultCallback_t callback)
    {
        CI_LOG_I("Solve for Young's Modulus at timestep " << deformedTimestep);
        // Create simulation
        worker->setStatus("Adjoint - Young's Modulus: create simulation");
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
        simulation.setRotationCorrection(SoftBodySimulation::RotationCorrection::None);
        if (worker->isInterrupted()) return InverseProblemOutput();

        //cost function:
        //F(youngsModulus) = 0.5*|| disp(youngsModulus) - dispRef ||^2 + priorWeight / youngsModulus

		VectorX outputU;
#if 0
		//Gradient Descent
        //define gradient of the cost function
        const auto gradient = [&simulation, &outputU, this, worker, deformedTimestep](const Vector1& youngsModulusV)
        {
            real youngsModulus = youngsModulusV.x();
            real cost;
            real gradient = gradientMesh(deformedTimestep, simulation, youngsModulus, outputU, cost, worker);
            return Vector1(gradient);
        };

        //run optimization
        GradientDescent<Vector1> gd(Vector1(initialYoungsModulus), gradient, 1e-10, 10);
        int oi;
        for (oi = 0; oi < numIterations; ++oi) {
            worker->setStatus(tfm::format("Adjoint - Young's Modulus: optimization %d/%d", (oi + 1), numIterations));
            if (gd.step()) break;
            if (worker->isInterrupted()) return InverseProblemOutput();
        }
        real finalValue = gd.getCurrentSolution()[0];
#else
		//LBFGS
		LBFGSpp::LBFGSParam<real> params;
		params.epsilon = 1e-15;
		params.max_iterations = numIterations;
		LBFGSpp::LBFGSSolver<real> lbfgs(params);
		//define gradient
		LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t fun([&simulation, &outputU, this, worker, deformedTimestep](const VectorX& x, VectorX& gradient) -> real {
			real youngsModulus = x[0];
			real cost;
			real grad = gradientMesh(deformedTimestep, simulation, youngsModulus, outputU, cost, worker);
			gradient[0] = grad;
			return cost;
		});
		LBFGSpp::LBFGSSolver<real>::CallbackFunction_t lbfgsCallback([worker, &simulation, &outputU, callback, this](const VectorX& x, const VectorX& g, const real& v, int k) -> bool {
			worker->setStatus(tfm::format("Adjoint - Young's Modulus: optimization %d/%d", k, numIterations));

			InverseProblemOutput output;
			output.youngsModulus_ = x[0];
			output.resultMeshDisp_ = SoftBodyMesh2D::Vector2List(simulation.getNumNodes());
			for (int i = 0; i<simulation.getNumNodes(); ++i)
			{
				if (simulation.getNodeToFreeMap()[i] < 0)
					output.resultMeshDisp_->at(i).setZero();
				else
					output.resultMeshDisp_->at(i) = outputU.segment<2>(2 * simulation.getNodeToFreeMap()[i]);
			}
			output.finalCost_ = v;
			callback(output);

			return !worker->isInterrupted();
		});
		//optimize
		real finalCost = 0;
		VectorX finalValueV = VectorX::Constant(1, initialYoungsModulus);
		int oi = lbfgs.minimize(fun, finalValueV, finalCost, lbfgsCallback);
		if (worker->isInterrupted()) return InverseProblemOutput();
		real finalValue = finalValueV[0];
#endif
        CI_LOG_I("Optimization: final Youngs' Modulus is " << finalValue << " after " << oi << " optimization steps, reference value is " << input->settings_.youngsModulus_);

        InverseProblemOutput output;
        output.youngsModulus_ = finalValue;
        output.resultMeshDisp_ = SoftBodyMesh2D::Vector2List(simulation.getNumNodes());
        for (int i=0; i<simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0)
                output.resultMeshDisp_->at(i).setZero();
            else
                output.resultMeshDisp_->at(i) = outputU.segment<2>(2 * simulation.getNodeToFreeMap()[i]);
        }
		output.finalCost_ = finalCost;
        return output;
    }

    real InverseProblem_Adjoint_YoungsModulus::gradientMesh(
        int deformedTimestep, const SoftBodyMesh2D& simulation,
        real youngsModulus, VectorX& outputU, real& outputCost, BackgroundWorker* worker) const
    {
        real poissonsRatio = input->settings_.poissonsRatio_;

        //FORWARD

        MatrixX K = MatrixX::Zero(2 * simulation.getNumFreeNodes(), 2 * simulation.getNumFreeNodes());
        VectorX F = VectorX::Zero(2 * simulation.getNumFreeNodes());

        //assemble force vector F and stiffness matrix K
        simulation.assembleForceVector(F);
        if (worker->isInterrupted()) return 0;
        real materialMu, materialLambda;
        SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
        Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);
        simulation.assembleStiffnessMatrix(C, K, &F, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
        if (worker->isInterrupted()) return 0;
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(K, tfm::format("AdjointYoung-GradientMesh-K-k%f.csv", youngsModulus));
#endif

        //assemble derivate of matrix K with respect to the youngsModulus
        MatrixX KDyoung = MatrixX::Zero(2 * simulation.getNumFreeNodes(), 2 * simulation.getNumFreeNodes());
        real muDyoung, lambdaDyoung;
        AdjointUtilities::computeMaterialParameters_D_YoungsModulus(youngsModulus, poissonsRatio, muDyoung, lambdaDyoung);
        Matrix3 CDyoung = SoftBodySimulation::computeMaterialMatrix(muDyoung, lambdaDyoung);
        simulation.assembleStiffnessMatrix(CDyoung, KDyoung, nullptr, SoftBodyMesh2D::Vector2List(simulation.getNumNodes(), Vector2::Zero()));
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(KDyoung, tfm::format("AdjointYoung-GradientMesh-KDyoung-k%f.csv", youngsModulus));
#endif

        //solve for the current displacement
        PartialPivLU<MatrixX> Klu = K.partialPivLu();
        VectorX u = Klu.solve(F);
        outputU = u;

        //compute the partial gradient of the operations with respect to Young's Modulus
        VectorX EDyoung = -KDyoung * u;
        //VectorX EDyoung = Klu.solve(-KDyoung * u);

        // BACKWARD

        //compute derivate of the cost function with respect to the output u (displacement)
        VectorX costDu(2 * simulation.getNumFreeNodes());
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0) continue;
            costDu.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                = u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                - input->meshResultsDisplacement_[deformedTimestep][i];
        }
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(costDu, tfm::format("AdjointYoung-GradientMesh-costDu-k%f.csv", youngsModulus));
#endif

        //solve for the dual
        //In theory, I have to use K', but because I know that K is symmetric,
        //I can use K directly and not the transpose.
        VectorX lambda = Klu.solve(costDu);

        //compute gradient of the prior term
        real priorDyoung = -youngsModulusPrior / (youngsModulus*youngsModulus);

        //evalue the final gradient
        real gradient = lambda.transpose() * EDyoung + priorDyoung;

        //For testing: evaluate cost
        real cost = youngsModulusPrior / youngsModulus;
        for (int i = 0; i < simulation.getNumNodes(); ++i)
        {
            if (simulation.getNodeToFreeMap()[i] < 0) continue;
            cost += (u.segment<2>(2 * simulation.getNodeToFreeMap()[i])
                - input->meshResultsDisplacement_[deformedTimestep][i]).squaredNorm() / 2;
        }
        outputCost = cost;
        CI_LOG_I("Optimization: Youngs Modulus " << youngsModulus << " -> cost " << cost << ", gradient " << gradient);

        return gradient;
    }

    real InverseProblem_Adjoint_YoungsModulus::gradientGrid(int deformedTimestep, const SoftBodyGrid2D& simulation,
        real youngsModulus, GridUtils2D::grid_t& outputSdf, real& outputCost, BackgroundWorker* worker) const
    {
        real poissonsRatio = input->settings_.poissonsRatio_;

        //FORWARD
        int resolution = input->gridResolution_;
        real h = 1.0 / (resolution - 1);
        Vector2 pos(0, 0); //lower left corner of the grid
        Vector2 size(h, h); //size of each cell

		//collect degrees of freedom in the stifness solve
		const Eigen::MatrixXi& posToIndex = simulation.getPosToIndex();
		const SoftBodyGrid2D::indexToPos_t& indexToPos = simulation.getIndexToPos();
		const int dof = simulation.getDoF();

        MatrixX K = MatrixX::Zero(dof * 2, dof * 2);
        VectorX f = VectorX::Zero(dof * 2);
		const VectorX prevU = VectorX::Zero(dof * 2); //previous displacements for rotation correction. Not needed here

        //assemble force vector F and stiffness matrix K
		VectorX collisionForces = VectorX::Zero(2 * dof);
        simulation.assembleForceVector(f, posToIndex, collisionForces);
        if (worker->isInterrupted()) return 0;
        real materialMu, materialLambda;
        SoftBodySimulation::computeMaterialParameters(youngsModulus, poissonsRatio, materialMu, materialLambda);
        Matrix3 C = SoftBodySimulation::computeMaterialMatrix(materialMu, materialLambda);
        simulation.assembleStiffnessMatrix(C, materialMu, materialLambda, K, &f, posToIndex, prevU);
        if (worker->isInterrupted()) return 0;
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(K, tfm::format("AdjointYoung-GradientGrid-K-k%f.csv", youngsModulus));
#endif

        //assemble derivate of matrix K with respect to the youngsModulus
        MatrixX KDyoung = MatrixX::Zero(dof * 2, dof * 2);
        real muDyoung, lambdaDyoung;
		AdjointUtilities::computeMaterialParameters_D_YoungsModulus(youngsModulus, poissonsRatio, muDyoung, lambdaDyoung);
        Matrix3 CDyoung = SoftBodySimulation::computeMaterialMatrix(muDyoung, lambdaDyoung);
        simulation.assembleStiffnessMatrix(CDyoung, muDyoung, lambdaDyoung, KDyoung, nullptr, posToIndex, prevU);
        if (worker->isInterrupted()) return 0;
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(KDyoung, tfm::format("AdjointYoung-GradientGrid-KDyoung-k%f.csv", youngsModulus));
#endif

        //solve for the current displacement
        PartialPivLU<MatrixX> Klu = K.partialPivLu();
        VectorX solution = Klu.solve(f);
        if (worker->isInterrupted()) return 0;
        
        //map back to a grid
        SoftBodyGrid2D::grid_t uGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t uGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        for (int i = 0; i < dof; ++i) {
            Vector2i p = indexToPos.at(i);
            uGridX(p.x(), p.y()) = solution[2 * i];
            uGridY(p.x(), p.y()) = solution[2 * i + 1];
        }
        if (worker->isInterrupted()) return 0;

        //perform diffusion step
        GridUtils2D::bgrid_t validCells = simulation.computeValidCells(uGridX, uGridY, posToIndex);
        uGridX = GridUtils2D::fillGridDiffusion(uGridX, validCells);
        uGridY = GridUtils2D::fillGridDiffusion(uGridY, validCells);
        if (worker->isInterrupted()) return 0;

#if 0
        //invert displacements
        SoftBodyGrid2D::grid_t uGridInvX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t uGridInvY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::invertDisplacement(uGridX, uGridY, uGridInvX, uGridInvY, h);
        if (worker->isInterrupted()) return 0;

        //advect levelset
        outputSdf = GridUtils2D::advectGridSemiLagrange(input->gridReferenceSdf_, uGridInvX, uGridInvY, -1 / h);
        if (worker->isInterrupted()) return 0;
#else
        //advect levelset
		GridUtils2D::grid_t advectionWeights(resolution, resolution);
        outputSdf = GridUtils2D::advectGridDirectForward(input->gridReferenceSdf_, uGridX, uGridY, -1 / h, &advectionWeights);
        if (worker->isInterrupted()) return 0;
#endif

        //compute the partial gradient of the operations with respect to Young's Modulus
        VectorX EDyoung = -KDyoung * solution;
        //VectorX EDyoung = Klu.solve(-KDyoung * solution);
        if (worker->isInterrupted()) return 0;

        // BACKWARD

#if 1
        //compute derivate of the cost function with respect to the output u (displacement)
        //SoftBodyGrid2D::grid_t costDu = outputSdf - input->gridResultsSdf_[deformedTimestep];
        SoftBodyGrid2D::grid_t costDu =
            (2 * ((2 / M_PI)*outputSdf.atan() - (2 / M_PI)*input->gridResultsSdf_[deformedTimestep].atan())) /
            (M_PI * (1 + outputSdf.square()));

#if 0
        //Adjoint: advect levelset
        SoftBodyGrid2D::grid_t adjInputSdf = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridInvX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridInvY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::advectGridSemiLagrangeAdjoint(
            adjInputSdf, adjUGridInvX, adjUGridInvY, -1 / h, costDu,
            uGridInvX, uGridInvY, input->gridReferenceSdf_);
        if (worker->isInterrupted()) return 0;

        //Adjoint: invert levelset
        SoftBodyGrid2D::grid_t adjUGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::invertDisplacementDirectShepardAdjoint(
            adjUGridX, adjUGridY, adjUGridInvX, adjUGridInvY, 
            uGridX, uGridY, uGridInvX, uGridInvY, h);
        if (worker->isInterrupted()) return 0;
#else
        SoftBodyGrid2D::grid_t adjInputSdf = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::advectGridDirectForwardAdjoint(adjInputSdf, adjUGridX, adjUGridY, 
            costDu, uGridX, uGridY, input->gridReferenceSdf_, outputSdf, -1 / h, advectionWeights);
        if (worker->isInterrupted()) return 0;
#endif

        //Adjoint: perform diffusion step
        GridUtils2D::fillGridDiffusionAdjoint(adjUGridX, adjUGridX, uGridX, validCells); //note that the adjoint is both input and output
        GridUtils2D::fillGridDiffusionAdjoint(adjUGridY, adjUGridY, uGridY, validCells); //it is modified in-place
        if (worker->isInterrupted()) return 0;

        //Adjoint: map back to a grid
        VectorX adjSolution(2 * dof);
        for (int i = 0; i < dof; ++i) {
            Vector2i p = indexToPos.at(i);
            adjSolution[2 * i] = adjUGridX(p.x(), p.y());
            adjSolution[2 * i + 1] = adjUGridY(p.x(), p.y());
        }
        if (worker->isInterrupted()) return 0;

        //Adjoint: solve for the current displacement
        //since K is symmetric, I can use K directly instead of K' as normally required by the Adjoint Method.
        VectorX lambda = Klu.solve(adjSolution);
        if (worker->isInterrupted()) return 0;

#else
        //Recompute inverted true solution
        SoftBodyGrid2D::grid_t gridResultsInvUx = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t gridResultsInvUy = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::invertDisplacementDirectShepard(
            input->gridResultsUx_[deformedTimestep], input->gridResultsUy_[deformedTimestep],
            gridResultsInvUx, gridResultsInvUy, h);

        //Reduced problem only on the levelset
        SoftBodyGrid2D::grid_t adjUGridInvX = uGridInvX - gridResultsInvUx;
        SoftBodyGrid2D::grid_t adjUGridInvY = uGridInvY - gridResultsInvUy;

        //Adjoint: invert levelset
        SoftBodyGrid2D::grid_t adjUGridX = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        SoftBodyGrid2D::grid_t adjUGridY = SoftBodyGrid2D::grid_t::Zero(resolution, resolution);
        GridUtils2D::invertDisplacementDirectShepardAdjoint(
            adjUGridX, adjUGridY, adjUGridInvX, adjUGridInvY,
            uGridX, uGridY, uGridInvX, uGridInvY, h);
        if (worker->isInterrupted()) return 0;

        //Adjoint: perform diffusion step
        GridUtils2D::fillGridDiffusionAdjoint(adjUGridX, adjUGridX, uGridX, validCells); //note that the adjoint is both input and output
        GridUtils2D::fillGridDiffusionAdjoint(adjUGridY, adjUGridY, uGridY, validCells); //it is modified in-place
        if (worker->isInterrupted()) return 0;

        //Adjoint: map back to a grid
        VectorX adjSolution(2 * dof);
        for (int i = 0; i < dof; ++i) {
            Vector2i p = indexToPos[i];
            adjSolution[2 * i] = adjUGridX(p.x(), p.y());
            adjSolution[2 * i + 1] = adjUGridY(p.x(), p.y());
        }
        if (worker->isInterrupted()) return 0;
#if DEBUG_SAVE_MATRICES==1
        saveAsCSV(adjSolution, tfm::format("AdjointYoung-GradientGrid-costDu-k%f.csv", youngsModulus));
#endif

        //Adjoint: solve for the current displacement
        //since K is symmetric, I can use K directly instead of K' as normally required by the Adjoint Method.
        VectorX lambda = -Klu.solve(adjSolution);
        if (worker->isInterrupted()) return 0;
#endif

        //Done with the adjoint computation
        //Now I can assemble the final gradient

        //compute gradient of the prior term
        real priorDyoung = -youngsModulusPrior / (youngsModulus*youngsModulus);

        //evalue the final gradient
        real gradient = -lambda.transpose() * EDyoung + priorDyoung;

        //For testing: evaluate cost
#if 1
        real cost = youngsModulusPrior / youngsModulus
            + ((2/M_PI)*outputSdf.atan() - (2/M_PI)*input->gridResultsSdf_[deformedTimestep].atan()).matrix().squaredNorm() / 2;
#else
        //reduced problem
        real cost = (uGridInvX - gridResultsInvUx).matrix().squaredNorm() / 2 +
                    (uGridInvY - gridResultsInvUy).matrix().squaredNorm() / 2;
#endif
        outputCost = cost;
        CI_LOG_I("Optimization: Youngs Modulus " << youngsModulus << " -> cost " << cost << ", gradient " << gradient);

        return gradient;
    }

    void InverseProblem_Adjoint_YoungsModulus::setupParams(cinder::params::InterfaceGlRef params,
        const std::string& group)
    {
        params->addParam("InverseProblem_Adjoint_YoungsModulus_InitialYoungsModulus", &initialYoungsModulus)
            .group(group).label("Initial Young's Modulus").min(0).step(0.01f);
        params->addParam("InverseProblem_Adjoint_YoungsModulus_YoungsModulusPrior", &youngsModulusPrior)
            .group(group).label("Prior on Young's Modulus").min(0).step(0.01f);
        params->addParam("InverseProblem_Adjoint_YoungsModulus_NumIterations", &numIterations)
            .group(group).label("Iterations").min(1);
    }

    void InverseProblem_Adjoint_YoungsModulus::setParamsVisibility(cinder::params::InterfaceGlRef params,
        bool visible) const
    {
        std::string option = visible ? "visible=true" : "visible=false";
        params->setOptions("InverseProblem_Adjoint_YoungsModulus_InitialYoungsModulus", option);
        params->setOptions("InverseProblem_Adjoint_YoungsModulus_YoungsModulusPrior", option);
        params->setOptions("InverseProblem_Adjoint_YoungsModulus_NumIterations", option);
    }

    std::vector<InverseProblem_Adjoint_YoungsModulus::DataPoint> InverseProblem_Adjoint_YoungsModulus::plotEnergy(int deformedTimestep, BackgroundWorker * worker) const
    {
        //control points
        real trueYoung = input->settings_.youngsModulus_;
        real minYoung = trueYoung / 4;
        real maxYoung = trueYoung + (trueYoung - minYoung);
        int steps = 25; // has to be odd
        std::vector<DataPoint> points(steps);
        for (int i = 0; i < steps; ++i)
            points[i].youngsModulus = minYoung + (maxYoung - minYoung) * i / (steps - 1.0);

        // Create grid simulation
        worker->setStatus("Test Plot Grid: create simulation");
        SoftBodyGrid2D simulationGrid;
        simulationGrid.setGridResolution(input->gridResolution_);
        simulationGrid.setSDF(input->gridReferenceSdf_);
        simulationGrid.setExplicitDiffusion(true);
        simulationGrid.setHardDirichletBoundaries(false);
        simulationGrid.resetBoundaries();
        for (const auto& b : input->gridDirichletBoundaries_)
            simulationGrid.addDirichletBoundary(b.first.first, b.first.second, b.second);
        for (const auto& b : input->gridNeumannBoundaries_)
            simulationGrid.addNeumannBoundary(b.first.first, b.first.second, b.second);
        simulationGrid.setGravity(input->settings_.gravity_);
        simulationGrid.setMass(input->settings_.mass_);
        simulationGrid.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
        simulationGrid.setRotationCorrection(input->settings_.rotationCorrection_);

        //Run grid simulation
        GridUtils2D::grid_t outputGrid;
        for (int i = 0; i < steps; ++i)
        {
            worker->setStatus(tfm::format("Test Plot Grid: simulation %d/%d",i+1,steps));
            real cost;
            real gradient = gradientGrid(deformedTimestep, simulationGrid, points[i].youngsModulus, outputGrid, cost, worker);
            if (worker->isInterrupted()) return {};
            points[i].costGrid = cost;
            points[i].gradientGrid = gradient;
        }

        //Create mesh simulation
        worker->setStatus("Test Plot Mesh: create simulation");
        SoftBodyMesh2D simulationMesh;
        simulationMesh.setMesh(input->meshReferencePositions_, input->meshReferenceIndices_);
        simulationMesh.resetBoundaries();
        for (const auto& b : input->meshDirichletBoundaries_)
            simulationMesh.addDirichletBoundary(b.first, b.second);
        for (const auto& b : input->meshNeumannBoundaries_)
            simulationMesh.addNeumannBoundary(b.first, b.second);
        simulationMesh.reorderNodes();
        simulationMesh.setGravity(input->settings_.gravity_);
        simulationMesh.setMass(input->settings_.mass_);
        simulationMesh.setDamping(input->settings_.dampingAlpha_, input->settings_.dampingBeta_);
        simulationMesh.setRotationCorrection(input->settings_.rotationCorrection_);

        //Run mesh simulation
        VectorX outputMesh;
        for (int i = 0; i < steps; ++i)
        {
            worker->setStatus(tfm::format("Test Plot Mesh: simulation %d/%d", i + 1, steps));
            real cost;
            real gradient = gradientMesh(deformedTimestep, simulationMesh, points[i].youngsModulus, outputMesh, cost, worker);
            if (worker->isInterrupted()) return {};
            points[i].costMesh = cost;
            points[i].gradientMesh = gradient;
        }

        return points;
    }

	void InverseProblem_Adjoint_YoungsModulus::testPlot(int deformedTimestep, BackgroundWorker* worker)
	{
		auto points = plotEnergy(deformedTimestep-1, worker);
		if (worker->isInterrupted()) return;
		fstream file("../plots/Adjoint-YoungsModulus.csv", fstream::out | fstream::trunc);
		file << "Youngs Modulus , Cost Mesh , Gradient Mesh , Cost Grid , Gradient Grid" << endl;
		file << std::fixed;
		for (auto point : points)
		{
			file << point.youngsModulus << " , "
				<< point.costMesh << " , " << point.gradientMesh << " , "
				<< point.costGrid << " , " << point.gradientGrid << endl;
		}
		file.close();
	}

}
