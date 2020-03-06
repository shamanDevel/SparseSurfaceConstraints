#include "SoftBodyGrid2D.h"

#include <cassert>
#include <map>
#include <array>
#include <vector>
#include <cinder/Log.h>
#include <Eigen/Dense>
#include <cinder/app/AppBase.h>

#include "Utils.h"
#include "Integration.h"
#include "GridUtils.h"
#include "GeometryUtils2D.h"

using namespace Eigen;
using namespace std;
using namespace ar::utils;

const std::string ar::SoftBodyGrid2D::advectionModeNames[] = {
    "only Semi-Lagrange",
    "Shepard-Inversion + Semi-Lagrange",
    "Direct Forward"
};
const std::string& ar::SoftBodyGrid2D::advectionModeName(AdvectionMode mode)
{
    return advectionModeNames[static_cast<int>(mode)];
}

ar::SoftBodyGrid2D::SoftBodyGrid2D()
	: resolution_(0)
	, h_(1)
	, dof_(0)
{
    //basis function matrices
    Phi1 << Matrix2::Identity(), Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero();
    Phi2 << Matrix2::Zero(), Matrix2::Identity(), Matrix2::Zero(), Matrix2::Zero();
    Phi3 << Matrix2::Zero(), Matrix2::Zero(), Matrix2::Identity(), Matrix2::Zero();
    Phi4 << Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(), Matrix2::Identity();
}

void ar::SoftBodyGrid2D::setGridResolution(int resolution)
{
    resolution_ = resolution;
    integrator_ = nullptr;
    resetBoundaries();
    resetSolution();

    //B's are dependent on the size
    h_ = 1.0 / (resolution_ - 1);
    size_ = Vector2(h_, h_); //size of each cell
    const real b = h_;
    const real a = h_;
    CI_LOG_V("a=" << a << ", b=" << b);
    B1 << -1 / a, 0, 1 / a, 0, 0, 0, 0, 0,
        0, -1 / b, 0, 0, 0, 1 / b, 0, 0,
        -1 / b, -1 / a, 0, 1 / a, 1 / b, 0, 0, 0;
    B2 << -1 / a, 0, 1 / a, 0, 0, 0, 0, 0,
        0, 0, 0, -1 / b, 0, 0, 0, 1 / b,
        0, -1 / a, -1 / b, 1 / a, 0, 0, 1 / b, 0;
    B3 << 0, 0, 0, 0, -1 / a, 0, 1 / a, 0,
        0, -1 / b, 0, 0, 0, 1 / b, 0, 0,
        -1 / b, 0, 0, 0, 1 / b, -1 / a, 0, 1 / a;
    B4 << 0, 0, 0, 0, -1 / a, 0, 1 / a, 0,
        0, 0, 0, -1 / b, 0, 0, 0, 1 / b,
        0, 0, -1 / b, 0, 0, -1 / a, 1 / b, 1 / a;
}

void ar::SoftBodyGrid2D::setSDF(const grid_t & sdf)
{
    assert(sdf.rows() == resolution_);
    assert(sdf.cols() == resolution_);
    sdf_ = sdf;

	//find number of free nodes (nodes of a grid cell that contain a part of the volume)
	//We can't solve anything in completely empty regions -> remove them
	dof_ = findDoF(posToIndex_, indexToPos_);

	resetSolution();
}

void ar::SoftBodyGrid2D::setExplicitDiffusion(bool enable)
{
	if (!enable) CI_LOG_W("implicit diffusion is deprecated!");
    gridSettings_.explicitDiffusion_ = enable;
}

void ar::SoftBodyGrid2D::resetBoundaries()
{
    gridNeumannX_ = grid_t::Zero(resolution_, resolution_);
    gridNeumannY_ = grid_t::Zero(resolution_, resolution_);
    gridDirichlet_ = bgrid_t::Constant(resolution_, resolution_, false);
}

void ar::SoftBodyGrid2D::addDirichletBoundary(int x, int y, const Vector2 & displacement)
{
    //TODO: support non-uniform dirichlet boundaries
    gridDirichlet_(x, y) = true;
}

void ar::SoftBodyGrid2D::addNeumannBoundary(int x, int y, const Vector2 & force)
{
    gridNeumannX_(x, y) = force.x();
    gridNeumannY_(x, y) = force.y();
}

void ar::SoftBodyGrid2D::resetSolution()
{
    uGridX_ = grid_t::Zero(resolution_, resolution_);
    uGridY_ = grid_t::Zero(resolution_, resolution_);
    uGridInvX_ = grid_t::Zero(resolution_, resolution_);
    uGridInvY_ = grid_t::Zero(resolution_, resolution_);
    sdfSolution_ = sdf_;
	currentU = VectorX::Zero(2 * dof_);
	currentUDot = VectorX::Zero(2 * dof_);
	collisionForces_ = VectorX::Zero(2 * dof_);
    integrator_ = nullptr;
    setHasSolution(false);
}

int ar::SoftBodyGrid2D::findDoF(Eigen::MatrixXi& posToIndex, indexToPos_t& indexToPos) const
{
    int dof;
    if (gridSettings_.explicitDiffusion_) {
        //computation only in non-empty cells,
        //displacements are diffused into empty cells afterwards
        dof = 0;
        posToIndex = MatrixXi::Constant(resolution_, resolution_, -1);
        for (int x = 0; x < resolution_ - 1; ++x) {
            for (int y = 0; y < resolution_ - 1; ++y) {
                //assemble SDF array
                array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
                if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3]))
                    continue; //completely outside
                              //Add these positions as indices
                if (posToIndex(x, y) < 0) {
                    posToIndex(x, y) = dof;
                    indexToPos[dof] = Vector2i(x, y);
                    dof++;
                }
                if (posToIndex(x + 1, y) < 0) {
                    posToIndex(x + 1, y) = dof;
                    indexToPos[dof] = Vector2i(x + 1, y);
                    dof++;
                }
                if (posToIndex(x, y + 1) < 0) {
                    posToIndex(x, y + 1) = dof;
                    indexToPos[dof] = Vector2i(x, y + 1);
                    dof++;
                }
                if (posToIndex(x + 1, y + 1) < 0) {
                    posToIndex(x + 1, y + 1) = dof;
                    indexToPos[dof] = Vector2i(x + 1, y + 1);
                    dof++;
                }
            }
        }
        CI_LOG_V("Degrees of freedom / nodes connected to non-empty cells: " << dof << " (of " << (resolution_*resolution_) << " nodes)");
    }
    else
    {
        //diffusion is performed within the matrix
        dof = resolution_ * resolution_;
    }
    return dof;
}

void ar::SoftBodyGrid2D::solve(bool dynamic, BackgroundWorker * worker)
{
    worker->setStatus("SoftBodyGrid2D: compute degrees of freedom");

	//Collision forces
	if (isEnableCollision())
	{
		collisionForces_ = resolveCollisions(posToIndex_, dof_, currentU, currentUDot);
	}
	else
	{
		collisionForces_.setZero();
	}

    //solve elasticity
#if SOFT_BODY_SUPPORT_SPARSE_MATRICES==1
    VectorX solution, velocity;
    if (isUseSparseMatrices())
        std::tie(solution, velocity) = solveImplSparse(dynamic, dof_, posToIndex_, currentU, worker);
    else
        std::tie(solution, velocity) = solveImplDense(dynamic, dof_, posToIndex_, currentU, worker);
#else
    VectorX solution = solveImplDense(timestep, size, dof, posToIndex, currentU, worker);
#endif
	currentU = solution;
    currentUDot = velocity;
    if (worker->isInterrupted()) return;

    //map displacements back to the whole grid
    worker->setStatus("SoftBodyGrid2D: map solution to the grid");
    if (gridSettings_.explicitDiffusion_) {
        uGridX_ = grid_t::Zero(resolution_, resolution_);
        uGridY_ = grid_t::Zero(resolution_, resolution_);
        for (int i = 0; i < dof_; ++i) {
            Vector2i p = indexToPos_[i];
            uGridX_(p.x(), p.y()) = solution[2 * i];
            uGridY_(p.x(), p.y()) = solution[2 * i + 1];
        }
    }
    else {
        uGridX_.resize(resolution_, resolution_);
        uGridY_.resize(resolution_, resolution_);
        for (int x = 0; x < resolution_; ++x) {
            for (int y = 0; y < resolution_; ++y) {
                int j = x + y * resolution_;
                uGridX_(x, y) = solution(2 * j);
                uGridY_(x, y) = solution(2 * j + 1);
            }
        }
    }
    CI_LOG_V("uGridX:\n" << uGridX_);
    CI_LOG_V("uGridY:\n" << uGridY_);
    if (worker->isInterrupted()) return;

    if (gridSettings_.explicitDiffusion_) {
        //compute valid cells
        GridUtils2D::bgrid_t validCells = computeValidCells(uGridX_, uGridY_, posToIndex_);
        if (worker->isInterrupted()) return;

        worker->setStatus("SoftBodyGrid2D: perform explicit diffusion");
        //diffuse displacement into invalid/not-computed nodes
        CI_LOG_V("valid cells:\n" << validCells);
#if 0
        uGridX_ = GridUtils2D::fillGrid(uGridX_, validCells);
        uGridY_ = GridUtils2D::fillGrid(uGridY_, validCells);
#else
        uGridX_ = GridUtils2D::fillGridDiffusion(uGridX_, validCells);
        uGridY_ = GridUtils2D::fillGridDiffusion(uGridY_, validCells);
#endif
        if (worker->isInterrupted()) return;
    }
    
    switch (gridSettings_.advectionMode_)
    {
    case AdvectionMode::SEMI_LAGRANGE_ONLY:
        {
        worker->setStatus("SoftBodyGrid2D: advect levelset");
        sdfSolution_ = GridUtils2D::advectGridSemiLagrange(sdf_, uGridX_, uGridY_, -1 / h_);
        CI_LOG_V("SDF advected directly by the displacements");
        } break;
    case AdvectionMode::SEMI_LAGRANGE_SHEPARD_INVERSION:
        {
        worker->setStatus("SoftBodyGrid2D: invert displacements");
        uGridInvX_ = grid_t::Zero(resolution_, resolution_);
        uGridInvY_ = grid_t::Zero(resolution_, resolution_);
        GridUtils2D::invertDisplacement(uGridX_, uGridY_, uGridInvX_, uGridInvY_, h_);
        CI_LOG_V("uGridInvX:\n" << uGridInvX_);
        CI_LOG_V("uGridInvY:\n" << uGridInvY_);
        if (worker->isInterrupted()) return;

        worker->setStatus("SoftBodyGrid2D: advect levelset");
        sdfSolution_ = GridUtils2D::advectGridSemiLagrange(sdf_, uGridInvX_, uGridInvY_, -1 / h_);
        CI_LOG_V("SDF advected by the inverse displacements");
        } break;
    case AdvectionMode::DIRECT_FORWARD:
        {
        worker->setStatus("SoftBodyGrid2D: advect levelset");
        sdfSolution_ = GridUtils2D::advectGridDirectForward(sdf_, uGridX_, uGridY_, -1 / h_, nullptr);
		//sdfSolution_ = sdf_;
		//int steps = 10;
		//for (int i=0; i<steps; ++i)
		//{
		//	sdfSolution_ = GridUtils2D::advectGridDirectForward(sdfSolution_, uGridX_, uGridY_, -1 / (h_*steps), nullptr, 1);
		//}
        CI_LOG_V("SDF advected by direct forward interpolation");
        } break;
    }
    if (worker->isInterrupted()) return;

	// reconstruct SDF
	//if (getSdfRecoveryIterations() > 0 && recoverSdf)
	//{
		sdfSolution_ = GridUtils2D::recoverSDFSussmann(sdfSolution_, real(0.01), getSdfRecoveryIterations());
	//}

    //DONE
    setHasSolution(true);
}

ar::real ar::SoftBodyGrid2D::computeElasticEnergy(const grid_t & uGridX, const grid_t & uGridY)
{
	real energy = 0;
	real h = 1.0 / (resolution_ - 1);
	Vector2 size(h, h); //size of each cell
	real area = size.prod();

	real mu = getMaterialMu();
	real lambda = getMaterialLambda();
    Matrix3 C = computeMaterialMatrix(mu, lambda);


#pragma omp parallel for schedule(dynamic) reduction(+:energy)
	for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
		int x = xy / (resolution_ - 1);
		int y = xy % (resolution_ - 1);

		array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
		if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
			//completely outside
			continue;
		}

		Vector8 u; u <<
			uGridX(x, y), uGridY(x, y), uGridX(x + 1, y), uGridY(x + 1, y),
			uGridX(x, y + 1), uGridY(x, y + 1), uGridY(x + 1, y + 1), uGridY(x + 1, y + 1);
		array<real, 4> Kex = {
			(B1 * u).dot(C * B1 * u),
			(B2 * u).dot(C * B2 * u),
			(B3 * u).dot(C * B3 * u),
			(B4 * u).dot(C * B4 * u),
		};
		energy += Integration2D<real>::integrateQuad(size, sdfs, Kex);
	}

	return energy;
}

// DEPRECATED !!!!
bool ar::SoftBodyGrid2D::computeElementMatrix_old(int x, int y, const Vector2& size, const Matrix3 & C, Matrix8 & Ke, Vector8 & MeVec, Vector8 & Fe, const VectorX& currentDisplacements) const
{
    //assemble SDF array
    array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
    if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
        //completely outside
        return false;
    }
    CI_LOG_V("element (" << x << "," << y << ") -> SDFs " << sdfs[0] << ", " << sdfs[1] << ", " << sdfs[2] << ", " << sdfs[3]);
    real area = size.prod();

    //The mass (assuming constant mass, this array is the same for all elements)
    real mass = getMass();
    array<Matrix8, 4> Mex = {
        mass * Phi1.transpose() * Phi1,
        mass * Phi2.transpose() * Phi2,
        mass * Phi3.transpose() * Phi3,
        mass * Phi4.transpose() * Phi4
    };
    //The mass matrix really is purely diagonal.
    //TODO: integrate then also only over vectors
    Matrix8 Me = Integration2D<real>::integrateQuad(size, sdfs, Mex);
    CI_LOG_V("mass Me:\n" << Me);
    MeVec = Me.diagonal();

    //The stiffness (assuming constant stiffness, this array is the same for all elements)
    array<Matrix8, 4> Kex = {
        B1.transpose() * C * B1,
        B2.transpose() * C * B2,
        B3.transpose() * C * B3,
        B4.transpose() * C * B4,
    };
    Ke = Integration2D<real>::integrateQuad(size, sdfs, Kex);
    CI_LOG_V("stiffness Ke:\n" << Ke);

    //Forces
    Vector8 bodyForces;
    bodyForces << getGravity(), getGravity(), getGravity(), getGravity();
    Fe = area * 0.25 * bodyForces;

    //Neumann boundaries
    Vector8 FsNeumann;
    FsNeumann << Vector2(gridNeumannX_(x, y), gridNeumannY_(x, y)),
        Vector2(gridNeumannX_(x + 1, y), gridNeumannY_(x + 1, y)),
        Vector2(gridNeumannX_(x, y + 1), gridNeumannY_(x, y + 1)),
        Vector2(gridNeumannX_(x + 1, y + 1), gridNeumannY_(x + 1, y + 1));
    array<Vector8, 4> FeNeumannX = {
        Phi1.transpose() * Phi1 * FsNeumann,
        Phi2.transpose() * Phi2 * FsNeumann,
        Phi3.transpose() * Phi3 * FsNeumann,
        Phi4.transpose() * Phi4 * FsNeumann
    };
    Vector8 FeNeumann = Integration2D<real>::integrateQuadBoundary(size, sdfs, FeNeumannX);
    CI_LOG_V("Neumann boundaries FeNeumman:\n" << FeNeumann.transpose());
    Fe += FeNeumann;

    //Dirichlet boundaries
    real c = 10000;
    real rho = c / size.x();
    Matrix8 KeDirichlet = Matrix8::Zero();
    Vector8 FeDirichlet = Vector8::Zero();
    if (!gridSettings_.hardDirichletBoundaries_ && (gridDirichlet_(x, y) || gridDirichlet_(x + 1, y) || gridDirichlet_(x, y + 1) || gridDirichlet_(x + 1, y + 1))) {
        //get target displacement, for now, set the to zero
        Vector8 u0 = Vector8::Zero();
        //compute normal vector (approximation: constant over the cell / normal in the center
        Vector2 normal(
            0.5*(sdf_(x + 1, y) + sdf_(x + 1, y + 1)) - 0.5*(sdf_(x, y) + sdf_(x, y + 1)),
            0.5*(sdf_(x, y + 1) + sdf_(x + 1, y + 1)) - 0.5*(sdf_(x, y) + sdf_(x + 1, y)));
        normal.normalize();
        //assemble matrix D, a modified version of C
        real mu = getMaterialMu();
        real lambda = getMaterialLambda();
        Matrix23 De;
        De << (2 * mu + lambda) * normal.x(), lambda * normal.x(), mu * normal.y(),
            lambda * normal.y(), (2 * mu + lambda) * normal.y(), mu * normal.x();
        //compute first integral P(u)*n*v = Phi'*D*B -> left hand side
        array<Matrix8, 4> KeD1a = {
            Phi1.transpose() * De * B1,
            Phi2.transpose() * De * B2,
            Phi3.transpose() * De * B3,
            Phi4.transpose() * De * B4,
        };
        Matrix8 KeD1 = Integration2D<real>::integrateQuadBoundary(size, sdfs, KeD1a);
        //compute second integral P(v)*n*u = B'*D'*Phi -> left hand side
        array<Matrix8, 4> KeD2a = {
            B1.transpose() * De.transpose() * Phi1,
            B2.transpose() * De.transpose() * Phi2,
            B3.transpose() * De.transpose() * Phi3,
            B4.transpose() * De.transpose() * Phi4,
        };
        Matrix8 KeD2 = Integration2D<real>::integrateQuadBoundary(size, sdfs, KeD2a);
        //compute third integral -P(v)*n*u_0 -> right hand side
        array<Vector8, 4> FeD1a = {
            B1.transpose() * De.transpose() * Phi1 * u0,
            B2.transpose() * De.transpose() * Phi2 * u0,
            B3.transpose() * De.transpose() * Phi3 * u0,
            B4.transpose() * De.transpose() * Phi4 * u0,
        };
        Vector8 FeD1 = Integration2D<real>::integrateQuadBoundary(size, sdfs, FeD1a);
        //compute forth integral rho*u*v -> left hand side
        array<Matrix8, 4> KeD3a = {
            Phi1.transpose() * Phi1,
            Phi2.transpose() * Phi2,
            Phi3.transpose() * Phi3,
            Phi4.transpose() * Phi4
        };
        Matrix8 KeD3 = rho * Integration2D<real>::integrateQuadBoundary(size, sdfs, KeD3a);
        //compute fifth integral -rho*u_0*v -> right hand side
        array<Vector8, 4> FeD2a = {
            Phi1.transpose() * Phi1 * u0,
            Phi2.transpose() * Phi2 * u0,
            Phi3.transpose() * Phi3 * u0,
            Phi4.transpose() * Phi4 * u0
        };
        Vector8 FeD2 = Integration2D<real>::integrateQuadBoundary(size, sdfs, FeD2a);
        //assemble print result
        KeDirichlet = -KeD1 - KeD2 - KeD3;
        FeDirichlet = FeD1 + FeD2;
        CI_LOG_V("Dirichlet boundary, KeDirichlet:\n" << KeDirichlet);
        CI_LOG_V("Dirichlet boundary, FeDirichlet:\n" << FeDirichlet.transpose());
    }
    Ke += KeDirichlet;
    Fe += FeDirichlet;

    //corotational reparation
    if (getRotationCorrection() == RotationCorrection::Corotation)
    {
        typedef Matrix<real, 1, 2> RowVector2;

        //compute jacobian of the displacements
		const auto getOrZero = [&currentDisplacements,this](int x, int y) {return posToIndex_(x, y) >= 0 ? currentDisplacements.segment<2>(2 * posToIndex_(x, y)) : Vector2(0, 0); };
		Matrix2 F = Matrix2::Identity() + 0.5*(
			getOrZero(x, y) * RowVector2(-1 / size_.x(), -1 / size_.y()) +
			getOrZero(x + 1, y) * RowVector2(1 / size_.x(), -1 / size_.y()) +
			getOrZero(x, y + 1) * RowVector2(-1 / size_.x(), 1 / size_.y()) +
			getOrZero(x + 1, y + 1) * RowVector2(1 / size_.x(), 1 / size_.y())
			);
        //compute polar decomposition
        Matrix2 R = abs(F.determinant()) < 1e-15 ? Matrix2::Identity() : polarDecomposition(F);
        //augment Ke
        Matrix8 Re;
        Re << R, Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(),
            Matrix2::Zero(), R, Matrix2::Zero(), Matrix2::Zero(),
            Matrix2::Zero(), Matrix2::Zero(), R, Matrix2::Zero(),
            Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(), R;
        Vector8 xe; 
        xe << Vector2(x*size.x(), y*size.y()), Vector2((x + 1)*size.x(), y*size.y()),
            Vector2(x*size.x(), (y + 1)*size.y()), Vector2((x + 1)*size.x(), (y + 1)*size.y());
        Fe -= Re * Ke * (Re.transpose() * xe - xe);
        Ke = Re * Ke * Re.transpose();
    }

#ifdef AR_NO_DEBUG_TESTS
    real minEntry = std::numeric_limits<real>::max();
    real maxEntry = 0;
    for (int i = 0; i<Ke.rows(); ++i)
        for (int j = 0; j<Ke.cols(); ++j)
            if (Ke(i, j) != 0)
            {
                minEntry = std::min(minEntry, std::abs(Ke(i, j)));
                maxEntry = std::max(maxEntry, std::abs(Ke(i, j)));
            }
    CI_LOG_V("Matrix min entry: " << minEntry << ", max entry: " << maxEntry);
    //if (minEntry < 1e-10)
        //__debugbreak();
#endif

    return true;
}

void ar::SoftBodyGrid2D::assembleForceVector(VectorX& f, const Eigen::MatrixXi& posToIndex, const VectorX& extraForces) const
{
    for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
        int x = xy / (resolution_ - 1);
        int y = xy % (resolution_ - 1);
        Vector8 Fe;
        
        //assemble SDF array
        array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
        if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
            //completely outside
            continue;
        }

        //Gravity
        Vector8 bodyForces;
        bodyForces << getGravity(), getGravity(), getGravity(), getGravity();
#if 0
		real area = size_.prod();
#else
		real area = Integration2D<real>::integrateQuad<real>(size_, sdfs, { 1,1,1,1 });
#endif
        Fe = area * 0.25 * bodyForces;

        //Neumann boundaries
        Vector8 FsNeumann;
        FsNeumann << Vector2(gridNeumannX_(x, y), gridNeumannY_(x, y)),
            Vector2(gridNeumannX_(x + 1, y), gridNeumannY_(x + 1, y)),
            Vector2(gridNeumannX_(x, y + 1), gridNeumannY_(x, y + 1)),
            Vector2(gridNeumannX_(x + 1, y + 1), gridNeumannY_(x + 1, y + 1));
		if (posToIndex(x, y) >= 0) FsNeumann.segment<2>(0) += extraForces.segment<2>(2 * posToIndex(x, y));
		if (posToIndex(x+1, y) >= 0) FsNeumann.segment<2>(2) += extraForces.segment<2>(2 * posToIndex(x+1, y));
		if (posToIndex(x, y+1) >= 0) FsNeumann.segment<2>(4) += extraForces.segment<2>(2 * posToIndex(x, y+1));
		if (posToIndex(x+1, y+1) >= 0) FsNeumann.segment<2>(6) += extraForces.segment<2>(2 * posToIndex(x+1, y+1));
        array<Vector8, 4> FeNeumannX = {
            Phi1.transpose() * Phi1 * FsNeumann,
            Phi2.transpose() * Phi2 * FsNeumann,
            Phi3.transpose() * Phi3 * FsNeumann,
            Phi4.transpose() * Phi4 * FsNeumann
        };
        Vector8 FeNeumann = Integration2D<real>::integrateQuadBoundary(size_, sdfs, FeNeumannX);
        CI_LOG_V("Neumann boundaries FeNeumman:\n" << FeNeumann.transpose());
        Fe += FeNeumann;

        //combine Me, Ke, Fe into full matrix
        int mapping[4] = {
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y) : (x + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y) : ((x + 1) + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y + 1) : (x + (y + 1) * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y + 1) : ((x + 1) + (y + 1) * resolution_)
        };
        bool dirichlet[4] = {
            gridDirichlet_(x, y),
            gridDirichlet_(x + 1, y),
            gridDirichlet_(x, y + 1),
            gridDirichlet_(x + 1, y + 1)
        };
        for (int i = 0; i < 4; ++i) {
            f.segment<2>(2 * mapping[i]) += Fe.segment<2>(2 * i);
        }

        //if (worker->isInterrupted()) return VectorX();
    }
}

bool ar::SoftBodyGrid2D::computeElementMatrix(
	int x, int y, Matrix8& Ke, Vector8& Fe, 
	real materialMu, real materialLambda, 
	RotationCorrection rotationCorrection, bool hardDirichlet, 
	const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements) const
{
	Ke.setZero();
	Fe.setZero();

	//assemble SDF array
	array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
	if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
		//completely outside
		return false;
	}

	const Integration2D<real>::IntegrationWeights volumeWeights
		= Integration2D<real>::getIntegrateQuadWeights(size_, sdfs);
	const Integration2D<real>::IntegrationWeights boundaryWeights
		= Integration2D<real>::getIntegrateQuadBoundaryWeights(size_, sdfs);

	//The stiffness (assuming constant stiffness, this array is the same for all elements)
	Matrix3 C = computeMaterialMatrix(materialMu, materialLambda);
	array<Matrix8, 4> Kex = {
		B1.transpose() * C * B1,
		B2.transpose() * C * B2,
		B3.transpose() * C * B3,
		B4.transpose() * C * B4,
	};
	Ke = Integration2D<real>::integrateQuad(volumeWeights, Kex);
	CI_LOG_V("stiffness Ke:\n" << Ke);

	//corotational reparation
	if (rotationCorrection == RotationCorrection::Corotation)
	{
		//compute jacobian of the displacements
		const auto getOrZero = [&posToIndex, &currentDisplacements](int x, int y) {return posToIndex(x, y) >= 0 ? currentDisplacements.segment<2>(2 * posToIndex(x, y)) : Vector2(0, 0); };
		array<Matrix2, 4> Fx = {
			getOrZero(x, y) * RowVector2(-1 / size_.x(), -1 / size_.y()),
			getOrZero(x + 1, y) * RowVector2(1 / size_.x(), -1 / size_.y()),
			getOrZero(x, y + 1) * RowVector2(-1 / size_.x(), 1 / size_.y()),
			getOrZero(x + 1, y + 1) * RowVector2(1 / size_.x(), 1 / size_.y())
		};
		Matrix2 F = 0.5 * (Fx[0] + Fx[1] + Fx[2] + Fx[3] + Matrix2::Identity());
		//Matrix2 F = Integration2D<real>::integrateQuad(volumeWeights, Fx) 
		//	/ Integration2D<real>::integrateQuad(volumeWeights, array<real, 4>({ 1, 1, 1, 1 }));
		//compute polar decomposition
		Matrix2 R = abs(F.determinant()) < 1e-15 ? Matrix2::Identity() : polarDecomposition(F);
		//augment Ke
		Matrix8 Re;
		Re << R, Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(),
			Matrix2::Zero(), R, Matrix2::Zero(), Matrix2::Zero(),
			Matrix2::Zero(), Matrix2::Zero(), R, Matrix2::Zero(),
			Matrix2::Zero(), Matrix2::Zero(), Matrix2::Zero(), R;
		Vector8 xe;
		xe << Vector2(x*size_.x(), y*size_.y()), Vector2((x + 1)*size_.x(), y*size_.y()),
			Vector2(x*size_.x(), (y + 1)*size_.y()), Vector2((x + 1)*size_.x(), (y + 1)*size_.y());
		Fe -= Re * Ke * (Re.transpose() * xe - xe);
		Ke = Re * Ke * Re.transpose();
	}

	//Dirichlet boundaries
	Matrix8 KeDirichlet = Matrix8::Zero();
	Vector8 FeDirichlet = Vector8::Zero();
	if (!gridSettings_.hardDirichletBoundaries_ && (gridDirichlet_(x, y) || gridDirichlet_(x + 1, y) || gridDirichlet_(x, y + 1) || gridDirichlet_(x + 1, y + 1))) {
		computeNietscheDirichletBoundaries(x, y, KeDirichlet, FeDirichlet, materialMu, materialLambda, boundaryWeights);
	}
	Ke += KeDirichlet;
	Fe += FeDirichlet;

	return true;
}

void ar::SoftBodyGrid2D::computeNietscheDirichletBoundaries(
	int x, int y, Matrix8& KeDirichlet, Vector8& FeDirichlet,
	real materialMu, real materialLambda,
	const Integration2D<real>::IntegrationWeights& boundaryWeights) const
{
	KeDirichlet.setZero();
	FeDirichlet.setZero();

	//constants
	static const real c = 1e10; //constant, to be tweaked
	real rho = c / size_.x();

	//get target displacement, for now, set the to zero
	Vector8 u0 = Vector8::Zero();
	//compute normal vector (approximation: constant over the cell / normal in the center
	Vector2 normal(
		0.5*(sdf_(x + 1, y) + sdf_(x + 1, y + 1)) - 0.5*(sdf_(x, y) + sdf_(x, y + 1)),
		0.5*(sdf_(x, y + 1) + sdf_(x + 1, y + 1)) - 0.5*(sdf_(x, y) + sdf_(x + 1, y)));
	normal.normalize();
	//assemble matrix D, a modified version of C
	Matrix23 De;
	De << (2 * materialMu + materialLambda) * normal.x(), materialLambda * normal.x(), materialMu * normal.y(),
		materialLambda * normal.y(), (2 * materialMu + materialLambda) * normal.y(), materialMu * normal.x();
	//compute first integral P(u)*n*v = Phi'*D*B -> left hand side
	array<Matrix8, 4> KeD1a = {
		Phi1.transpose() * De * B1,
		Phi2.transpose() * De * B2,
		Phi3.transpose() * De * B3,
		Phi4.transpose() * De * B4,
	};
	Matrix8 KeD1 = Integration2D<real>::integrateQuadBoundary(boundaryWeights, KeD1a);
	//compute second integral P(v)*n*u = B'*D'*Phi -> left hand side
	array<Matrix8, 4> KeD2a = {
		B1.transpose() * De.transpose() * Phi1,
		B2.transpose() * De.transpose() * Phi2,
		B3.transpose() * De.transpose() * Phi3,
		B4.transpose() * De.transpose() * Phi4,
	};
	Matrix8 KeD2 = Integration2D<real>::integrateQuadBoundary(boundaryWeights, KeD2a);
	//compute third integral -P(v)*n*u_0 -> right hand side
	array<Vector8, 4> FeD1a = {
		B1.transpose() * De.transpose() * Phi1 * u0,
		B2.transpose() * De.transpose() * Phi2 * u0,
		B3.transpose() * De.transpose() * Phi3 * u0,
		B4.transpose() * De.transpose() * Phi4 * u0,
	};
	Vector8 FeD1 = Integration2D<real>::integrateQuadBoundary(boundaryWeights, FeD1a);
	//compute forth integral rho*u*v -> left hand side
	array<Matrix8, 4> KeD3a = {
		Phi1.transpose() * Phi1,
		Phi2.transpose() * Phi2,
		Phi3.transpose() * Phi3,
		Phi4.transpose() * Phi4
	};
	Matrix8 KeD3 = rho * Integration2D<real>::integrateQuadBoundary(boundaryWeights, KeD3a);
	//compute fifth integral -rho*u_0*v -> right hand side
	array<Vector8, 4> FeD2a = {
		Phi1.transpose() * Phi1 * u0,
		Phi2.transpose() * Phi2 * u0,
		Phi3.transpose() * Phi3 * u0,
		Phi4.transpose() * Phi4 * u0
	};
	Vector8 FeD2 = rho * Integration2D<real>::integrateQuadBoundary(boundaryWeights, FeD2a);
	//assemble print result
	KeDirichlet = -KeD1 - KeD2 - KeD3;
	FeDirichlet = FeD1 + FeD2;
	CI_LOG_V("Dirichlet boundary, KeDirichlet:\n" << KeDirichlet);
	CI_LOG_V("Dirichlet boundary, FeDirichlet:\n" << FeDirichlet.transpose());
}

void ar::SoftBodyGrid2D::assembleStiffnessMatrix(
    const Matrix3& C, real materialMu, real materialLambda,
    MatrixX& K, VectorX* F, const Eigen::MatrixXi& posToIndex,
	const VectorX& currentDisplacements) const
{
    for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
        int x = xy / (resolution_ - 1);
        int y = xy % (resolution_ - 1);
        Matrix8 Ke;
        Vector8 Fe;

		if (!computeElementMatrix(x, y, Ke, Fe, materialMu, materialLambda,
			getRotationCorrection(), isHardDirichletBoundaries(),
			posToIndex, currentDisplacements))
		{
			continue; //completely outside
		}

        //combine Me, Ke, Fe into full matrix
        int mapping[4] = {
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y) : (x + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y) : ((x + 1) + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y + 1) : (x + (y + 1) * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y + 1) : ((x + 1) + (y + 1) * resolution_)
        };
        bool dirichlet[4] = {
            gridDirichlet_(x, y),
            gridDirichlet_(x + 1, y),
            gridDirichlet_(x, y + 1),
            gridDirichlet_(x + 1, y + 1)
        };
        for (int i = 0; i < 4; ++i) {
            if (gridSettings_.hardDirichletBoundaries_ && dirichlet[i])
            {
                K.block<2, 2>(2 * mapping[i], 2 * mapping[i]) = Matrix<real, 2, 2>::Identity();
                continue;
            }
            for (int j = 0; j < 4; ++j) {
                if (gridSettings_.hardDirichletBoundaries_ && dirichlet[j]) continue;
                K.block<2, 2>(2 * mapping[i], 2 * mapping[j]) += Ke.block<2, 2>(2 * i, 2 * j);
            }
            if (F)
                F->segment<2>(2 * mapping[i]) += Fe.segment<2>(2 * i);
        }

        //if (worker->isInterrupted()) return VectorX();
    }
}

ar::real ar::SoftBodyGrid2D::assembleMassMatrix(VectorX& Mvec, const Eigen::MatrixXi& posToIndex) const
{
	real totalMass = 0;
    for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
        int x = xy / (resolution_ - 1);
        int y = xy % (resolution_ - 1);
        Vector8 MeVec;
        
        //assemble SDF array
        array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
        if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
            //completely outside
            continue;
        }
        real area = size_.prod();

        //The mass (assuming constant mass, this array is the same for all elements)
        real mass = getMass();
        array<Matrix8, 4> Mex = {
            mass * Phi1.transpose() * Phi1,
            mass * Phi2.transpose() * Phi2,
            mass * Phi3.transpose() * Phi3,
            mass * Phi4.transpose() * Phi4
        };
        //The mass matrix really is purely diagonal.
        //TODO: integrate then also only over vectors
        Matrix8 Me = Integration2D<real>::integrateQuad(size_, sdfs, Mex);
        CI_LOG_V("mass Me:\n" << Me);
        MeVec = Me.diagonal();
		totalMass += MeVec.sum() / 2;

        //combine Me, Ke, Fe into full matrix
        int mapping[4] = {
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y) : (x + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y) : ((x + 1) + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y + 1) : (x + (y + 1) * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y + 1) : ((x + 1) + (y + 1) * resolution_)
        };
        bool dirichlet[4] = {
            gridDirichlet_(x, y),
            gridDirichlet_(x + 1, y),
            gridDirichlet_(x, y + 1),
            gridDirichlet_(x + 1, y + 1)
        };
        for (int i = 0; i < 4; ++i) {
            Mvec.segment<2>(2 * mapping[i]) += MeVec.segment<2>(2 * i);
        }
    }
	return totalMass;
}

ar::SoftBodyGrid2D::bgrid_t ar::SoftBodyGrid2D::computeValidCells(const grid_t& uGridX, const grid_t& uGridY,
    const Eigen::MatrixXi& posToIndex) const
{
#if 1
    //valid vertices are all vertices that are computed
    GridUtils2D::bgrid_t validCells = posToIndex.array() >= 0;
#if 0
    //fix an issue that in certain cases, the displacements are zero, even if the node is marked as valid
    //This happens if the node is on the edge of the object, outside of the actual object
    for (int y = 0; y < validCells.cols(); ++y)
    {
        for (int x = 0; x < validCells.rows(); ++x)
        {
            if (validCells(x, y) && uGridX(x, y) == 0 && uGridY(x, y) == 0
                && ((x > 0 && !validCells(x - 1, y) && y > 0 && !validCells(x, y - 1))
                    || (x > 0 && !validCells(x - 1, y) && y < validCells.cols() - 1 && !validCells(x, y + 1))
                    || (x < validCells.rows() - 1 && !validCells(x + 1, y) && y>0 && !validCells(x, y - 1))
                    || (x < validCells.rows() - 1 && !validCells(x + 1, y) && y < validCells.cols() - 1 && !validCells(x, y + 1))
                    ))
            {
                validCells(x, y) = false;
            }
        }
    }
#endif

#else
    //valid vertices are only inside vertices. Seems more stable
    GridUtils2D::bgrid_t validCells = sdf_ <= 0;
#endif
    return validCells;
}

void ar::SoftBodyGrid2D::augmentMatrixWithImplicitDiffusion(MatrixX & K) const
{
    //for numerical stability, scale the diffusion terms so that it matches the matrix K
    real epsilon = K.array().cwiseAbs().maxCoeff() * 0.01;
    CI_LOG_V("diffusion epsilon: " << epsilon);
    //plug in diffusion terms into the matrix
    for (int x = 0; x < resolution_; ++x) {
        for (int y = 0; y < resolution_; ++y) {
            int j = x + y * resolution_;

            ////check if all 4 neighboring cells are outside, i.e. the matrix entries are empty
            bool outside = K(2 * j, 2 * j) == 0; //inside cells always have something on the diagonal

            if (outside) {
                int count = 0;
                if (x > 0) {
                    count++;
                    int i = (x - 1) + y * resolution_;
                    K(2 * j, 2 * i) -= epsilon;
                    K(2 * j + 1, 2 * i + 1) -= epsilon;
                }
                if (x < resolution_ - 1) {
                    count++;
                    int i = (x + 1) + y * resolution_;
                    K(2 * j, 2 * i) -= epsilon;
                    K(2 * j + 1, 2 * i + 1) -= epsilon;
                }
                if (y > 0) {
                    count++;
                    int i = x + (y - 1) * resolution_;
                    K(2 * j, 2 * i) -= epsilon;
                    K(2 * j + 1, 2 * i + 1) -= epsilon;
                }
                if (y < resolution_ - 1) {
                    count++;
                    int i = x + (y + 1) * resolution_;
                    K(2 * j, 2 * i) -= epsilon;
                    K(2 * j + 1, 2 * i + 1) -= epsilon;
                }
                K(2 * j, 2 * j) += epsilon * count;
                K(2 * j + 1, 2 * j + 1) += epsilon * count;
            }
        }
    }
    //NOTE: The matrix K is not symmetric now!
    //(Diffusion only into invalid cells is not symmetric)
}

std::optional<std::pair<ar::real, ar::Vector2>> ar::SoftBodyGrid2D::getGroundCollisionSimpleNeumann(
	int x, int y, const std::array<real, 4>& phi, const Vector8& ue) const
{
	//the boundary length of the current cell
	real boundary = Integration2D<real>::integrateQuadBoundary(size_,
		std::array<real, 4>(phi),
		std::array<real, 4>({ 1.0f, 1.0f, 1.0f, 1.0f }));
	assert(boundary > 0);

	//the actual position of the element
	Vector8 xe;
	xe << Vector2(x*size_.x(), y*size_.y()), Vector2((x + 1)*size_.x(), y*size_.y()),
		Vector2(x*size_.x(), (y + 1)*size_.y()), Vector2((x + 1)*size_.x(), (y + 1)*size_.y());
	xe += ue;
	std::array<Vector2, 4> corners = { Vector2(xe.segment<2>(0)), Vector2(xe.segment<2>(2)), Vector2(xe.segment<2>(4)), Vector2(xe.segment<2>(6)) };

	//the two points of the current cell's boundary that cut the cell sides
	auto points1 = Integration2D<ar::real>::getIntersectionPoints(phi, corners);
	assert(points1.has_value());

	//check them against collisions
	auto col1 = groundCollision(points1->first);
	auto col2 = groundCollision(points1->second);
	Vector2 normal = (col1.second + col2.second) * 0.5;

	bool i1 = std::get<0>(col1) <= 0;
	bool i2 = std::get<0>(col2) <= 0;
	if (i1 && i2)
		return std::make_pair(boundary, normal); //boundary fully inside the ground
	else if (!i1 && !i2)
		return {}; //no collision with the ground
	 //else: partial collision

	std::array<real, 4> distances = {
		groundCollision(corners[0]).first,
		groundCollision(corners[1]).first,
		groundCollision(corners[2]).first,
		groundCollision(corners[3]).first
	};
	auto points2 = Integration2D<ar::real>::getIntersectionPoints(std::array<ar::real, 4>(distances), corners);
	assert(points2.has_value()); //we only have a partial collision, so there should be points here

	geom2d::line l1 = geom2d::lineByPoints({ points1->first.x(), points1->first.y() }, { points1->second.x(), points1->second.y() });
	geom2d::line l2 = geom2d::lineByPoints({ points2->first.x(), points2->first.y() }, { points2->second.x(), points2->second.y() });
	geom2d::point i = geom2d::intersection(l1, l2);
	float len;
	if (i1)
		len = (points1->first - ar::Vector2(i.first, i.second)).norm();
	else
		len = (points1->second - ar::Vector2(i.first, i.second)).norm();
	return make_pair(boundary * (len / (points1->first - points1->second).norm()), normal);

}

std::array<ar::Vector2, 4> ar::SoftBodyGrid2D::implicitCollisionForces(
	int x, int y, const std::array<real, 4>& sdfs,
	const std::array<Vector2, 4>& positions, const std::array<Vector2, 4>& velocities, 
	bool implicit, real groundHeight, real groundAngle, real groundStiffness, real softminAlpha,
	real timestep, real newmarkTheta) const
{
	//forces
	std::array<ar::Vector2, 4> forces {
		Vector2::Zero(),
		Vector2::Zero(),
		Vector2::Zero(),
		Vector2::Zero()
	};

	//the two points of the current cell's boundary that cut the cell sides
	auto points1 = Integration2D<ar::real>::getIntersectionPoints(sdfs, positions);
	if (!points1.has_value()) return forces;

	//the basis points for interpolation
	auto points2 = Integration2D<ar::real>::getIntersectionPoints(sdfs, { Vector2(0,0), Vector2(1,0), Vector2(0,1), Vector2(1,1) }).value();

	//check them against collisions
	auto col1 = groundCollision(points1->first, groundHeight, groundAngle);
	auto col2 = groundCollision(points1->second, groundHeight, groundAngle);
	Vector2 normal = col1.second; //or col2.second, they are the same
	real dist1 = col1.first;
	real dist2 = col2.first;

	//compute the forces for these two points
	Vector2 f1, f2;
	if (!implicit) { //CollisionResolution::SPRING_EXPLICT
		f1 = (-groundStiffness * ar::utils::softmin(dist1, softminAlpha)) * normal;
		f2 = (-groundStiffness * ar::utils::softmin(dist2, softminAlpha)) * normal;
	}
	else //CollisionResolution::SPRING_IMPLICIT
	{
		//current forces
		Vector2 f1Current = (-groundStiffness * ar::utils::softmin(dist1, softminAlpha)) * normal;
		Vector2 f2Current = (-groundStiffness * ar::utils::softmin(dist2, softminAlpha)) * normal;
		//velocity at the two points
		Vector2 vel1 = ar::utils::bilinearInterpolate(velocities, points2.first.x(), points2.first.y());
		Vector2 vel2 = ar::utils::bilinearInterpolate(velocities, points2.second.x(), points2.second.y());
		//time derivative
		Vector2 f1Dt = (-groundStiffness * ar::utils::softminDx(dist1, softminAlpha) * groundCollisionDt(vel1, groundHeight, groundAngle)) * normal;
		Vector2 f2Dt = (-groundStiffness * ar::utils::softminDx(dist2, softminAlpha) * groundCollisionDt(vel2, groundHeight, groundAngle)) * normal;
		//approximate next force
		Vector2 f1Next = f1Current + timestep * f1Dt;
		Vector2 f2Next = f2Current + timestep * f2Dt;
		//blend them together to the final force
		f1 = newmarkTheta * f1Next + (1 - newmarkTheta) * f1Current;
		f2 = newmarkTheta * f2Next + (1 - newmarkTheta) * f2Current;
	}

	//blend them into the forces
	forces[0] += (1 - points2.first.x()) * (1 - points2.first.y()) * 0.5 * f1;
	forces[1] += points2.first.x() * (1 - points2.first.y()) * 0.5 * f1;
	forces[2] += (1 - points2.first.x()) * points2.first.y() * 0.5 * f1;
	forces[3] += points2.first.x() * points2.first.y() * 0.5 * f1;
	forces[0] += (1 - points2.second.x()) * (1 - points2.second.y()) * 0.5 * f2;
	forces[1] += points2.second.x() * (1 - points2.second.y()) * 0.5 * f2;
	forces[2] += (1 - points2.second.x()) * points2.second.y() * 0.5 * f2;
	forces[3] += points2.second.x() * points2.second.y() * 0.5 * f2;

	return forces;
}

ar::VectorX ar::SoftBodyGrid2D::resolveCollisions(const Eigen::MatrixXi& posToIndex,
	int dof, const VectorX& currentDisplacements, const VectorX& currentVelocities) const
{
	VectorX forces = VectorX::Zero(2 * dof);

	real totalBoundaryLength = 0; //the total length of the boundary that collides

	for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
		int x = xy / (resolution_ - 1);
		int y = xy % (resolution_ - 1);

		//assemble SDF array
		array<real, 4> sdfs = { sdf_(x, y), sdf_(x + 1,y), sdf_(x, y + 1), sdf_(x + 1, y + 1) };
		if (outside(sdfs[0]) && outside(sdfs[1]) && outside(sdfs[2]) && outside(sdfs[3])) {
			//completely outside
			continue;
		}
		if (inside(sdfs[0]) && inside(sdfs[1]) && inside(sdfs[2]) && inside(sdfs[3])) {
			//completely inside -> no boundary
			continue;
		}

		const auto getOrZero = [&posToIndex, &currentDisplacements](int x, int y) {return posToIndex(x, y) >= 0 ? currentDisplacements.segment<2>(2 * posToIndex(x, y)) : Vector2(0, 0); };
        const auto getOrZeroV = [&posToIndex, &currentVelocities](int x, int y) {return posToIndex(x, y) >= 0 ? currentVelocities.segment<2>(2 * posToIndex(x, y)) : Vector2(0, 0); };
		Vector8 ue;
		ue << getOrZero(x, y), getOrZero(x + 1, y), getOrZero(x, y + 1), getOrZero(x + 1, y + 1);

        switch (getCollisionResolution())
        {
        case CollisionResolution::POST_REPAIR:
            {
            CI_LOG_I("PostRepair not supported on the grid");
            break;
            }
        case CollisionResolution::REPAIR_PLUS_NEUMANN:
            {
            CI_LOG_I("Repair+Neumann not supported on the grid");
            break;
            }
        case CollisionResolution::SIMPLE_NEUMANN:
            {
            auto collision = getGroundCollisionSimpleNeumann(x, y, sdfs, ue);
            if (collision.has_value())
            {
                real len = collision->first;        //boundary length
                Vector2 normal = collision->second; //collision normal
                totalBoundaryLength += len;
                if (posToIndex(x, y) >= 0) forces.segment<2>(2 * posToIndex(x, y)) = normal; //TODO: better += and then normalize later?
                if (posToIndex(x + 1, y) >= 0) forces.segment<2>(2 * posToIndex(x + 1, y)) = normal;
                if (posToIndex(x, y + 1) >= 0) forces.segment<2>(2 * posToIndex(x, y + 1)) = normal;
                if (posToIndex(x + 1, y + 1) >= 0) forces.segment<2>(2 * posToIndex(x + 1, y + 1)) = normal;
            }
            break;
            }
        case CollisionResolution::SPRING_EXPLICIT:
        case CollisionResolution::SPRING_IMPLICIT:
            {
            //the actual position of the element
            Vector8 xe;
            xe << Vector2(x*size_.x(), y*size_.y()), Vector2((x + 1)*size_.x(), y*size_.y()),
                Vector2(x*size_.x(), (y + 1)*size_.y()), Vector2((x + 1)*size_.x(), (y + 1)*size_.y());
            xe += ue;
            std::array<Vector2, 4> corners = { Vector2(xe.segment<2>(0)), Vector2(xe.segment<2>(2)), Vector2(xe.segment<2>(4)), Vector2(xe.segment<2>(6)) };
			std::array<Vector2, 4> velocities = { getOrZeroV(x, y), getOrZeroV(x + 1, y), getOrZeroV(x, y + 1), getOrZeroV(x + 1, y + 1) };

			std::array<Vector2, 4> forcesArray = implicitCollisionForces(x, y, sdfs,
				corners, velocities, getCollisionResolution() == CollisionResolution::SPRING_IMPLICIT,
				getGroundPlaneHeight(), getGroundPlaneAngle(),
				getGroundStiffness(), getCollisionSoftmaxAlpha(),
				getTimestep(), TimeIntegrator_Newmark1::DefaultTheta);

			if (posToIndex(x, y) >= 0)      forces.segment<2>(2 * posToIndex(x, y))     += forcesArray[0];
			if (posToIndex(x+1, y) >= 0)    forces.segment<2>(2 * posToIndex(x+1, y))   += forcesArray[1];
			if (posToIndex(x, y+1) >= 0)    forces.segment<2>(2 * posToIndex(x, y+1))   += forcesArray[2];
			if (posToIndex(x+1, y+1) >= 0)  forces.segment<2>(2 * posToIndex(x+1, y+1)) += forcesArray[3];

            break;
            }
        }
		
	}

    if (getCollisionResolution() == CollisionResolution::SIMPLE_NEUMANN) {
        real scaling = ((1 - getCollisionVelocityDamping()) / getTimestep());
        forces *= scaling;
    }
	return forces;
}

std::pair<ar::VectorX, ar::VectorX> ar::SoftBodyGrid2D::solveImplDense(bool dynamic, int dof, const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements, BackgroundWorker* worker)
{
    worker->setStatus("SoftBodyGrid2D-dense: prepare matrices");

    //allocate matrices (dense at the moment)
    VectorX Mvec = VectorX::Zero(dof * 2);
    MatrixX K = MatrixX::Zero(dof * 2, dof * 2);
    VectorX f = VectorX::Zero(dof * 2);

    //assemble factor matrix C
    real mu = getMaterialMu();
    real lambda = getMaterialLambda();
    Matrix3 C = computeMaterialMatrix(mu, lambda);
    CI_LOG_V("C:\n" << C);

    //assemble matrices
    worker->setStatus("SoftBodyGrid2D-dense: assemble matrices");
    assembleForceVector(f, posToIndex, collisionForces_);
    assembleStiffnessMatrix(C, mu, lambda, K, &f, posToIndex, currentDisplacements);
    totalMass_  = assembleMassMatrix(Mvec, posToIndex);
    if (worker->isInterrupted()) return {};

    //Implicit diffusion
    if (!gridSettings_.explicitDiffusion_)
    {
        worker->setStatus("SoftBodyGrid2D-dense: add diffusion to matrix");
        augmentMatrixWithImplicitDiffusion(K);
        if (worker->isInterrupted()) return {};
    }

#if 0
    {
        //TEST: compute min and max entry
        real minEntry = std::numeric_limits<real>::max();
        real maxEntry = 0;
        for (int i=0; i<K.rows(); ++i)
            for (int j=0; j<K.cols(); ++j)
                if (K(i,j) != 0)
                {
                    minEntry = std::min(minEntry, std::abs(K(i, j)));
                    maxEntry = std::max(maxEntry, std::abs(K(i, j)));
                }
        CI_LOG_I("Matrix min entry: " << minEntry << ", max entry: " << maxEntry);
        JacobiSVD<MatrixX> svd(K);
        real cond = svd.singularValues()(0)
            / svd.singularValues()(svd.singularValues().size() - 1);
        CI_LOG_I("Condition number of the matrix: " << cond);
        //TEST: check for extreme values in the SDF
        real sdfZeroPos = 1;
        real sdfZeroNeg = -1;
        real sdfNegOnePos = 0;
        real sdfNegOneNeg = -2;
        for (int x=0; x<resolution_; ++x)
            for (int y=0; y<resolution_; ++y) {
                real sdf = sdf_(x, y);
                if (sdf <= 0) sdfZeroNeg = std::max(sdfZeroNeg, sdf);
                if (sdf >= 0) sdfZeroPos = std::min(sdfZeroPos, sdf);
                if (sdf <= -1) sdfNegOneNeg = std::max(sdfNegOneNeg, sdf);
                if (sdf >= -1) sdfNegOnePos = std::min(sdfNegOnePos, sdf);
            }
        CI_LOG_I("SDF values closest to zero: " << sdfZeroNeg << " : " << sdfZeroPos);
        CI_LOG_I("SDF values closest to -1:   " << sdfNegOneNeg << " : " << sdfNegOnePos);
    }
#endif

#if 0
    {
        worker->setStatus("SoftBodyGrid2D-dense: save matrix to file");
        //save it to a file
        const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
        fstream os1("../../Kdense.csv", fstream::out | fstream::trunc);
        os1 << K.format(CSVFormat);
        os1.close();
        fstream os2("../../Fdense.csv", fstream::out | fstream::trunc);
        os2 << f.format(CSVFormat);
        os2.close();
    }
#endif

    CI_LOG_V("K:\n" << K);
    CI_LOG_V("f:\n" << f.transpose());

    //solve it
    if (integrator_ == nullptr) {
        integrator_ = TimeIntegrator::createIntegrator(getTimeIntegrator(), dof * 2);
        if (worker->isInterrupted()) return {};
    }
    integrator_->setDenseLinearSolver(getDenseLinearSolver());
    VectorX solution;
    VectorX velocity;
    if (!dynamic) {
        //static solution
        worker->setStatus("SoftBodyGrid2D-dense: solve for static solution");
        solution = integrator_->solveDense(K, f);
        velocity = VectorX::Zero(solution.size());
        CI_LOG_V("Static solution:\n" << Map<Matrix2X>(solution.data(), 2, dof));
    }
    else {
        //dynamic solution
        worker->setStatus("SoftBodyGrid2D-dense: solve for dynamic solution");

        //Raylight Damping C = alpha*M + beta*K
        MatrixX M = Mvec.asDiagonal();
        MatrixX C = getDampingAlpha() * M + getDampingBeta() * K;
        CI_LOG_V("M:\n" << Mvec.transpose());
        CI_LOG_V("C:\n" << C);

        //integrate / solve
        integrator_->performStep(getTimestep(), Mvec, C, K, f);
        solution = integrator_->getCurrentU();
        velocity = integrator_->getCurrentUDot();
        CI_LOG_V("Dynamic solution:\n" << Map<Matrix2X>(solution.data(), 2, dof));
    }
    if (worker->isInterrupted()) return {};
    return std::make_pair(solution, velocity);
}

#if SOFT_BODY_SUPPORT_SPARSE_MATRICES==1

void ar::SoftBodyGrid2D::augmentMatrixWithImplicitDiffusion(const std::vector<bool>& insideCells, real epsilon, std::vector<Eigen::Triplet<real>>& Kentries) const
{
    CI_LOG_V("diffusion epsilon: " << epsilon);
    //plug in diffusion terms into the matrix
    for (int x = 0; x < resolution_; ++x) {
        for (int y = 0; y < resolution_; ++y) {
            int j = x + y * resolution_;

            bool outside = !insideCells[j];

            if (outside) {
                int count = 0;
                if (x > 0) {
                    count++;
                    int i = (x - 1) + y * resolution_;
                    Kentries.emplace_back(2 * j, 2 * i, -epsilon);
                    Kentries.emplace_back(2 * j + 1, 2 * i + 1, -epsilon);
                }
                if (x < resolution_ - 1) {
                    count++;
                    int i = (x + 1) + y * resolution_;
                    Kentries.emplace_back(2 * j, 2 * i, -epsilon);
                    Kentries.emplace_back(2 * j + 1, 2 * i + 1, -epsilon);
                }
                if (y > 0) {
                    count++;
                    int i = x + (y - 1) * resolution_;
                    Kentries.emplace_back(2 * j, 2 * i, -epsilon);
                    Kentries.emplace_back(2 * j + 1, 2 * i + 1, -epsilon);
                }
                if (y < resolution_ - 1) {
                    count++;
                    int i = x + (y + 1) * resolution_;
                    Kentries.emplace_back(2 * j, 2 * i, -epsilon);
                    Kentries.emplace_back(2 * j + 1, 2 * i + 1, -epsilon);
                }
                Kentries.emplace_back(2 * j, 2 * j, count * epsilon);
                Kentries.emplace_back(2 * j + 1, 2 * j + 1, count * epsilon);
            }
        }
    }
    //NOTE: The matrix K is not symmetric now!
    //(Diffusion only into invalid cells is not symmetric)
}

std::pair<ar::VectorX, ar::VectorX> ar::SoftBodyGrid2D::solveImplSparse(bool dynamic, int dof, const Eigen::MatrixXi& posToIndex, const VectorX& currentDisplacements, BackgroundWorker* worker)
{
    worker->setStatus("SoftBodyGrid2D-sparse: prepare matrices");
    typedef TimeIntegrator::SparseMatrixRowMajor SMatrix;

    //allocate matrices (dense at the moment)
    VectorX Mvec = VectorX::Zero(dof * 2);
    SMatrix K(dof * 2, dof * 2);
    std::vector<Triplet<real>> Kentries;
    Kentries.reserve(6 * 2 * dof);
    VectorX f = VectorX::Zero(dof * 2);
    std::vector<bool> insideCells(dof, false);

    //assemble factor matrix C
    real mu = getMaterialMu();
    real lambda = getMaterialLambda();
    Matrix3 C;
    C << (2 * mu + lambda), lambda, 0,
        lambda, (2 * mu + lambda), 0,
        0, 0, lambda;

    worker->setStatus("SoftBodyGrid2D-sparse: assemble matrix entries");
    real implicitDiffusionEpsilon = 0;
	totalMass_ = 0;
#pragma omp parallel for schedule(dynamic)
    for (int xy = 0; xy < (resolution_ - 1)*(resolution_ - 1); ++xy) {
        int x = xy / (resolution_ - 1);
        int y = xy % (resolution_ - 1);
        Vector8 MeVec;
        Matrix8 Ke;
        Vector8 Fe;
        if (!computeElementMatrix_old(x, y, size_, C, Ke, MeVec, Fe, currentDisplacements)) continue;

        //combine Me, Ke, Fe into full matrix
        int mapping[4] = {
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y) : (x + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y) : ((x + 1) + y * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x, y + 1) : (x + (y + 1) * resolution_),
            gridSettings_.explicitDiffusion_ ? posToIndex(x + 1, y + 1) : ((x + 1) + (y + 1) * resolution_)
        };
        bool dirichlet[4] = {
            gridDirichlet_(x, y),
            gridDirichlet_(x + 1, y),
            gridDirichlet_(x, y + 1),
            gridDirichlet_(x + 1, y + 1)
        };
#pragma omp critical
		totalMass_ += MeVec.sum() / 2;
        for (int i = 0; i < 4; ++i) {
            Mvec.segment<2>(2 * mapping[i]) += MeVec.segment<2>(2 * i);
            if (gridSettings_.hardDirichletBoundaries_ && dirichlet[i])
            {
                Kentries.emplace_back(2 * mapping[i], 2 * mapping[i], 1);
                Kentries.emplace_back(2 * mapping[i] + 1, 2 * mapping[i] + 1, 1);
                continue;
            }
            for (int j = 0; j < 4; ++j) {
                if (gridSettings_.hardDirichletBoundaries_ && dirichlet[j]) continue;
                Kentries.emplace_back(2 * mapping[i], 2 * mapping[j], Ke(2 * i, 2 * j));
                Kentries.emplace_back(2 * mapping[i] + 1, 2 * mapping[j], Ke(2 * i + 1, 2 * j));
                Kentries.emplace_back(2 * mapping[i], 2 * mapping[j] + 1, Ke(2 * i, 2 * j + 1));
                Kentries.emplace_back(2 * mapping[i] + 1, 2 * mapping[j] + 1, Ke(2 * i + 1, 2 * j + 1));
            }
            implicitDiffusionEpsilon = max({ implicitDiffusionEpsilon, abs(Ke(2 * i, 2 * i)), abs(Ke(2 * i + 1, 2 * i + 1)) });
            f.segment<2>(2 * mapping[i]) += Fe.segment<2>(2 * i);
                
            insideCells[mapping[i]] = true;
        }

        //if (worker->isInterrupted()) return VectorX();
    }

    //Implicit diffusion
    if (!gridSettings_.explicitDiffusion_)
    {
        worker->setStatus("SoftBodyGrid2D-sparse: add diffusion to matrix");
        augmentMatrixWithImplicitDiffusion(insideCells, implicitDiffusionEpsilon * 0.01, Kentries);
        if (worker->isInterrupted()) return {};
    }

    worker->setStatus("SoftBodyGrid2D-sparse: create sparse matrix");
    K.setFromTriplets(Kentries.begin(), Kentries.end());

#if 0
    {
        //save it to a file
        const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
        MatrixX Kdense = K;
        fstream os("../../Ksparse.csv", fstream::out | fstream::trunc);
        os << Kdense.format(CSVFormat);
        os.close();
    }
#endif

    //solve it
    if (integrator_ == nullptr) {
        integrator_ = TimeIntegrator::createIntegrator(getTimeIntegrator(), dof * 2);
        if (worker->isInterrupted()) return {};
    }
    integrator_->setSparseLinearSolver(getSparseLinearSolver());
    integrator_->setSparseSolveIterations(getSparseSolverIterations());
    integrator_->setSparseSolveTolerance(getSparseSolverTolerance());
    VectorX solution;
    VectorX velocity;
    if (dynamic && gridSettings_.explicitDiffusion_) {
        CI_LOG_W("dynamic cutFEM system only works with implicit diffusion");
    }
    if (!dynamic || gridSettings_.explicitDiffusion_) {
        //static solution
        worker->setStatus("SoftBodyGrid2D-sparse: solve for static solution");
        solution = integrator_->solveDense(K, f);
        velocity = VectorX::Zero(solution.size());
        CI_LOG_V("Static solution:\n" << Map<Matrix2X>(solution.data(), 2, dof));
    }
    else {
        //dynamic solution
        worker->setStatus("SoftBodyGrid2D-sparse: solve for dynamic solution");

        //Raylight Damping C = alpha*M + beta*K
        SMatrix M(dof * 2, dof * 2);
        M.setIdentity();
        M.diagonal() = Mvec;
        SMatrix C = getDampingAlpha() * M + getDampingBeta() * K;

        //integrate / solve
        //TODO: why does this not work??
        //integrator_->performStep(timestep, Mvec, C, K, f);
        //solution = integrator_->getCurrentU();
        CI_LOG_V("Dynamic solution:\n" << Map<Matrix2X>(solution.data(), 2, dof));
    }
    if (worker->isInterrupted()) return {};
    return std::make_pair(solution, velocity);
}

ar::SoftBodyGrid2D ar::SoftBodyGrid2D::CreateBar(
	const Vector2 & rectCenter, const Vector2 & rectHalfSize, int resolution, bool fitObjectToGrid, 
	bool enableDirichlet, const Vector2 & dirichletBoundaries, const Vector2 & neumannBoundaries)
{
    SoftBodyGrid2D gridSolver;
    gridSolver.setGridResolution(resolution);

    //for testing if Grid==Mesh: fit to grid
    Vector2 center = rectCenter;
    Vector2 halfSize = rectHalfSize;
    if (fitObjectToGrid)
    {
        halfSize = (rectHalfSize * resolution + Vector2(0.9, 0.9)).array().floor().matrix() / resolution + Vector2(0.000001, 0.000001);
        center = ((rectCenter * resolution).array().floor().matrix() + Vector2(0.5, 0.5)) / resolution;
    }
    //create SDF
    grid_t sdf = grid_t::Zero(resolution, resolution);
    int numberOfClamped = 0;
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            Vector2d pos((x + 0.5) / resolution, (y + 0.5) / resolution);
            double val = 0;
            if (pos.x() >= center.x() - halfSize.x() && pos.x() <= center.x() + halfSize.x()
                && pos.y() >= center.y() - halfSize.y() && pos.y() <= center.y() + halfSize.y())
            {
                //inside
                val = -10000;
                val = std::max(val, center.x() - halfSize.x() - pos.x());
                val = std::max(val, pos.x() - center.x() - halfSize.x());
                val = std::max(val, center.y() - halfSize.y() - pos.y());
                val = std::max(val, pos.y() - center.y() - halfSize.y());
                //neumann boundaries
                if (pos.x() + 1.0 / resolution > center.x() + halfSize.x()
                    && pos.y() - 1.0 / resolution < center.y() - halfSize.y()) {
                    gridSolver.addNeumannBoundary(x, y, neumannBoundaries);
                }
                //dirichlet bounds
                if (pos.x() - 1.0 / resolution <= center.x() - halfSize.x() && enableDirichlet) {
                    gridSolver.addDirichletBoundary(x, y, Vector2(0, 0));
                }
            }
            else
            {
                //outside
                if (pos.x() < center.x() - halfSize.x() && pos.y() < center.y() - halfSize.y())
                    val = (pos - Vector2d(center.x() - halfSize.x(), center.y() - halfSize.y())).norm();
                else if (pos.x() < center.x() - halfSize.x() && pos.y() > center.y() + halfSize.y())
                    val = (pos - Vector2d(center.x() - halfSize.x(), center.y() + halfSize.y())).norm();
                else if (pos.x() > center.x() + halfSize.x() && pos.y() < center.y() - halfSize.y())
                    val = (pos - Vector2d(center.x() + halfSize.x(), center.y() - halfSize.y())).norm();
                else if (pos.x() > center.x() + halfSize.x() && pos.y() > center.y() + halfSize.y())
                    val = (pos - Vector2d(center.x() + halfSize.x(), center.y() + halfSize.y())).norm();
                else {
                    val = std::max(val, center.x() - halfSize.x() - pos.x());
                    val = std::max(val, pos.x() - center.x() - halfSize.x());
                    val = std::max(val, center.y() - halfSize.y() - pos.y());
                    val = std::max(val, pos.y() - center.y() - halfSize.y());
                }
            }
            val *= resolution;
            static const real epsilon = 0.0001;
            if (abs(val) < epsilon) {
                val = 0;
                numberOfClamped++;
            }
            sdf(x, y) = val;
        }
    }
    gridSolver.setSDF(sdf);

    return gridSolver;
}

ar::SoftBodyGrid2D ar::SoftBodyGrid2D::CreateTorus(
	real torusOuterRadius, real torusInnerRadius, int resolution, 
	bool enableDirichlet, const Vector2 & dirichletBoundaries, const Vector2 & neumannBoundaries)
{
    SoftBodyGrid2D gridSolver;

    gridSolver.setGridResolution(resolution);

    //create SDF
    grid_t sdf = grid_t::Zero(resolution, resolution);
    Vector2d torusCenter(0.5, 1 - ((torusOuterRadius + torusInnerRadius) * 1.5));
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            Vector2d pos((x + 0.5) / resolution, (y + 0.5) / resolution);
            double val = 0;
            //compute sdf to torus
            double dist = (pos - torusCenter).norm();
            if (torusInnerRadius >= torusOuterRadius) {
                //circle
                val = dist - torusOuterRadius - torusInnerRadius;
            }
            else {
                //disk / torus
                if (dist >= torusOuterRadius + torusInnerRadius)
                    val = dist - torusOuterRadius - torusInnerRadius; //outside
                else if (dist <= torusOuterRadius - torusInnerRadius)
                    val = torusOuterRadius - torusInnerRadius - dist; //outside
                else if (dist > torusOuterRadius)
                    val = dist - torusOuterRadius - torusInnerRadius; //inside
                else
                    val = torusOuterRadius - dist - torusInnerRadius; //inside
            }
            sdf(x, y) = val * resolution;
        }
    }
    gridSolver.setSDF(sdf);

    //Dirichlet boundaries
	if (enableDirichlet) {
		for (int y = resolution - 1; y >= 0; --y)
		{
			bool found = false;
			for (int x = 0; x < resolution; ++x)
			{
				if (sdf(x, y) <= 0)
				{
					gridSolver.addDirichletBoundary(x, y, Vector2(0, 0)); //non-uniform Dirichlet not supported yet
					found = true;
				}
			}
			if (found) break;
		}
	}
    //Neumann boundaries
    for (int y = 0; y < resolution; ++y)
    {
        bool found = false;
        for (int x = 0; x<resolution; ++x)
        {
            if (sdf(x, y) <= 0)
            {
                gridSolver.addNeumannBoundary(x, y, neumannBoundaries);
                found = true;
            }
        }
        if (found) break;
    }

    return gridSolver;
}

#endif
