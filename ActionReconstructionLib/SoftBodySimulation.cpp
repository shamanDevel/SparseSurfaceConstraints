#include "SoftBodySimulation.h"

#include "Utils.h"

const std::string ar::SoftBodySimulation::RotationCorrectionNames[] = {
    "None",
    "Corotation"
};
const std::string ar::SoftBodySimulation::CollisionResolutionNames[] = {
	"PostRepair",
	"NeumannImpulse",
	"Repair+Neumann",
    "Spring (Implicit)",
    "Spring (Explicit)"
};

ar::SoftBodySimulation::Settings::Settings()
    : gravity_(0, -10)
    , youngsModulus_(200)
    , poissonsRatio_(0.45)
    , mass_(1)
    , dampingAlpha_(0.001)
    , dampingBeta_(0.001)
    , materialLambda_(0)
    , materialMu_(0)
    , rotationCorrection_(RotationCorrection::None)
    , timestep_(0.01)
{
}

ar::SoftBodySimulation::SoftBodySimulation()
    : settings_()
    , denseLinearSolverType_(TimeIntegrator::DenseLinearSolver::PartialPivLU)
    , sparseLinearSolverType_(TimeIntegrator::SparseLinearSolver::BiCGSTAB)
    , integratorType_(TimeIntegrator::Integrator::Newmark1)
    , hasSolution_(false)
    , useSparseMatrixes_(false)
    , sparseSolverIterations_(100)
    , sparseSolverTolerance_(1e-5)
{
    setMaterialParameters(settings_.youngsModulus_, settings_.poissonsRatio_);
}

void ar::SoftBodySimulation::computeMaterialParameters(real youngsModulus, real poissonsRatio, real& materialMu,
                                                       real& materialLambda)
{
    materialMu = youngsModulus / (2 * (1 + poissonsRatio));
    materialLambda = (youngsModulus * poissonsRatio) / ((1 + poissonsRatio) * (1 - 2 * poissonsRatio));
}

void ar::SoftBodySimulation::setMaterialParameters(real youngsModulus, real poissonsRatio)
{
    assert(youngsModulus > 0);
    assert(poissonsRatio > 0 && poissonsRatio < 0.5);
    settings_.youngsModulus_ = youngsModulus;
    settings_.poissonsRatio_ = poissonsRatio;
    computeMaterialParameters(youngsModulus, poissonsRatio, settings_.materialMu_, settings_.materialLambda_);
}

void ar::SoftBodySimulation::setMass(real mass)
{
    assert(mass > 0);
    settings_.mass_ = mass;
}

void ar::SoftBodySimulation::setDamping(real dampingOnMass, real dampingOnStiffness)
{
    assert(dampingOnMass >= 0);
    assert(dampingOnStiffness >= 0);
    settings_.dampingAlpha_ = dampingOnMass;
    settings_.dampingBeta_ = dampingOnStiffness;
}

void ar::SoftBodySimulation::setTimestep(real timestep)
{
    assert(timestep > 0);
    settings_.timestep_ = timestep;
}

std::pair<ar::real, ar::Vector2> ar::SoftBodySimulation::groundCollision(const Vector2& p) const
{
	return groundCollision(p, getGroundPlaneHeight(), getGroundPlaneAngle());
}

std::pair<ar::real, ar::Vector2> ar::SoftBodySimulation::groundCollision(const Vector2& p, real groundHeight,
	real groundAngle)
{
	real dx = cos(groundAngle);
	real dy = sin(groundAngle);
	real distance = -(dy*p.x() - dx * p.y() - (dy*0.5 - dx * groundHeight));
	Vector2 normal(-dy, dx);
	return { distance, normal };
}

ar::real ar::SoftBodySimulation::groundCollisionDt(const Vector2& pDot, real groundHeight, real groundAngle)
{
    real dx = cos(groundAngle);
    real dy = sin(groundAngle);
    real distanceDt = -dy * pDot.x() + dx * pDot.y();
    return distanceDt;
}

ar::Matrix3 ar::SoftBodySimulation::computeMaterialMatrix(real mu, real lambda)
{
    Matrix3 C;
    C << (2 * mu + lambda), lambda, 0,
        lambda, (2 * mu + lambda), 0,
        0, 0, lambda;
    return C;
}

ar::Matrix2 ar::SoftBodySimulation::polarDecomposition(const Matrix2& F)
{
    //Matrix Animation and Polar Decomposition, Ken Shoemake & Tom Duff, page 3
    Matrix2 m;
    m << F(1,1), -F(1,0),
         -F(0,1), F(0,0);
    Matrix2 R = F + ar::utils::sgn(F.determinant()) * m;
    R.array() /= R.col(0).norm(); //scale to make the column unit vectors

	real det = R.determinant();
    assert(abs(det - 1) <= 0.0001); //Check if it is a rotation matrix (det=1, no mirroring)

    return R;
}

ar::Matrix3 ar::SoftBodySimulation::polarDecomposition(const Matrix3& F, int iterations)
{
    //Corotational Simulation of Deformable Solids, Michael Hauth & Wolfgang Strasser, eq. 4.16 and 4.17
    Matrix3 R = F;
    for (int i=0; i<iterations; ++i)
    {
        R = (0.5*(R + R.transpose().inverse())).eval();
    }
    return R;
}
