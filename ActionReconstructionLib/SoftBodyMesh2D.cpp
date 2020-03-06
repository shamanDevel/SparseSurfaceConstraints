#include "SoftBodyMesh2D.h"

#include <cassert>
#include <cinder/Log.h>
#include <cinder/TriMesh.h>
#include <cinder/app/AppBase.h>

#include "Utils.h"

using namespace Eigen;

void ar::SoftBodyMesh2D::setMesh(const Vector2List & positions, const Vector3iList & indices)
{
    referencePositions_ = positions;
    triIndices_ = indices;
    numNodes_ = (int)positions.size();
    numTris_ = (int)indices.size();
    resetBoundaries();
    resetSolution();
}

void ar::SoftBodyMesh2D::resetBoundaries()
{
    nodeStates_ = Eigen::ArrayXi::Constant(numNodes_, FREE);
    boundaries_.resize(numNodes_);
    for (int i = 0; i < numNodes_; ++i) boundaries_[i].setZero();
    numFreeNodes_ = numNodes_;
    needsReordering_ = true;
    integrator_ = nullptr;
}

void ar::SoftBodyMesh2D::addDirichletBoundary(int node, const Vector2 & displacement)
{
    nodeStates_[node] = DIRICHLET;
    boundaries_[node] = displacement;
}

void ar::SoftBodyMesh2D::addNeumannBoundary(int node, const Vector2 & force)
{
    nodeStates_[node] = NEUMANN;
    boundaries_[node] = force;
}

void ar::SoftBodyMesh2D::resetSolution()
{
    currentPositions_ = referencePositions_;
    currentDisplacements_.resize(numNodes_);
    for (int i = 0; i < numNodes_; ++i) currentDisplacements_[i] = Vector2::Zero();
	currentVelocities_.resize(numNodes_);
	for (int i = 0; i < numNodes_; ++i) currentVelocities_[i] = Vector2::Zero();
    setHasSolution(false);
}

const ar::SoftBodyMesh2D::Vector2List & ar::SoftBodyMesh2D::getCurrentPositions() const
{
    assert(hasSolution());
    return currentPositions_;
}

const ar::SoftBodyMesh2D::Vector2List & ar::SoftBodyMesh2D::getCurrentDisplacements() const
{
    assert(hasSolution());
    return currentDisplacements_;
}

const ar::SoftBodyMesh2D::Vector2List & ar::SoftBodyMesh2D::getCurrentVelocities() const
{
	assert(hasSolution());
	return currentVelocities_;
}

ar::real ar::SoftBodyMesh2D::computeElasticEnergy(const Vector2List& displacements)
{
	real energy = 0;

	Matrix3 C;
	real mu = getMaterialMu();
	real lambda = getMaterialLambda();
	C << (2 * mu + lambda), lambda, 0,
		lambda, (2 * mu + lambda), 0,
		0, 0, lambda;

#pragma omp parallel for reduction(+:energy)
	for (int e = 0; e < numTris_; ++e) {
		//compute elastic energy of that element
		int a = triIndices_[e][0];
		int b = triIndices_[e][1];
		int c = triIndices_[e][2];
		Vector2 posA = referencePositions_[a];
		Vector2 posB = referencePositions_[b];
		Vector2 posC = referencePositions_[c];
		real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
		real area = abs(signedArea);

		//compute Straing and stress vector
		Matrix<real, 3, 6> Be;
		Be << (posB.y() - posC.y()), 0, (posC.y() - posA.y()), 0, (posA.y() - posB.y()), 0,
			0, (posC.x() - posB.x()), 0, (posA.x() - posC.x()), 0, (posB.x() - posA.x()),
			(posC.x() - posB.x()), (posB.y() - posC.y()), (posA.x() - posC.x()), (posC.y() - posA.y()), (posB.x() - posA.x()), (posA.y() - posB.y());
		Be *= (1 / signedArea);
		Vector6 u; u << displacements[a], displacements[b], displacements[c];
		Vector3 Strain = Be * u;
		Vector3 Stress = C * Strain;
		
		energy += area * Strain.dot(Stress);
	}

	return energy;
}

bool ar::SoftBodyMesh2D::isReadyForSimulation() const
{
    return !needsReordering_;
}

void ar::SoftBodyMesh2D::reorderNodes()
{
    if (!needsReordering_) return;
    int numNeumann = (int)(nodeStates_ == NEUMANN).count();
    int numDirichlet = (int)(nodeStates_ == DIRICHLET).count();

    numFreeNodes_ = numNodes_ - numDirichlet;
    nodeToFree_.resize(numNodes_);
    freeToNode_.resize(numFreeNodes_);

    int j = 0;
    for (int i = 0; i < numNodes_; ++i) {
        if (nodeStates_[i] != DIRICHLET) {
            nodeToFree_[i] = j;
            freeToNode_[j] = i;
            j++;
        }
        else
            nodeToFree_[i] = -1;
    }

    needsReordering_ = false;
}

void ar::SoftBodyMesh2D::computeElementMatrix(int e, const Matrix3 & C, Matrix6 & Ke, Vector6& Fe, const Vector2List& currentDisplacements) const
{
    int a = triIndices_[e][0];
    int b = triIndices_[e][1];
    int c = triIndices_[e][2];
    Vector2 posA = referencePositions_[a];
    Vector2 posB = referencePositions_[b];
    Vector2 posC = referencePositions_[c];

    real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
    real area = abs(signedArea);

    Fe.setZero();

    //Stiffness
    Matrix36 Be;
    Be << (posB.y() - posC.y()), 0, (posC.y() - posA.y()), 0, (posA.y() - posB.y()), 0,
        0, (posC.x() - posB.x()), 0, (posA.x() - posC.x()), 0, (posB.x() - posA.x()),
        (posC.x() - posB.x()), (posB.y() - posC.y()), (posA.x() - posC.x()), (posC.y() - posA.y()), (posB.x() - posA.x()), (posA.y() - posB.y());
    Be *= (1 / signedArea);
    Ke = area * Be.transpose() * C * Be;

    //rotation correction
    if (getRotationCorrection() == RotationCorrection::Corotation)
    {
        //compute jacobian of the displacements
        Vector2 defA = currentDisplacements[a] + posA;
        Vector2 defB = currentDisplacements[b] + posB;
        Vector2 defC = currentDisplacements[c] + posC;
        Matrix2 Ddef, Dref;
        Ddef << defA.x()-defC.x(), defB.x()-defC.x(),
                defA.y()-defC.y(), defB.y()-defC.y();
        Dref << posA.x() - posC.x(), posB.x() - posC.x(),
                posA.y() - posC.y(), posB.y() - posC.y();
        Matrix2 F = Ddef * Dref.inverse();
        //compute polar decomposition
        Matrix2 R = abs(F.determinant()) < 1e-15 ? Matrix2::Identity() : polarDecomposition(F);
		//cinder::app::console() << e << " (forward) F={{" << F(0, 0) << "," << F(0, 1) << "},{" << F(1, 0) << "," << F(1, 1) << "}}, R={{" << R(0, 0) << "," << R(0, 1) << "},{" << R(1, 0) << "," << R(1, 1) << "}}" << std::endl;
        //augment Ke
        Matrix6 Re;
        Re << R, Matrix2::Zero(), Matrix2::Zero(),
              Matrix2::Zero(), R, Matrix2::Zero(),
              Matrix2::Zero(), Matrix2::Zero(), R;
        Vector6 xe; xe << posA, posB, posC;
        Fe -= Re * Ke * (Re.transpose() * xe - xe);
        Ke = Re * Ke * Re.transpose();
    }
}

void ar::SoftBodyMesh2D::assembleForceVector(VectorX& F, const VectorX* additionalForces) const
{
    //Neumann Forces
    for (int i = 0; i < numNodes_; ++i) {
        //if (nodeStates_[i] == NEUMANN) {
		if (nodeStates_[i] != DIRICHLET) {
            real len = computeBoundaryIntegral(i);
            F.segment<2>(nodeToFree_[i] * 2) += len * (boundaries_[i] + (additionalForces ? additionalForces->segment<2>(nodeToFree_[i] * 2) : Vector2(0, 0)));
        }
    }

    //Body Forces (Gravity)
    for (int e = 0; e < numTris_; ++e) {
        int a = triIndices_[e][0];
        int b = triIndices_[e][1];
        int c = triIndices_[e][2];
        Vector2 posA = referencePositions_[a];
        Vector2 posB = referencePositions_[b];
        Vector2 posC = referencePositions_[c];
        real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
        real area = abs(signedArea);

        Vector2 gravity = area * getGravity();
        if (nodeStates_[a] != DIRICHLET) F.segment<2>(nodeToFree_[a] * 2) += gravity;
        if (nodeStates_[b] != DIRICHLET) F.segment<2>(nodeToFree_[b] * 2) += gravity;
        if (nodeStates_[c] != DIRICHLET) F.segment<2>(nodeToFree_[c] * 2) += gravity;
    }
}

void ar::SoftBodyMesh2D::assembleStiffnessMatrix(const Matrix3& C, MatrixX& K, VectorX* F, const Vector2List& currentDisplacements) const
{
    //loop over all elements
    for (int e = 0; e < numTris_; ++e) {
        Matrix6 Ke;
        Vector6 Fe;

        computeElementMatrix(e, C, Ke, Fe, currentDisplacements);

        placeIntoMatrix(e, &Ke, &K, &Fe, F);
    }
}

void ar::SoftBodyMesh2D::placeIntoMatrix(int e, const Matrix6* Ke, MatrixX* K, const Vector6* Fe, VectorX* F) const
{
    assert(EIGEN_IMPLIES(K != nullptr, Ke != nullptr));
    assert(EIGEN_IMPLIES(F != nullptr, Fe != nullptr));
    for (int i = 0; i < 3; ++i) {
        int nodeI = triIndices_[e][i];
        bool dirichletI = nodeStates_[nodeI] == DIRICHLET;
        for (int j = 0; j < 3; ++j) {
            int nodeJ = triIndices_[e][j];
            bool dirichletJ = nodeStates_[nodeJ] == DIRICHLET;
            if (i == j && dirichletI) {
                //ignore it, v_i is zero by definition of V
            }
            else if (i == j && !dirichletI) {
                //regular, free nodes
                if (K)
                    K->block<2, 2>(nodeToFree_[nodeI] * 2, nodeToFree_[nodeI] * 2) += Ke->block<2, 2>(i * 2, i * 2);
            }
            else if (dirichletI && dirichletJ) {
                //ignore it, v_i is zero by def of V
            }
            else if (dirichletI && !dirichletJ) {
                //ignore it, v_i is zero by def of V
            }
            else if (!dirichletI && dirichletJ) {
                //add it to the force
                if (F && K)
                    F->segment<2>(nodeToFree_[nodeI] * 2) += Ke->block<2, 2>(i * 2, j * 2) * boundaries_[nodeJ];
            }
            else {
                //regular, free nodes
                if (K)
                    K->block<2, 2>(nodeToFree_[nodeI] * 2, nodeToFree_[nodeJ] * 2) += Ke->block<2, 2>(i * 2, j * 2);
            }
        }
        if (!dirichletI) {
            if (F)
                F->segment<2>(nodeToFree_[nodeI] * 2) += Fe->segment<2>(2 * i);
        }
    }
}

ar::VectorX ar::SoftBodyMesh2D::linearize(const Vector2List & list) const
{
    VectorX v(2*getNumFreeNodes());
    for (int i=0; i<numFreeNodes_; ++i)
    {
        v.segment<2>(2 * i) = list[freeToNode_[i]];
    }
    return v;
}

ar::SoftBodyMesh2D::Vector2List ar::SoftBodyMesh2D::delinearize(const VectorX & vector) const
{
    Vector2List list(getNumNodes(), Vector2::Zero());
    for (int i = 0; i < numNodes_; ++i) {
        if (nodeToFree_[i] < 0) continue;
        list[i] = vector.segment<2>(2 * nodeToFree_[i]);
    }
    return list;
}

std::pair<ar::Vector2, ar::Vector2> ar::SoftBodyMesh2D::resolveSingleCollision_PostRepair(real dist, const Vector2 & normal, const Vector2 & u, const Vector2 & uDot, real alpha, real beta)
{
	Vector2 uPost = u - alpha * dist * normal; //project the position out of the plane
	Vector2 uDotPos = (1 - beta) * (uDot - 2 * (uDot.dot(normal))*normal);
	return std::make_pair(uPost, uDotPos);
}

ar::SoftBodyMesh2D ar::SoftBodyMesh2D::CreateBar(
	const Vector2 & rectCenter, const Vector2 & rectHalfSize, int resolution, bool fitObjectToGrid, 
	bool enableDirichlet, const Vector2 & dirichletBoundaries, const Vector2 & neumannBoundaries)
{
    SoftBodyMesh2D meshSolver;

    Vector2 center = rectCenter;
    Vector2 halfSize = rectHalfSize;
    if (fitObjectToGrid)
    {
        halfSize = (rectHalfSize * resolution + Vector2(0.9, 0.9)).cast<int>().cast<real>() / resolution;
        center = ((rectCenter * resolution).cast<int>().cast<real>() + Vector2(0.5, 0.5)) / resolution;
    }

    typedef Matrix<real, 2, 1> Vector2;
    int resolutionX = (int)std::round(resolution * 2 * halfSize.x());
    int resolutionY = (int)std::round(resolution * 2 * halfSize.y());
    int numNodes = (resolutionX + 1) * (resolutionY + 1);
    int numTris = 2 * resolutionX * resolutionY;

    Vector2List vertices(numNodes);
    for (int x = 0; x <= resolutionX; ++x)
    {
        for (int y = 0; y <= resolutionY; ++y)
        {
            vertices[x*(resolutionY + 1) + y] = (center - halfSize + Vector2(2 * x * halfSize.x() / resolutionX, 2 * y * halfSize.y() / resolutionY));
        }
    }

    Vector3iList triangles(numTris);
    for (int x = 0; x<resolutionX; ++x)
    {
        for (int y = 0; y<resolutionY; ++y)
        {
            int a = x * (resolutionY + 1) + y;
            int b = x * (resolutionY + 1) + y + 1;
            int c = (x + 1) * (resolutionY + 1) + y;
            int d = (x + 1) * (resolutionY + 1) + y + 1;
            triangles[2 * (x*resolutionY + y)] = Vector3i(a, b, c);
            triangles[2 * (x*resolutionY + y) + 1] = Vector3i(b, d, c);
        }
    }

    meshSolver.setMesh(vertices, triangles);

    //boundary conditions
    for (int x = 0; x <= resolutionX; ++x)
    {
        for (int y = 0; y <= resolutionY; ++y)
        {
            if (x == 0) {
				if (enableDirichlet)
					meshSolver.addDirichletBoundary(x*(resolutionY + 1) + y, dirichletBoundaries);
            }
            else if (x == resolutionX && y == 0) {
                meshSolver.addNeumannBoundary(x*(resolutionY + 1) + y, neumannBoundaries.eval());
            }
        }
    }

    return meshSolver;
}

ar::SoftBodyMesh2D ar::SoftBodyMesh2D::CreateTorus(
	real torusOuterRadius, real torusInnerRadius, int resolution, 
	bool enableDirichlet, const Vector2 & dirichletBoundaries, const Vector2 & neumannBoundaries)
{
    SoftBodyMesh2D meshSolver;

    Vector2d torusCenter(0.5, 1 - ((torusOuterRadius + torusInnerRadius) * 1.5));
    cinder::TriMeshRef mesh;
    if (torusInnerRadius >= torusOuterRadius) {
        //circle
        cinder::geom::Circle c = cinder::geom::Circle().center(glm::vec2(torusCenter.x(), torusCenter.y())).radius(torusOuterRadius + torusInnerRadius).subdivisions(resolution);
        mesh = cinder::TriMesh::create(c, cinder::TriMesh::Format().positions(2));
    }
    else {
        //disk
        //cinder::geom::Ring r = cinder::geom::Ring().center(vec2(torusCenter.x(), torusCenter.y())).radius(torusOuterRadius).width(2*torusInnerRadius).subdivisions(resolution);
        //mesh = cinder::TriMesh::create(r, TriMesh::Format().positions(2));
        mesh = cinder::TriMesh::create(cinder::TriMesh::Format().positions(2));
        float innerRadius = torusOuterRadius - torusInnerRadius;
        float outerRadius = torusOuterRadius + torusInnerRadius;
        const float tDelta = 1 / (float)resolution * 2.0f * M_PI;
        float t = 0;
        for (int s = 0; s < resolution; s++) {
            glm::vec2 unit(cinder::math<float>::cos(t), cinder::math<float>::sin(t));
            mesh->appendPosition(glm::vec2(torusCenter.x(), torusCenter.y()) + unit * innerRadius);
            mesh->appendPosition(glm::vec2(torusCenter.x(), torusCenter.y()) + unit * outerRadius);
            mesh->appendTriangle(2 * s, 2 * s + 1, (2 * s + 3) % (2 * resolution));
            mesh->appendTriangle(2 * s, (2 * s + 3) % (2 * resolution), (2 * s + 2) % (2 * resolution));
            t += tDelta;
        }
    }

    //copy to solver
    Vector2List vertices(mesh->getNumVertices());
    Vector3iList triangles(mesh->getNumTriangles());
    for (size_t i = 0; i < mesh->getNumVertices(); ++i) {
        vertices[i] = Vector2(mesh->getPositions<2>()[i].x, mesh->getPositions<2>()[i].y);
    }
    for (size_t i = 0; i < mesh->getNumTriangles(); ++i) {
        triangles[i] = Vector3i(mesh->getIndices()[3 * i], mesh->getIndices()[3 * i + 1], mesh->getIndices()[3 * i + 2]);
    }
    meshSolver.setMesh(vertices, triangles);

    //Boundary conditions
    float minY = 1;
    float maxY = 0;
    for (int i = 0; i<mesh->getNumVertices(); ++i)
    {
        float y = mesh->getPositions<2>()[i].y;
        if (y < minY)
        {
            minY = y;
        }
        if (y > maxY)
        {
            maxY = y;
        }
    }

    //Dirichlet boundary
	if (enableDirichlet) {
		for (int i = 0; i < mesh->getNumVertices(); ++i)
		{
			float y = mesh->getPositions<2>()[i].y;
			if (y >= (maxY - 0.1*(maxY - minY)))
			{
				meshSolver.addDirichletBoundary(i, dirichletBoundaries);
			}
		}
	}

    //Neumann boundary
    for (int i = 0; i<mesh->getNumVertices(); ++i)
    {
        float y = mesh->getPositions<2>()[i].y;
        if (y <= (minY + 0.1*(maxY - minY)))
        {
            meshSolver.addNeumannBoundary(i, neumannBoundaries);
        }
    }

    return meshSolver;
}

void ar::SoftBodyMesh2D::assembleMassMatrix(VectorX& Mvec) const
{
    //loop over all elements
    for (int e = 0; e < numTris_; ++e) {
        int a = triIndices_[e][0];
        int b = triIndices_[e][1];
        int c = triIndices_[e][2];
        Vector2 posA = referencePositions_[a];
        Vector2 posB = referencePositions_[b];
        Vector2 posC = referencePositions_[c];

        real signedArea = ((posB.x() - posA.x())*(posC.y() - posA.y()) - (posC.x() - posA.x())*(posB.y() - posA.y())) / 2;
        real area = abs(signedArea);

        Vector2 mass = Vector2::Constant(getMass() * area / 3);
        if (nodeStates_[a] != DIRICHLET) Mvec.segment<2>(nodeToFree_[a] * 2) += mass;
        if (nodeStates_[b] != DIRICHLET) Mvec.segment<2>(nodeToFree_[b] * 2) += mass;
        if (nodeStates_[c] != DIRICHLET) Mvec.segment<2>(nodeToFree_[c] * 2) += mass;
    }
}

ar::real ar::SoftBodyMesh2D::computeBoundaryIntegral(int node) const
{
    Vector2 boundary(0, 0);
    for (int e = 0; e < numTris_; ++e) {
        for (int i=0; i<3; ++i)
        {
            int a = triIndices_[e][i];
            int b = triIndices_[e][(i+1) % 3];
            if (a==node || b==node)
            {
                Vector2 posA = referencePositions_[a];
                Vector2 posB = referencePositions_[b];
                Vector2 v = Vector2(posB.y() - posA.y(), posA.x() - posB.x()) / 2;
                boundary += v;
            }
        }
    }
    return boundary.norm();
}

void ar::SoftBodyMesh2D::solve(bool dynamic, BackgroundWorker * worker)
{
    if (needsReordering_ && hasSolution()) {
        CI_LOG_E("changed boundaries after the simulation started. Reset the simulation first");
        return;
    }
    reorderNodes();
    if (worker->isInterrupted()) return;

    // Assemble matrices
    worker->setStatus("SoftBodyMesh2D: Assemble matrices");
    VectorX Mvec = VectorX::Zero(2 * numFreeNodes_);
    MatrixX K = MatrixX::Zero(2 * numFreeNodes_, 2 * numFreeNodes_);
    VectorX F = VectorX::Zero(2 * numFreeNodes_);
    real mu = getMaterialMu();
    real lambda = getMaterialLambda();
    Matrix3 C = computeMaterialMatrix(mu, lambda);

	//Collision 1
	VectorX collisionForces = VectorX::Zero(2 * numFreeNodes_);
	if (isEnableCollision())
	{
		//compute forces for each node (lumped mass)
		for (int i = 0; i<numFreeNodes_; ++i)
		{
			Vector2 u = currentDisplacements_.at(freeToNode_[i]);
			Vector2 uDot = currentVelocities_.at(freeToNode_[i]);
			Vector2 pos = referencePositions_.at(freeToNode_[i]) + u;
			auto[dist, normal] = groundCollision(pos);
			real penetration = 0.7;
			switch (getCollisionResolution())
			{
			case CollisionResolution::SIMPLE_NEUMANN:
			case CollisionResolution::REPAIR_PLUS_NEUMANN:
			{
				if (dist < 0) {
					real boundary = computeBoundaryIntegral(freeToNode_[i]);
					if (boundary > 0) {
						collisionForces.segment<2>(2 * i) = ((1 - getCollisionVelocityDamping())) * normal;
					}
				}
			} break;
			case CollisionResolution::SPRING_EXPLICIT:
			{
				//force magnitude
				real softmin = ar::utils::softmin(dist, getCollisionSoftmaxAlpha());
				real f = -getGroundStiffness() * softmin;
				//final force
				collisionForces.segment<2>(2 * i) += f * normal;
			} break;
			case CollisionResolution::SPRING_IMPLICIT:
			{
				//force magnitude
				real softmin = ar::utils::softmin(dist, getCollisionSoftmaxAlpha());
				real distDt = groundCollisionDt(uDot, getGroundPlaneHeight(), getGroundPlaneAngle());
				real fCurrent = -getGroundStiffness() * softmin; //current timestep
				real fDt = -getGroundStiffness() * (ar::utils::softminDx(dist, getCollisionSoftmaxAlpha()) * distDt); //time derivative
				real fNext = fCurrent + getTimestep() * fDt; //next timestep
				real f = TimeIntegrator_Newmark1::DefaultTheta * fNext + (1 - TimeIntegrator_Newmark1::DefaultTheta) * fCurrent; //average force magnitude
				//final force:
				collisionForces.segment<2>(2 * i) += f * normal;
			} break;
			}
		}
		//cinder::app::console() << "Force: " << collisionForces.transpose() << std::endl;
	}

    //forces
    assembleForceVector(F, &collisionForces);
    if (worker->isInterrupted()) return;

    //stiffness
    assembleStiffnessMatrix(C, K, &F, currentDisplacements_);
    if (worker->isInterrupted()) return;

    //mass
    if (dynamic)
    {
        assembleMassMatrix(Mvec);
        if (worker->isInterrupted()) return;
    }

    CI_LOG_D("K:\n" << K);
    CI_LOG_D("f:\n" << F.transpose());

    // Solve the system
	VectorX velocities = VectorX::Zero(2 * numFreeNodes_);
    if (integrator_ == nullptr) {
        integrator_ = TimeIntegrator::createIntegrator(getTimeIntegrator(), numFreeNodes_ * 2);
        if (worker->isInterrupted()) return;
    }
    integrator_->setDenseLinearSolver(getDenseLinearSolver());
    VectorX solution;
    if (!dynamic) {
        worker->setStatus("SoftBodyMesh2D: solve for static solution");
        solution = integrator_->solveDense(K, F);
    }
    else {
        worker->setStatus("SoftBodyMesh2D: solve for dynamic solution");
        MatrixX C = TimeIntegrator::rayleighDamping(getDampingAlpha(), Mvec, getDampingBeta(), K);

        CI_LOG_D("M:\n" << Mvec.transpose());
        CI_LOG_D("C:\n" << C);

        integrator_->performStep(getTimestep(), Mvec, C, K, F);
        solution = integrator_->getCurrentU();
		velocities = integrator_->getCurrentUDot();
    }
    if (worker->isInterrupted()) return;
    CI_LOG_D("Solution:\n" << Map<Matrix2X>(solution.data(), 2, numFreeNodes_));

	//Collisions 2
	if (isEnableCollision())
	{
		for (int i=0; i<numFreeNodes_; ++i)
		{
			Vector2 pos = referencePositions_.at(freeToNode_[i]) + solution.segment<2>(2 * i);
			auto[dist, normal] = groundCollision(pos);
			real penetration = 0.7;
			switch (getCollisionResolution())
			{
			case CollisionResolution::POST_REPAIR:
				{
                if (dist < 0) {
                    auto[newU, newUDot] = resolveSingleCollision_PostRepair(
                        dist, normal, solution.segment<2>(2 * i), velocities.segment<2>(2 * i),
                        penetration, getCollisionVelocityDamping());
                    solution.segment<2>(2 * i) = newU;
                    velocities.segment<2>(2 * i) = newUDot;
                    //solution.segment<2>(2 * i) += penetration * -dist * normal; //project the position out of the plane
                    //velocities.segment<2>(2 * i) = (1-getCollisionVelocityDamping()) * (velocities.segment<2>(2 * i) - 2 * (velocities.segment<2>(2 * i).dot(normal))*normal); //mirror velocity
                }
				} break;
			case CollisionResolution::REPAIR_PLUS_NEUMANN:
			{
                if (dist < 0) {
                    solution.segment<2>(2 * i) += penetration * -dist * normal; //project the position out of the plane
                    velocities.segment<2>(2 * i) = (1 - getCollisionVelocityDamping()) * (velocities.segment<2>(2 * i) - 2 * (velocities.segment<2>(2 * i).dot(normal))*normal); //mirror velocity
                }
			} break;
			}
		}
		//send updated positions back to the integrator
		integrator_->setCurrentU(solution);
		integrator_->setCurrentUDot(velocities);
	}

    //Map linear array back
    for (int i = 0; i < numNodes_; ++i) {
        if (nodeToFree_[i] < 0) continue;
        currentDisplacements_[i] = solution.segment<2>(2 * nodeToFree_[i]);
		currentVelocities_[i] = velocities.segment<2>(2 * nodeToFree_[i]);
        currentPositions_[i] = referencePositions_[i] + currentDisplacements_[i];
    }

    setHasSolution(true);
}
