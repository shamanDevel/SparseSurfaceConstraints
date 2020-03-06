#include "SoftBodyMesh3D.h"

#include <vector>
#include <set>
#include <assert.h>
#include <algorithm>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cinder/app/AppBase.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuMat/src/ConjugateGradient.h>

#include <cinder/Log.h>

#include "helper_matrixmath.h"
#include "CommonKernels.h"
#include "DebugUtils.h"
#include "CudaTimer.h"

#ifndef NDEBUG
#include <Eigen/Dense>
#endif

namespace ar3d {
    void SoftBodyMesh3D::Input::assertSizes() const
    {
        assert(indices_.rows() > 0);
        assert(indices_.rows() == numElements_);
        assert(referencePositions_.rows() > 0);
        assert(numTotalNodes_ == referencePositions_.rows());
        assert(numFreeNodes_ <= numTotalNodes_);
        assert(neumannForces_.rows() <= numFreeNodes_);
        assert(numFreeNodes_ > 0);
    }

    SoftBodyMesh3D::Precomputed SoftBodyMesh3D::allocatePrecomputed(const Input& input)
    {
        input.assertSizes();
        Precomputed p;
        p.bodyForces_ = Vector3X(input.numFreeNodes_); p.bodyForces_.setZero();
        p.lumpedMass_ = VectorX(input.numFreeNodes_); p.lumpedMass_.setZero();
        return p;
    }

    SoftBodyMesh3D::State SoftBodyMesh3D::allocateState(const Input& input)
    {
        State s;
        s.displacements_ = Vector3X(input.numFreeNodes_); s.displacements_.setZero();
        s.velocities_ = Vector3X(input.numFreeNodes_); s.velocities_.setZero();
        return s;
    }

	cuMat::SparsityPattern<cuMat::CSR> SoftBodyMesh3D::computeSparsityPattern(const std::vector<int4>& indices, int numFreeNodes)
	{
		typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
		SPattern pattern;
		pattern.rows = numFreeNodes;
		pattern.cols = numFreeNodes;

        //create entry set
        typedef std::pair<int, int> entry_t;
        std::set<entry_t> entries;
        for (const int4& e : indices)
        {
            const std::array<int, 4> ix = {e.x, e.y, e.z, e.w};
            for (int i = 0; i < 4; ++i) {
                const int nodeI = ix[i];
                const bool dirichletI = nodeI >= numFreeNodes;
                for (int j = 0; j < 4; ++j) {
                    const int nodeJ = ix[j];
                    const bool dirichletJ = nodeJ >= numFreeNodes;
                    if (!dirichletI && !dirichletJ) {
                        entries.insert(std::make_pair(nodeI, nodeJ));
                    }
                }
            }
        }
        SMatrix3x3::StorageIndex nnz = static_cast<SMatrix3x3::StorageIndex>(entries.size());
        pattern.nnz = nnz;

        //allocate indices on the host
        std::vector<SMatrix3x3::StorageIndex> JA(pattern.rows + 1, 0); //outer
        std::vector<SMatrix3x3::StorageIndex> IA; IA.reserve(nnz); //inner

        //loop through all sorted entries and build indices
        entry_t lastEntry(-1,-1);
        for (const entry_t& e : entries)
        {
            //assert sorted
            assert(lastEntry.first < e.first || (lastEntry.first==e.first && lastEntry.second<e.second));
            lastEntry = e;
            //increment outer index, add inner index
            JA[lastEntry.first + 1]++;
            IA.push_back(lastEntry.second);
        }
		assert(IA.size() == nnz);
        for (int i=0; i<pattern.rows; ++i)
            JA[i+1] += JA[i]; //prefix sum

        //copy to device
		pattern.JA = SPattern::IndexVector(pattern.rows + 1);
        pattern.JA.copyFromHost(JA.data());
		pattern.IA = SPattern::IndexVector(nnz);
		pattern.IA.copyFromHost(IA.data());

		CI_LOG_I("Sparsity pattern created, matrix size: " << pattern.rows << ", non-zeros: " << nnz
			<< " (" << (100.0*nnz / pattern.rows / pattern.rows) << "%, avg " << real(nnz/pattern.rows) << " per row)");
		pattern.assertValid();
		return pattern;
	}

	SoftBodyMesh3D::Input SoftBodyMesh3D::createBar(const InputBarSettings& settings)
    {
		real3 min = settings.center - settings.halfsize;
		real3 max = settings.center + settings.halfsize;
		real3 minDirichlet = settings.centerDirichlet - settings.halfsizeDirichlet;
		real3 maxDirichlet = settings.centerDirichlet + settings.halfsizeDirichlet;
        real3 size = max - min;
        int3 resolution = make_int3(
            std::max(1, static_cast<int>(round(size.x * settings.resolution))),
            std::max(1, static_cast<int>(round(size.y * settings.resolution))),
            std::max(1, static_cast<int>(round(size.z * settings.resolution))));
        real3 invRes = make_real3(1.0/(resolution.x), 1.0/(resolution.y), 1.0/(resolution.z));

        //compute counts
        Input input;
        input.numTotalNodes_ = (resolution.x+1)*(resolution.y+1)*(resolution.z+1);
        input.numElements_ = 6 * resolution.x * resolution.y * resolution.z;
        input.numFreeNodes_ = input.numTotalNodes_;

        //index conversion + create vertices
        std::vector<real3> vertices(input.numTotalNodes_);
        std::vector<int> indexMap(input.numTotalNodes_);
#define IDX(ix, iy, iz) ((ix) + (resolution.x+1)*((iy) + (resolution.y+1)*(iz)))

        int i=0;
        for (int z=0; z<=resolution.z; ++z) for (int y=0; y<=resolution.y; ++y) for (int x=0; x<=resolution.x; ++x)
        {
            real3 pos = min + size * make_real3(x, y, z) * invRes;
            if (settings.enableDirichlet && 
                pos.x>=minDirichlet.x && pos.y>=minDirichlet.y && pos.z>=minDirichlet.z &&
                pos.x<=maxDirichlet.x && pos.y<=maxDirichlet.y && pos.z<=maxDirichlet.z)
            {
                //dirichlet boundary (add them to the end)
                input.numFreeNodes_--;
                vertices[input.numFreeNodes_] = pos;
                indexMap[IDX(x, y, z)] = input.numFreeNodes_;
            } else
            {
                //free
                vertices[i] = pos;
                indexMap[IDX(x, y, z)] = i;
                i++;
            }
        }

        //create indices
        std::vector<int4> indices; indices.reserve(input.numElements_);
        for (int z=0; z<resolution.z; ++z) for (int y=0; y<resolution.y; ++y) for (int x=0; x<resolution.x; ++x)
        {
			//cinder::app::console() << IDX(x, y, z) << " " << IDX(x + 1, y + 1, z + 1) << std::endl;
            indices.push_back(make_int4(indexMap.at(IDX(x, y, z)), indexMap.at(IDX(x+1, y, z)), indexMap.at(IDX(x, y+1, z)), indexMap.at(IDX(x, y, z+1))));
            indices.push_back(make_int4(indexMap.at(IDX(x+1, y, z)), indexMap.at(IDX(x, y+1, z)), indexMap.at(IDX(x, y, z+1)), indexMap.at(IDX(x+1, y, z+1))));
            indices.push_back(make_int4(indexMap.at(IDX(x, y+1, z)), indexMap.at(IDX(x, y, z+1)), indexMap.at(IDX(x+1, y, z+1)), indexMap.at(IDX(x, y+1, z+1))));
            indices.push_back(make_int4(indexMap.at(IDX(x+1, y, z)), indexMap.at(IDX(x+1, y+1, z)), indexMap.at(IDX(x, y+1, z)), indexMap.at(IDX(x+1, y, z+1))));
            indices.push_back(make_int4(indexMap.at(IDX(x, y+1, z)), indexMap.at(IDX(x+1, y+1, z)), indexMap.at(IDX(x, y+1, z+1)), indexMap.at(IDX(x+1, y, z+1))));
			indices.push_back(make_int4(indexMap.at(IDX(x+1, y+1, z)), indexMap.at(IDX(x, y+1, z+1)), indexMap.at(IDX(x+1, y, z+1)), indexMap.at(IDX(x+1, y+1, z+1))));
        }
#undef IDX

        //copy to the gpu
        input.indices_ = Vector4Xi(input.numElements_); input.indices_.copyFromHost(indices.data());
        input.referencePositions_ = Vector3X(input.numTotalNodes_); input.referencePositions_.copyFromHost(vertices.data());
        input.neumannForces_ = Vector3X::Constant(input.numFreeNodes_, make_real3(0,0,0));

        //compute sparsity pattern
        input.sparsityPattern_ = computeSparsityPattern(indices, input.numFreeNodes_);

		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
        return input;
    }

	//---------------------------------------------
	// The actual instances:
	// They only store the settings for simple access
	// No logic is implemented here
	//---------------------------------------------

	SoftBodyMesh3D::SoftBodyMesh3D(const Input& input)
		: input_(input)
		, precomputed_(allocatePrecomputed(input))
		, state_(allocateState(input))
	{
		allocateTemporary(input_);

		//fill statistics
		statistics_.numElements = input_.numElements_;
		statistics_.numFreeNodes = input_.numFreeNodes_;
		statistics_.numFixedNodes = input_.numTotalNodes_ - input_.numFreeNodes_;
		statistics_.avgEntriesPerRow = input_.sparsityPattern_.nnz / double(input_.sparsityPattern_.rows);
	}


	SoftBodyMesh3D::~SoftBodyMesh3D()
	{
	}

	void SoftBodyMesh3D::reset()
	{
		state_.displacements_.setZero();
		state_.velocities_.setZero();
		resetTimings();
	}

	void SoftBodyMesh3D::solve(bool dynamic, BackgroundWorker2* worker)
	{
		resetTemporary();
		CudaTimer timer;

		//1. Forces
		worker->setStatus("Mesh: compute forces");
		if (isRecordTimings()) timer.start();
		forces_.inplace() = precomputed_.bodyForces_;
		if (settings_.enableCollision_)
		{
			applyCollisionForces(input_, settings_, state_, forces_);
		}
		if (isRecordTimings()) { timer.stop(); statistics_.collisionForcesTime.push_back(timer.duration()); }
		if (worker->isInterrupted()) return;

		//2. stiffness matrix
		worker->setStatus("Mesh: compute stiffness matrix");
		if (isRecordTimings()) timer.start();
		computeStiffnessMatrix(input_, state_, settings_, stiffness_, forces_);
		if (isRecordTimings()) { timer.stop(); statistics_.matrixAssembleTime.push_back(timer.duration()); }
		if (worker->isInterrupted()) return;

		//3. Solve
		if (dynamic)
		{
			worker->setStatus("Mesh: Newmark compute matrices");
			CommonKernels::newmarkTimeIntegration(
				stiffness_, forces_, precomputed_.lumpedMass_,
				state_.displacements_, state_.velocities_,
				settings_.dampingAlpha_, settings_.dampingBeta_, settings_.timestep_,
				newmarkA_, newmarkB_, settings_.newmarkTheta_);

			worker->setStatus("Mesh: CG solve");
			Vector3X currentDisplacement = state_.displacements_ + make_real3(settings_.timestep_) * state_.velocities_; //initial guess
			int iterations = settings_.solverIterations_;
			real tolError = settings_.solverTolerance_;
			if (isRecordTimings()) timer.start();
			CommonKernels::solveCG(newmarkA_, newmarkB_, currentDisplacement, iterations, tolError);
			if (isRecordTimings()) { timer.stop(); statistics_.cgTime.push_back(timer.duration()); statistics_.cgIterations.push_back(iterations); }

			worker->setStatus("Mesh: Newmark compute velocity");
			Vector3X currentVelocity(input_.numFreeNodes_);
			CommonKernels::newmarkComputeVelocity(
				state_.displacements_, state_.velocities_,
				currentDisplacement, currentVelocity,
				settings_.timestep_, settings_.newmarkTheta_);

			state_.displacements_.inplace() = currentDisplacement;
			state_.velocities_.inplace() = currentVelocity;

		} else
		{
			worker->setStatus("Mesh: CG solve");
			state_.displacements_.setZero();
			int iterations = settings_.solverIterations_;
			real tolError = settings_.solverTolerance_;
			if (isRecordTimings()) timer.start();
			CommonKernels::solveCG(stiffness_, forces_, state_.displacements_, iterations, tolError);
			if (isRecordTimings()) { timer.stop(); statistics_.cgTime.push_back(timer.duration()); statistics_.cgIterations.push_back(iterations); }
#if 0
			Eigen::MatrixXf eigenStiffness = DebugUtils::matrixToEigen(stiffness_);
			Eigen::VectorXf eigenForce = DebugUtils::vectorToEigen(forces_);
			cinder::app::console() << "Stiffness matrix:\n" << eigenStiffness << std::endl;
			if (!eigenStiffness.isApprox(eigenStiffness.transpose())) cinder::app::console() << "  Stiffness matrix is not symmetric!!" << std::endl;
			cinder::app::console() << "Force vector:\n" << eigenForce.transpose() << std::endl;
			cinder::app::console() << "Solution:\n" << DebugUtils::vectorToEigen(state_.displacements_).transpose() << std::endl;
			Eigen::VectorXf eigenSolution = eigenStiffness.fullPivLu().solve(eigenForce);
			cinder::app::console() << "Solution using Eigen:\n" << eigenSolution.transpose() << std::endl;
			DebugUtils::eigenToVector(eigenSolution, state_.displacements_);
#endif

		}
		worker->setStatus("Mesh: done");
	}

	void SoftBodyMesh3D::updateSettings()
	{
		precomputed_.bodyForces_.setZero();
		precomputed_.lumpedMass_.setZero();

		computeMassMatrix(input_, settings_, precomputed_.lumpedMass_);
		computeBodyForces(input_, settings_, precomputed_.bodyForces_);

		CI_LOG_I("Settings updated, mass matrix and body forces recomputed");
	}

	void SoftBodyMesh3D::allocateTemporary(const Input& input)
	{
		forces_ = Vector3X(input.numFreeNodes_);
		stiffness_ = SMatrix3x3(input.sparsityPattern_);
		newmarkA_ = SMatrix3x3(input.sparsityPattern_);
		newmarkB_ = Vector3X(input.numFreeNodes_);
	}

	void SoftBodyMesh3D::resetTemporary()
	{
		forces_.setZero();
		stiffness_.setZero();
	}
}
