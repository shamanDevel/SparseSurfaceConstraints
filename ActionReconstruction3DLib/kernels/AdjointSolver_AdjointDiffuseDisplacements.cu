#include "../AdjointSolver.h"
#include <cinder/app/AppBase.h>
#include <iostream>

namespace ar3d
{
    typedef WorldGridData<int>::DeviceArray_t PosToIndex_t;

    __global__ void AdjointGridDiffusionMappingKernel(dim3 size, 
        const PosToIndex_t posToIndex, const WorldGridData<real3>::DeviceArray_t adjGridDisp, SoftBodyGrid3D::DiffusionRhs adjDiffusedRhs)
    {
        CUMAT_KERNEL_3D_LOOP(x, y, z, size)
            int i = posToIndex.coeff(x, y, z, -1);
            if (i < 0)
            {
                real3 adjDisp = adjGridDisp.coeff(x, y, z, -1);
                adjDiffusedRhs.coeff(-i - 1, 0, 0, -1) = adjDisp.x;
                adjDiffusedRhs.coeff(-i - 1, 0, 1, -1) = adjDisp.y;
                adjDiffusedRhs.coeff(-i - 1, 0, 2, -1) = adjDisp.z;
            }
        CUMAT_KERNEL_3D_LOOP_END
    }

    __global__ void AdjointGridDiffusionFillRhsKernel(dim3 size, const PosToIndex_t posToIndex, 
        const WorldGridData<real3>::DeviceArray_t adjGridDisplacements, const SoftBodyGrid3D::DiffusionRhs adjRhs, Vector3X adjDisplacements)
    {
        static const int neighborsX[] = { -1, 1, 0, 0, 0, 0 };
        static const int neighborsY[] = { 0, 0, -1, 1, 0, 0 };
        static const int neighborsZ[] = { 0, 0, 0, 0, -1, 1 };
        CUMAT_KERNEL_3D_LOOP(x, y, z, size)
            const int row = posToIndex.coeff(x, y, z, -1);
            if (row > 0)
                atomicAddReal3(adjDisplacements.data() + row-1, adjGridDisplacements.coeff(x, y, z, -1));
            else if (row < 0) {
                real3 val = make_real3(adjRhs.coeff(-row-1, 0, 0, -1), adjRhs.coeff(-row-1, 0, 1, -1), adjRhs.coeff(-row-1, 0, 2, -1));
                for (int i = 0; i < 6; ++i) {
                    int ix = x + neighborsX[i];
                    int iy = y + neighborsY[i];
                    int iz = z + neighborsZ[i];
                    if (ix < 0 || ix >= size.x || iy < 0 || iy >= size.y || iz < 0 || iz >= size.z) continue;
                    int col = posToIndex.coeff(ix, iy, iz, -1) - 1;
                    if (col >= 0)
                        atomicAddReal3(adjDisplacements.data() + col, val);
                }
            }
        CUMAT_KERNEL_3D_LOOP_END
    }

    void AdjointSolver::adjointDiffuseDisplacements(const Input& input,
        const WorldGridData<real3>::DeviceArray_t& adjGridDisplacements, Vector3X& adjDisplacementsOut)
    {
        cuMat::Context& ctx = cuMat::Context::current();
        const Eigen::Vector3i& size = input.grid_->getSize();
        unsigned numDiffusedNodes = input.numDiffusedNodes_;

        //create RHS
        CI_LOG_D("Norm of adjGridDisplacements: " << std::sqrt(static_cast<real>(adjGridDisplacements.cwiseAbs2().sum<cuMat::Axis::All>())));
        SoftBodyGrid3D::DiffusionRhs adjDiffusedRhs = SoftBodyGrid3D::DiffusionRhs(numDiffusedNodes, 1, 3);
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(size.x(), size.y(), size.z(), AdjointGridDiffusionMappingKernel);
        AdjointGridDiffusionMappingKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.posToIndex_->getDeviceMemory(), adjGridDisplacements, adjDiffusedRhs);
		CUMAT_CHECK_ERROR();
		CI_LOG_D("Adjoint-Diffusion: RHS assembled");

        //cinder::app::console() << "adjRhs: \n" << std::scientific << (adjDiffusedRhs.cwiseDiv(adjDiffusedRhs.norm())).eval() << std::endl;

        //Solve CG for every dimension
        CI_LOG_D("Norms of the RHS: " << static_cast<real>(adjDiffusedRhs.slice(0).norm()) << ", " << static_cast<real>(adjDiffusedRhs.slice(1).norm()) << ", " << static_cast<real>(adjDiffusedRhs.slice(2).norm()));
        VectorX tmp(numDiffusedNodes);
        SoftBodyGrid3D::DiffusionRhs adjDiffusedX = SoftBodyGrid3D::DiffusionRhs(numDiffusedNodes, 1, 3);
        for (int i = 0; i<3; ++i)
        {
            const real scaling = static_cast<real>(adjDiffusedRhs.slice(i).norm());
            if (scaling > 1e-7) {
                tmp = input.diffusionCG_.solve(adjDiffusedRhs.slice(i).cwiseDiv(scaling));
                adjDiffusedX.slice(i) = tmp.cwiseMul(scaling);
                CI_LOG_D("Adjoint-Diffusion: Dimension " << i << ", CG finished after " << input.diffusionCG_.iterations() << " iterations with an error of " << input.diffusionCG_.error() << ", scaling=" << (1.0 / static_cast<real>(scaling)));
            }
            else
                adjDiffusedX.slice(i) = VectorX::Zero(numDiffusedNodes);
        }

        //map back to active nodes
        cfg = ctx.createLaunchConfig3D(size.x(), size.y(), size.z(), AdjointGridDiffusionFillRhsKernel);
        AdjointGridDiffusionFillRhsKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.posToIndex_->getDeviceMemory(), adjGridDisplacements, adjDiffusedX, adjDisplacementsOut);
		CUMAT_CHECK_ERROR();
		CI_LOG_D("Adjoint-Diffusion: solution mapped back to active nodes");
    }
}
