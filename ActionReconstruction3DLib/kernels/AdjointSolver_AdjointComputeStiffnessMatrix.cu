#include "../AdjointSolver.h"
#include "ComputeStiffnessMatrixCommons.cuh"

#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

#include "../PolarDecomposition.h"

namespace ar3d
{
    //Worksize = 64
    template <bool Dirichlet, bool Corotation>
    __global__ void GridAdjointComputeStiffnessMatrixKernel(
        const Vector8X interpolationVolumeWeights, const Vector8X interpolationBoundaryWeights,
        const Vector3X surfaceNormals, const VectorXc dirichlet, 
        const Vector4Xi mapping, const Vector3X referencePositions,
        const Vector3X lastDisplacements,
        const real lambda, const real mu, const real h,
        SMatrix3x3 adjStiffnessMatrix, Vector3X adjForce,
        Vector3X adjLastDisplacements,
        real* adjLambda, real* adjMu)
    {
        const int elementIdx = blockIdx.x;
        const int thread = threadIdx.x;
        const int i = thread % 8;
        const int j = thread / 8;
        CUMAT_ASSERT_CUDA(blockDim.x == 64);
		CUMAT_ASSERT_CUDA(thread >= 0 && thread <= 63);

        //information about the current element that is always needed
        const real8 volumeWeight = interpolationVolumeWeights.getRawCoeff(elementIdx);
        const real weights[8] = {
            volumeWeight.first.x, volumeWeight.first.y, volumeWeight.first.z, volumeWeight.first.w,
            volumeWeight.second.x, volumeWeight.second.y, volumeWeight.second.z, volumeWeight.second.w
        };
        const int4 map = mapping.getRawCoeff(elementIdx);
        const int nodeIdx[8] = { map.x, map.x + 1, map.y, map.y + 1, map.z, map.z + 1, map.w, map.w + 1 };

        //shared memory configuration
        __shared__ real3x3 Rx[polarDecompositionIterations+1];
        __shared__ bool hasRotation;
        __shared__ real3x3 adjRotationReduction[2];
		__shared__ real3x3 adjFPart;

        //adjoint output
        real adjLambdaPart = 0;
        real adjMuPart = 0;
        real3 adjDispPart = make_real3(0);

        //ADJOINT: Place into matrix
        real3x3 adjKePart(0);
        SMatrix3x3::StorageIndex start = adjStiffnessMatrix.getSparsityPattern().JA.getRawCoeff(nodeIdx[i]);
        SMatrix3x3::StorageIndex end = adjStiffnessMatrix.getSparsityPattern().JA.getRawCoeff(nodeIdx[i] + 1);
        for (SMatrix3x3::StorageIndex k = start; k < end; ++k)
        {
            SMatrix3x3::StorageIndex inner = adjStiffnessMatrix.getSparsityPattern().IA.getRawCoeff(k);
            if (inner == nodeIdx[j])
            {
                //entry found
                adjKePart = adjStiffnessMatrix.getData().getRawCoeff(k);
                break;
            }
        }

        //ADJOINT: DIRICHLET BOUNDARIES
        if (Dirichlet)
        {
            if (dirichlet.getRawCoeff(elementIdx))
            {
                //we are on a dirichlet boundary

                //fetch normal and surface interpolation weights
                const real3 normal = surfaceNormals.getRawCoeff(elementIdx);
                const real8 surfaceWeight = interpolationBoundaryWeights.getRawCoeff(elementIdx);
                const real sweights[8] = {
                    surfaceWeight.first.x, surfaceWeight.first.y, surfaceWeight.first.z, surfaceWeight.first.w,
                    surfaceWeight.second.x, surfaceWeight.second.y, surfaceWeight.second.z, surfaceWeight.second.w
                };

                const real3x3 adjKeD1 = -sweights[i] * adjKePart;
                const real3x3 adjKeD2 = -sweights[j] * adjKePart;

                //derivatives for lambda
                adjLambdaPart +=
                      adjKeD1.r1.x * cGridB[i][j][0] * normal.x
                    + adjKeD1.r1.y * cGridB[i][j][1] * normal.x
                    + adjKeD1.r1.z * cGridB[i][j][2] * normal.x
                    + adjKeD1.r2.x * cGridB[i][j][0] * normal.y
                    + adjKeD1.r2.y * cGridB[i][j][1] * normal.y
                    + adjKeD1.r2.z * cGridB[i][j][2] * normal.y
                    + adjKeD1.r3.x * cGridB[i][j][0] * normal.z
                    + adjKeD1.r3.y * cGridB[i][j][1] * normal.z
                    + adjKeD1.r3.z * cGridB[i][j][2] * normal.z
                    + adjKeD2.r1.x * cGridB[j][i][0] * normal.x
                    + adjKeD2.r1.y * cGridB[j][i][1] * normal.x
                    + adjKeD2.r1.z * cGridB[j][i][2] * normal.x
                    + adjKeD2.r2.x * cGridB[j][i][0] * normal.y
                    + adjKeD2.r2.y * cGridB[j][i][1] * normal.y
                    + adjKeD2.r2.z * cGridB[j][i][2] * normal.y
                    + adjKeD2.r3.x * cGridB[j][i][0] * normal.z
                    + adjKeD2.r3.y * cGridB[j][i][1] * normal.z
                    + adjKeD2.r3.z * cGridB[j][i][2] * normal.z;
                //derivatives for mu
                adjMuPart +=
                      adjKeD1.r1.x * (2 * cGridB[i][j][0] * normal.x + cGridB[i][j][1] * normal.y + cGridB[i][j][2] * normal.z)
                    + adjKeD1.r1.y *  cGridB[i][j][0] * normal.y
                    + adjKeD1.r1.z *  cGridB[i][j][0] * normal.z
                    + adjKeD1.r2.x *  cGridB[i][j][1] * normal.x
                    + adjKeD1.r2.y * (cGridB[i][j][0] * normal.x + 2 * cGridB[i][j][1] * normal.y + cGridB[i][j][2] * normal.z)
                    + adjKeD1.r2.z *  cGridB[i][j][1] * normal.z
                    + adjKeD1.r3.x *  cGridB[i][j][2] * normal.x
                    + adjKeD1.r3.y *  cGridB[i][j][2] * normal.y
                    + adjKeD1.r3.z * (cGridB[i][j][0] * normal.x + cGridB[i][j][1] * normal.y + 2 * cGridB[i][j][2] * normal.z)
                    + adjKeD2.r1.x * (2 * cGridB[j][i][0] * normal.x + cGridB[j][i][1] * normal.y + cGridB[j][i][2] * normal.z)
                    + adjKeD2.r1.y *  cGridB[j][i][0] * normal.y
                    + adjKeD2.r1.z *  cGridB[j][i][0] * normal.z
                    + adjKeD2.r2.x *  cGridB[j][i][1] * normal.x
                    + adjKeD2.r2.y * (cGridB[j][i][0] * normal.x + 2 * cGridB[j][i][1] * normal.y + cGridB[j][i][2] * normal.z)
                    + adjKeD2.r2.z *  cGridB[j][i][1] * normal.z
                    + adjKeD2.r3.x *  cGridB[j][i][2] * normal.x
                    + adjKeD2.r3.y *  cGridB[j][i][2] * normal.y
                    + adjKeD2.r3.z * (cGridB[j][i][0] * normal.x + cGridB[j][i][1] * normal.y + 2 * cGridB[j][i][2] * normal.z);
            }
        }

        //ADJOINT COROTATION
        if (Corotation)
        {
            //fetch adjoint force
            real3 adjFe = adjForce.getRawCoeff(nodeIdx[i]);

            //forward: compute Ke_ij
            //Duplication of GridComputeStiffnessMatrixKernel
            real3x3 KePart(0);
            for (int v = 0; v < 8; ++v)
            {
                //row 1
                KePart.r1 += weights[v] * make_real3(
                    (2 * mu + lambda) * cGridB[v][i][0] * cGridB[v][j][0] + mu * (cGridB[v][i][1] * cGridB[v][j][1] + cGridB[v][i][2] * cGridB[v][j][2]),
                    mu * cGridB[v][i][1] * cGridB[v][j][0] + lambda * cGridB[v][i][0] * cGridB[v][j][1],
                    mu * cGridB[v][i][2] * cGridB[v][j][0] + lambda * cGridB[v][i][0] * cGridB[v][j][2]
                );
                //row 2
                KePart.r2 += weights[v] * make_real3(
                    mu * cGridB[v][i][0] * cGridB[v][j][1] + lambda * cGridB[v][i][1] * cGridB[v][j][0],
                    (2 * mu + lambda) * cGridB[v][i][1] * cGridB[v][j][1] + mu * (cGridB[v][i][0] * cGridB[v][j][0] + cGridB[v][i][2] * cGridB[v][j][2]),
                    mu * cGridB[v][i][2] * cGridB[v][j][1] + lambda * cGridB[v][i][1] * cGridB[v][j][2]
                );
                //row 3
                KePart.r3 += weights[v] * make_real3(
                    mu * cGridB[v][i][0] * cGridB[v][j][2] + lambda * cGridB[v][i][2] * cGridB[v][j][0],
                    mu * cGridB[v][i][1] * cGridB[v][j][2] + lambda * cGridB[v][i][2] * cGridB[v][j][1],
                    (2 * mu + lambda) * cGridB[v][i][2] * cGridB[v][j][2] + mu * (cGridB[v][i][0] * cGridB[v][j][0] + cGridB[v][i][1] * cGridB[v][j][1])
                );
            }

            //forward: compute rotation
            if (j == 0)
            {
                //4. Thread 0-7 compute their contributions to the deformation gradient
                //compute part of FPart
                const real3 displacement = lastDisplacements.getRawCoeff(nodeIdx[i]);
                real3x3 FPart = real3x3::OuterProduct(displacement, make_real3(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1) / h) * 0.25;
                //printf("element=%d, i=%d -> FPart=[%5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f]\n", elementIdx, i, FPart.r1.x, FPart.r1.y, FPart.r1.z, FPart.r2.x, FPart.r2.y, FPart.r2.z, FPart.r3.x, FPart.r3.y, FPart.r3.z);
                //Warp-Reduction to sum up FPart
                const unsigned int mask = __activemask();
                for (unsigned int offset = 4; offset > 0; offset /= 2) {
                    FPart += __shfl_down_sync_real3x3(mask, FPart, offset, 8);
                }

                //4. Thread 0 computes the polar decomposition -> rotation
                if (i == 0)
                {
                    real3x3 F = real3x3::Identity() + FPart;
                    real det = F.det();
                    if (det < POLAR_DECOMPOSITION_THRESHOLD)
                    {
                        hasRotation = false;
                    } else
                    {
						hasRotation = true;
                        Rx[0] = F;
                        for (int ii = 0; ii<polarDecompositionIterations; ++ii)
                        { //copy-paste from PolarDecomposition::polarDecompositionIterative
                            Rx[ii + 1] = 0.5*(Rx[ii] + Rx[ii].transpose().inverse());
                        }
                    }
                }
            }
            __syncthreads();
            if (hasRotation) {
                const real3x3& rotation = Rx[polarDecompositionIterations];

				//adjoint: reduction of the subforces into outputForces
				//The adjoint code is simply the inversion of adjFe
				adjFe = -adjFe;

                //adjoint: KePart = rotation.matmul(KePart.matmulT(rotation));
				//wrong:
                //real3x3 adjRotation = adjKePart.transpose().matmul(rotation).matmulT(KePart)
                //    + adjKePart.matmul(rotation).matmul(KePart);
				//correct:
				real3x3 adjRotation(0);
#pragma unroll
				for (int ii = 0; ii < 3; ++ii) for (int jj = 0; jj < 3; ++jj)
				{
					real3x3 Rd = real3x3::SingleEntry(ii, jj);
					adjRotation.entry(ii, jj) = (Rd.matmul(KePart).matmulT(rotation) + rotation.matmul(KePart).matmulT(Rd)).vecProd(adjKePart);
				}
                adjKePart = rotation.transpose().matmul(adjKePart).matmul(rotation);

                //adjoint: Fe = rotation.matmul(KePart).matmul(rotation.transpose().matmul(position) - position);
                const real3 position = h * make_real3(j & 1 ? 1 : 0, j & 2 ? 1 : 0, j & 4 ? 1 : 0);
				//adjR_ij = I_ij K R^T p + R K I_ji p - I_ij K p   with I_ij = e_i*e_j^T
				adjRotation +=
					real3x3::OuterProduct(adjFe, KePart.matmulLeftT(rotation.matmulLeft(position))) +
					real3x3::OuterProduct(position, KePart.transpose().matmulT(rotation).matmul(adjFe)) -
					real3x3::OuterProduct(adjFe, KePart.matmulLeftT(position));
                adjKePart += rotation.transpose().matmul(real3x3::OuterProduct(adjFe, rotation.transpose().matmul(position) - position));

                //reduce adjRotation into thread 1
				__syncthreads();
                for (int offset = 16; offset >= 1; offset /= 2)
                    adjRotation += __shfl_down_sync_real3x3(0xffffffff, adjRotation, offset, 32);
				__syncthreads();

            	//if ((thread & 0x1f) == 0)
				//	adjRotationReduction[thread >> 5] = adjRotation;
				// verbose version of above
				if (thread == 32) adjRotationReduction[1] = adjRotation;
				else if (thread == 0) adjRotationReduction[0] = adjRotation;

            	__syncthreads();

                if (j == 0 && i == 0)
                {
                    //adjoint: Thread 0 computes the polar decomposition -> rotation
					real3x3 adjF = adjRotationReduction[0] + adjRotationReduction[1];
                    for (int ii = polarDecompositionIterations - 1; ii >= 0; --ii)
                    {
                        adjF = 0.5 * (adjF + PolarDecomposition::adjointMatrixTrInv(Rx[ii], adjF));
                    }
					adjFPart = 0.25 * adjF;
                }
                __syncthreads();
                if (j == 0)
                {
                    //adjoint: compute part of FPart
                    real3 basis = make_real3(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1) / h;
                    adjDispPart = adjFPart.matmul(basis);
                }
            }
        }

        //ADJOINT STIFFNESS
        for (int v=0; v<8; ++v)
        {
            adjLambdaPart += weights[v] * (
                adjKePart.r1.x * cGridB[v][i][0] * cGridB[v][j][0] +
                adjKePart.r1.y * cGridB[v][i][0] * cGridB[v][j][1] +
                adjKePart.r1.z * cGridB[v][i][0] * cGridB[v][j][2] +
                adjKePart.r2.x * cGridB[v][i][1] * cGridB[v][j][0] +
                adjKePart.r2.y * cGridB[v][i][1] * cGridB[v][j][1] +
                adjKePart.r2.z * cGridB[v][i][1] * cGridB[v][j][2] +
                adjKePart.r3.x * cGridB[v][i][2] * cGridB[v][j][0] +
                adjKePart.r3.y * cGridB[v][i][2] * cGridB[v][j][1] +
                adjKePart.r3.z * cGridB[v][i][2] * cGridB[v][j][2]
            );
            adjMuPart += weights[v] * (
                adjKePart.r1.x * (2 * cGridB[v][i][0] * cGridB[v][j][0] + cGridB[v][i][1] * cGridB[v][j][1] + cGridB[v][i][2] * cGridB[v][j][2]) +
                adjKePart.r1.y * cGridB[v][i][1] * cGridB[v][j][0] +
                adjKePart.r1.z * cGridB[v][i][2] * cGridB[v][j][0] +
                adjKePart.r2.x * cGridB[v][i][0] * cGridB[v][j][1] +
                adjKePart.r2.y * (2 * cGridB[v][i][1] * cGridB[v][j][1] + cGridB[v][i][0] * cGridB[v][j][0] + cGridB[v][i][2] * cGridB[v][j][2]) +
                adjKePart.r2.z * cGridB[v][i][2] * cGridB[v][j][1] +
                adjKePart.r3.x * cGridB[v][i][0] * cGridB[v][j][2] +
                adjKePart.r3.y * cGridB[v][i][1] * cGridB[v][j][2] +
                adjKePart.r3.z * (2 * cGridB[v][i][2] * cGridB[v][j][2] + cGridB[v][i][0] * cGridB[v][j][0] + cGridB[v][i][1] * cGridB[v][j][1])
            );
        }

        //write adjoint
        if (adjLambdaPart != 0) atomicAddReal(adjLambda, adjLambdaPart);
        if (adjMuPart != 0) atomicAddReal(adjMu, adjMuPart);
        if (j==0 && any(adjDispPart != make_real3(0))) atomicAddReal3(adjLastDisplacements.data() + nodeIdx[i], adjDispPart);
    }

    void AdjointSolver::adjointComputeStiffnessMatrix(const Input& input, const Vector3X& lastDisplacements,
        const SoftBodySimulation3D::Settings& settings, const SMatrix3x3& adjStiffnessMatrix, const Vector3X& adjForce,
        Vector3X& adjLastDisplacementsOut, DeviceScalar& adjLambdaOut, DeviceScalar& adjMuOut)
    {
        cuMat::Context& ctx = cuMat::Context::current();
        dim3 gridDim = dim3(input.numActiveCells_, 1, 1);
        dim3 blockDim = dim3(64, 1, 1);
        real h = static_cast<real>(input.grid_->getVoxelSize());
        if (input.hasDirichlet_ && settings.enableCorotation_)
        {
            GridAdjointComputeStiffnessMatrixKernel<true, true>
                <<< gridDim, blockDim, 0, ctx.stream() >>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                input.dirichlet_, input.mapping_, input.referencePositions_,
                lastDisplacements, settings.materialLambda_, settings.materialMu_, h,
                adjStiffnessMatrix, adjForce,
                adjLastDisplacementsOut, adjLambdaOut.data(), adjMuOut.data());
        }
        else if (!input.hasDirichlet_ && settings.enableCorotation_)
        {
            GridAdjointComputeStiffnessMatrixKernel<false, true>
                <<< gridDim, blockDim, 0, ctx.stream() >>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                input.dirichlet_, input.mapping_, input.referencePositions_,
                lastDisplacements, settings.materialLambda_, settings.materialMu_, h,
                adjStiffnessMatrix, adjForce,
                adjLastDisplacementsOut, adjLambdaOut.data(), adjMuOut.data());
        }
        else if (input.hasDirichlet_ && !settings.enableCorotation_)
        {
            GridAdjointComputeStiffnessMatrixKernel<true, false>
                <<< gridDim, blockDim, 0, ctx.stream() >>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                input.dirichlet_, input.mapping_, input.referencePositions_,
                lastDisplacements, settings.materialLambda_, settings.materialMu_, h,
                adjStiffnessMatrix, adjForce,
                adjLastDisplacementsOut, adjLambdaOut.data(), adjMuOut.data());
        }
        else if (!input.hasDirichlet_ && !settings.enableCorotation_)
        {
            GridAdjointComputeStiffnessMatrixKernel<false, false>
                <<< gridDim, blockDim, 0, ctx.stream() >>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                input.dirichlet_, input.mapping_, input.referencePositions_,
                lastDisplacements, settings.materialLambda_, settings.materialMu_, h,
                adjStiffnessMatrix, adjForce,
                adjLastDisplacementsOut, adjLambdaOut.data(), adjMuOut.data());
        }
        CUMAT_CHECK_ERROR();
        cudaPrintfDisplay_D(cinder::app::console());
    }

}
