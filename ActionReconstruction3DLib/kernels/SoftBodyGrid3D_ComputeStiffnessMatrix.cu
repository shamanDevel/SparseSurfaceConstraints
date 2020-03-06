#include "../SoftBodyGrid3D.h"
#include "ComputeStiffnessMatrixCommons.cuh"

#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>
#include "../PolarDecomposition.h"

//Specifies if flipped rotations shall be logged to the console (=1 to enable)
#define LOG_FLIPPED_ROTATIONS 0

namespace ar3d
{
    __constant__ real cGridB[8][8][3]; //Vertex(interpolation), Basis, Coordinate

    void GridComputeBasisFunctions(int resolution)
    {
        static int lastResolution = 0;
        if (lastResolution == resolution) return; //cGridB already up-to-date
        lastResolution = resolution;

        //see CubeSoftBody.nb
        real h = 1.0 / resolution;
        real gridB[8][8][3] = 
   {{{-(1/h), -(1/h), -(1/h)}, {1/h, 0, 0}, {0, 1/h, 0}, {0, 0, 0}, {0, 
   0, 1/h}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{-(1/h), 0, 0}, {1/
   h, -(1/h), -(1/h)}, {0, 0, 0}, {0, 1/h, 0}, {0, 0, 0}, {0, 0, 1/
   h}, {0, 0, 0}, {0, 0, 0}}, {{0, -(1/h), 0}, {0, 0, 0}, {-(1/h), 1/
   h, -(1/h)}, {1/h, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1/h}, {0, 0, 
   0}}, {{0, 0, 0}, {0, -(1/h), 0}, {-(1/h), 0, 0}, {1/h, 1/
   h, -(1/h)}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1/h}}, {{0, 
   0, -(1/h)}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {-(1/h), -(1/h), 1/
   h}, {1/h, 0, 0}, {0, 1/h, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 
   0, -(1/h)}, {0, 0, 0}, {0, 0, 0}, {-(1/h), 0, 0}, {1/h, -(1/h), 1/
   h}, {0, 0, 0}, {0, 1/h, 0}}, {{0, 0, 0}, {0, 0, 0}, {0, 
   0, -(1/h)}, {0, 0, 0}, {0, -(1/h), 0}, {0, 0, 0}, {-(1/h), 1/h, 1/
   h}, {1/h, 0, 0}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 
   0, -(1/h)}, {0, 0, 0}, {0, -(1/h), 0}, {-(1/h), 0, 0}, {1/h, 1/h, 
   1/h}}};
        assert(sizeof(gridB) == 8*8*3*sizeof(real));
        
        cudaMemcpyToSymbol(cGridB, gridB, sizeof(gridB));
        CI_LOG_I("Grid Basis functions recomputed");
    }

    //Worksize = 64
    template <bool Dirichlet, bool Corotation>
    __global__ void GridComputeStiffnessMatrixKernel(
        Vector8X interpolationVolumeWeights, Vector8X interpolationBoundaryWeights,
        Vector3X surfaceNormals, VectorXc dirichlet, Vector4Xi mapping, Vector3X referencePositions,
        Vector3X lastDisplacements,
        real lambda, real mu, real h,
        SMatrix3x3 outputMatrix, Vector3X outputForce)
    {
        const int elementIdx = blockIdx.x;
        const int thread = threadIdx.x;
        const int i = thread % 8;
        const int j = thread / 8;
        CUMAT_ASSERT_CUDA(blockDim.x == 64);

        //information about the current element that is always needed
        const real8 volumeWeight = interpolationVolumeWeights.getRawCoeff(elementIdx);
        const real weights[8] = {
            volumeWeight.first.x, volumeWeight.first.y, volumeWeight.first.z, volumeWeight.first.w,
            volumeWeight.second.x, volumeWeight.second.y, volumeWeight.second.z, volumeWeight.second.w
        };
        const int4 map = mapping.getRawCoeff(elementIdx);
        const int nodeIdx[8] = {map.x, map.x + 1, map.y, map.y + 1, map.z, map.z + 1, map.w, map.w + 1};

        //shared memory configuration
        __shared__ __device__ real3x3 rotation;
        __shared__ __device__ real3 subforces[8][8];

        //3. Each thread computes Ke_ij
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
		//printf("element=%d, i=%d, j=%d -> KePart=[%5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f]\n", elementIdx, i, j, KePart.r1.x, KePart.r1.y, KePart.r1.z, KePart.r2.x, KePart.r2.y, KePart.r2.z, KePart.r3.x, KePart.r3.y, KePart.r3.z);

        if (Corotation)
		{
			if (j==0)
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
				if (i==0)
				{
					real3x3 F = real3x3::Identity() + FPart;
                    //cuPrintf("element=%d -> Freduced=[%5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f]\n", elementIdx, FPart.r1.x, FPart.r1.y, FPart.r1.z, FPart.r2.x, FPart.r2.y, FPart.r2.z, FPart.r3.x, FPart.r3.y, FPart.r3.z);
					real det = F.det();
					if (det < 0)
					{
#if LOG_FLIPPED_ROTATIONS == 1
                        cuPrintf("[element %d] det(F)<0: w0=%7.5f, w1=%7.5f, w2=%7.5f, w3=%7.5f, w4=%7.5f, w5=%7.5f, w6=%7.5f, w7=%7.5f\n", elementIdx, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7]);
                        cuPrintf("[element %d] det(F)<0: u0=[4.2f 4.2f 4.2f], u1=[4.2f 4.2f 4.2f], u2=[4.2f 4.2f 4.2f], u3=[4.2f 4.2f 4.2f], u4=[4.2f 4.2f 4.2f], u5=[4.2f 4.2f 4.2f], u6=[4.2f 4.2f 4.2f], u7=[4.2f 4.2f 4.2f]\n", elementIdx,
                            lastDisplacements.getRawCoeff(nodeIdx[0]).x, lastDisplacements.getRawCoeff(nodeIdx[0]).y, lastDisplacements.getRawCoeff(nodeIdx[0]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[1]).x, lastDisplacements.getRawCoeff(nodeIdx[1]).y, lastDisplacements.getRawCoeff(nodeIdx[1]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[2]).x, lastDisplacements.getRawCoeff(nodeIdx[2]).y, lastDisplacements.getRawCoeff(nodeIdx[2]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[3]).x, lastDisplacements.getRawCoeff(nodeIdx[3]).y, lastDisplacements.getRawCoeff(nodeIdx[3]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[4]).x, lastDisplacements.getRawCoeff(nodeIdx[4]).y, lastDisplacements.getRawCoeff(nodeIdx[4]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[5]).x, lastDisplacements.getRawCoeff(nodeIdx[5]).y, lastDisplacements.getRawCoeff(nodeIdx[5]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[6]).x, lastDisplacements.getRawCoeff(nodeIdx[6]).y, lastDisplacements.getRawCoeff(nodeIdx[6]).z,
                            lastDisplacements.getRawCoeff(nodeIdx[7]).x, lastDisplacements.getRawCoeff(nodeIdx[7]).y, lastDisplacements.getRawCoeff(nodeIdx[7]).z);
						cuPrintf("[element %d] det(F)<0: f=[%5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f], det=%5.3f\n", elementIdx, F.r1.x, F.r1.y, F.r1.z, F.r2.x, F.r2.y, F.r2.z, F.r3.x, F.r3.y, F.r3.z, det);
#endif
						det = 0;
					}

#if 1
				    rotation = det < POLAR_DECOMPOSITION_THRESHOLD 
						? real3x3::Identity() 
						: PolarDecomposition::polarDecompositionIterative(F, polarDecompositionIterations);
#else
                    //less stable
                    if (abs(det) < 1e-15)
                        rotation = real3x3::Identity();
                    else
                    {
                        real3x3 S;
                        PolarDecomposition::polarDecompositionFullSquareRoot(F, rotation, S);
                    }
#endif

					//printf("element=%d -> R=[%5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f; %5.3f %5.3f %5.3f]\n", elementIdx, rotation.r1.x, rotation.r1.y, rotation.r1.z, rotation.r2.x, rotation.r2.y, rotation.r2.z, rotation.r3.x, rotation.r3.y, rotation.r3.z);
				}
			}
			__syncthreads();

            //5. Each thread computes the contribution to the force -> subforces
			const real3 position = h * make_real3(j & 1 ? 1 : 0, j & 2 ? 1 : 0, j & 4 ? 1 : 0);
			//const real3 position = referencePositions.getRawCoeff(nodeIdx[j]);
			subforces[i][j] = rotation.matmul(KePart).matmul(rotation.transpose().matmul(position) - position);
			__syncthreads();

            //6. Each thread updates their Ke_ij
			KePart = rotation.matmul(KePart.matmulT(rotation));

            //7. Threads j=0 reduce the subforces and update outputForce
			if (j == 0)
			{
				real3 Fi = //warp reduce?
					- subforces[i][0] - subforces[i][1] - subforces[i][2] - subforces[i][3]
					- subforces[i][4] - subforces[i][5] - subforces[i][6] - subforces[i][7];
				atomicAddReal3(outputForce.data() + nodeIdx[i], Fi);
			}
        }

        if (Dirichlet)
        {
            if (dirichlet.getRawCoeff(elementIdx))
            {
                //we are on a dirichlet boundary
                
                //8. fetch normal and surface interpolation weights
                const real3 normal = surfaceNormals.getRawCoeff(elementIdx);
                const real8 surfaceWeight = interpolationBoundaryWeights.getRawCoeff(elementIdx);
                const real sweights[8] = {
                    surfaceWeight.first.x, surfaceWeight.first.y, surfaceWeight.first.z, surfaceWeight.first.w,
                    surfaceWeight.second.x, surfaceWeight.second.y, surfaceWeight.second.z, surfaceWeight.second.w
                };

                //9. Nietsche conditions
                real3x3 KeD1( //j,i
                    make_real3(
                        cGridB[i][j][0]*(lambda*normal.x+2*mu*normal.x) + cGridB[i][j][1]*mu*normal.y + cGridB[i][j][2]*mu*normal.z,
                        cGridB[i][j][1]*lambda*normal.x + cGridB[i][j][0]*mu*normal.y,
                        cGridB[i][j][2]*lambda*normal.x + cGridB[i][j][0]*mu*normal.z),
                    make_real3(
                        cGridB[i][j][1]*mu*normal.x + cGridB[i][j][0]*lambda*normal.y,
                        cGridB[i][j][0]*mu*normal.x + cGridB[i][j][1]*(lambda*normal.y+2*mu*normal.y) + cGridB[i][j][2]*mu*normal.z,
                        cGridB[i][j][2]*lambda*normal.y + cGridB[i][j][1]*mu*normal.z),
                    make_real3(
                        cGridB[i][j][2]*mu*normal.x + cGridB[i][j][0]*lambda*normal.z,
                        cGridB[i][j][2]*mu*normal.y + cGridB[i][j][1]*lambda*normal.z,
                        cGridB[i][j][0]*mu*normal.x + cGridB[i][j][1]*mu*normal.y + cGridB[i][j][2]*(lambda*normal.z+2*mu*normal.z))
                );
                real3x3 KeD2( //i,j, transposed
                    make_real3(
                        cGridB[j][i][0]*(lambda*normal.x+2*mu*normal.x) + cGridB[j][i][1]*mu*normal.y + cGridB[j][i][2]*mu*normal.z,
                        cGridB[j][i][1]*mu*normal.x + cGridB[j][i][0]*lambda*normal.y,
                        cGridB[j][i][2]*mu*normal.x + cGridB[j][i][0]*lambda*normal.z),
                    make_real3(
                        cGridB[j][i][1]*lambda*normal.x + cGridB[j][i][0]*mu*normal.y,
                        cGridB[j][i][0]*mu*normal.x + cGridB[j][i][1]*(lambda*normal.y+2*mu*normal.y) + cGridB[j][i][2]*mu*normal.z,
                        cGridB[j][i][2]*mu*normal.y + cGridB[j][i][1]*lambda*normal.z),
                    make_real3(
                        cGridB[j][i][2]*lambda*normal.x + cGridB[j][i][0]*mu*normal.z,
                        cGridB[j][i][2]*lambda*normal.y + cGridB[j][i][1]*mu*normal.z,
                        cGridB[j][i][0]*mu*normal.x + cGridB[j][i][1]*mu*normal.y + cGridB[j][i][2]*(lambda*normal.z+2*mu*normal.z))
                );
                KePart.r1 -= sweights[i] * KeD1.r1 + sweights[j] * KeD2.r1;
                KePart.r2 -= sweights[i] * KeD1.r2 + sweights[j] * KeD2.r2;
                KePart.r3 -= sweights[i] * KeD1.r3 + sweights[j] * KeD2.r3;

                //10. Regularizer
                if (i==j)
                {
                    KePart.r1.x -= dirichletRho * sweights[i];
                    KePart.r2.y -= dirichletRho * sweights[i];
                    KePart.r3.z -= dirichletRho * sweights[i];
                }
            }
        }

        //11. Each thread updates ouputMatrix
        //find the correct entry in the sparse matrix
#if !defined(NDEBUG) && !defined(_NDEBUG)
        bool found = false;
#endif
        SMatrix3x3::StorageIndex start = outputMatrix.getSparsityPattern().JA.getRawCoeff(nodeIdx[i]);
        SMatrix3x3::StorageIndex end = outputMatrix.getSparsityPattern().JA.getRawCoeff(nodeIdx[i] + 1);
        for (SMatrix3x3::StorageIndex k = start; k < end; ++k)
        {
            SMatrix3x3::StorageIndex inner = outputMatrix.getSparsityPattern().IA.getRawCoeff(k);
            if (inner == nodeIdx[j])
            {
                //entry found
#if !defined(NDEBUG) && !defined(_NDEBUG)
                assert(k < outputMatrix.getData().size());
                found = true;
#endif
                atomicAddReal3x3(outputMatrix.getData().data() + k, KePart);
                break;
            }
        }
#if !defined(NDEBUG) && !defined(_NDEBUG)
        assert(found);
#endif
    }

    void SoftBodyGrid3D::computeStiffnessMatrix(const Input& input, const State& lastState, const Settings& settings,
                                                SMatrix3x3& outputMatrix, Vector3X& outputForce)
    {
        GridComputeBasisFunctions(input.grid_->getVoxelResolution());
        cuMat::Context& ctx = cuMat::Context::current();
        dim3 gridDim = dim3(input.numActiveCells_, 1, 1);
        dim3 blockDim = dim3(64, 1, 1);
		real h = static_cast<real>(input.grid_->getVoxelSize());
        if (input.hasDirichlet_ && settings.enableCorotation_)
        {
            GridComputeStiffnessMatrixKernel<true, true>
                <<<gridDim, blockDim, 0, ctx.stream()>>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                 input.dirichlet_, input.mapping_, input.referencePositions_,
                 lastState.displacements_, settings.materialLambda_, settings.materialMu_, h,
				 outputMatrix, outputForce);
        }
        else if (!input.hasDirichlet_ && settings.enableCorotation_)
        {
            GridComputeStiffnessMatrixKernel<false, true>
                <<<gridDim, blockDim, 0, ctx.stream()>>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                 input.dirichlet_, input.mapping_, input.referencePositions_,
                 lastState.displacements_, settings.materialLambda_, settings.materialMu_, h,
				 outputMatrix, outputForce);
        }
        else if (input.hasDirichlet_ && !settings.enableCorotation_)
        {
            GridComputeStiffnessMatrixKernel<true, false>
                <<<gridDim, blockDim, 0, ctx.stream()>>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                 input.dirichlet_, input.mapping_, input.referencePositions_,
                 lastState.displacements_, settings.materialLambda_, settings.materialMu_, h,
				 outputMatrix, outputForce);
        }
        else if (!input.hasDirichlet_ && !settings.enableCorotation_)
        {
            GridComputeStiffnessMatrixKernel<false, false>
                <<<gridDim, blockDim, 0, ctx.stream()>>>
                (input.interpolationVolumeWeights_, input.interpolationBoundaryWeights_, input.surfaceNormals_,
                 input.dirichlet_, input.mapping_, input.referencePositions_,
                 lastState.displacements_, settings.materialLambda_, settings.materialMu_, h,
				 outputMatrix, outputForce);
        }
        CUMAT_CHECK_ERROR();
#if LOG_FLIPPED_ROTATIONS==1
		cudaPrintfDisplay(cinder::app::console());
#endif
    }
}
