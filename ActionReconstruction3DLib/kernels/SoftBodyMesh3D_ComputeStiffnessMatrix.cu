#include "../SoftBodyMesh3D.h"
#include "../PolarDecomposition.h"

namespace ar3d
{
	//Worksize = 16
	template<bool Corotation>
	__global__ void MeshComputeStiffnessMatrixKernel(
		/*const SoftBodyMesh3D::Input input,*/ Vector4Xi indices, Vector3X refPos, int numFreeNodes,
		/*const SoftBodyMesh3D::State lastState,*/ Vector3X lastDisplacements,
		real lambda, real mu,
		SMatrix3x3 outputMatrix, Vector3X outputForce)
    {
		int elementIdx = blockIdx.x;
		int thread = threadIdx.x;
		CUMAT_ASSERT_CUDA(blockDim.x == 16);

		//information about the current element
		int4 elementVertexIndices = indices.getRawCoeff(elementIdx);//input.indices_.getRawCoeff(elementIdx);
		int vertIdx[4] = { elementVertexIndices.x, elementVertexIndices.y, elementVertexIndices.z, elementVertexIndices.w };
#ifndef NDEBUG
		if (thread == 0)
			printf("element %d: indices %d,%d,%d,%d\n", elementIdx, vertIdx[0], vertIdx[1], vertIdx[2], vertIdx[3]);
#endif

		//shared memory configuration
		__shared__ __device__ real3 positions[4];
		__shared__ __device__ real3 displacements[4];
		__shared__ __device__ real deriv[4][3];
		__shared__ __device__ real3x3 rotation;
		__shared__ __device__ real3 subforces[4][4];

		//1. Threads 0-3 load the four positions into shared memory
		if (thread<4)
		{
			positions[thread] = refPos.getRawCoeff(vertIdx[thread]);
			if (Corotation && vertIdx[thread]<numFreeNodes)
				displacements[thread] = lastDisplacements.getRawCoeff(vertIdx[thread]);
		}
		__syncthreads();

		//2. Thread 0-3 compute the subdeterminants that lead to \partial\phi / \partial x,y,z
		const real vol = SoftBodyMesh3D::tetSize(positions[0], positions[1], positions[2], positions[3]);
		assert(vol > 0);
		if (thread<4)
		{
			static const int DetIdx[4][3] {
				{ 3,2,1 },
				{ 3,0,2 },
				{ 1,0,3 },
				{ 1,2,0 }
			};
			deriv[thread][0] = (
				  (positions[DetIdx[thread][0]].y - positions[DetIdx[thread][2]].y) * (positions[DetIdx[thread][1]].z - positions[DetIdx[thread][2]].z)
				- (positions[DetIdx[thread][1]].y - positions[DetIdx[thread][2]].y) * (positions[DetIdx[thread][0]].z - positions[DetIdx[thread][2]].z)
				) / (6 * vol);
			deriv[thread][1] = (
				  (positions[DetIdx[thread][1]].x - positions[DetIdx[thread][2]].x) * (positions[DetIdx[thread][0]].z - positions[DetIdx[thread][2]].z)
				- (positions[DetIdx[thread][0]].x - positions[DetIdx[thread][2]].x) * (positions[DetIdx[thread][1]].z - positions[DetIdx[thread][2]].z)
				) / (6 * vol);
			deriv[thread][2] = (
				  (positions[DetIdx[thread][0]].x - positions[DetIdx[thread][2]].x) * (positions[DetIdx[thread][1]].y - positions[DetIdx[thread][2]].y)
				- (positions[DetIdx[thread][1]].x - positions[DetIdx[thread][2]].x) * (positions[DetIdx[thread][0]].y - positions[DetIdx[thread][2]].y)
				) / (6 * vol);
		}
		__syncthreads();

		//3. Each thread computes Ke_ij
		const int i = thread % 4;
		const int j = thread / 4;
		real3x3 KePart(
			//row 1
			vol * make_real3(
			    (2 * mu + lambda) * deriv[i][0] * deriv[j][0] + mu * (deriv[i][1] * deriv[j][1] + deriv[i][2] * deriv[j][2]),
				mu*deriv[i][1] * deriv[j][0] + lambda * deriv[i][0] * deriv[j][1],
				mu*deriv[i][2] * deriv[j][0] + lambda * deriv[i][0] * deriv[j][2]
			),
			//row 2
			vol * make_real3(
				mu*deriv[i][0] * deriv[j][1] + lambda * deriv[i][1] * deriv[j][0],
				(2 * mu + lambda) * deriv[i][1] * deriv[j][1] + mu * (deriv[i][0] * deriv[j][0] + deriv[i][2] * deriv[j][2]),
				mu*deriv[i][2] * deriv[j][1] + lambda * deriv[i][1] * deriv[j][2]
			),
			//row 3
			vol * make_real3(
				mu*deriv[i][0] * deriv[j][2] + lambda * deriv[i][2] * deriv[j][0],
				mu*deriv[i][1] * deriv[j][2] + lambda * deriv[i][2] * deriv[j][1],
				(2 * mu + lambda) * deriv[i][2] * deriv[j][2] + mu * (deriv[i][0] * deriv[j][0] + deriv[i][1] * deriv[j][1])
			)
		);

		if (Corotation)
		{
			//4. Thread 0 computes the polar decomposition -> rotation
			if (thread == 0)
			{
				real3x3 DrefInv = real3x3(
					positions[0] - positions[3], 
					positions[1] - positions[3], 
					positions[2] - positions[3]
				).transpose().inverse();
				real3x3 Ddef = real3x3(
					positions[0] + displacements[0] - positions[3] - displacements[3],
					positions[1] + displacements[1] - positions[3] - displacements[3],
					positions[2] + displacements[2] - positions[3] - displacements[3]
				).transpose();
				real3x3 F = Ddef.matmul(DrefInv);
				real det = F.det();
				assert(det > 0);
				rotation = det < 1e-15 ? real3x3::Identity() : PolarDecomposition::polarDecomposition(F);
				//rotation = real3x3::Identity();
			}
			__syncthreads();

			//5. Each thread computes the contribution to the force -> subforces
			//subforces[i][j] = KePart.matmul(rotation.matmul(rotation.transpose().matmul(positions[j]) - positions[j]));
			subforces[i][j] = rotation.matmul(KePart).matmul(rotation.transpose().matmul(positions[j]) - positions[j]);
			__syncthreads();

			//6. Each thread updates their Ke_ij
			//KePart = KePart.matmul(rotation.matmulT(rotation));
			KePart = rotation.matmul(KePart.matmulT(rotation));

			//7. Thread j=0 reduce the subforces and update outputForce
			if (j == 0)
			{
				real3 Fi = -subforces[i][0] - subforces[i][1] - subforces[i][2] - subforces[i][3];
				if (vertIdx[i] < numFreeNodes)
					atomicAddReal3(outputForce.data() + vertIdx[i], Fi);
			}
		}
		
    	//8. Each thread updates ouputMatrix
		if (vertIdx[i]<numFreeNodes && vertIdx[j]<numFreeNodes)
		{
			//no dirichlet boundary
			//find the correct entry in the sparse matrix
#if !defined(NDEBUG) && !defined(_NDEBUG)
			bool found = false;
#endif
			SMatrix3x3::StorageIndex start = outputMatrix.getSparsityPattern().JA.getRawCoeff(vertIdx[i]);
			SMatrix3x3::StorageIndex end = outputMatrix.getSparsityPattern().JA.getRawCoeff(vertIdx[i] +1);
			for (SMatrix3x3::StorageIndex k = start; k < end; ++k)
			{
				SMatrix3x3::StorageIndex inner = outputMatrix.getSparsityPattern().IA.getRawCoeff(k);
				if (inner == vertIdx[j])
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
    }

	void SoftBodyMesh3D::computeStiffnessMatrix(const Input& input, const State& lastState, const Settings& settings,
		SMatrix3x3& outputMatrix, Vector3X& outputForce)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		dim3 gridDim = dim3(input.numElements_, 1, 1);
		dim3 blockDim = dim3(16, 1, 1);
		if (settings.enableCorotation_)
		{
			MeshComputeStiffnessMatrixKernel<true>
				<<<gridDim, blockDim, 0, ctx.stream()>>>
				(input.indices_, input.referencePositions_, input.numFreeNodes_, lastState.displacements_, settings.materialLambda_, settings.materialMu_, outputMatrix, outputForce);
		} else
		{
			MeshComputeStiffnessMatrixKernel<false>
				<<<gridDim, blockDim, 0, ctx.stream()>>>
				(input.indices_, input.referencePositions_, input.numFreeNodes_, lastState.displacements_, settings.materialLambda_, settings.materialMu_, outputMatrix, outputForce);
		}
		CUMAT_CHECK_ERROR();
	}
}
