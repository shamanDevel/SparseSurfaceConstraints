#include "../CommonKernels.h"

#include <cuMat/Core>
#include <cuMat/Sparse>
//#include "../DebugUtils.h"
//#include <cinder/app/AppBase.h>

namespace ar3d
{
	void CommonKernels::newmarkTimeIntegration(const SMatrix3x3& stiffness, const Vector3X& forces, const VectorX& mass,
		const Vector3X& prevDisplacement, const Vector3X& prevVelocity, real dampingMass, real dampingStiffness,
		real timestep, SMatrix3x3& A, Vector3X& b, real theta)
	{
		//matrix
		A.inplace() = ((1 / (theta*timestep) + dampingMass) * mass).cast<real3>().asDiagonal() //mass + Rayleight mass
			+ real3x3(dampingStiffness + theta * timestep) * stiffness.direct(); //Rayleight damping: stiffness + Whole stiffness matrix

		//vector
        b.inplace()
            = (
                ((1 / (theta*timestep) + dampingMass) * mass).cast<real3>().asDiagonal() + 
                real3x3(dampingStiffness - (1 - theta) * timestep) * stiffness.direct()
              ).sparseView<cuMat::CSR>(stiffness.getSparsityPattern())
              * prevDisplacement
            + ((1 / theta) * mass).cast<real3>().cwiseMul(prevVelocity)
            + make_real3(timestep) * forces;
		//SMatrix3x3 tmp(A.getSparsityPattern());
		//b.setZero();
		//tmp.inplace() = ((1 / (theta*timestep) + dampingMass) * mass).cast<real3>().asDiagonal() + real3x3(dampingStiffness + (1 - theta) * timestep) * stiffness.direct();
		//b += tmp * prevDisplacement; //I have to save the matrix in SMatrix3x3 in between, otherwise, the Dense Matmul would be triggered
		//b += ((1 / theta) * mass).cast<real3>().cwiseMul(prevVelocity) + make_real3(timestep) * forces;
	}

    void CommonKernels::adjointNewmarkTimeIntegration(
        const SMatrix3x3& stiffness, const Vector3X& forces,
        const VectorX& mass, const Vector3X& prevDisplacement, const Vector3X& prevVelocity, real dampingMass,
        real dampingStiffness, const SMatrix3x3& adjA, const Vector3X& adjB, SMatrix3x3& adjStiffness,
        Vector3X& adjForces, VectorX& adjMass, Vector3X& adjPrevDisplacement, Vector3X& adjPrevVelocity,
        DeviceScalar& adjDampingMass, DeviceScalar& adjDampingStiffness, real timestep, real theta)
    {
        adjForces += make_real3(timestep) * adjB;

        adjPrevVelocity += ((1 / theta) * mass).cast<real3>().cwiseMul(adjB);

        adjPrevDisplacement += (
                ((1 / (theta*timestep) + dampingMass) * mass).cast<real3>().asDiagonal() +
                real3x3(dampingStiffness - (1 - theta) * timestep) * stiffness.direct()
            ).sparseView<cuMat::CSR>(stiffness.getSparsityPattern()) * adjB;
        
	    adjMass += (
            make_real3(1 / theta)*adjB*prevVelocity.transpose() +
            make_real3(1 / (theta*timestep) + dampingMass)*adjB*prevDisplacement.transpose() +
            real3x3(1 / (theta*timestep) + dampingMass)*adjA).diagonal().template unaryExpr<Real3SumFunctor>();
        
	    adjStiffness +=
            make_real3(dampingStiffness - (1 - theta)*timestep)*adjB*prevDisplacement.transpose() +
            real3x3(dampingStiffness + theta * timestep)*adjA.direct();

        const DeviceScalar adjDampingMassTmp = mass.dot((adjA + adjB * prevDisplacement.transpose()).diagonal().template unaryExpr<Real3SumFunctor>());
        adjDampingMass += adjDampingMassTmp; //Reduction does not support compound-assignment yet (issue #2 in cuMat)

	    SMatrix3x3 tmp(stiffness.getSparsityPattern());
        tmp = adjA.direct() + adjB * prevDisplacement.transpose();
        //CI_LOG_I("stiffness:\n" << DebugUtils::matrixToEigen(stiffness));
        //CI_LOG_I("tmp:\n" << DebugUtils::matrixToEigen(tmp));
        const DeviceScalar adjDampingStiffnessTmp = stiffness.getData().dot(tmp.getData());
        //cudaPrintfDisplay(cinder::app::console());
        //CI_LOG_I("adjDampingStiffnessTmp=" << static_cast<float>(adjDampingStiffnessTmp));
        adjDampingStiffness += adjDampingStiffnessTmp;
    }

	void CommonKernels::newmarkComputeVelocity(const Vector3X& prevDisplacement, const Vector3X& prevVelocity,
		const Vector3X& currentDisplacement, Vector3X& currentVelocity, real timestep, real theta)
	{
        //                                                                                                       (1-theta)/theta = (1/theta)-1
		currentVelocity.inplace() = make_real3(1 / (theta*timestep)) * (currentDisplacement - prevDisplacement) - make_real3((1 / theta) - 1) * prevVelocity;
	}

	void CommonKernels::adjointNewmarkComputeVelocity(const Vector3X& adjCurrentVelocity,
		Vector3X& adjCurrentDisplacement, Vector3X& adjPrevVelocity, Vector3X& adjPrevDisplacement, real timestep,
		real theta)
	{
		adjCurrentDisplacement += make_real3(1 / (theta * timestep)) * adjCurrentVelocity;
		adjPrevVelocity += make_real3(1 - 1 / theta) * adjCurrentVelocity;
		adjPrevDisplacement += make_real3(-1 / (theta * timestep)) * adjCurrentVelocity;
	}
}
