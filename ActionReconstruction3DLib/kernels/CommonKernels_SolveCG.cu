#include "../CommonKernels.h"

#include <cuMat/src/ConjugateGradient.h>
#include <cinder/Log.h>

namespace ar3d
{
	
	void CommonKernels::solveCG(const SMatrix3x3& A, const Vector3X& b, Vector3X& x, 
		int& iterations, real& tolerance)
	{
		cuMat::ConjugateGradient<SMatrix3x3> cg(A);
		cg.setMaxIterations(iterations);
		cg.setTolerance(tolerance);
		x = cg.solveWithGuess(b, x);
		if (cg.error() > tolerance) {
			CI_LOG_E("CG failed to converge after " << iterations << " iterations, error is " << cg.error());
			//std::cout << "CG failed to converge after " << iterations << " iterations, error is " << cg.error() << std::endl;
		}
		else if (cg.iterations() == 0)
		{
			CI_LOG_E("CG finished after zero iterations, RHS too small?");
			std::cout << "CG finished after zero iterations, RHS too small? Tolerance was " << tolerance << std::endl;
		}
		else {
			CI_LOG_I("CG finished after " << cg.iterations() << " iterations with an error of " << cg.error());
			//std::cout << "CG finished after " << cg.iterations() << " iterations with an error of " << cg.error() << std::endl;
		}
        iterations = cg.iterations();
        tolerance = cg.error();
	}

	bool CommonKernels::adjointSolveCG(
		const SMatrix3x3& A, const Vector3X& b, const Vector3X& x,
		const Vector3X& adjX, SMatrix3x3& adjA, Vector3X& adjB,
		const int iterations, real tolerance)
	{
		Vector3X adjTmp = adjX.deepClone();
        int itCopy = iterations;
		solveCG(A, adjX, adjTmp, itCopy, tolerance);
		adjB += adjTmp;
		adjA += (-adjTmp) * x.transpose();
        return itCopy < iterations; // itCopy==iterations -> not converged -> return false
	}
}
