#include "PolarDecomposition.h"
#include "Utils3D.h"
#include <cmath>
#include <cinder/CinderMath.h>

ar3d::real3x3 ar3d::PolarDecomposition::polarDecomposition(const real3x3& F)
{
    return polarDecompositionIterative(F);
}

ar3d::real3x3 ar3d::PolarDecomposition::polarDecompositionIterative(const real3x3& F, int iterations)
{
    //Corotational Simulation of Deformable Solids, Michael Hauth & Wolfgang Strasser, eq. 4.16 and 4.17
    real3x3 R = F;
    for (int i = 0; i<iterations; ++i)
    {
        R = (0.5*(R + R.transpose().inverse()));
    }
    return R;
}

void ar3d::PolarDecomposition::polarDecompositionFullIterative(const real3x3& F, real3x3& R, real3x3& S, int iterations)
{
    R = polarDecompositionIterative(F, iterations);
    S = R.inverse().matmul(F);
}

void ar3d::PolarDecomposition::polarDecompositionFullSquareRoot(const real3x3& F, real3x3& R, real3x3& S)
{
    S = matrixSquareRoot(F.transpose().matmul(F), 10);
    R = F.matmul(S.inverse());
}

ar3d::real3x3 ar3d::PolarDecomposition::adjointPolarDecompositionAnalytic(const real3x3& F, const real3x3& R, const real3x3& S,
    const real3x3& X)
{
    //eigendecomposition of S
    real eig1, eig2, eig3;
    eigenvalues(S, eig1, eig2, eig3);
    real3 vec1 = eigenvector(S, eig1, eig2, eig3);
    real3 vec2 = eigenvector(S, eig2, eig1, eig3);
    real3 vec3 = eigenvector(S, eig3, eig1, eig2);
    real eig[] = { eig1, eig2, eig3 };
    real3 vec[] = { vec1, vec2, vec3 };

    real3x3 deriv(0);
    real3x3 sk = skew(R.transpose().matmul(X));
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
    {
        deriv += (2 / (eig[i] + eig[j])) * dot3(vec[i], sk.matmul(vec[j]))
            * R.matmul(real3x3::OuterProduct(vec[i], vec[j]) - real3x3::OuterProduct(vec[j], vec[i]));
    }
    return deriv;
}

ar3d::real3x3 ar3d::PolarDecomposition::adjointPolarDecompositionIterative(
    const real3x3& F, const real3x3& R,
    const real3x3& S, const real3x3& X, int iterations)
{
    std::vector<real3x3> Rx(iterations + 1);

    //Forward code:
    Rx[0] = F;
    for (int i = 0; i<iterations; ++i)
    {
        Rx[i+1] = 0.5*(Rx[i] + Rx[i].transpose().inverse());
    }

    //adjoint code
    real3x3 adjR = X;
    for (int i=iterations-1; i>=0; --i)
    {
        adjR = 0.5 * (adjR + adjointMatrixTrInv(Rx[i], adjR));
    }
    return adjR;
}

ar3d::real3x3 ar3d::PolarDecomposition::matrixSquareRoot(const real3x3& A, int iterations)
{
    real3x3 Y = A;
    real3x3 Z = real3x3::Identity();
    for (int i=0; i<iterations; ++i)
    {
        real3x3 Ynew = 0.5 * (Y + Z.inverse());
        real3x3 Znew = 0.5 * (Z + Y.inverse());
        Y = Ynew; Z = Znew;
    }
    return Y;
}

void ar3d::PolarDecomposition::eigenvalues(const real3x3& A, real& eig1, real& eig2, real& eig3)
{
    //source: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices

    // Given a real symmetric 3x3 matrix A, compute the eigenvalues
    // Note that acos and cos operate on angles in radians

    const real p1 = utils::square(A.r1.y) + utils::square(A.r1.z) + utils::square(A.r2.z);
    if (p1 == 0) {
        // A is diagonal.
        eig1 = A.r1.x;
        eig2 = A.r2.y;
        eig3 = A.r3.z;
    }
    else {
        const real q = A.trace() / 3;               // trace(A) is the sum of all diagonal values
        const real p2 = utils::square(A.r1.x - q) + utils::square(A.r2.y - q) + utils::square(A.r3.z - q) + 2 * p1;
        const real p = std::sqrt(p2 / 6);
        const real3x3 B = (1 / p) * (A - q * real3x3::Identity());
        const real r = B.det() / 2;

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        real phi;
        if (r <= -1)
            phi = M_PI / 3;
        else if (r >= 1)
            phi = 0;
        else
            phi = std::acos(r) / 3;

        // the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * std::cos(phi);
        eig3 = q + 2 * p * std::cos(phi + (2 * M_PI / 3));
        eig2 = 3 * q - eig1 - eig3;     // since trace(A) = eig1 + eig2 + eig3
    }
}

ar3d::real3 ar3d::PolarDecomposition::eigenvector(const real3x3& A, real eig1, real eig2, real eig3)
{
    const real3x3 prod = (A - eig2 * real3x3::Identity()).matmul(A - eig3 * real3x3::Identity());
    const real3 vec = make_real3(prod.r1.x, prod.r2.x, prod.r3.x);
    return normalize3(vec);
}

ar3d::real3x3 ar3d::PolarDecomposition::adjointMatrixTrInv(const real3x3& A, const real3x3& adjB)
{
    //real3x3 B = A.transpose().inverse();
    real3x3 Atrinv = A.transpose().inverse();
    //real3x3 adjA =
    //    real3x3::OuterProduct(-Atrinv.matmul(adjB.r1), B.r1) +
    //    real3x3::OuterProduct(-Atrinv.matmul(adjB.r2), B.r2) +
    //    real3x3::OuterProduct(-Atrinv.matmul(adjB.r3), B.r3);
    real3x3 adjA = -Atrinv.matmulT(adjB).matmul(Atrinv);
    return adjA;
}
