#pragma once

#include <cuda_runtime.h>
#include "helper_matrixmath.h"

namespace ar3d
{
    /**
     * \brief Trilinear interpolation.
     * 
     * The corners are ordered as
     * (X,Y,Z): (-,-,-),(+,-,-),(-,+,-),(+,+,-),(-,-,+),(+,-,+),(-,+,+),(+,+,+)
     * \tparam T the value type
     * \param xyz the interpolation weights
     * \param corners the values at the corners
     * \return the interpolated value
     */
    template <typename T>
    __host__ __device__ __inline__ T trilinear(real3 xyz, T corners[8])
    {
        return (1 - xyz.x)*((1 - xyz.y)*((1 - xyz.z)*corners[0] + xyz.z * corners[4]) + xyz.y * ((1 - xyz.z)*corners[2] + xyz.z * corners[6])) 
            + xyz.x * ((1 - xyz.y)*((1 - xyz.z)*corners[1] + xyz.z * corners[5]) + xyz.y * ((1 - xyz.z)*corners[3] + xyz.z * corners[7]));
    }

    /**
     * \brief Computes the inverse of the trilinear interpolation.
     * It solves for the interpolation weights given the interpolated value
     * 
     * \param pos the interpolated value
     * \param corners the values at the corners
     * \param iterations the number of iterations 
     * \return the interpolation weights
     */
    __host__ __device__ __inline__ real3 trilinearInverse(const real3& pos, const real3 corners[8], int iterations = 5)
    {
        const real3 z0 = corners[0] - pos;
        const real3 z1 = corners[1] - corners[0];
        const real3 z2 = corners[2] - corners[0];
        const real3 z3 = corners[4] - corners[0];
        const real3 z4 = corners[3] - corners[2] - corners[1] + corners[0];
        const real3 z5 = corners[5] - corners[4] - corners[1] + corners[0];
        const real3 z6 = corners[6] - corners[4] - corners[2] + corners[0];
        const real3 z7 = corners[7] - corners[6] - corners[5] - corners[3] + corners[4] + corners[2] + corners[1] - corners[0];
        real3 abc = make_real3(0.5, 0.5, 0.5);
        for (int i=0; i<iterations; ++i)
        {
            real3 F = z0 + abc.x*z1 + abc.y*z2 + abc.z*z3 + abc.x*abc.y*z4 + abc.x*abc.z*z5 + abc.y*abc.z*z6 + abc.x*abc.y*abc.z*z7;
            real3x3 J = real3x3(
                z1 + abc.y*z4 + abc.z*z5 + abc.y*abc.z*z7,
                z2 + abc.x*z4 + abc.z*z6 + abc.x*abc.z*z7,
                z3 + abc.x*z5 + abc.y*z6 + abc.x*abc.y*z7).transpose();
            abc -= J.inverse().matmul(F); //Newton step
            if (abc.x < -1 || abc.y < -1 || abc.z < -1 || abc.x>2 || abc.y>2 || abc.z>2) break; //divergence
        }
        return abc;
    }

    /**
     * \brief Computes the adjoint of the trilinear interpolation
     * \param xyz the interpolation weights
     * \param corners the values at the corners
     * \param xyz the interpolation weights
     * \param adjResult the adjoint of the interpolated value
     * \param adjXYZOut the adjoint of the interpolation weights
     */
    __host__ __device__ __inline__ void trilinearAdjoint(const real3& xyz, const real corners[8], const real& adjResult, real3& adjXYZOut)
    {
        const real z0 = corners[0];
        const real z1 = corners[1] - corners[0];
        const real z2 = corners[2] - corners[0];
        const real z3 = corners[4] - corners[0];
        const real z4 = corners[3] - corners[2] - corners[1] + corners[0];
        const real z5 = corners[5] - corners[4] - corners[1] + corners[0];
        const real z6 = corners[6] - corners[4] - corners[2] + corners[0];
        const real z7 = corners[7] - corners[6] - corners[5] - corners[3] + corners[4] + corners[2] + corners[1] - corners[0];
        adjXYZOut.x += adjResult * (z1 + xyz.y*z4 + xyz.z*z5 + xyz.y*xyz.z*z7);
        adjXYZOut.y += adjResult * (z2 + xyz.x*z4 + xyz.z*z6 + xyz.x*xyz.z*z7);
        adjXYZOut.z += adjResult * (z3 + xyz.x*z5 + xyz.y*z6 + xyz.x*xyz.y*z7);
    }

    /**
     * \brief Computes the adjoint of the inverse of the trilinear interpolation
     * \param xyz the interpolation weights
     * \param corners the values at the corners
     * \param adjXYZ the adjoint of the interpolation weights
     * \param adjCorners the adjoint of the values at the corners
     */
    __host__ __device__ __inline__ void trilinearInvAdjoint(const real3& xyz, const real3 corners[8], const real3& adjXYZ, real3 adjCorners[8])
    {
        const real3 z0 = corners[0];
        const real3 z1 = corners[1] - corners[0];
        const real3 z2 = corners[2] - corners[0];
        const real3 z3 = corners[4] - corners[0];
        const real3 z4 = corners[3] - corners[2] - corners[1] + corners[0];
        const real3 z5 = corners[5] - corners[4] - corners[1] + corners[0];
        const real3 z6 = corners[6] - corners[4] - corners[2] + corners[0];
        const real3 z7 = corners[7] - corners[6] - corners[5] - corners[3] + corners[4] + corners[2] + corners[1] - corners[0];

        const real3x3 Atr = real3x3(
            z1 + xyz.y*z4 + xyz.z*z5 + xyz.y*xyz.z*z7,
            z2 + xyz.x*z4 + xyz.z*z6 + xyz.x*xyz.z*z7,
            z3 + xyz.x*z5 + xyz.y*z6 + xyz.x*xyz.y*z7);
        const real3 xyzPrime = Atr.inverse().matmul(adjXYZ);

        const real3 adjZ0 = -xyzPrime;
        const real3 adjZ1 = -xyz.x * xyzPrime;
        const real3 adjZ2 = -xyz.y * xyzPrime;
        const real3 adjZ3 = -xyz.z * xyzPrime;
        const real3 adjZ4 = -xyz.x*xyz.y * xyzPrime;
        const real3 adjZ5 = -xyz.x*xyz.z * xyzPrime;
        const real3 adjZ6 = -xyz.y*xyz.z * xyzPrime;
        const real3 adjZ7 = -xyz.x*xyz.y*xyz.z * xyzPrime;

        adjCorners[0] += adjZ0 - adjZ1 - adjZ2 - adjZ3 + adjZ4 + adjZ5 + adjZ6 - adjZ7;
        adjCorners[1] += adjZ1 - adjZ4 - adjZ5 + adjZ7;
        adjCorners[2] += adjZ2 - adjZ4 - adjZ6 + adjZ7;
        adjCorners[4] += adjZ3 - adjZ5 - adjZ6 + adjZ7;
        adjCorners[3] += adjZ4 - adjZ7;
        adjCorners[5] += adjZ5 - adjZ7;
        adjCorners[6] += adjZ6 - adjZ7;
        adjCorners[7] += adjZ7;
    }
}