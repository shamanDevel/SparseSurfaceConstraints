#include "../AdjointSolver.h"
#include "../CommonKernels.h"
#include "../Utils3D.h"

namespace ar3d
{
    __device__ __inline__ void groundDistanceAdjoint(
        const real4& groundPlane, const real3& position,
        const real3& adjNormal, real adjDistance,
        real4& adjGroundPlane, real3& adjPosition)
    {
        adjGroundPlane.x += adjNormal.x + adjDistance * position.x;
        adjGroundPlane.y += adjNormal.y + adjDistance * position.y;
        adjGroundPlane.z += adjNormal.z + adjDistance * position.z;
        adjGroundPlane.w -= adjDistance;
        adjPosition += adjDistance * groundPlane;
    }

    __device__ __inline__ void groundDistanceDtAdjoint(
        const real4& groundPlane, const real3& velocity,
        real adjDistance,
        real4& adjGroundPlane, real3& adjVelocity)
    {
        adjGroundPlane.x += adjDistance * velocity.x;
        adjGroundPlane.y += adjDistance * velocity.y;
        adjGroundPlane.z += adjDistance * velocity.z;
        adjVelocity += adjDistance * make_real3(groundPlane.x, groundPlane.y, groundPlane.z);
    }

    __global__ void GridAdjointApplyCollisionForcesKernel(dim3 size,
        const real3* refPositions, const real3* displacements, const real3* velocities,
        const int4* mapping, const real8* sdfs, const real8* surfaceWeights,
        real4 groundPlane, real groundStiffness, real softminAlpha, real timestep, real theta,
        const real3* adjForces,
        real3* adjDisplacementsOut, real3* adjVelocitiesOut, real4* adjGroundPlaneOut)
    {
        CUMAT_KERNEL_1D_LOOP(elementIdx, size)
            //node indices
            const int4 map = mapping[elementIdx];
            const int nodeIdx[8] = { map.x, map.x + 1, map.y, map.y + 1, map.z, map.z + 1, map.w, map.w + 1 };
            //load position + displacement + velocity + init forces
            real3 posx[8];
            real3 velx[8];
            float3 forcex[8];
            #pragma unroll
            for (int i = 0; i<8; ++i)
            {
                posx[i] = refPositions[nodeIdx[i]] + displacements[nodeIdx[i]];
                velx[i] = velocities[nodeIdx[i]];
                forcex[i] = make_float3(0);
            }

            //prepare adjoint output
            real4 adjGroundPlane = make_real4(0, 0, 0, 0);
            real3 adjPosx[8] = {};
            real3 adjVelx[8] = {};

            //prepare intersection points
            static int EDGES[12][2] = {
                { 0, 1 },
                { 2, 3 },
                { 0, 2 },
                { 1, 3 },
                { 4, 5 },
                { 6, 7 },
                { 4, 6 },
                { 5, 7 },
                { 0, 4 },
                { 1, 5 },
                { 2, 6 },
                { 3, 7 }
            };
            real8 phiTmp = sdfs[elementIdx];
            real phi[8] = { phiTmp.first.x, phiTmp.first.y, phiTmp.first.z, phiTmp.first.w, phiTmp.second.x, phiTmp.second.y, phiTmp.second.z, phiTmp.second.w };

            //adjoint: integrate the force over the surface
            real8 sw = surfaceWeights[elementIdx];
            real3 adjForceX[8] = {
                adjForces[nodeIdx[0]] * sw.first.x,
                adjForces[nodeIdx[1]] * sw.first.y,
                adjForces[nodeIdx[2]] * sw.first.z,
                adjForces[nodeIdx[3]] * sw.first.w,
                adjForces[nodeIdx[4]] * sw.second.x,
                adjForces[nodeIdx[5]] * sw.second.x,
                adjForces[nodeIdx[6]] * sw.second.x,
                adjForces[nodeIdx[7]] * sw.second.x
            };

            //for each of those intersection points, compute the adjoint
            for (int i=0; i<12; ++i)
            {
                //find intersection
                real intersection = phi[EDGES[i][0]] / (phi[EDGES[i][0]] - phi[EDGES[i][1]]);
                if (intersection < 0 || intersection > 1 || isnan(intersection)) continue;

                //get interpolated collision point on the edge
                real3 pos = posx[EDGES[i][0]] * (1 - intersection) + posx[EDGES[i][1]] * intersection;
                real3 vel = velx[EDGES[i][0]] * (1 - intersection) + velx[EDGES[i][1]] * intersection;

                //collide them against the ground -> compute force
                real4 normalDist = SoftBodySimulation3D::groundDistance(groundPlane, pos);
                real softmin = ar3d::utils::softmin(normalDist.w, softminAlpha);
                real distDt = SoftBodySimulation3D::groundDistanceDt(groundPlane, vel);
                real fCurrent = -groundStiffness * softmin;
                real fDt = -groundStiffness * (ar3d::utils::softminDx(normalDist.w, softminAlpha) * distDt);
                real fNext = fCurrent + timestep * fDt;
                real f = theta * fNext + (1 - theta) * fCurrent;
                if (f <= 1e-10) continue;
                real3 fVec = make_real3(normalDist.x, normalDist.y, normalDist.z) * f;

                //adjoint: blend them into the forces
                real adjIntersection = dot3(adjForceX[EDGES[i][0]] - adjForceX[EDGES[i][1]], fVec);
                real3 adjFVec = (1 - intersection) * adjForceX[EDGES[i][0]] + intersection * adjForceX[EDGES[i][1]];

                //adjoint: collide them against the ground
                real3 adjNormal = f * adjFVec; real adjF = dot3(normalDist, adjFVec);
                real adjFNext = theta * adjF; real adjFCurrent = (1 - theta) * adjF;
                adjFCurrent += adjFNext; real adjFDt = timestep * adjFNext;
                real adjDistDt = -groundStiffness * ar3d::utils::softminDx(normalDist.w, softminAlpha) * adjFDt;
                real adjDist = utils::softminDxAdjoint(normalDist.w, softminAlpha, -groundStiffness * distDt * adjFDt);
                real adjSoftmin = -groundStiffness * adjFCurrent;
                real3 adjVel = make_real3(0), adjPos = make_real3(0);
                groundDistanceDtAdjoint(groundPlane, vel, adjDistDt, adjGroundPlane, adjVel);
                adjDist += utils::softminAdjoint(normalDist.w, softminAlpha, adjSoftmin);
                groundDistanceAdjoint(groundPlane, pos, adjNormal, adjDist, adjGroundPlane, adjPos);

                //adjoint: interpolated collision point
                adjVelx[EDGES[i][0]] = (1 - intersection) * adjVel;
                adjVelx[EDGES[i][1]] = intersection * adjVel;
                adjPosx[EDGES[i][0]] = (1 - intersection) * adjPos;
                adjPosx[EDGES[i][1]] = intersection * adjPos;
            }

            //write result
            atomicAddReal4(adjGroundPlaneOut, adjGroundPlane);
            #pragma unroll
            for (int i = 0; i<8; ++i)
            {
                atomicAddReal3(adjDisplacementsOut + nodeIdx[i], adjPosx[i]);
                atomicAddReal3(adjVelocitiesOut + nodeIdx[i], adjVelx[i]);
            }

        CUMAT_KERNEL_1D_LOOP_END
    }

    void AdjointSolver::adjointApplyCollisionForces(const Input& input, const SoftBodySimulation3D::Settings& settings,
        const Vector3X& displacements, const Vector3X& velocities, const Vector3X& adjBodyForces,
        Vector3X& adjDisplacementsOut, Vector3X& adjVelocitiesOut, double4& adjGroundPlaneOut)
    {
        cuMat::Context& ctx = cuMat::Context::current();
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveCells_, GridAdjointApplyCollisionForcesKernel);

        cuMat::Matrix<real4, 1, 1, 1, 0> adjGroundPlane;
        adjGroundPlane.setZero();

        GridAdjointApplyCollisionForcesKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
            cfg.virtual_size,
            input.referencePositions_.data(), displacements.data(), velocities.data(),
            input.mapping_.data(), input.cellSdfs_.data(), input.interpolationBoundaryWeights_.data(),
            settings.groundPlane_, settings.groundStiffness_, settings.softmaxAlpha_, settings.timestep_, settings.newmarkTheta_,
            adjBodyForces.data(),
            adjDisplacementsOut.data(), adjVelocitiesOut.data(), adjGroundPlane.data());
        CUMAT_CHECK_ERROR();

        real4 adjPlane = static_cast<real4>(adjGroundPlane);
        adjGroundPlaneOut += make_double4(adjPlane.x, adjPlane.y, adjPlane.z, adjPlane.w);
    }
}
