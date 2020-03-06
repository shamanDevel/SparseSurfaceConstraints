#include "../CostFunctions.h"

#include <cuda_runtime.h>
#include <cuMat/Core>
#include "Utils.h"
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>
#include "../MarchingCubes.h"
#include <random>

#if 1

namespace ar3d
{

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

    /* the original jgt code */
    int intersect_triangle(double orig[3], double dir[3],
        double vert0[3], double vert1[3], double vert2[3],
        double *t, double *u, double *v)
    {
        double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
        double det, inv_det;

        /* find vectors for two edges sharing vert0 */
        SUB(edge1, vert1, vert0);
        SUB(edge2, vert2, vert0);

        /* begin calculating determinant - also used to calculate U parameter */
        CROSS(pvec, dir, edge2);

        /* if determinant is near zero, ray lies in plane of triangle */
        det = DOT(edge1, pvec);

        if (det > -EPSILON && det < EPSILON)
            return 0;
        inv_det = 1.0 / det;

        /* calculate distance from vert0 to ray origin */
        SUB(tvec, orig, vert0);

        /* calculate U parameter and test bounds */
        *u = DOT(tvec, pvec) * inv_det;
        if (*u < 0.0 || *u > 1.0)
            return 0;

        /* prepare to test V parameter */
        CROSS(qvec, tvec, edge1);

        /* calculate V parameter and test bounds */
        *v = DOT(dir, qvec) * inv_det;
        if (*v < 0.0 || *u + *v > 1.0)
            return 0;

        /* calculate t, ray intersects triangle */
        *t = DOT(edge2, qvec) * inv_det;

        return 1;
    }

    void CostFunctionPartialObservations::preprocessSetSdfCPU(WorldGridDataPtr<real> advectedSdf)
    {
    }

    CostFunctionPartialObservations::Image CostFunctionPartialObservations::preprocessCreateObservationCPU(
        SimulationResults3DPtr results, int timestep, const DataCamera& camera, int resolution, real noise)
    {
        // MARCHING CUBES ON THE CPU
        std::vector<int> indexBuffer;
        std::vector<MarchingCubes::Vertex> vertexBuffer;
        MarchingCubes::polygonizeGrid(results->input_.referenceSdf_, results->input_.posToIndex_, indexBuffer, vertexBuffer);
        Vector3X positionsDevice = results->input_.referencePositions_ + results->states_[timestep].displacements_;
        std::vector<real3> positionsHost(positionsDevice.size());
        std::vector<real3> vertices;
        positionsDevice.copyToHost(&positionsHost[0]);
        for (const auto& v : vertexBuffer)
        {
            real3 pos = (1 - v.weight) * positionsHost[v.indexA] + v.weight * positionsHost[v.indexB];
            vertices.push_back(pos);
        }

		//prepare noise
		static std::default_random_engine rnd;
		std::normal_distribution<real> distr(real(0), noise>0?noise:1);

        // Raytrace triangles
        typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ImageHost_t;
        ImageHost_t imageHost = ImageHost_t::Constant(resolution, resolution, -1);
#pragma omp parallel for schedule(dynamic)
        for (int x=0; x<resolution; ++x) for (int y=0; y<resolution; ++y)
        {
            glm::vec3 screenPos(x / float(resolution), y / float(resolution), 0.5f);
            glm::vec3 worldPosGlm = camera.getWorldCoordinates(screenPos);
            real3 worldPos = make_real3(worldPosGlm.x, worldPosGlm.y, worldPosGlm.z);
            real3 origin = make_real3(camera.location.x, camera.location.y, camera.location.z);
            real3 dir = worldPos - origin; dir /= length3(dir);

            double originA[3] = { origin.x, origin.y, origin.z };
            double dirA[3] = { dir.x, dir.y, dir.z };
            for (int i=0; i<indexBuffer.size()/3; ++i)
            {
                double vert0[3] = { vertices[indexBuffer[3 * i + 0]].x, vertices[indexBuffer[3 * i + 0]].y, vertices[indexBuffer[3 * i + 0]].z };
                double vert1[3] = { vertices[indexBuffer[3 * i + 1]].x, vertices[indexBuffer[3 * i + 1]].y, vertices[indexBuffer[3 * i + 1]].z };
                double vert2[3] = { vertices[indexBuffer[3 * i + 2]].x, vertices[indexBuffer[3 * i + 2]].y, vertices[indexBuffer[3 * i + 2]].z };
                double u, v, t;
                if (intersect_triangle(originA, dirA, vert0, vert1, vert2, &t, &u, &v)==1)
                {
                    real3 intersection = (t + (noise>0 ? distr(rnd) : 0)) * dir + origin;
                    glm::vec3 screenPos2 = camera.getScreenCoordinates(glm::vec3(intersection.x, intersection.y, intersection.z));
                    assert(screenPos2.z > 0);
                    if (imageHost(x, y) < 0 || imageHost(x, y) > screenPos2.z)
                        imageHost(x, y) = screenPos2.z;
                }
            }
        }

        Image img = Image::fromEigen(imageHost);
        return img;
    }

}

#endif