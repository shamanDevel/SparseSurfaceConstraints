#include "../CostFunctions.h"

#include <cuda_runtime.h>
#include <cuMat/Core>
#include "Utils.h"
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

#if 1

namespace ar3d
{
    namespace
    {
        //texture<float, cudaTextureType3D, cudaReadModeElementType> texRef;
        //surface<void, cudaSurfaceType3D> surfRef;
        //cudaArray_t array = nullptr;

        //__global__ void copySdfToTextureKernel(dim3 size, WorldGridData<real>::DeviceArray_t sdf)
        //{
        //    CUMAT_KERNEL_3D_LOOP(i, j, k, size)
        //        surf3Dwrite(static_cast<float>(sdf.coeff(i, j, k, -1)), surfRef, i*sizeof(float), j, k);
        //    CUMAT_KERNEL_3D_LOOP_END
        //}

        typedef WorldGridData<real>::DeviceArray_t Grid_t;

        struct RaytracingFunctor
        {
        private:
            glm::vec3 cameraLocation;
            glm::mat4 cameraMatrix;
            glm::mat4 cameraInvMatrix;
            glm::vec3 boxMin;
            glm::vec3 boxSize;
            glm::ivec2 screenSize;
            float stepsize;
            Grid_t grid;
            CostFunctionPartialObservations::Image noise1;
            CostFunctionPartialObservations::Image noise2;
            real noise;

        public:
            RaytracingFunctor(const glm::vec3& camera_location, const glm::mat4& camera_matrix, const glm::mat4& camera_inv_matrix,
                const glm::vec3& box_min, const glm::vec3& box_size, const glm::ivec2& size, float stepsize, const Grid_t& grid,
                CostFunctionPartialObservations::Image noise1, CostFunctionPartialObservations::Image noise2, real noise)
                : cameraLocation(camera_location),
                  cameraMatrix(camera_matrix),
                  cameraInvMatrix(camera_inv_matrix),
                  boxMin(box_min),
                  boxSize(box_size),
                  screenSize(size),
                  stepsize(stepsize),
                  grid(grid),
                  noise1(noise1),
                  noise2(noise2),
                  noise(noise)
            {}

            __device__ __inline__ real at(float x, float y, float z) const
            {
                x *= grid.rows();
                y *= grid.cols();
                z *= grid.batches();
                const cuMat::Index ix = static_cast<int>(x);
                const cuMat::Index iy = static_cast<int>(y);
                const cuMat::Index iz = static_cast<int>(z);
                const real fx = x - ix;
                const real fy = y - iy;
                const real fz = z - iz;
                const real v000 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix)),     max(cuMat::Index(0), min(grid.cols() - 1, iy)),     max(cuMat::Index(0), min(grid.batches() - 1, iz)), -1);
                const real v100 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix + 1)), max(cuMat::Index(0), min(grid.cols() - 1, iy)),     max(cuMat::Index(0), min(grid.batches() - 1, iz)), -1);
                const real v010 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix)),     max(cuMat::Index(0), min(grid.cols() - 1, iy + 1)), max(cuMat::Index(0), min(grid.batches() - 1, iz)), -1);
                const real v110 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix + 1)), max(cuMat::Index(0), min(grid.cols() - 1, iy + 1)), max(cuMat::Index(0), min(grid.batches() - 1, iz)), -1);
                const real v001 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix)),     max(cuMat::Index(0), min(grid.cols() - 1, iy)),     max(cuMat::Index(0), min(grid.batches() - 1, iz + 1)), -1);
                const real v101 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix + 1)), max(cuMat::Index(0), min(grid.cols() - 1, iy)),     max(cuMat::Index(0), min(grid.batches() - 1, iz + 1)), -1);
                const real v011 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix)),     max(cuMat::Index(0), min(grid.cols() - 1, iy + 1)), max(cuMat::Index(0), min(grid.batches() - 1, iz + 1)), -1);
                const real v111 = grid.coeff(max(cuMat::Index(0), min(grid.rows() - 1, ix + 1)), max(cuMat::Index(0), min(grid.cols() - 1, iy + 1)), max(cuMat::Index(0), min(grid.batches() - 1, iz + 1)), -1);
                return (1 - fx)*((1 - fy)*((1 - fz)*v000 + fz * v001) + fy * ((1 - fz)*v010 + fz * v011)) + fx * ((1 - fy)*((1 - fz)*v100 + fz * v101) + fy * ((1 - fz)*v110 + fz * v111));
            }

            __device__ __inline__ real getNoise(cuMat::Index x, cuMat::Index y) const
            {
                //Box-Muller transform
                real Z0 = ::sqrt(-2 * ::logf(noise1.coeff(x, y, 0, -1))) * ::cos(2 * M_PI * noise2.coeff(x, y, 0, -1));
                //add to image
                return Z0 * noise;
            }

            typedef float ReturnType;
            __device__ ReturnType operator()(cuMat::Index x, cuMat::Index y, cuMat::Index /*z*/) const
            {
                //world position
                glm::vec4 screenPos = glm::vec4(glm::vec2(x / float(screenSize.x), y / float(screenSize.y)) * 2.0f - glm::vec2(1.0, 1.0), 0.999999, 1);
                glm::vec4 worldPos = cameraInvMatrix * screenPos;
                worldPos.x /= worldPos.w;
                worldPos.y /= worldPos.w;
                worldPos.z /= worldPos.w;

                //ray direction
                glm::vec3 rayDir = normalize(glm::vec3(worldPos) - cameraLocation);
                float depth = length(glm::vec3(worldPos) - cameraLocation);
                glm::vec3 invRayDir = glm::vec3(1,1,1) / rayDir;

                //entry, exit points
                float t1 = (boxMin.x - cameraLocation.x) * invRayDir.x;
                float t2 = (boxMin.x + boxSize.x - cameraLocation.x) * invRayDir.x;
                float t3 = (boxMin.y - cameraLocation.y) * invRayDir.y;
                float t4 = (boxMin.y + boxSize.y - cameraLocation.y) * invRayDir.y;
                float t5 = (boxMin.z - cameraLocation.z) * invRayDir.z;
                float t6 = (boxMin.z + boxSize.z - cameraLocation.z) * invRayDir.z;
                float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
                float tmax = min(depth, min(min(max(t1, t2), max(t3, t4)), max(t5, t6)));
                if (tmax < 0 || tmin > tmax)
                {
                    return -1;
                }

                //perform stepping
                glm::vec3 pos = cameraLocation + tmin * rayDir;
                //cuPrintf_D("x=%d, y=%d -> start=(%5.3f,%5.3f,%5.3f) dir=(%5.3f,%5.3f,%5.3f), tmin=%5.3f, tmax=%5.3f\n", x, y, pos.x, pos.y, pos.z, rayDir.x, rayDir.y, rayDir.z, tmin, tmax);
                for (float sampleDepth = max(0.0f, tmin); sampleDepth < tmax; sampleDepth += stepsize)
                {
                    pos = cameraLocation + sampleDepth * rayDir;
                    glm::vec3 volPos = (pos - boxMin) / boxSize;
                    float val = at(volPos.x, volPos.y, volPos.z);
                    if (val < 0) {
                        if (noise>0)
                        {
                            //add noise
                            pos = cameraLocation + float(sampleDepth + getNoise(x, y)) * rayDir;
                        }
                        //compute depth
                        glm::vec4 posScreen = cameraMatrix * glm::vec4(pos, 1.0);
                        return ((posScreen.z / posScreen.w) + 1.0f) / 2.0f;
                    }
                }
                return -1;
            }

        };
    }
    

    void CostFunctionPartialObservations::preprocessSetSdfGPU(WorldGridDataPtr<real> advectedSdf)
    {
        //static WorldGridPtr lastGrid = nullptr;

        //if (advectedSdf==nullptr)
        //{
        //    //cleanup
        //    if (array!=nullptr)
        //        CUMAT_SAFE_CALL(cudaFreeArray(array));
        //    array = nullptr;
        //    CI_LOG_I("Cleaned up");
        //    return;
        //}

        //size_t w = advectedSdf->getGrid()->getSize().x();
        //size_t h = advectedSdf->getGrid()->getSize().y();
        //size_t d = advectedSdf->getGrid()->getSize().z();

        //if (lastGrid==nullptr || (lastGrid->getSize().array() != advectedSdf->getGrid()->getSize().array()).any())
        //{
        //    //clean old
        //    if (array != nullptr) {
        //        CUMAT_SAFE_CALL(cudaFreeArray(array));
        //        CI_LOG_I("old array deleted");
        //    }

        //    //allocate new
        //    lastGrid = advectedSdf->getGrid();
        //    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        //    CUMAT_SAFE_CALL(cudaMalloc3DArray(&array, &channelDesc, cudaExtent{ w, h, d }, cudaArraySurfaceLoadStore));
        //    CUMAT_SAFE_CALL(cudaBindSurfaceToArray(surfRef, array));
        //    CUMAT_SAFE_CALL(cudaBindTextureToArray(texRef, array));

        //    texRef.channelDesc = channelDesc;
        //    texRef.filterMode = cudaFilterModeLinear;
        //    texRef.addressMode[0] = cudaAddressModeClamp;
        //    texRef.addressMode[1] = cudaAddressModeClamp;
        //    texRef.addressMode[2] = cudaAddressModeClamp;
        //    CI_LOG_I("new array created of size " << w << "," << h << "," << d);
        //}

        ////Invoke copy kernel
        //cuMat::Context& ctx = cuMat::Context::current();
        //cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
        //    static_cast<unsigned>(w), static_cast<unsigned>(h), static_cast<unsigned>(d),
        //    copySdfToTextureKernel);
        //copySdfToTextureKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
        //    (cfg.virtual_size, advectedSdf->getDeviceMemory());
        //CUMAT_CHECK_ERROR();
        //CI_LOG_I("SDF copied to texture memory");
    }

    struct TestFunctor
    {
        CostFunctionPartialObservations::Image noise1;
        CostFunctionPartialObservations::Image noise2;
        real noise;
        typedef real ReturnValue;
        __device__ __inline__ real operator()(cuMat::Index x, cuMat::Index y, cuMat::Index z) const
        {
            //Box-Muller transform
            real Z0 = ::sqrt(-2 * ::logf(noise1.coeff(x, y, 0, -1))) * ::cos(2 * M_PI * noise2.coeff(x, y, 0, -1));
            //add to image
            return Z0 * noise;
        }
    };

    CostFunctionPartialObservations::Image CostFunctionPartialObservations::preprocessCreateObservationGPU(
        SimulationResults3DPtr results, int timestep, const DataCamera& camera, int resolution,
        cuMat::SimpleRandom& rnd, real noise)
    {
        cuMat::Index w = results->states_[timestep].advectedSDF_->getGrid()->getSize().x();
        cuMat::Index h = results->states_[timestep].advectedSDF_->getGrid()->getSize().y();
        cuMat::Index d = results->states_[timestep].advectedSDF_->getGrid()->getSize().z();

        //create two noise images for gaussian noise
        Image noise1;
        Image noise2;
        if (noise > 0) {
            noise1 = Image(resolution, resolution);
            noise2 = Image(resolution, resolution);
            rnd.fillUniform(noise1, real(1e-10), real(1));
            rnd.fillUniform(noise2, real(0), real(1));
            ////Test noise
            //TestFunctor testFunctor = { noise1, noise2, noise };
            //Image testNoise = Image::NullaryOp_t<TestFunctor>(resolution, resolution, 1, testFunctor);
            //cinder::app::console() << "Test Noise:\n" << testNoise << std::endl;
        }

        RaytracingFunctor functor(
            camera.location,
            camera.viewProjMatrix,
            camera.invViewProjMatrix,
            ar::utils::toGLM(((results->states_[timestep].advectedSDF_->getGrid()->getOffset().cast<double>()/* - Eigen::Vector3d(0.5, 0.5, 0.5)*/) * results->states_[timestep].advectedSDF_->getGrid()->getVoxelSize()).eval()),
            ar::utils::toGLM((results->states_[timestep].advectedSDF_->getGrid()->getSize().cast<double>() * results->states_[timestep].advectedSDF_->getGrid()->getVoxelSize()).eval()),
            glm::ivec2(resolution, resolution),
            static_cast<float>(results->states_[timestep].advectedSDF_->getGrid()->getVoxelSize() * 0.01),
            results->states_[timestep].advectedSDF_->getDeviceMemory(),
            noise1, noise2, noise
        );

        Image img = Image::NullaryOp_t<RaytracingFunctor>(
            resolution, resolution, 1, functor);
        //cudaPrintfDisplay_D(cinder::app::console()); cinder::app::console() << std::endl;
        return img;
    }

}

#endif
