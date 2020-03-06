#pragma once

#include "Commons3D.h"
#include <mutex>
#include <Eigen/Core>
#include <cuMat/Core>
#include <memory>
#include <cstring>
#include <cinder/CinderMath.h>
#include <cinder/gl/Texture.h>
#include <cinder/Log.h>
#include <cuda_gl_interop.h>
#include <istream>
#include <ostream>
#include "release_assert.h"

#define WORLD_GRID_NO_INTEROP 1

namespace ar3d
{
    /**
     * \brief Describes a grid of any data within the 3d world.
     * The grid can lie anywhere in the world, but it is aligned to an underlying meta-grid. This allows multiple grids to be combined into a bigger grid.
     * This grid does not actually store any data, this is done in WorldGridData
     */
    class WorldGrid
    {
    private:
        //The size of each voxel as a fraction of 1 world unit
        //voxelResolution=4 -> voxel side length = 1/4 world units
        int voxelResolution_;

        //The offsets in voxels from where the grid starts (lower corner, -x, -y, -z)
        Eigen::Vector3i offset_;

        //The size of the grid in voxels
        Eigen::Vector3i size_;

    public:
        WorldGrid(int voxelResolution, const Eigen::Vector3i& offset, const Eigen::Vector3i& size);

        int getVoxelResolution() const { return voxelResolution_; }
        double getVoxelSize() const { return 1.0 / voxelResolution_; }
        const Eigen::Vector3i& getOffset() const { return offset_; }
        void setOffset(const Eigen::Vector3i& offset) { offset_ = offset; }
        const Eigen::Vector3i& getSize() const { return size_; }
		int getNumVoxels() const { return size_.prod(); }

		//Loads the WorldGrid from a binary input stream
		static std::shared_ptr<WorldGrid> load(std::istream& i)
		{
			int resolution;
			Eigen::Vector3i offset, size;
			i.read((char*)&resolution, sizeof(int));
			i.read((char*)&offset, 3 * sizeof(int));
			i.read((char*)&size, 3 * sizeof(int));
			return std::make_shared<WorldGrid>(resolution, offset, size);
		}
		//Saves this WorldGrid to a binary output stream
		void save(std::ostream& o)
		{
			o.write((char*)&voxelResolution_, sizeof(int));
			o.write((char*)&offset_, 3*sizeof(int));
			o.write((char*)&size_, 3*sizeof(int));
		}
    };
    typedef std::shared_ptr<WorldGrid> WorldGridPtr;

    namespace internal {
        //Provides the conversion from CPU-storage to GPU 3d texture
        template<typename T> struct TypeToOpenGL
        {
            enum {
                dataType = GL_NONE,
                internalFormat = GL_NONE,
                dataFormat = GL_NONE,
                bytesPerPixel = 0,
				needsConversion = false,
            };
            //static void convert(void* dst, const float* src, size_t size) {
            //    static_assert(false, "Unknown type");
            //}
        };
        template<>
        struct TypeToOpenGL<float>
        {
            enum {
                dataType = GL_FLOAT,
                internalFormat = GL_R32F,
                dataFormat = GL_RED,
                bytesPerPixel = 4,
				needsConversion = false,
            };
            static void convert(void* dst, const float* src, size_t size) {
                memcpy(dst, src, size * 4);
            }
        };
        template<>
        struct TypeToOpenGL<double>
        {
            enum {
                dataType = GL_FLOAT,
                internalFormat = GL_R32F,
                dataFormat = GL_RED,
                bytesPerPixel = 4,
				needsConversion = true,
            };
            static void convert(void* dst, const double* src, size_t size) {
                float* dstf = static_cast<float*>(dst);
                for (size_t i = 0; i < size; ++i) dstf[i] = static_cast<float>(src[i]);
            }
        };
    }

    //Container for the data in the grid
    template<typename T>
    class WorldGridData
    {
    public:
		typedef WorldGridData<T> Type;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> HostArray_t;
		typedef cuMat::Matrix<T, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> DeviceArray_t;
		CUMAT_DISALLOW_COPY_AND_ASSIGN(WorldGridData);

    private:
		WorldGridPtr grid_;
        HostArray_t hostMemory_;
		DeviceArray_t deviceMemory_;

        //texture reference
        cinder::gl::Texture3dRef tex_;
        bool texValid_;
        std::mutex mutex_;

    public:
        //Creates a new data grid for the specified world grid
        WorldGridData(WorldGridPtr grid)
            : grid_(grid)
            , texValid_(false)
        {}
		~WorldGridData()
        {}

		WorldGridPtr getGrid() const { return grid_; }

		bool hasHostMemory() const { return hostMemory_.size() > 0; }
		bool hasDeviceMemory() const { return deviceMemory_.size() > 0; }

		void allocateHostMemory() { hostMemory_.resize(grid_->getNumVoxels()); }
		void allocateDeviceMemory() { deviceMemory_ = DeviceArray_t(grid_->getSize().x(), grid_->getSize().y(), grid_->getSize().z()); }

		const HostArray_t& getHostMemory() const { assert(hasHostMemory());  return hostMemory_; }
		HostArray_t& getHostMemory() { assert(hasHostMemory()); return hostMemory_; }
		template<typename Derived> 
    	void setHostMemory(const Eigen::EigenBase<Derived>& mem)
        {
	        hostMemory_ = mem.derived(); 
        	release_assert(hostMemory_.size() == grid_->getNumVoxels());
        }

		const DeviceArray_t& getDeviceMemory() const { assert(hasDeviceMemory()); return deviceMemory_; }
		DeviceArray_t& getDeviceMemory() { assert(hasDeviceMemory()); return deviceMemory_; }
		template<typename Derived>
		void setDeviceMemory(const cuMat::MatrixBase<Derived>& mem)
		{
			deviceMemory_ = mem.derived();
			release_assert(deviceMemory_.rows() == grid_->getSize().x());
			release_assert(deviceMemory_.cols() == grid_->getSize().y());
			release_assert(deviceMemory_.batches() == grid_->getSize().z());
		}

		void copyHostToDevice()
        {
			release_assert(hasHostMemory());
			if (!hasDeviceMemory()) allocateDeviceMemory();
			deviceMemory_.copyFromHost(hostMemory_.data());
        }
		void copyDeviceToHost()
        {
			release_assert(hasDeviceMemory());
			if (!hasHostMemory()) allocateHostMemory();
			deviceMemory_.copyToHost(hostMemory_.data());
        }

        void requireHostMemory()
        {
            if (hasHostMemory()) return;
            copyDeviceToHost();
        }
        void requireDeviceMemory()
        {
            if (hasDeviceMemory()) return;
            copyHostToDevice();
        }

		//Loads the WorldGridData from a binary input stream.
		//If you only need the dimensions, then WorldGrid::load(i) can be used.
		//The data format is independent of the floating point type (float or double)
		static std::shared_ptr<Type> load(std::istream& i)
		{
			WorldGridPtr grid = WorldGrid::load(i);
			std::cout << "grid info loaded: " << grid->getSize().x()
        		<< "," << grid->getSize().y() << "," << grid->getSize().z() << std::endl;
			std::shared_ptr<Type> sdf = std::make_shared<Type>(grid);
			sdf->allocateHostMemory();

			const size_t n = grid->getSize().prod();
			std::cout << "load " << n << " floats" << std::endl;
			std::vector<float> data(n);
			i.read(reinterpret_cast<char*>(&data[0]), sizeof(float) * n);
			for (size_t j = 0; j < n; ++j) {
				sdf->getHostMemory()[j] = static_cast<real>(data[j]);
			}
			if (i)
				std::cout << "all characters read successfully." << std::endl;
			else {
				std::cout << "error: only " << i.gcount() << " could be read" << std::endl;
				throw std::exception("Unable to read SDF file");
			}

			return sdf;
		}
		//Saves this WorldGridData to a binary output stream
		void save(std::ostream& o)
		{
			grid_->save(o);

			const size_t n = grid_->getSize().prod();
			std::vector<float> data(n);
			o.write(reinterpret_cast<char*>(&data[0]), sizeof(float) * n);
			for (size_t j = 0; j < n; ++j) {
				data[j] = static_cast<float>(hostMemory_[j]);
			}
		}

    private:
        void assertBounds(int x, int y, int z) const
        {
            if (x < 0 || x >= grid_->getSize().x()) throw std::exception("Out-Of-Bounds: 0 <= x < grid->getSize().x() violated");
            if (y < 0 || y >= grid_->getSize().y()) throw std::exception("Out-Of-Bounds: 0 <= y < grid->getSize().y() violated");
            if (z < 0 || z >= grid_->getSize().z()) throw std::exception("Out-Of-Bounds: 0 <= z < grid->getSize().z() violated");
        }

    public:
        //Read-only access to a voxel in the grid
        const T& getHost(int x, int y, int z) const
        {
			assert(hasHostMemory());
            assertBounds(x, y, z);
            return hostMemory_[x + grid_->getSize().x() * (y + grid_->getSize().y() * z)];
        }
        //Read-write access to a voxel in the grid
        T& atHost(int x, int y, int z)
        {
			assert(hasHostMemory());
            assertBounds(x, y, z);
            return hostMemory_[x + grid_->getSize().x() * (y + grid_->getSize().y() * z)];
        }
        size_t toLinear(int x, int y, int z) const
        {
            assertBounds(x, y, z);
            return x + grid_->getSize().x() * (y + grid_->getSize().y() * z);
        }

        //Invalidates the texture, can be called from any thread.
        void invalidateTexture() { texValid_ = false; }

		enum class DataSource
        {
	        HOST, 
        	DEVICE,
			//First Device is attempted, then Host
			AUTO
        };

#if WORLD_GRID_NO_INTEROP==1
        /**
         * \brief Returns the grid as a 3D texture, must be called from the main OpenGL thread.
         * \param source The data source if the texture has to be updated (invalidateTexture() was called).
         *	It can either be HOST (copied from host memory, Eigen) or DEVICE (copied from device memory, cuMat).
         * \return the OpenGL Texture
         */
        cinder::gl::Texture3dRef getTexture(DataSource source)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!tex_) {
                //create texture
                cinder::gl::Texture3d::Format f;
                f.setInternalFormat(internal::TypeToOpenGL<T>::internalFormat);
                f.setDataType(internal::TypeToOpenGL<T>::dataType);
                tex_ = cinder::gl::Texture3d::create(grid_->getSize().x(), grid_->getSize().y(), grid_->getSize().z(), f);
                texValid_ = false;
                CI_LOG_D("Texture created");
            }

            if (!texValid_) {
                //update texture
                size_t size = grid_->getSize().prod();
                if (source == DataSource::AUTO)
                {
                    if (hasDeviceMemory())
                        source = DataSource::DEVICE;
                    else if (hasHostMemory())
                        source = DataSource::HOST;
                    else
                        throw std::runtime_error("no data available as source");
                }
                if (source == DataSource::DEVICE)
                {
                    //copy device to host
                    if (!hasHostMemory()) allocateHostMemory();
                    copyDeviceToHost();
                }
                //update texture with host data
				release_assert(hasHostMemory() && "Host memory not allocated / initialized");
                std::unique_ptr<char[]> mem(new char[size * internal::TypeToOpenGL<T>::bytesPerPixel]);
                internal::TypeToOpenGL<T>::convert(static_cast<void*>(mem.get()), hostMemory_.data(), size);
                tex_->update(static_cast<void*>(mem.get()), internal::TypeToOpenGL<T>::dataFormat, internal::TypeToOpenGL<T>::dataType, 0, grid_->getSize().x(), grid_->getSize().y(), grid_->getSize().z());

                texValid_ = true;
                CI_LOG_D("Texture updated");
            }

            return tex_;
        }

        /**
         * \brief Deletes the texture.
         * Has to be called from the OpenGL render thread.
         */
        void deleteTexture()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tex_) {
                tex_ = nullptr;
                CI_LOG_D("Texture deleted");
            }
        }
#else
	    /**
         * \brief Returns the grid as a 3D texture, must be called from the main OpenGL thread.
         * \param source The data source if the texture has to be updated (invalidateTexture() was called).
         *	It can either be HOST (copied from host memory, Eigen) or DEVICE (copied from device memory, cuMat).
         * \return the OpenGL Texture
         */
        cinder::gl::Texture3dRef getTexture(DataSource source)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!tex_) {
                //create texture
                cinder::gl::Texture3d::Format f;
                f.setInternalFormat(internal::TypeToOpenGL<T>::internalFormat);
                f.setDataType(internal::TypeToOpenGL<T>::dataType);
                tex_ = cinder::gl::Texture3d::create(grid_->getSize().x(), grid_->getSize().y(), grid_->getSize().z(), f);
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				CUMAT_SAFE_CALL(cudaGLRegisterBufferObject(tex_->getId()));
                texValid_ = false;
                CI_LOG_D("Texture created");
            }

            if (!texValid_) {
                //update texture
				size_t size = grid_->getSize().prod();
				if (source == DataSource::AUTO)
				{
					if (hasDeviceMemory())
						source = DataSource::DEVICE;
					else if (hasHostMemory())
						source = DataSource::HOST;
					else
						throw std::runtime_error("no data available as source");
				}
				if (source == DataSource::HOST) {
					assert(hasHostMemory() && "Host memory not allocated / initialized");
					std::unique_ptr<char[]> mem(new char[size * internal::TypeToOpenGL<T>::bytesPerPixel]);
					internal::TypeToOpenGL<T>::convert(static_cast<void*>(mem.get()), hostMemory_.data(), size);
					tex_->update(static_cast<void*>(mem.get()), internal::TypeToOpenGL<T>::dataFormat, internal::TypeToOpenGL<T>::dataType, 0, grid_->getSize().x(), grid_->getSize().y(), grid_->getSize().z());
				} 
            	else //Device
				{
					assert(!internal::TypeToOpenGL<T>::needsConversion && "DataSource::DEVICE can only be used if the type is directly supported by OpenGL, i.e. it is float");
					assert(hasDeviceMemory() && "Device memory not allocated / initialized");
					void* mem;
					CUMAT_SAFE_CALL(cudaDeviceSynchronize());
					CUMAT_SAFE_CALL(cudaGLMapBufferObject(&mem, tex_->getId()));
					CUMAT_SAFE_CALL(cudaMemcpy(mem, deviceMemory_.data(), sizeof(T) * size, cudaMemcpyDeviceToDevice));
					CUMAT_SAFE_CALL(cudaDeviceSynchronize());
					CUMAT_SAFE_CALL(cudaGLUnmapBufferObject(tex_->getId()));
				}
                texValid_ = true;
                CI_LOG_D("Texture updated");
            }

            return tex_;
        }

	    /**
		 * \brief Deletes the texture.
		 * Has to be called from the OpenGL render thread.
		 */
		void deleteTexture()
        {
            std::lock_guard<std::mutex> lock(mutex_);
			if (tex_) {
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(tex_->getId()));
				tex_ = nullptr;
				CI_LOG_D("Texture deleted");
			}
        }
#endif
    };

	template<typename T>
	using WorldGridDataPtr = std::shared_ptr<WorldGridData<T> >;
    typedef std::shared_ptr<WorldGridData<float> > WorldGridFloatDataPtr;
    typedef std::shared_ptr<WorldGridData<glm::float3> > WorldGridFloat3DataPtr;
    typedef std::shared_ptr<WorldGridData<double> > WorldGridDoubleDataPtr;
    typedef std::shared_ptr<WorldGridData<real> > WorldGridRealDataPtr;

    //explicit template instantiations, the declaration
    template class WorldGridData<float>;
    template class WorldGridData<double>;
}
