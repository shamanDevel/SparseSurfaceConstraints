#pragma once

#include <mutex>
#include <cinder/app/App.h>
#include <cinder/gl/gl.h>
#include <cinder/Camera.h>
#include <cinder/params/Params.h>

#include "SoftBodyGrid3D.h"

namespace ar3d
{
	class VolumeVisualizationParams
	{
	public:
		enum class Mode
		{
			Slice,
			Volume,
			MCSurface,
			HighRes,
		};
		static inline Mode ToMode(const std::string& str)
		{
			if (str == "Slice") return Mode::Slice;
			else if (str == "Volume") return Mode::Volume;
			else if (str == "MCSurface") return Mode::MCSurface;
			else if (str == "HighRes") return Mode::HighRes;
			throw std::exception("Unknown mode");
		}
		static inline std::string FromMode(const Mode& mode)
		{
			if (mode == Mode::Slice) return "Slice";
			else if (mode == Mode::Volume) return "Volume";
			else if (mode == Mode::MCSurface) return "MCSurface";
			else if (mode == Mode::HighRes) return "HighRes";
			throw std::exception("Unknown mode");
		}

		//global properties
		Mode mode_;

		//light
		cinder::vec3 directionalLightDir_;
		cinder::ColorAf directionalLightColor_;
		cinder::ColorAf ambientLightColor_;

		//slice (Deprecated)
		double slicePosition_; //distance from the center of the grid
		int sliceAxis_;
		double rangeMin_;
		double rangeMax_;

		//volume
		bool showNormals_;
		double stepSize_;

		//grid cells
		cinder::ColorA8u gridCellColor_;
		bool showGridCells_;

		VolumeVisualizationParams();
		void addParams(const cinder::params::InterfaceGlRef& params);
		void load(const cinder::JsonTree& parent);
		void save(cinder::JsonTree& parent) const;
	private:
		void updateVisualizationMode();
		cinder::params::InterfaceGlRef params_;
	};

    
    class VolumeVisualization
    {
    private:
		VolumeVisualizationParams* params_;
		
        std::mutex inputMutex_;
        SoftBodyGrid3D::Input input_;

		Vector3X positions_;
		WorldGridDataPtr<real> advectedSdf_;
        WorldGridRealDataPtr sdfData_;
		WorldGridData<real3>::DeviceArray_t gridDisplacements_;

		cinder::gl::Texture3dRef sdfTexture_;
        cinder::gl::Texture3dRef colorTexture_;
		bool volumeValid_;

        const cinder::Camera* cam_;
        cinder::app::WindowRef window_;

        //slice
        cinder::gl::Texture1dRef transferFunctionTexture_;
        cinder::gl::GlslProgRef sliceShader_;
        cinder::gl::BatchRef sliceBatch_;

        //volume raytracer
        cinder::gl::GlslProgRef surfaceShader_;
        cinder::gl::BatchRef surfaceBatch_;

        //marching cubes
		cinder::TriMeshRef mcMesh_;
		cinder::gl::GlslProgRef mcShader_;
		cinder::gl::BatchRef mcBatch_;
		cinder::gl::BufferObjRef mcPositionBuffer_;

		//high-res mesh
		cinder::TriMeshRef highResMesh_;
		cinder::TriMeshRef modifiedHighResMesh_;
		cinder::gl::BatchRef highResBatch_;

        //grid cells
        cinder::gl::GlslProgRef cellShader_;
        cinder::gl::BatchRef cellBatch_;
        cinder::gl::VboRef cellVertexBuffer_;
        bool cellValid_;

    public:
        VolumeVisualization(const cinder::app::WindowRef& window,
			const cinder::Camera* cam, VolumeVisualizationParams* params);
		~VolumeVisualization();

		//Sets the input SDF.
		//This also resets the high res input mesh.
		void setInput(const SoftBodyGrid3D::Input& input);
		//Sets a high-res input mesh, if available.
		//This enables the display of the deformation on that high res mesh.
		//The mesh has to be properly aligned with the grid, e.g. as produced by Object2SDF.
		//If this is not available, pass an empty string.
		void setHighResInputMesh(const std::string& file);
		bool hasHighResMesh() const;

		//updates the rendering with the new simulation state
        void update(const SoftBodyGrid3D::State& state);

        bool needsTransferFunction() const;
        void setTransferFunction(cinder::gl::Texture1dRef transferFunction,
            double rangeMin, double rangeMax);

        void draw();
        void reloadResources();

		//Saves the current SDF as binary file
		void saveSdf(const std::string& filename);
		//Saves the current Marching cubes mesh as .obj file
		void saveMCMesh(const std::string& filename);
		//Saves the current high resolution mesh as .obj file
		void saveHighResultMesh(const std::string& filename, bool includeNormals, bool includeOriginalPositions);

    private:
        void drawSlice();
        void drawSurface();
		void updateMarchingCubes();
		void drawMarchingCubes();
		void updateHighResMesh();
		void drawHighResMesh();
        void drawCells();

        void mouseWheel(cinder::app::MouseEvent &event);
    };

}
