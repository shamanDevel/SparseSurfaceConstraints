#pragma once

#include <cinder/gl/Texture.h>
#include <cinder/params/Params.h>
#include <cinder/Json.h>

#include "InputConfig.h"
#include "InputDataLoader.h"
#include "WorldGrid.h"
#include "BackgroundWorker2.h"
#include "GeometryUtils3D.h"

namespace ar3d
{
    /**
     * \brief Performs dense 3D reconstruction of the scene from the aquired RGB-D images.
     * This reconstruction is frame based. No time consistency yet.
     */
    class MeshReconstruction
    {
    public:
        typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> SegmentationMatrix;
        typedef std::vector<SegmentationMatrix> SegmentationMatrices;

    public:
        //Settings
        struct Settings {
            //Number of voxels that are added to grid along the borders
            // to ensure that really everything is inside
            int gridBorder = 2;

            //Grid resultion, see WorldGrid::gridResolution
            //Low for the first test
            int gridResolution = 64;

            //The distance after which the SDF is truncated (in voxel)
            //This is the parameter delta in the paper
            double truncationDistance = 4;

            //The connecting factor theta of the primal and dual variable
            double primalDualConnection = 0.02;

            //The distance behind the silhouette on how far the projected sdf is certain
            double certaintyDistance = 4;

            //How accurate should the data be matched?
            double dataFidelity = 10;

            int maxPrimalDualIterations = 5;
            int maxROFIteratons = 10;

            //Number of iterations in the SDF recovery
            int sdfIterations = 20;
            //Artificial viscosity (damping) in the SDF recovery
            double sdfViscosity = 0.2;
        };
        Settings settings;

		class SettingsGui
		{
		private:
			Settings settings_;
		public:
			SettingsGui() : settings_({}) {}
			SettingsGui(const Settings& settings) : settings_(settings) {}
			const Settings& getSettings() const { return settings_; }
			void setSettings(const Settings& settings) { settings_ = settings; }

			void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
			void load(const cinder::JsonTree& parent);
			void save(cinder::JsonTree& parent) const;
		};

    private:
        //Input configuration with the cameras
        InputConfigPtr inputConfig;

        //The frame to be processed
        InputDataLoader::FramePtr frame;

        //segementation result
        SegmentationMatrices segmentedImages;
        std::vector<cinder::gl::Texture2dRef> segmentedTextures; //testing

        //bounding box that includes the object.
        //Used to constrain the region of reconstruction
	    ar::geom3d::AABBBox boundingBox;
        std::vector<ar::geom3d::Pyramid> maskPyramids; //testing

        //3d reconstruction result: the parent grid
        WorldGridPtr worldGrid;
        //3d reconstruction result: signed distance function of the volume, type=Double
        //positive outside, negative inside
        WorldGridRealDataPtr sdfData;
        //3d reconstruction result: color, type=Float3
        WorldGridFloat3DataPtr colorData;
        //The projected SDF per camera, type=Double
        std::vector<WorldGridRealDataPtr> projectedSdfPerCamera;
        //The certainty of the projection, type=Double
        std::vector<WorldGridRealDataPtr> certaintyPerCamera;

    public:
        MeshReconstruction(InputConfigPtr inputConfig, InputDataLoader::FramePtr frame);

        //Performs a segmentation of the input images into content (what should be reconstructed) and background
        //Returns true if successfull (false if cancelled)
        //Note: This is the first, simple version that only works on the simulation
        bool runSegmentation(BackgroundWorker2* worker);
        const SegmentationMatrices& getSegmentationResult() const { return segmentedImages; }
        cinder::gl::Texture2dRef getSegmentationTexture(int cam);

        //Finds a bounding box in world space that contains the segmented result by
        //intersecting the view frustrum of the segmented mask
        //Assumption: Each camera shows always the whole object
        bool runBoundingBoxExtraction(BackgroundWorker2* worker);
        const ar::geom3d::AABBBox& getBoundingBox() const { return boundingBox; }
        const std::vector<ar::geom3d::Pyramid>& getMaskPyramids() const { return maskPyramids; }

        //Initializes the reconstruction
        // - First guess for the result
        // - Projected SDF for each camera
        bool runInitReconstruction(BackgroundWorker2* worker);
        WorldGridPtr getWorldGrid() const { return worldGrid; }
		//The reconstructed SDF with the world grid
        WorldGridRealDataPtr getSdfData() const { return sdfData; }
        const std::vector<WorldGridRealDataPtr>& getProjectedSdfPerCamera() const
        {
            return projectedSdfPerCamera;
        }

        //Runs the actual reconstruction optimization for the resulting sdf
        bool runReconstruction(BackgroundWorker2* worker);
        bool runReconstructionTV1(BackgroundWorker2* worker);
        bool runReconstructionKinectFusion(BackgroundWorker2* worker);

        //Reconstruct signed distance function
        bool runSignedDistanceReconstruction(BackgroundWorker2* worker);

		//Test: get extracted points from the camera
		static std::vector<ci::vec3> getProjectedPoints(InputConfigPtr inputConfig, InputDataLoader::FramePtr frame);

    private:
        //computes div(p) with p=(pX,pY,pZ)
        static double getDivergence(const Eigen::Vector3i& pos, const WorldGridData<real>* pX, const WorldGridData<real>* pY, const WorldGridData<real>* pZ, const Eigen::Vector3i& size);
        //computes grad(v)
        static Eigen::Vector3d getGradient(const Eigen::Vector3i& pos, const WorldGridData<real>* v, const Eigen::Vector3i& size);
        //computes div(grad(v))
        static double getLaplacian(const Eigen::Vector3i& pos, const WorldGridData<real>* v, const Eigen::Vector3i& size);
        //computes grad(div(p)) with p=(pX,pY,pZ)
        static Eigen::Vector3d getGradientOfDivergence(const Eigen::Vector3i& pos, const WorldGridData<real>* pX, const WorldGridData<real>* pY, const WorldGridData<real>* pZ, const Eigen::Vector3i& size);

        template <typename T> static int sgn(T val) {
            return (T(0) < val) - (val < T(0));
        }
    };
    typedef std::shared_ptr<MeshReconstruction> MeshReconstructionPtr;
}