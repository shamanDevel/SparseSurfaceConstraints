#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <cinder/gl/TextureFont.h>

#include "InputConfig.h"
#include "BackgroundWorker2.h"

namespace ar3d
{
	
	class InputDataLoader
	{
	public:
		typedef float colorComponent_t;
		typedef double depth_t;
		typedef Eigen::Array<colorComponent_t, Eigen::Dynamic, Eigen::Dynamic> colorComponentMatrix_t;
		typedef Eigen::Array<depth_t, Eigen::Dynamic, Eigen::Dynamic> depthMatrix_t;

		struct CameraImage
		{
        public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			const colorComponentMatrix_t colorRedMatrix;
			const colorComponentMatrix_t colorGreenMatrix;
			const colorComponentMatrix_t colorBlueMatrix;
			const depthMatrix_t depthMatrix;
        private:
			//for visualization
            cinder::Surface8u colorSurface;
			cinder::Surface8u depthSurface;
			cinder::gl::Texture2dRef colorTexture;
			cinder::gl::Texture2dRef depthTexture;

        public:
			CameraImage(const colorComponentMatrix_t& color_red_matrix, const colorComponentMatrix_t& color_green_matrix,
				const colorComponentMatrix_t& color_blue_matrix, const depthMatrix_t& depth_matrix,
				const cinder::Surface8u& color_surface, const cinder::Surface8u& depth_surface);

            cinder::gl::Texture2dRef getColorTexture();
			cinder::gl::Texture2dRef getDepthTexture();
		};

		struct Frame
		{
			int width;
			int height;
			int frameIndex;
			std::vector<CameraImage> cameraImages;
		};
		typedef std::shared_ptr<Frame> FramePtr;
		typedef std::weak_ptr<Frame> FrameWPtr;

	private:
		InputConfigPtr inputConfig;
		std::map<int, FrameWPtr> frameCache;

	public:
		InputDataLoader(InputConfigPtr inputConfig);

		FramePtr loadFrame(int frame, BackgroundWorker2* worker = nullptr);
	};
	typedef std::shared_ptr<InputDataLoader> InputDataLoaderPtr;
}
