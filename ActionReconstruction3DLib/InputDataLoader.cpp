#include "InputDataLoader.h"
#include <cinder/Log.h>

namespace ar3d {
	InputDataLoader::CameraImage::CameraImage(const colorComponentMatrix_t& color_red_matrix,
	                                          const colorComponentMatrix_t& color_green_matrix,
	                                          const colorComponentMatrix_t& color_blue_matrix,
	                                          const depthMatrix_t& depth_matrix, 
	                                          const cinder::Surface8u& color_surface,
	                                          const cinder::Surface8u& depth_surface):
		colorRedMatrix(color_red_matrix),
		colorGreenMatrix(color_green_matrix),
		colorBlueMatrix(color_blue_matrix),
		depthMatrix(depth_matrix),
		colorSurface(color_surface),
		depthSurface(depth_surface)
	{
	}

	InputDataLoader::InputDataLoader(InputConfigPtr inputConfig)
		: inputConfig(inputConfig)
	{
	}

	InputDataLoader::FramePtr InputDataLoader::loadFrame(int frame, BackgroundWorker2* worker)
	{
		if (frame < 0 || frame >= inputConfig->duration)
		{
			return InputDataLoader::FramePtr();
		}

		FramePtr fp = frameCache[frame].lock();
		if (!fp)
		{
			//load frame
			Frame* f = new Frame();
			f->width = inputConfig->width;
			f->height = inputConfig->height;
			f->frameIndex = frame;
			f->cameraImages.reserve(inputConfig->cameras.size());
			//for all cameras
			for (size_t i=0; i<inputConfig->cameras.size(); ++i)
			{
                if (worker && worker->isInterrupted()) return nullptr;
				cinder::Surface8u colorSurface = cinder::loadImage(inputConfig->getPathToFrame(i, frame, false));
                if (worker && worker->isInterrupted()) return nullptr;
				cinder::Surface8u depthSurface = cinder::loadImage(inputConfig->getPathToFrame(i, frame, true));
                if (worker && worker->isInterrupted()) return nullptr;
				colorComponentMatrix_t red(f->width, f->height);
				colorComponentMatrix_t green(f->width, f->height);
				colorComponentMatrix_t blue(f->width, f->height);
				depthMatrix_t depth(f->width, f->height);
				cinder::Surface8u::Iter colorIter = colorSurface.getIter();
				const bool flipY = inputConfig->cameras[i].flipY;
				while (colorIter.line()) {
					while (colorIter.pixel()) {
						int y = flipY ? f->height - colorIter.y() - 1 : colorIter.y();
						red(colorIter.x(), y) = colorIter.r() / 255.0;
						green(colorIter.x(), y) = colorIter.g() / 255.0;
						blue(colorIter.x(), y) = colorIter.b() / 255.0;
						if (colorSurface.hasAlpha()) colorIter.a() = 255;
					}
                    if (worker && worker->isInterrupted()) return nullptr;
				}
				cinder::Surface8u::Iter depthIter = depthSurface.getIter();
				while (depthIter.line()) {
					while (depthIter.pixel()) {
						int y = flipY ? f->height - depthIter.y() - 1 : depthIter.y();
						depth(depthIter.x(), y) = depthIter.r() / 255.0;
					}
                    if (worker && worker->isInterrupted()) return nullptr;
				}
				ar3d::InputDataLoader::CameraImage img(red, green, blue, depth, colorSurface, depthSurface);
				f->cameraImages.push_back(img);
			}
			//place into cache
			fp = FramePtr(f);
			frameCache[frame] = fp;
			CI_LOG_D("Frame " << frame << " loaded");
		} else
		{
			CI_LOG_D("Frame " << frame << " reused from the cache");
		}
		return fp;
	}

    cinder::gl::Texture2dRef InputDataLoader::CameraImage::getColorTexture()
    {
        if (colorTexture) {
            return colorTexture;
        }
        else {
            colorTexture = cinder::gl::Texture::create(colorSurface);
            return colorTexture;
        }
    }

	cinder::gl::Texture2dRef InputDataLoader::CameraImage::getDepthTexture()
	{
		if (depthTexture) {
			return depthTexture;
		}
		else {
			depthTexture = cinder::gl::Texture::create(depthSurface);
			return depthTexture;
		}
	}

}
