#pragma once

#include <cinder/app/App.h>
#include <cinder/gl/gl.h>
#include <cinder/params/Params.h>
#include <vector>

namespace ar
{
    class TransferFunctionEditor
    {
    public:
        struct Config
        {
            float min;
            float max;
            std::vector<std::pair<float, cinder::ColorAf> > controlPoints;
            Config();
        };

    private:
        cinder::app::WindowRef window;
        cinder::params::InterfaceGlRef paramsLeft; //min, max, load, save
        cinder::params::InterfaceGlRef paramsMain;
        cinder::params::InterfaceGlRef paramsRight; //color
        cinder::gl::Texture1dRef texture;
        bool textureValid;
        cinder::gl::BatchRef batchTF;
        cinder::gl::GlslProgRef	shaderTF;
        Config config;
        bool visible;

        int selectedControlPoint;
        int dragOffset;

        static const int offX = 20;
        static const int offY = 20;
        static const int heightLeft = 100;
        static const int heightMain = 70;
        static const int heightRight = 150;
        static const int widthLeft = 140;
        static const int widthRight = 140;
        static const int widthMain = 300;
        static const int textureResolution = 256;

    public:
        TransferFunctionEditor(cinder::app::WindowRef window);

        void setVisible(bool visible);
        void update();
        void draw();

        cinder::gl::Texture1dRef getTexture() const;
        double getRangeMin() const { return config.min; }
        double getRangeMax() const { return config.max; }

        const Config& getConfig() const;
        void setConfig(const Config& config);

    private:
        bool isInside(cinder::app::MouseEvent& event);
        void mouseDown(cinder::app::MouseEvent &event);
        void mouseUp(cinder::app::MouseEvent &event);
        void mouseDrag(cinder::app::MouseEvent &event);
        void mouseMove(cinder::app::MouseEvent &event);

        void controlPointSelected(int index);
        void sortPoints();

        cinder::ColorAf getParamColor();
        void setParamColor(const cinder::ColorAf& col);
        float getParamPos();
        void setParamPos(float pos);
        void saveConfig();
        void loadConfig();

        void updateTexture();
        void invalidateTexture();
    };
}