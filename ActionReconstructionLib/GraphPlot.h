#pragma once

#include <vector>
#include <cinder/Color.h>
#include <cinder/Rect.h>
#include <cinder/Font.h>
#include <cinder/PolyLine.h>

class GraphPlot
{
private:
    enum Colors
    {
        BACKGROUND,
        AXIS,
        TEXT,
        GROUND_TRUTH,
        LINE,
        __COLOR_COUNT__
    };
    static const cinder::ColorAf colors[__COLOR_COUNT__][2];

    std::string name_;
    bool printMode_;
    cinder::Rectf boundingRect_;
    int numSteps_;
    float minValue_;
    float maxValue_;
    float trueValue_;
    cinder::PolyLine2f values_;
    cinder::Font font_;

public:
    GraphPlot(const std::string& name);
    const std::string& getName() const { return name_; }

    void clear();
    void setTrueValue(float value);
    void addPoint(float point);
    int getNumPoints() const;
    float getPoint(int index) const;
    void setMaxPoints(int maxPoints);

    void setPrintMode(bool printMode);
    void setBoundingRect(const cinder::Rectf& boundingRect);

    void draw();
};

