#include "GraphPlot.h"
#include <cinder/gl/gl.h>

using namespace cinder;

const ColorAf GraphPlot::colors[5][2] =
{
    {ColorAf(0.1,0.1,0.1,0.2), ColorAf(0.9,0.9,0.9,0.2)}, //Background
    {ColorAf(1,1,1,1), ColorAf(0,0,0,1)}, //Axis
    {ColorAf(1,1,1,1), ColorAf(0,0,0,1) }, //Text
    {ColorAf(1,0,0,1), ColorAf(0.7,0,0,1)}, //Ground Truth
    {ColorAf(1,1,0,1), ColorAf(1,0.8,0,1)} //Line
};

GraphPlot::GraphPlot(const std::string& name) :
    name_(name),
    printMode_(false),
    boundingRect_(0,0,1,1),
    numSteps_(1),
    minValue_(-1),
    maxValue_(1),
    trueValue_(0),
    font_("Arial", 20)
{
}

void GraphPlot::clear()
{
    values_.getPoints().clear();
    constexpr float BIG = 1e7;
    minValue_ = trueValue_;
    maxValue_ = trueValue_;
}

void GraphPlot::setTrueValue(float value)
{
    trueValue_ = value;
    minValue_ = trueValue_;
    maxValue_ = trueValue_;
}

void GraphPlot::addPoint(float point)
{
    values_.push_back(vec2(values_.size(), point));
    minValue_ = std::min(minValue_, point);
    maxValue_ = std::max(maxValue_, point);
}

int GraphPlot::getNumPoints() const
{
    return values_.size();
}

float GraphPlot::getPoint(int index) const
{
    return values_.getPoints()[index].y;
}

void GraphPlot::setMaxPoints(int maxPoints)
{
    numSteps_ = maxPoints;
}

void GraphPlot::setPrintMode(bool printMode)
{
    printMode_ = printMode;
}

void GraphPlot::setBoundingRect(const cinder::Rectf& boundingRect)
{
    boundingRect_ = boundingRect;
}

void GraphPlot::draw()
{
    static const float minMaxGap = 0.1;
    const auto valueToY = [this](float value) -> float
    {
        return (1 - 2*minMaxGap) * boundingRect_.getHeight() - 
            (value - minValue_) / (maxValue_ - minValue_ + 1e-10f)
            * (boundingRect_.getHeight() * (1 - 2*minMaxGap)) + minMaxGap * boundingRect_.getHeight();
    };

    gl::ScopedMatrices scoped_matrices;
    gl::ScopedColor scoped_color;

    gl::translate(boundingRect_.getUpperLeft());

    //Background
    gl::color(colors[BACKGROUND][printMode_]);
    gl::drawSolidRect(Rectf(0, 0, boundingRect_.getWidth(), boundingRect_.getHeight()));

    //Lines
    gl::color(colors[AXIS][printMode_]);
    gl::drawLine(vec2(0, 0), vec2(0, boundingRect_.getHeight()));
    gl::drawLine(vec2(-5, valueToY(minValue_)), vec2(0, valueToY(minValue_)));
    gl::drawLine(vec2(-5, valueToY(maxValue_)), vec2(0, valueToY(maxValue_)));
    gl::color(colors[GROUND_TRUTH][printMode_]);
    gl::drawLine(vec2(0, valueToY(trueValue_)), vec2(boundingRect_.getWidth(), valueToY(trueValue_)));

    //Text
    gl::drawStringCentered(name_ + ": " + std::to_string(values_.size()==0 ? trueValue_ : values_.getPoints()[values_.size()-1].y),
        vec2(boundingRect_.getWidth()/2, 0), colors[TEXT][printMode_], font_);
    gl::drawStringRight(std::to_string(minValue_), vec2(-5, valueToY(minValue_) - 8), colors[TEXT][printMode_], font_);
    gl::drawStringRight(std::to_string(maxValue_), vec2(-5, valueToY(maxValue_) - 8), colors[TEXT][printMode_], font_);

    //Values
    if (values_.size() > 0)
    {
        gl::ScopedMatrices scoped_matrices2;
        gl::color(colors[LINE][printMode_]);
        gl::translate(vec2(0, valueToY(0)));
        gl::scale(boundingRect_.getWidth() / float(numSteps_), -(1 - 2*minMaxGap) * boundingRect_.getHeight() / (maxValue_ - minValue_));
        gl::draw(values_);
    }
}
