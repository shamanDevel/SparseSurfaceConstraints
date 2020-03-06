#include "TransferFunctionEditor.h"

#include <cinder/Log.h>
#include <cinder/Json.h>
#include <algorithm>

ar::TransferFunctionEditor::Config::Config()
    : min(0)
    , max(1)
{
    controlPoints.emplace_back(0.0f, cinder::ColorAf(0, 0, 0, 0));
    controlPoints.emplace_back(1.0f, cinder::ColorAf(1, 1, 1, 1));
}

ar::TransferFunctionEditor::TransferFunctionEditor(cinder::app::WindowRef window)
    : window(window)
    , visible(true)
    , selectedControlPoint(-1)
    , textureValid(false)
{
    //create param bars
    paramsLeft = cinder::params::InterfaceGl::create(window, "tfeLeft ", cinder::ivec2(widthLeft, heightLeft));
    paramsMain = cinder::params::InterfaceGl::create(window, "tfeMain", cinder::ivec2(widthMain, heightMain));
    paramsRight = cinder::params::InterfaceGl::create(window, "tfeRight", cinder::ivec2(widthRight, heightRight));
    paramsLeft->setOptions("", " label=' ' iconifiable=false movable=false resizable=false ");
    paramsMain->setOptions("", " label='Transfer Function Editor' iconifiable=false movable=false resizable=false ");
    paramsRight->setOptions("", " label=' ' iconifiable=false movable=false resizable=false ");

    //add parameters
    paramsLeft->addParam("min", &config.min).label("min").step(0.001f);
    paramsLeft->addParam("max", &config.max).label("max").step(0.001f);
    paramsLeft->addButton("save", [this]() {this->saveConfig(); });
    paramsLeft->addButton("load", [this]() {this->loadConfig(); });
    paramsRight->addParam("color", 
        std::function<void(cinder::ColorAf)>([this](cinder::ColorAf c) {this->setParamColor(c); }), 
        std::function<cinder::ColorAf()>([this]() {return this->getParamColor(); })
    ).label("color").optionsStr("opened=true");
    paramsRight->addParam("pos",
        std::function<void(float)>([this](float d) {this->setParamPos(d); }),
        std::function<float()>([this]() {return this->getParamPos(); })
    ).label("pos").min(0).max(1).step(0.001f);
    paramsRight->setOptions("color", "visible=false");
    paramsRight->setOptions("pos", "visible=false");

    //connect input
    window->getSignalMouseDown().connect(1000, [this](cinder::app::MouseEvent& event) {this->mouseDown(event); });
    window->getSignalMouseUp().connect(1000, [this](cinder::app::MouseEvent& event) {this->mouseUp(event); });
    window->getSignalMouseDrag().connect(1000, [this](cinder::app::MouseEvent& event) {this->mouseDrag(event); });
    window->getSignalMouseMove().connect(1000, [this](cinder::app::MouseEvent& event) {this->mouseMove(event); });

    //create texture
    updateTexture();

    //create shader
    shaderTF = cinder::gl::GlslProg::create( cinder::gl::GlslProg::Format()
	.vertex(R"GLSL(
#version 150
uniform mat4	ciModelViewProjection;
in vec4			ciPosition;
in vec2			ciTexCoord0;
out vec2		TexCoord0;
		
void main( void ) {
	gl_Position	= ciModelViewProjection * ciPosition;
	TexCoord0 = ciTexCoord0;
}
)GLSL")
	.fragment(R"GLSL(
#version 150
uniform sampler1D   uTex0;		
in vec2		        TexCoord0;
out vec4            oColor;

void main( void ) {
    vec4 texCol = texture( uTex0, TexCoord0.x );
    vec4 chessCol = (mod(floor(TexCoord0.x*16) + floor(TexCoord0.y*4), 2.0)==0.0) ? vec4(0.8) : vec4(0.4);
    vec4 finalCol = mix(chessCol, texCol, texCol.a);
    finalCol.a = 1.0;
	oColor = finalCol;
}
)GLSL"));
	batchTF = cinder::gl::Batch::create( cinder::geom::Rect(cinder::Rectf(0, 0, 1, 1)), shaderTF );
}

void ar::TransferFunctionEditor::setVisible(bool visible)
{
    this->visible = visible;
    if (visible)
    {
        paramsLeft->show();
        paramsMain->show();
        paramsRight->show();
    } else
    {
        paramsLeft->hide();
        paramsMain->hide();
        paramsRight->hide();
    }
}

void ar::TransferFunctionEditor::update()
{
    //position bars
    if (visible) {
        int winWidth = window->getWidth();
        int winHeight = window->getHeight();
        paramsLeft->setPosition(cinder::ivec2(winWidth - widthLeft - widthMain - widthRight - offX, winHeight - heightLeft - offY));
        paramsMain->setPosition(cinder::ivec2(winWidth - widthMain - widthRight - offX, winHeight - heightMain - offY));
        paramsRight->setPosition(cinder::ivec2(winWidth - widthRight - offX, winHeight - heightRight - offY));
    }
    updateTexture();
}

void ar::TransferFunctionEditor::draw()
{
if (visible) {
    //all bars are drawn in ActionReconstructionApp
    /*paramsLeft->draw();
    paramsMain->draw();
    paramsRight->draw();*/

    int winWidth = window->getWidth();
    int winHeight = window->getHeight();
    int tfWidth = widthMain - 20;
    int tfHeight = heightMain - 40;
    int tfOffX = winWidth - widthMain - widthRight - offX + 10;
    int tfOffY = winHeight - offY - tfHeight - 10;
    {
        cinder::gl::ScopedMatrices m;
        cinder::gl::ScopedDepth d(false);
        cinder::gl::ScopedTextureBind t(texture, 0);
        cinder::gl::ScopedColor c;
        cinder::gl::setMatricesWindow(window->getSize(), true);

        //draw bar
        cinder::gl::setViewMatrix(glm::mat4(tfWidth, 0, 0, 0, 0, tfHeight, 0, 0, 0, 0, 1, 0, tfOffX, tfOffY, 0, 1));
        shaderTF->uniform("uTex0", 0);
        batchTF->draw();

        //handles
        cinder::gl::setViewMatrix(glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));
        for (size_t i = 0; i < config.controlPoints.size(); ++i) {
            if (selectedControlPoint == i) {
                cinder::gl::color(0.8f, 0.8f, 0.8f, 1.0f);
            }
            else {
                cinder::gl::color(0.0f, 0.0f, 0.0f, 1.0f);
            }
            double pos = tfOffX + config.controlPoints[i].first * tfWidth;
            cinder::gl::drawSolidTriangle(glm::vec2(pos - 5, tfOffY - 5), glm::vec2(pos + 5, tfOffY - 5), glm::vec2(pos, tfOffY + tfHeight / 2));
            cinder::gl::drawSolidTriangle(glm::vec2(pos - 5, tfOffY + tfHeight + 5), glm::vec2(pos + 5, tfOffY + tfHeight + 5), glm::vec2(pos, tfOffY + tfHeight / 2));
        }
    }
}
}

cinder::gl::Texture1dRef ar::TransferFunctionEditor::getTexture() const
{
    return texture;
}

const ar::TransferFunctionEditor::Config& ar::TransferFunctionEditor::getConfig() const
{
    return config;
}
void ar::TransferFunctionEditor::setConfig(const Config& config)
{
    this->config = config;
    controlPointSelected(-1);
    invalidateTexture();
}

bool ar::TransferFunctionEditor::isInside(cinder::app::MouseEvent& event)
{
    if (!visible) return false;
    if (event.isHandled()) return false;
    int winWidth = window->getWidth();
    int winHeight = window->getHeight();
    if (event.getX() < winWidth - widthMain - widthRight - offX
        || event.getX() > winWidth - widthRight - offX
        || event.getY() < winHeight - heightMain - offY
        || event.getY() > winHeight - offY)
    {
        return false;
    }
    event.setHandled();
    return true;
}

void ar::TransferFunctionEditor::mouseDown(cinder::app::MouseEvent & event)
{
    if (!isInside(event)) return;
    CI_LOG_V("mouse down");

    //find control point
    int winWidth = window->getWidth();
    int winHeight = window->getHeight();
    int tfWidth = widthMain - 20;
    int tfHeight = heightMain - 40;
    int tfOffX = winWidth - widthMain - widthRight - offX + 10;
    int tfOffY = winHeight - offY - tfHeight - 10;

    int inputPosX = event.getX() - tfOffX;
    if (inputPosX < -5 || inputPosX > tfWidth + 5) {
        controlPointSelected(-1);
        return;
    }

    int inputPosY = event.getY() - tfOffY;
    if (inputPosY < -5 || inputPosY > tfHeight + 5) {
        controlPointSelected(-1);
        return;
    }

    int bestMatch = -1;
    float distance = 10000;
    for (int i = 0; i<int(config.controlPoints.size()); ++i) {
        float x = config.controlPoints[i].first * tfWidth;
        float dist = abs(x - inputPosX);
        if (dist < distance && dist <= 5) {
            bestMatch = i;
            distance = dist;
            dragOffset = int(x - inputPosX);
        }
    }

    if (bestMatch >= 0 && event.isRight()) {
        //delete point
        if (bestMatch == 0 || bestMatch == config.controlPoints.size() - 1) {
            CI_LOG_D("can't delete border points");
            controlPointSelected(-1);
        }
        else {
            config.controlPoints.erase(config.controlPoints.begin() + bestMatch);
            invalidateTexture();
            CI_LOG_D("control point deleted");
        }
    }
    else if (bestMatch >= 0 && event.isLeft()) {
        //select exisiting point
        controlPointSelected(bestMatch);
    }
    else if (bestMatch == -1 && event.isLeft()) {
        static double lastClick = -1000;
        double currentTime = cinder::app::getElapsedSeconds();
        if (currentTime < lastClick + 0.2) { //0.2sec for double click
            //create new point
            float pos = inputPosX / float(tfWidth);
            assert(pos > 0);
            assert(pos < 1);
            int right;
            for (right = 1; right < int(config.controlPoints.size()); right++) {
                if (config.controlPoints[right - 1].first < pos && config.controlPoints[right].first > pos) {
                    cinder::ColorAf c = config.controlPoints[right - 1].second.lerp(
                        (pos - config.controlPoints[right - 1].first) / (config.controlPoints[right].first - config.controlPoints[right - 1].first), 
                        config.controlPoints[right].second);
                    config.controlPoints.insert(config.controlPoints.begin() + right, { pos, c });
                    break;
                }
            }
            invalidateTexture();
            controlPointSelected(right);
            lastClick = -1000;
            CI_LOG_D("new point created");
        }
        else {
            lastClick = currentTime;
            controlPointSelected(-1);
        }
    }

}

void ar::TransferFunctionEditor::mouseUp(cinder::app::MouseEvent & event)
{
    if (!isInside(event)) return;
    CI_LOG_V("mouse up");
}

void ar::TransferFunctionEditor::mouseDrag(cinder::app::MouseEvent & event)
{
    if (!isInside(event)) return;
    CI_LOG_V("mouse drag");

    if (selectedControlPoint > 0 && selectedControlPoint < config.controlPoints.size() - 1) {
        int winWidth = window->getWidth();
        int tfWidth = widthMain - 20;
        int tfOffX = winWidth - widthMain - widthRight - offX + 10;

        double newPos = (event.getX() - tfOffX + dragOffset) / double(tfWidth);
        double epsilon = 0.001;
        double posMin = config.controlPoints[selectedControlPoint - 1].first + epsilon;
        double posMax = config.controlPoints[selectedControlPoint + 1].first - epsilon;
        newPos = std::max(posMin, std::min(posMax, newPos));
        config.controlPoints[selectedControlPoint].first = static_cast<float>(newPos);
        invalidateTexture();
    }
}

void ar::TransferFunctionEditor::mouseMove(cinder::app::MouseEvent & event)
{
    if (!isInside(event)) return;
    CI_LOG_V("mouse move");
}

void ar::TransferFunctionEditor::controlPointSelected(int index)
{
    selectedControlPoint = index;
    CI_LOG_D("control point: " << index);
    if (index >= 0) {
        paramsRight->setOptions("color", "visible=true");
        if (index == 0 || index == config.controlPoints.size()-1)
            paramsRight->setOptions("pos", "visible=true readonly=true");
        else
            paramsRight->setOptions("pos", "visible=true readonly=false");
    }
    else {
        paramsRight->setOptions("color", "visible=false");
        paramsRight->setOptions("pos", "visible=false");
    }
}

void ar::TransferFunctionEditor::sortPoints()
{
    std::sort(config.controlPoints.begin(), config.controlPoints.end(), [](const std::pair<float, cinder::ColorAf>& a, const std::pair<float, cinder::ColorAf>& b)
    {
        return a.first < b.first;
    });
}

cinder::ColorAf ar::TransferFunctionEditor::getParamColor()
{
    if (selectedControlPoint >= 0) {
        return config.controlPoints[selectedControlPoint].second;
    }
    else {
        return {};
    }
}
void ar::TransferFunctionEditor::setParamColor(const cinder::ColorAf& col)
{
    if (selectedControlPoint >= 0) {
        config.controlPoints[selectedControlPoint].second = col;
        invalidateTexture();
    }
}
float ar::TransferFunctionEditor::getParamPos()
{
    if (selectedControlPoint >= 0) {
        return config.controlPoints[selectedControlPoint].first;
    }
    else {
        return -1;
    }
}
void ar::TransferFunctionEditor::setParamPos(float pos)
{
    if (selectedControlPoint >= 0) {
        config.controlPoints[selectedControlPoint].first = pos;
        sortPoints();
        invalidateTexture();
    }
}

void ar::TransferFunctionEditor::updateTexture()
{
    if (!texture) {
        //create texture
        cinder::gl::Texture1d::Format f;
        f.setInternalFormat(GL_RGBA32F);
        f.setDataType(GL_FLOAT);
        texture = cinder::gl::Texture1d::create(textureResolution, f);
        textureValid = false;
    }

    if (!textureValid) {
        //compute texture
        std::unique_ptr<float[]> mem = std::make_unique<float[]>(textureResolution * 4);
        for (size_t i = 1; i < config.controlPoints.size(); ++i) {
            const std::pair<float, cinder::ColorAf>& l = config.controlPoints[i - 1];
            const std::pair<float, cinder::ColorAf>& r = config.controlPoints[i];
            int lp = std::max(0, int(floor(l.first * textureResolution)));
            int rp = std::min(textureResolution - 1, int(ceil(r.first * textureResolution)));
            for (int x = lp; x <= rp; ++x) {
                float v = (x - lp) / float(rp - lp);
                cinder::ColorAf c = l.second.lerp(v, r.second);
                mem[4 * x] = c.r;
                mem[4 * x + 1] = c.g;
                mem[4 * x + 2] = c.b;
                mem[4 * x + 3] = c.a;
            }
        }

        //update texture
        texture->update(mem.get(), GL_RGBA, GL_FLOAT, 0, textureResolution);

        CI_LOG_D("texture updated");
        textureValid = true;
    }
}

void ar::TransferFunctionEditor::invalidateTexture()
{
    textureValid = false;
}

void ar::TransferFunctionEditor::saveConfig()
{
    std::vector<std::string> extensions = { "json" };
    auto path = cinder::app::getSaveFilePath(cinder::app::getAppPath(), extensions);
    if (path.empty()) return;
    path.replace_extension("json");
    std::string pathString = path.string();
    CI_LOG_I("Save transfer function to " << pathString);

    cinder::JsonTree jsonTree;
    jsonTree.addChild(cinder::JsonTree("min", config.min));
    jsonTree.addChild(cinder::JsonTree("max", config.max));
    cinder::JsonTree points = cinder::JsonTree::makeArray("points");
    for (const auto& p : config.controlPoints) {
        cinder::JsonTree point = cinder::JsonTree::makeObject();
        point.addChild(cinder::JsonTree("pos", p.first));
        point.addChild(cinder::JsonTree("r", p.second.r));
        point.addChild(cinder::JsonTree("g", p.second.g));
        point.addChild(cinder::JsonTree("b", p.second.b));
        point.addChild(cinder::JsonTree("a", p.second.a));
        points.addChild(point);
    }
    jsonTree.addChild(points);
    jsonTree.write(path);
}
void ar::TransferFunctionEditor::loadConfig()
{
    std::vector<std::string> extensions = { "json" };
    auto path = cinder::app::getOpenFilePath(cinder::app::getAppPath(), extensions);
    if (path.empty()) return;

    std::string pathString = path.string();
    CI_LOG_I("Load transfer function from " << pathString);

    cinder::JsonTree jsonTree(cinder::loadFile(path));
    if (!jsonTree.hasChildren()) {
        CI_LOG_W("Unable to load json file");
    }
    
    Config newConfig;
    newConfig.min = jsonTree.getValueForKey<float>("min");
    newConfig.max = jsonTree.getValueForKey<float>("max");
    newConfig.controlPoints.clear();
    auto points = jsonTree.getChild("points");
    for (int i = 0; i < points.getNumChildren(); ++i) {
        auto point = points.getChild(i);
        float pos = point.getValueForKey<float>("pos");
        cinder::ColorAf c(
            point.getValueForKey<float>("r"),
            point.getValueForKey<float>("g"),
            point.getValueForKey<float>("b"),
            point.getValueForKey<float>("a")
        );
        newConfig.controlPoints.emplace_back(pos, c);
    }
    setConfig(newConfig);
}