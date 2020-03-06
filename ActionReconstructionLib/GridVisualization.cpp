#include "GridVisualization.h"
#include "Integration.h"
#include "SoftBodyMesh2D.h"

using namespace ar;
using namespace cinder;

const cinder::Color GridVisualization::colors[] = {
	Color(1, 0, 0),
	Color(1, 1, 0),
	Color(0, 1, 0.2)
};

void GridVisualization::setup()
{
	//transfer function editor
	tfe_ = std::make_unique<TransferFunctionEditor>(app::getWindow());
	TransferFunctionEditor::Config tfeConfig;
	tfeConfig.min = -4;
	tfeConfig.max = 4;
	tfeConfig.controlPoints.clear();
	tfeConfig.controlPoints.emplace_back(0.0f, ColorAf(0, 0, 1, 0.117647f));
	tfeConfig.controlPoints.emplace_back(0.49f, ColorAf(0, 0, 1, 1));
	tfeConfig.controlPoints.emplace_back(0.51f, ColorAf(1, 0, 0, 1));
	tfeConfig.controlPoints.emplace_back(1.0f, ColorAf(1, 0, 0, 0.392157));
	tfe_->setConfig(tfeConfig);

	//SDF shader
	sdfShader_ = gl::GlslProg::create(gl::GlslProg::Format()
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
uniform sampler1D tfTex;
uniform sampler2D sdfTex;
uniform float tfMin;
uniform float tfMax;	
in vec2	TexCoord0;
out vec4 oColor;

void main(void) {
    float val = texture(sdfTex, TexCoord0).r;
    val = -val; //level sets are outside outside

    //get color from transfer function
    vec4 col = texture(tfTex, (val - tfMin) / (tfMax - tfMin));
    oColor = col;
}
)GLSL"
));
	sdfBatch_ = gl::Batch::create(geom::Rect(cinder::Rectf(-0.5f, -0.5f, 0.5f, 0.5f)), sdfShader_);
}

void GridVisualization::setTfeVisible(bool visible)
{
	tfe_->setVisible(visible);
}

void GridVisualization::update()
{
	tfe_->update();
}

void GridVisualization::applyTransformation()
{
	//grid bounds
	int windowWidth = app::getWindow()->getWidth();
	int windowHeight = app::getWindow()->getHeight();
	int gridBoundary = 50;
	int gridSize = std::min(windowWidth, windowHeight) - 2 * gridBoundary;
	int gridOffsetX = windowWidth / 2;
	int gridOffsetY = windowHeight / 2;
	//apply transformation
	gl::translate(gridOffsetX, gridOffsetY);
	gl::scale(gridSize, gridSize);
}

void GridVisualization::setGrid(const GridUtils2D::grid_t& referenceSdf, const GridUtils2D::grid_t& currentSdf,
	const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy)
{
	referenceSdf_ = referenceSdf;
	currentSdf_ = currentSdf;
	currentUx_ = currentUx;
	currentUy_ = currentUy;
	sdfTextureValid_ = false;
	sdfLinesValid_ = false;
}

void GridVisualization::gridDrawSdf()
{
	//update texture
	if (!sdfTextureValid_) {
		//check if we need to allocate a new texture
		if (!sdfTexture_
			|| sdfTexture_->getWidth() != currentSdf_.cols()
			|| sdfTexture_->getHeight() != currentSdf_.rows()) {
			gl::Texture2d::Format format;
			format.setInternalFormat(GL_R32F);
			format.setDataType(GL_FLOAT);
			format.setMinFilter(GL_LINEAR);
			format.setMagFilter(GL_LINEAR);
			sdfTexture_ = gl::Texture2d::create(currentSdf_.cols(), currentSdf_.rows(), format);
		}
		//update texture data
		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> sdfFloat = currentSdf_.cast<float>();
		sdfTexture_->update(sdfFloat.data(), GL_RED, GL_FLOAT, 0, currentSdf_.cols(), currentSdf_.rows());
		CI_LOG_I("sdf texture updated");
		//now the texture is valid again
		sdfTextureValid_ = true;
	}
	if (sdfTextureValid_) {
		//draw the texture
		gl::ScopedMatrices m;
		applyTransformation();
		gl::ScopedTextureBind t0(sdfTexture_, 0);
		sdfShader_->uniform("sdfTex", 0);
		gl::ScopedTextureBind t1(tfe_->getTexture(), 1);
		sdfShader_->uniform("tfTex", 1);
		sdfShader_->uniform("tfMin", static_cast<float>(tfe_->getRangeMin()));
		sdfShader_->uniform("tfMax", static_cast<float>(tfe_->getRangeMax()));
		sdfBatch_->draw();
	}
}

void GridVisualization::gridDrawBoundaryConditions(const GridUtils2D::bgrid_t& dirichlet,
	const GridUtils2D::grid_t& neumannX, const GridUtils2D::grid_t& neumannY)
{
	gl::ScopedMatrices m;
	gl::ScopedColor c;
	applyTransformation();
	const int gridResolution = referenceSdf_.rows();

	for (int x = 0; x < gridResolution; ++x) {
		for (int y = 0; y < gridResolution; ++y) {
			if (dirichlet(x, y)) {
				gl::color(colors[SoftBodyMesh2D::DIRICHLET]);
				gl::drawSolidCircle(vec2((x + 0.5) / gridResolution - 0.5, (gridResolution - y - 0.5) / gridResolution - 0.5), 0.2 / gridResolution, 8);
			}
			vec2 v(neumannX(x, y), neumannY(x, y));
			if (v.x!=0 || v.y!=0) {				
				v /= sqrt(v.x*v.x + v.y*v.y);
				v *= 0.8;
				gl::color(colors[SoftBodyMesh2D::NEUMANN]);
				gl::drawVector(
					vec3((x + 0.5) / gridResolution - 0.5, (gridResolution - y - 0.5) / gridResolution - 0.5, 0),
					vec3((x + 0.5 + v.x) / gridResolution - 0.5, (gridResolution - y - v.y - 0.5) / gridResolution - 0.5, 0),
					0.2 / gridResolution, 0.05 / gridResolution);
			}
		}
	}
}

void GridVisualization::gridDrawObjectBoundary()
{
	if (!sdfLinesValid_)
	{
		sdfLines_ = getGridBoundaryLines(referenceSdf_, currentUx_, currentUy_);
		sdfLinesValid_ = true;
	}

	gl::ScopedMatrices m;
	applyTransformation();
	gl::ScopedColor c;
	gl::color(1, 1, 0);
	real h = 1.0 / (referenceSdf_.rows());
	for (const auto& points : sdfLines_)
	{
		gl::drawLine(
			vec2(points.first.x() - 0.5 + h / 2, 0.5 - points.first.y() - h / 2),
			vec2(points.second.x() - 0.5 + h / 2, 0.5 - points.second.y() - h / 2));
	}
}

GridVisualization::GridBoundaryLines
GridVisualization::getGridBoundaryLines(const GridUtils2D::grid_t& referenceSdf,
	const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy)
{
	GridBoundaryLines list;

	const int gridResolution = referenceSdf.rows();
	for (int x = 0; x<gridResolution - 1; ++x) for (int y = 0; y<gridResolution - 1; ++y)
	{
		std::array<real, 4> sdfs = { referenceSdf(x, y), referenceSdf(x + 1,y), referenceSdf(x, y + 1), referenceSdf(x + 1, y + 1) };
		if (utils::outside(sdfs[0]) && utils::outside(sdfs[1]) && utils::outside(sdfs[2]) && utils::outside(sdfs[3])) continue;
		if (utils::inside(sdfs[0]) && utils::inside(sdfs[1]) && utils::inside(sdfs[2]) && utils::inside(sdfs[3])) continue;

		real h = 1.0 / (gridResolution);
		Vector8 xe;
		xe << Vector2(x*h, y*h), Vector2((x + 1)*h, y*h),
			Vector2(x*h, (y + 1)*h), Vector2((x + 1)*h, (y + 1)*h);
		Vector8 ue;
		ue << currentUx(x, y), currentUy(x, y),
			  currentUx(x + 1, y), currentUy(x + 1, y),
			  currentUx(x, y + 1), currentUy(x, y + 1),
			  currentUx(x + 1, y + 1), currentUy(x + 1, y + 1);
		xe += ue;
		std::array<Vector2, 4> corners = { Vector2(xe.segment<2>(0)), Vector2(xe.segment<2>(2)), Vector2(xe.segment<2>(4)), Vector2(xe.segment<2>(6)) };
		auto points1 = Integration2D<real>::getIntersectionPoints(sdfs, corners);
		if (points1.has_value())
		    list.push_back(points1.value());
	}

	return list;
}

void GridVisualization::gridDrawGridLines()
{
	gl::ScopedMatrices m;
	gl::ScopedColor c;
	applyTransformation();
	const int gridResolution = currentSdf_.rows();

	//draw inner grid
	gl::color(0, 0.1f, 0.4f, 1);
	for (int i = 0; i < gridResolution; ++i) {
		double p = (i + 0.5) / double(gridResolution) - 0.5;
		double off = 1 / double(gridResolution) / 2;
		gl::drawLine(vec2(-0.5 + off, p), vec2(0.5 - off, p));
		gl::drawLine(vec2(p, -0.5 + off), vec2(p, 0.5 - off));
	}

	//draw outer grid
	gl::color(0, 0.6f, 0.2f, 1);
	gl::drawLine(vec2(-0.5, -0.5), vec2(0.5, -0.5));
	gl::drawLine(vec2(-0.5, 0.5), vec2(0.5, 0.5));
	gl::drawLine(vec2(-0.5, -0.5), vec2(-0.5, 0.5));
	gl::drawLine(vec2(0.5, -0.5), vec2(0.5, 0.5));
}

void GridVisualization::gridDrawDisplacements()
{
	gl::ScopedMatrices m;
	applyTransformation();
	const int gridResolution = currentSdf_.rows();
	for (int x = 0; x < gridResolution; ++x) {
		for (int y = 0; y < gridResolution; ++y) {
			vec2 v(currentUx_(x, y), currentUy_(x, y));
			float length = sqrt(v.x*v.x + v.y*v.y);
			if (length == 0) continue;
			gl::drawVector(
				vec3((x + 0.5) / gridResolution - 0.5, (gridResolution - y - 0.5) / gridResolution - 0.5, 0),
				vec3((x + 0.5) / gridResolution - 0.5 + v.x, (gridResolution - y - 0.5) / gridResolution - 0.5 - v.y, 0),
				4.0 / gridResolution * length, 1.5 / gridResolution * length);
		}
	}
}

void GridVisualization::setMesh(const SoftBodyMesh2D::Vector2List& positions,
	const SoftBodyMesh2D::Vector3iList& indices, const Eigen::ArrayXi& nodeStates)
{
	positions_ = positions;
	indices_ = indices;
	nodeStates_ = nodeStates;
	triMeshValid_ = false;
}

void GridVisualization::meshDraw()
{
	if (!triMeshValid_) {
		//update tri mesh
		if (!triMesh_) triMesh_ = cinder::TriMesh::create(cinder::TriMesh::Format().positions(2).colors(3));
		triMesh_->clear();
		for (size_t i = 0; i < positions_.size(); ++i) {
			triMesh_->appendPosition(vec2(positions_[i].x(), positions_[i].y()));
			triMesh_->appendColorRgb(colors[nodeStates_(i)]);
		}
		for (size_t i = 0; i < indices_.size(); ++i) {
			triMesh_->appendTriangle(indices_[i][0], indices_[i](1), indices_[i](2));
		}
		triMeshValid_ = true;
	}
	if (triMeshValid_) {
		//draw tri mesh
		gl::ScopedMatrices m;
		int windowWidth = app::getWindow()->getWidth();
		int windowHeight = app::getWindow()->getHeight();
		int gridBoundary = 50;
		int gridSize = std::min(windowWidth, windowHeight) - 2 * gridBoundary;
		int gridOffsetX = windowWidth / 2;
		int gridOffsetY = windowHeight / 2;
		gl::translate(gridOffsetX - 0.5*gridSize, gridOffsetY + 0.5*gridSize);
		gl::scale(gridSize, -gridSize);
		gl::enableWireframe();
		cinder::gl::VboMeshRef vbo = cinder::gl::VboMesh::create(*triMesh_);
		gl::draw(vbo);
		gl::disableWireframe();
	}
}

void GridVisualization::drawGround(real height, real angle)
{
	gl::ScopedMatrices m;
	applyTransformation();
	gl::ScopedColor c;
	gl::color(0.5, 0.3, 0.05); //brown
	vec2 center(0.0f, 0.5f-height);
	float len = 0.6 / cos(-angle);
	vec2 dir = len * vec2(cos(-angle), sin(-angle));
	gl::drawLine(center - dir, center + dir);
}

void GridVisualization::drawTransferFunctionEditor()
{
	tfe_->draw();
}
