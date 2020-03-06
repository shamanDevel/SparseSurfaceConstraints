#include "VolumeVisualization.h"

#include <cinder/Log.h>

#include "Utils.h"
#include "MarchingCubes.h"
#include "helper_matrixmath.h"
#include "TrilinearInterpolation.h"
#include <ObjectLoader.h>

using namespace Eigen;

ar3d::VolumeVisualizationParams::VolumeVisualizationParams()
	: mode_(Mode::Volume)
	, slicePosition_(0)
	, sliceAxis_(0)
	, rangeMin_(0)
	, rangeMax_(1)
	, stepSize_(0.02)
	, showGridCells_(false)
	, gridCellColor_(100, 150, 150)
{
	//default light, may be reconstructed later
	directionalLightDir_ = ar::utils::toGLM(Vector3d(0.2f, -1.6f, -0.4f).normalized());
	directionalLightColor_ = cinder::vec4(1, 1, 1, 1);
	ambientLightColor_ = cinder::vec4(0.25f, 0.25f, 0.25f, 1.0f);
	showNormals_ = false;
}

void ar3d::VolumeVisualizationParams::addParams(const cinder::params::InterfaceGlRef& params)
{
	this->params_ = params;
	std::vector<std::string> visualizationModeEnums = { "slice", "raytracing", "marching cubes", "high resolution" };
	params->addParam("SdfVisualizationMode", visualizationModeEnums, reinterpret_cast<int*>(&mode_)).group("Rendering").label("SDF Rendering").updateFn([this]() {this->updateVisualizationMode(); });

	std::vector<std::string> sliceAxisEnums = { "free", "X", "Y", "Z" };
	params->addParam("SdfVisualizationSliceAxis", sliceAxisEnums, &sliceAxis_).group("Rendering").label("Slice Axis");

	params->addParam("SdfVisualizationLightDir", &directionalLightDir_).group("Rendering").label("Light direction");
	params->addParam("SdfVisualizationLightColor", &directionalLightColor_).group("Rendering").label("Light color");
	params->addParam("SdfVisualizationAmbientColor", &ambientLightColor_).group("Rendering").label("Ambient color");
	params->addParam("SdfVisualizationShowNormals", &showNormals_).group("Rendering").label("Show normals");
	params->addParam("SdfVisualizationStepSize", &stepSize_).min(0.01f).max(1).step(0.01).group("Rendering").label("Step size");
	params->addParam("SdfVisualizationGridCells", &showGridCells_).group("Rendering").label("Show grid cells");

	updateVisualizationMode();
}

void ar3d::VolumeVisualizationParams::load(const cinder::JsonTree& parent)
{
	mode_ = ToMode(parent.getValueForKey("Mode"));
	directionalLightDir_.x = parent.getChild("LightDirection").getValueAtIndex<float>(0);
	directionalLightDir_.y = parent.getChild("LightDirection").getValueAtIndex<float>(1);
	directionalLightDir_.z = parent.getChild("LightDirection").getValueAtIndex<float>(2);
	directionalLightColor_.r = parent.getChild("LightColor").getValueAtIndex<float>(0);
	directionalLightColor_.g = parent.getChild("LightColor").getValueAtIndex<float>(1);
	directionalLightColor_.b = parent.getChild("LightColor").getValueAtIndex<float>(2);
	ambientLightColor_.r = parent.getChild("AmbientColor").getValueAtIndex<float>(0);
	ambientLightColor_.g = parent.getChild("AmbientColor").getValueAtIndex<float>(1);
	ambientLightColor_.b = parent.getChild("AmbientColor").getValueAtIndex<float>(2);
	showNormals_ = parent.getValueForKey<bool>("ShowNormals");
	stepSize_ = parent.getValueForKey<double>("StepSize");
	if (parent.hasChild("ShowGridCells")) showGridCells_ = parent.getValueForKey<bool>("ShowGridCells");
	updateVisualizationMode();
}

void ar3d::VolumeVisualizationParams::save(cinder::JsonTree& parent) const
{
	parent.addChild(cinder::JsonTree("Mode", FromMode(mode_)));
	parent.addChild(cinder::JsonTree::makeArray("LightDirection")
		.addChild(cinder::JsonTree("", directionalLightDir_.x))
		.addChild(cinder::JsonTree("", directionalLightDir_.y))
		.addChild(cinder::JsonTree("", directionalLightDir_.z)));
	parent.addChild(cinder::JsonTree::makeArray("LightColor")
		.addChild(cinder::JsonTree("", directionalLightColor_.r))
		.addChild(cinder::JsonTree("", directionalLightColor_.g))
		.addChild(cinder::JsonTree("", directionalLightColor_.b)));
	parent.addChild(cinder::JsonTree::makeArray("AmbientColor")
		.addChild(cinder::JsonTree("", ambientLightColor_.r))
		.addChild(cinder::JsonTree("", ambientLightColor_.g))
		.addChild(cinder::JsonTree("", ambientLightColor_.b)));
	parent.addChild(cinder::JsonTree("ShowNormals", showNormals_));
	parent.addChild(cinder::JsonTree("StepSize", stepSize_));
	parent.addChild(cinder::JsonTree("ShowGridCells", showGridCells_));
}

void ar3d::VolumeVisualizationParams::updateVisualizationMode()
{
	if (mode_ == Mode::Slice)
	{
		params_->setOptions("SdfVisualizationSliceAxis", "visible=true");
		params_->setOptions("SdfVisualizationLightDir", "visible=false");
		params_->setOptions("SdfVisualizationLightColor", "visible=false");
		params_->setOptions("SdfVisualizationAmbientColor", "visible=false");
		params_->setOptions("SdfVisualizationShowNormals", "visible=false");
		params_->setOptions("SdfVisualizationStepSize", "visible=false");
	}
	else
	{
		params_->setOptions("SdfVisualizationSliceAxis", "visible=false");
		params_->setOptions("SdfVisualizationLightDir", "visible=true");
		params_->setOptions("SdfVisualizationLightColor", "visible=true");
		params_->setOptions("SdfVisualizationAmbientColor", "visible=true");
		params_->setOptions("SdfVisualizationShowNormals", "visible=true");
		params_->setOptions("SdfVisualizationStepSize", "visible=true");
	}
}





ar3d::VolumeVisualization::VolumeVisualization(const cinder::app::WindowRef& window, 
	const cinder::Camera* cam, VolumeVisualizationParams* params)
	: volumeValid_(false)
    , cam_(cam)
    , window_(window)
    , cellValid_(false)
	, params_(params)
{
	if (window) {
		//connect input
		window->getSignalMouseWheel().connect(900, [this](cinder::app::MouseEvent& event) {this->mouseWheel(event); });
		//load resources
		reloadResources();
	}
	//else: running in window-less mode (only for export)
}

ar3d::VolumeVisualization::~VolumeVisualization()
{
    if (sdfData_)
        sdfData_->deleteTexture();
	//This fails because when the destructor is called, the OpenGL-context is already destroyed.
    //if (cellVertexBuffer_)
    //    CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(cellVertexBuffer_->getId()));
}


void ar3d::VolumeVisualization::setInput(const SoftBodyGrid3D::Input& input)
{
    std::lock_guard<std::mutex> guard(inputMutex_);
    input_ = input;
	positions_ = input_.referencePositions_.deepClone();
	advectedSdf_ = nullptr;
	volumeValid_ = false;
	mcBatch_ = nullptr;
	highResBatch_ = nullptr;
	highResMesh_ = nullptr;
    cellValid_ = false;
    cellBatch_ = nullptr;
}

void ar3d::VolumeVisualization::setHighResInputMesh(const std::string & file)
{
	if (file.empty()) {
		//no high resolution mesh
		highResMesh_ = nullptr;
		modifiedHighResMesh_ = nullptr;
	}
	else {
		//load mesh
		highResMesh_ = ObjectLoader::loadCustomObj(file);
		highResBatch_ = nullptr;
		if (highResMesh_->hasTexCoords0()) {
			//use original texture coordinates
			modifiedHighResMesh_ = cinder::TriMesh::create(*highResMesh_);
		}
		else {
			//use generated texture coordinates (the original vertex position)
			//to provide a mapping back to the undeformed mesh.
			//If the original mesh has texture coordinates, write original positions into texCoords1,
			//else, write them into tex coords 0
			auto format = cinder::TriMesh::formatFromSource(*highResMesh_).normals();
			if (highResMesh_->hasTexCoords0()) format = format.texCoords0(2).texCoords1(3);
			else format = format.texCoords0(3);
			auto mesh = cinder::TriMesh::create(*highResMesh_, format);
			size_t n = mesh->getNumVertices();
			if (highResMesh_->hasTexCoords0()) {
				mesh->getBufferTexCoords1().clear();
				for (size_t i = 0; i < n; ++i) {
					mesh->appendTexCoord1(mesh->getPositions<3>()[i]);
				}
			}
			else {
				mesh->getBufferTexCoords0().clear();
				for (size_t i = 0; i < n; ++i) {
					mesh->appendTexCoord0(mesh->getPositions<3>()[i]);
				}
			}
			modifiedHighResMesh_ = mesh;
		}
	}
}

bool ar3d::VolumeVisualization::hasHighResMesh() const
{
	return modifiedHighResMesh_ != nullptr && highResMesh_!=nullptr;
}

void ar3d::VolumeVisualization::update(const SoftBodyGrid3D::State& state)
{
	positions_ = input_.referencePositions_ + state.displacements_;
	advectedSdf_ = state.advectedSDF_;
	gridDisplacements_ = state.gridDisplacements_;
	volumeValid_ = false;
	mcBatch_ = nullptr;
    cellValid_ = false;
	highResBatch_ = nullptr;

	real3 centerOfMass;
	positions_.segment<1>(input_.centerOfMassIndex_).eval().copyToHost(&centerOfMass);
	CI_LOG_I("center of mass: " << centerOfMass);
}

bool ar3d::VolumeVisualization::needsTransferFunction() const
{
    return params_->mode_ == VolumeVisualizationParams::Mode::Slice;
}

void ar3d::VolumeVisualization::setTransferFunction(cinder::gl::Texture1dRef transferFunction,
    double rangeMin, double rangeMax)
{
    this->transferFunctionTexture_ = transferFunction;
    params_->rangeMin_ = rangeMin;
    params_->rangeMax_ = rangeMax;
}

void ar3d::VolumeVisualization::draw()
{
    if (!volumeValid_)
    {
		//update volume / create textures
        std::lock_guard<std::mutex> guard(inputMutex_);

        if (input_.grid_ == nullptr) return; //no input yet

        WorldGridRealDataPtr currentSdf = advectedSdf_ != nullptr ? advectedSdf_ : input_.referenceSdf_;

		//invalidate volume
        if (sdfData_ == nullptr || currentSdf->getGrid()->getSize() != sdfData_->getGrid()->getSize())
        {
            //allocate new texture
            if (sdfData_ != nullptr) sdfData_->deleteTexture();
            sdfData_ = std::make_shared<WorldGridData<real>>(currentSdf->getGrid());
        } else
        {
            //just update grid
            sdfData_->getGrid()->setOffset(currentSdf->getGrid()->getOffset());
        }
		
        //update data
        if (currentSdf->hasHostMemory())
        {
			sdfData_->setHostMemory(currentSdf->getHostMemory());
			sdfData_->invalidateTexture();
			sdfTexture_ = sdfData_->getTexture(WorldGridData<real>::DataSource::HOST);
        } else
        {
			//sdfData_->setDeviceMemory(currentSdf->getDeviceMemory());
			//sdfData_->invalidateTexture();
			//sdfTexture_ = sdfData_->getTexture(WorldGridData<float>::DataSource::DEVICE);
			sdfData_->setDeviceMemory(currentSdf->getDeviceMemory());
			sdfData_->copyDeviceToHost();
			sdfData_->invalidateTexture();
			sdfTexture_ = sdfData_->getTexture(WorldGridData<real>::DataSource::HOST);
        }
		
        volumeValid_ = true;
    }

    //actual drawing
    if (params_->showGridCells_)
        drawCells();
    switch (params_->mode_)
    {
	case VolumeVisualizationParams::Mode::Slice: drawSlice(); break;
    case VolumeVisualizationParams::Mode::Volume: drawSurface(); break;
	case VolumeVisualizationParams::Mode::MCSurface: drawMarchingCubes(); break;
	case VolumeVisualizationParams::Mode::HighRes: drawHighResMesh(); break;
    default: throw std::exception("Unknown visualization mode");
    }
}

void ar3d::VolumeVisualization::reloadResources()
{
    try {
        sliceShader_ = cinder::gl::GlslProg::create(
            cinder::app::loadAsset("shaders/VolumeVisualizationSlice.vert"),
            cinder::app::loadAsset("shaders/VolumeVisualizationSlice.frag"));
        sliceBatch_ = cinder::gl::Batch::create(cinder::geom::Rect(), sliceShader_);
        CI_LOG_I("slice shader (re)loaded");

        surfaceShader_ = cinder::gl::GlslProg::create(
            cinder::app::loadAsset("shaders/VolumeVisualizationSurface.vert"),
            cinder::app::loadAsset("shaders/VolumeVisualizationSurface.frag"));
        surfaceBatch_ = cinder::gl::Batch::create(cinder::geom::Rect(cinder::Rectf(-1,-1, +1,+1)), surfaceShader_);
        CI_LOG_I("surface shader (re)loaded");

		cinder::gl::GlslProg::Format mcShaderFormat;
		mcShaderFormat.vertex(cinder::app::loadAsset("shaders/VolumeVisualizationMC.vert"));
		mcShaderFormat.geometry(cinder::app::loadAsset("shaders/VolumeVisualizationMC.geom"));
		mcShaderFormat.fragment(cinder::app::loadAsset("shaders/VolumeVisualizationMC.frag"));
		mcShaderFormat.attrib(cinder::geom::CUSTOM_0, "in_NodeIndices");
		mcShaderFormat.attrib(cinder::geom::CUSTOM_1, "in_InterpWeight");
		mcShader_ = cinder::gl::GlslProg::create(mcShaderFormat);

		mcBatch_ = nullptr;
		CI_LOG_I("marching cubes shader (re)loaded");
    } catch (const  cinder::gl::GlslProgExc& ex)
    {
        CI_LOG_EXCEPTION("Unable to load shaders", ex);
    }
}

void ar3d::VolumeVisualization::saveSdf(const std::string & filename)
{
	std::ofstream o(filename, std::ofstream::binary);
	sdfData_->save(o);
	o.close();
}

void ar3d::VolumeVisualization::saveMCMesh(const std::string & filename)
{
	updateMarchingCubes();

	//old: save .obj
	//ObjectLoader::saveCustomObj(mcMesh_, filename);

	//new: save .ply
	std::string path = filename;
	std::string::size_type i = path.rfind('.', path.length());
	if (i != std::string::npos) {
		path.replace(i + 1, 3, "ply");
	}
	ObjectLoader::saveMeshPly(mcMesh_, path);
}

void ar3d::VolumeVisualization::saveHighResultMesh(const std::string & filename, bool includeNormals, bool includeOriginalPositions)
{
	if (hasHighResMesh()) {
		updateHighResMesh();
		auto format = ci::TriMesh::formatFromSource(*modifiedHighResMesh_);
		if (!includeNormals) format.mNormalsDims = 0;
		if (!includeOriginalPositions && format.mTexCoords1Dims > 0) format.mTexCoords1Dims = 0;
		else if (!includeOriginalPositions && format.mTexCoords1Dims == 0) format.mTexCoords0Dims = 0;
		ci::TriMeshRef copy = ci::TriMesh::create(*modifiedHighResMesh_, format);

		//old: save .obj
		//ObjectLoader::saveCustomObj(copy, filename);

		//new: save .ply
		std::string path = filename;
		std::string::size_type i = path.rfind('.', path.length());
		if (i != std::string::npos) {
			path.replace(i + 1, 3, "ply");
		}
		ObjectLoader::saveMeshPly(copy, path);
	}
}

void ar3d::VolumeVisualization::drawSlice()
{
    double maxSlicePos = (sdfData_->getGrid()->getSize().cast<double>()).norm() * 0.5;
	params_->slicePosition_ = std::max(-maxSlicePos, std::min(maxSlicePos, params_->slicePosition_));

    using namespace ar::utils;
    Vector3d eyePos = toEigen(cam_->getEyePoint());
    Vector3d gridCenter = (sdfData_->getGrid()->getOffset().cast<double>() + (Vector3i(1, 1, 1) + sdfData_->getGrid()->getSize()).cast<double>() * 0.5) * sdfData_->getGrid()->getVoxelSize();
    Vector3d focusPoint;
    Vector3d planeNormal;
    switch (params_->sliceAxis_)
    {
    case 0: //free
        planeNormal = toEigen(-cam_->getViewDirection()).normalized(); 
        focusPoint = gridCenter + params_->slicePosition_ * sdfData_->getGrid()->getVoxelSize() * (gridCenter - eyePos).normalized();
        break;
    case 1: //X
        planeNormal = Vector3d(1, 0, 0); 
        focusPoint = gridCenter + params_->slicePosition_ * sdfData_->getGrid()->getVoxelSize() * planeNormal;
        break;
    case 2: //Y
        planeNormal = Vector3d(0, 1, 0);
        focusPoint = gridCenter + params_->slicePosition_ * sdfData_->getGrid()->getVoxelSize() * planeNormal;
        break;
    case 3: //Z
        planeNormal = Vector3d(0, 0, 1);
        focusPoint = gridCenter + params_->slicePosition_ * sdfData_->getGrid()->getVoxelSize() * planeNormal;
        break;
    }
    
    double size = sdfData_->getGrid()->getSize().cast<double>().sum() * sdfData_->getGrid()->getVoxelSize(); //estimate of the size of the plane
    {
        //we want to draw the slice through 'focusPoint' with normal 'planeNormal'
        cinder::gl::ScopedMatrices m;
        cinder::gl::ScopedDepthTest dt(true);
        cinder::gl::ScopedDepthWrite dw(false);
        Quaterniond rot = Quaterniond::FromTwoVectors(Vector3d(0, 0, 1), planeNormal);
        cinder::gl::translate(toGLM(focusPoint));
        cinder::gl::rotate(toGLM(rot));
        cinder::gl::scale(float(size), float(size), float(size));

        cinder::gl::ScopedTextureBind t0(transferFunctionTexture_, 0);
        sliceShader_->uniform("tfTex", 0);
        cinder::gl::ScopedTextureBind t1(sdfTexture_, 1);
        sliceShader_->uniform("volTex", 1);
        sliceShader_->uniform("tfMin", static_cast<float>(params_->rangeMin_));
        sliceShader_->uniform("tfMax", static_cast<float>(params_->rangeMax_));
        sliceShader_->uniform("boxMin", toGLM(((sdfData_->getGrid()->getOffset().cast<double>() + Vector3d(0.5, 0.5, 0.5)) * sdfData_->getGrid()->getVoxelSize()).eval()));
        sliceShader_->uniform("boxSize", toGLM((sdfData_->getGrid()->getSize().cast<double>() * sdfData_->getGrid()->getVoxelSize()).eval()));
        sliceBatch_->draw();
    }
}

void ar3d::VolumeVisualization::drawSurface()
{
    using namespace ar::utils;
    cinder::gl::ScopedMatrices m;
    cinder::gl::ScopedDepthTest dt(true);
    cinder::gl::ScopedDepthWrite dw(true);

    cinder::gl::ScopedTextureBind t1(sdfTexture_, 0);
    surfaceShader_->uniform("volTex", 0);
    surfaceShader_->uniform("boxMin", toGLM(((sdfData_->getGrid()->getOffset().cast<double>() - Vector3d(0.5, 0.5, 0.5)) * sdfData_->getGrid()->getVoxelSize()).eval()));
    surfaceShader_->uniform("boxSize", toGLM((sdfData_->getGrid()->getSize().cast<double>() * sdfData_->getGrid()->getVoxelSize()).eval()));
    surfaceShader_->uniform("stepSize", static_cast<float>(sdfData_->getGrid()->getVoxelSize() * params_->stepSize_));
    surfaceShader_->uniform("directionalLightDir", params_->directionalLightDir_);
    surfaceShader_->uniform("directionalLightColor", params_->directionalLightColor_);
    surfaceShader_->uniform("ambientLightColor", params_->ambientLightColor_);
    surfaceShader_->uniform("showNormals", params_->showNormals_);

    surfaceBatch_->draw();
}

void ar3d::VolumeVisualization::updateMarchingCubes()
{
	if (mcBatch_ == nullptr) {
		std::vector<int> indexBuffer;
		std::vector<MarchingCubes::Vertex> vertexBuffer;
		MarchingCubes::polygonizeGrid(input_.referenceSdf_, input_.posToIndex_, indexBuffer, vertexBuffer);
		mcMesh_ = cinder::TriMesh::create(
			cinder::TriMesh::Format()
			.positions(3)
			.normals()
		);
		int maxIndex = 0;
		for (const auto& v : vertexBuffer)
			maxIndex = std::max(maxIndex, std::max(v.indexA, v.indexB));
		if (maxIndex >= positions_.size())
		{
			CI_LOG_E("max index (" << maxIndex << ") >= positions.size() (" << positions_.size() << ")");
			return;
		}
		std::vector<real3> positions(positions_.size());
		positions_.copyToHost(&positions[0]);
		for (const auto& v : vertexBuffer)
		{
			real3 pos = (1 - v.weight) * positions[v.indexA] + v.weight * positions[v.indexB];
			mcMesh_->appendPosition(glm::vec3(pos.x, pos.y, pos.z));
		}
		for (size_t i = 0; i < indexBuffer.size() / 3; ++i)
			mcMesh_->appendTriangle(indexBuffer[3 * i], indexBuffer[3 * i + 1], indexBuffer[3 * i + 2]);
		mcMesh_->recalculateNormals();
		if (window_!=nullptr)
			mcBatch_ = cinder::gl::Batch::create(*mcMesh_, cinder::gl::getStockShader(cinder::gl::ShaderDef().lambert()));
	}
}

void ar3d::VolumeVisualization::drawMarchingCubes()
{
#if 1
	updateMarchingCubes();
	ci::gl::BatchRef batch = mcBatch_;
	if (batch) batch->draw();
#endif


	//TODO: nothing is displayed. Why??
#if 0
	if (mcBatch_ == nullptr)
	{
		//create new marching cubes mesh
		std::vector<int> indexBuffer;
		std::vector<MarchingCubes::Vertex> vertexBuffer;
		MarchingCubes::polygonizeGrid(input_.referenceSdf_, input_.posToIndex_, indexBuffer, vertexBuffer);

		std::vector<std::pair<cinder::geom::BufferLayout, cinder::gl::VboRef>> vertexVbos;
		std::vector<int> vertexIndices(2*vertexBuffer.size());
		for (size_t i=0; i<vertexBuffer.size(); ++i)
		{
			vertexIndices[2 * i + 0] = vertexBuffer[i].indexA;
			vertexIndices[2 * i + 1] = vertexBuffer[i].indexB;
		}
		std::vector<float> vertexWeights(vertexBuffer.size());
		for (size_t i = 0; i < vertexBuffer.size(); ++i)
			vertexWeights[i] = static_cast<float>(vertexBuffer[i].weight);
		cinder::geom::BufferLayout layout1, layout2;
		layout1.append(cinder::geom::Attrib::CUSTOM_0, cinder::geom::INTEGER, 2, 2 * sizeof(int), 0);
		layout2.append(cinder::geom::Attrib::CUSTOM_1, cinder::geom::FLOAT, 1, sizeof(float), 0);
		vertexVbos.emplace_back(layout1, cinder::gl::Vbo::create(GL_ARRAY_BUFFER, vertexIndices, GL_STATIC_DRAW));
		vertexVbos.emplace_back(layout2, cinder::gl::Vbo::create(GL_ARRAY_BUFFER, vertexWeights, GL_STATIC_DRAW));
		cinder::gl::VboRef indexVbo = cinder::gl::Vbo::create(GL_ELEMENT_ARRAY_BUFFER, indexBuffer, GL_STATIC_DRAW);

		cinder::gl::VboMeshRef mesh = cinder::gl::VboMesh::create(vertexBuffer.size(), GL_TRIANGLES, vertexVbos, indexBuffer.size(), GL_INT, indexVbo);
		mcBatch_ = cinder::gl::Batch::create(mesh, mcShader_);
		CI_LOG_D("Marching cubes mesh created, " << vertexBuffer.size() << " vertices, " << indexBuffer.size() << " indices");

		//allocate position buffer
		if (mcPositionBuffer_ != nullptr) {
			CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(mcPositionBuffer_->getId()));
			mcPositionBuffer_.reset();
			CI_LOG_D("position buffer deleted");
		}
		mcPositionBuffer_ = cinder::gl::BufferObj::create(GL_SHADER_STORAGE_BUFFER, sizeof(real3) * positions_.size(), nullptr, GL_DYNAMIC_DRAW);
		CUMAT_SAFE_CALL(cudaGLRegisterBufferObject(mcPositionBuffer_->getId()));
		CI_LOG_D("position buffer allocated of size " << sizeof(real3) << " * " << positions_.size());
		mcValid_ = false;
	}

	if (!stateValid_)
	{
		//write position buffer
		void* mem;
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		CUMAT_SAFE_CALL(cudaGLMapBufferObject(&mem, mcPositionBuffer_->getId()));
		CUMAT_SAFE_CALL(cudaMemcpy(mem, positions_.data(), sizeof(real3) * positions_.size(), cudaMemcpyDeviceToDevice));
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		CUMAT_SAFE_CALL(cudaGLUnmapBufferObject(mcPositionBuffer_->getId()));
		CI_LOG_D("position buffer updated, " << positions_.size() << " entries written");
		std::vector<real3> testData(positions_.size());
		positions_.copyToHost(&testData[0]);
		for (size_t i = 0; i < positions_.size(); ++i) cinder::app::console() << "  " << testData[i].x << " " << testData[i].y << " " << testData[i].z << std::endl;
		mcValid_ = true;
	}

	//draw
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcPositionBuffer_->getId());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mcPositionBuffer_->getId());
	cinder::gl::ScopedFaceCulling c(false);
	mcBatch_->draw();
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#endif
}

void ar3d::VolumeVisualization::updateHighResMesh()
{
	if (!hasHighResMesh()) return;

	if (highResBatch_ == nullptr) {
		//create deformed mesh
		if (gridDisplacements_.size() > 0) {
			//1. copy grid displacements to host
			WorldGridData<real3> gridDisplacements(input_.grid_);
			gridDisplacements.setDeviceMemory(gridDisplacements_);
			gridDisplacements.copyDeviceToHost();
			//2. update vertex positions

			const size_t n = modifiedHighResMesh_->getNumVertices();
			for (size_t i = 0; i < n; ++i) {
				glm::vec3 cellPos = highResMesh_->getTexCoords1<3>()[i];
				//interpolate deformation
				real3 xyz; int3 ijk; long double tmp;
				xyz.x = modf(cellPos.x, &tmp); ijk.x = int(tmp);
				xyz.y = modf(cellPos.y, &tmp); ijk.y = int(tmp);
				xyz.z = modf(cellPos.z, &tmp); ijk.z = int(tmp);
				real3 corners[] = {
					gridDisplacements.getHost(ijk.x + 0, ijk.y + 0, ijk.z + 0),
					gridDisplacements.getHost(ijk.x + 1, ijk.y + 0, ijk.z + 0),
					gridDisplacements.getHost(ijk.x + 0, ijk.y + 1, ijk.z + 0),
					gridDisplacements.getHost(ijk.x + 1, ijk.y + 1, ijk.z + 0),
					gridDisplacements.getHost(ijk.x + 0, ijk.y + 0, ijk.z + 1),
					gridDisplacements.getHost(ijk.x + 1, ijk.y + 0, ijk.z + 1),
					gridDisplacements.getHost(ijk.x + 0, ijk.y + 1, ijk.z + 1),
					gridDisplacements.getHost(ijk.x + 1, ijk.y + 1, ijk.z + 1)
				};
				real3 displacement = ar3d::trilinear(xyz, corners);
				//update position
				modifiedHighResMesh_->getPositions<3>()[i] =
					highResMesh_->getPositions<3>()[i] +
					glm::vec3(displacement.x, displacement.y, displacement.z);
			}
		}
		//2. update normals
		modifiedHighResMesh_->recalculateNormals();
		//3. create batch
		if (window_) {
			if (modifiedHighResMesh_->hasColors())
				highResBatch_ = cinder::gl::Batch::create(*modifiedHighResMesh_, cinder::gl::getStockShader(cinder::gl::ShaderDef().lambert().color()));
			else
				highResBatch_ = cinder::gl::Batch::create(*modifiedHighResMesh_, cinder::gl::getStockShader(cinder::gl::ShaderDef().lambert()));
		}
	}
}

void ar3d::VolumeVisualization::drawHighResMesh()
{
	if (!hasHighResMesh()) {
		//no high resolution mesh available, fallback to marching cubes
		drawMarchingCubes();
		return;
	}

	//update mesh
	updateHighResMesh();

	//draw it
	highResBatch_->draw();
}

void ar3d::VolumeVisualization::drawCells()
{
	if (input_.numActiveCells_ == 0 || input_.cellSdfs_.size() == 0 && input_.mapping_.size() == 0) return;
    if (cellBatch_ == nullptr)
    {
        //create index buffer
        int numCells = input_.numActiveCells_;
        std::vector<real8> sdfHost(input_.cellSdfs_.size()); input_.cellSdfs_.copyToHost(&sdfHost[0]);
        std::vector<int4> indexHost(input_.mapping_.size()); input_.mapping_.copyToHost(&indexHost[0]);
        std::vector<unsigned> indexBuffer;
        static const int CubeEdges[] = {
            0,1, 0,2, 1,3, 2,3, 4,5, 4,6, 5,7, 6,7, 0,4, 1,5, 2,6, 3,7
        };
        for (int i=0; i<numCells; ++i)
        {
            real8 sdf = sdfHost[i];
            int c = 0;
            if (sdf.first.x < 0) c |= 1;
            if (sdf.first.y < 0) c |= 2;
            if (sdf.first.z < 0) c |= 4;
            if (sdf.first.w < 0) c |= 8;
            if (sdf.second.x < 0) c |= 16;
            if (sdf.second.y < 0) c |= 32;
            if (sdf.second.z < 0) c |= 64;
            if (sdf.second.w < 0) c |= 128;
            if (c == 0 || c == 255) continue; //no boundary
            //create cube
            int4 index = indexHost[i];
            int cellIndices[] = {
                index.x, index.x + 1,
                index.y, index.y + 1,
                index.z, index.z + 1,
                index.w, index.w + 1
            };
            for (int j=0; j<12; ++j)
            {
                indexBuffer.push_back(cellIndices[CubeEdges[2*j]]);
                indexBuffer.push_back(cellIndices[CubeEdges[2*j+1]]);
            }
        }

        //allocate vertex buffer
        if (cellVertexBuffer_)
            CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(cellVertexBuffer_->getId()));
        cellVertexBuffer_ = cinder::gl::Vbo::create(GL_ARRAY_BUFFER, input_.numActiveNodes_ * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
        CUMAT_SAFE_CALL(cudaGLRegisterBufferObject(cellVertexBuffer_->getId()));
        cellValid_ = false;

        //create vbo and batch
        cinder::gl::VboRef indexBufferSolidSurface = cinder::gl::Vbo::create(GL_ELEMENT_ARRAY_BUFFER, indexBuffer, GL_STATIC_DRAW);
        cinder::geom::BufferLayout layout;
        layout.append(cinder::geom::Attrib::POSITION, cinder::geom::DataType::FLOAT, 3, sizeof(float3), 0);
        cinder::gl::VboMeshRef vbo = cinder::gl::VboMesh::create(
            input_.numActiveNodes_, GL_LINES, std::vector<std::pair<cinder::geom::BufferLayout, cinder::gl::VboRef>>({ std::make_pair(layout, cellVertexBuffer_) }),
            static_cast<uint32_t>(indexBuffer.size()), GL_UNSIGNED_INT, indexBufferSolidSurface);
        auto shader = cinder::gl::ShaderDef().color();
        cellBatch_ = cinder::gl::Batch::create(vbo, cinder::gl::getStockShader(shader));
    }

    if (!cellValid_)
    {
        //update vertex buffer
        cuMat::Matrix<float3, cuMat::Dynamic, 1, 1, cuMat::RowMajor> posFloat = positions_.cast<float3>();
        void* dst;
        CUMAT_SAFE_CALL(cudaGLMapBufferObject(&dst, cellVertexBuffer_->getId()));
        CUMAT_SAFE_CALL(cudaDeviceSynchronize());
        CUMAT_SAFE_CALL(cudaMemcpy(dst, posFloat.data(), sizeof(float3)*input_.numActiveNodes_, cudaMemcpyDeviceToDevice));
        CUMAT_SAFE_CALL(cudaDeviceSynchronize());
        CUMAT_SAFE_CALL(cudaGLUnmapBufferObject(cellVertexBuffer_->getId()));
        cellValid_ = true;
    }

    {
        //draw
        cinder::gl::ScopedColor col(params_->gridCellColor_);
        cinder::gl::ScopedFaceCulling c(false);
        cellBatch_->draw();
    }
}

void ar3d::VolumeVisualization::mouseWheel(cinder::app::MouseEvent& event)
{
    if (event.isHandled()) return; //already handled
    if (params_->mode_ != VolumeVisualizationParams::Mode::Slice) return; //only needed for slice rendering
    if (!event.isControlDown()) return; //control must be pressed

    //move slice
	params_->slicePosition_ += event.getWheelIncrement() * 0.5;

    event.setHandled();
}
