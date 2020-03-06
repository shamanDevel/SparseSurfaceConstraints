#include "TetMeshVisualization.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuMat/Core>
#include <map>
#include <vector>
#include <cinder/Log.h>
#include <cinder/app/AppBase.h>

#include "helper_matrixmath.h"

using namespace cinder::gl;
using namespace std;

namespace ar3d {

TetMeshVisualization::TetMeshVisualization()
    : mode_(VisualizationMode::SOLID_SURFACE)
	, hasInput_(false)
	, inputValid_(false)
    , positionValid_(true)
{
	reloadShaders();
}

TetMeshVisualization::~TetMeshVisualization()
{
	if (vertexBuffer_)
		CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(vertexBuffer_->getId()));
}

void TetMeshVisualization::setInput(const SoftBodyMesh3D::Input& input)
{
	//copy indices to the CPU
	std::vector<int4> indices(input.numElements_);
	input.indices_.copyToHost(&indices[0]);

	//create index buffer and batch for VisualizationMode::SOLID_SURFACE
	{
		//first, find the indices of the surface elements
		map<array<int, 3>, int> triToCount;
		for (const int4 tet : indices)
		{
			triToCount[createSortedIndex(tet.x, tet.y, tet.z)]++;
			triToCount[createSortedIndex(tet.x, tet.y, tet.w)]++;
			triToCount[createSortedIndex(tet.x, tet.z, tet.w)]++;
			triToCount[createSortedIndex(tet.y, tet.z, tet.w)]++;
		}
		surfaceIndices_.clear();
		for (const auto& e : triToCount)
		{
			assert(e.second == 1 || e.second == 2);
			if (e.second == 1)
			{
				surfaceIndices_.push_back(e.first[0]);
				surfaceIndices_.push_back(e.first[1]);
				surfaceIndices_.push_back(e.first[2]);
			}
		}
	}

	input_ = input;
	inputValid_ = false;
	hasInput_ = true;
}

void TetMeshVisualization::update(const SoftBodyMesh3D::State& state)
{
    displacements_.inplace() = state.displacements_;
    positionValid_ = false;
}

void TetMeshVisualization::draw()
{
	if (!hasInput_) return;

	if (!inputValid_)
	{
		//create displacement buffer
		displacements_ = Vector3X(input_.numFreeNodes_);
		displacements_.setZero();

		//create vertex buffer with the positions
		CUMAT_CHECK_ERROR();
		if (vertexBuffer_)
			CUMAT_SAFE_CALL(cudaGLUnregisterBufferObject(vertexBuffer_->getId()));
		vertexBuffer_ = Vbo::create(GL_ARRAY_BUFFER, input_.numTotalNodes_ * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
		CUMAT_SAFE_CALL(cudaGLRegisterBufferObject(vertexBuffer_->getId()));
		positionValid_ = false;
		//data will be written in update()

		//create vbo
		VboRef indexBufferSolidSurface = Vbo::create(GL_ELEMENT_ARRAY_BUFFER, surfaceIndices_, GL_STATIC_DRAW);
		//create vbo mesh
		cinder::geom::BufferLayout layout;
        layout.append(cinder::geom::Attrib::POSITION, cinder::geom::DataType::FLOAT, 3, sizeof(float3), 0);
		solidSurfaceVbo_ = VboMesh::create(
			input_.numTotalNodes_, GL_TRIANGLES, vector<pair<cinder::geom::BufferLayout, VboRef>>({ make_pair(layout, vertexBuffer_) }),
			static_cast<uint32_t>(surfaceIndices_.size()), GL_UNSIGNED_INT, indexBufferSolidSurface);
		surfaceIndices_.clear();

		inputValid_ = true;
	}

    if (!positionValid_)
    {
        //write new positions
        Vector3X pos = input_.referencePositions_.deepClone();
        pos.head(input_.numFreeNodes_) += displacements_;
        cuMat::Matrix<float3, cuMat::Dynamic, 1, 1, cuMat::RowMajor> posFloat = pos.cast<float3>();

#if( CI_MIN_LOG_LEVEL <= 0 ) //verbose
		std::vector<real3> dispHostDebug(pos.rows());
        pos.copyToHost(&dispHostDebug[0]);
		cinder::app::console() << "Positions:" << std::endl;
		for (int i=0; i<pos.rows(); ++i)
		{
			cinder::app::console() << "  " << dispHostDebug[i].x << "  " << dispHostDebug[i].y << "  " << dispHostDebug[i].z << std::endl;
		}
#endif

        void* dst;
        CUMAT_SAFE_CALL(cudaGLMapBufferObject(&dst, vertexBuffer_->getId()));
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
        CUMAT_SAFE_CALL(cudaMemcpy(dst, posFloat.data(), sizeof(float3)*input_.numTotalNodes_, cudaMemcpyDeviceToDevice));
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
        CUMAT_SAFE_CALL(cudaGLUnmapBufferObject(vertexBuffer_->getId()));
		solidSurfaceBatch_ = BatchRef();
        positionValid_ = true;
    }

	if (!solidSurfaceBatch_ && surfaceShader_)
		solidSurfaceBatch_ = Batch::create(solidSurfaceVbo_, surfaceShader_);

    //render
    switch (mode_)
    {
    case VisualizationMode::SOLID_SURFACE:
		if (solidSurfaceBatch_) {
			//ScopedDepth d(false);
			//ScopedBlend b(false);
			ScopedFaceCulling c(false);
			solidSurfaceBatch_->draw();
		}
        break;
    }
}

void TetMeshVisualization::reloadShaders()
{
	try {
		surfaceShader_ = GlslProg::create(
			cinder::app::loadAsset("shaders/TetMeshVisualizationSurface.vert"),
			cinder::app::loadAsset("shaders/TetMeshVisualizationSurface.frag"),
			cinder::app::loadAsset("shaders/TetMeshVisualizationSurface.geom"));
		solidSurfaceBatch_ = BatchRef();
		CI_LOG_I("surface shaders (re)loaded");
	}
	catch (const  GlslProgExc& ex)
	{
		CI_LOG_EXCEPTION("Unable to load shaders", ex);
	}
}

std::array<int, 3> TetMeshVisualization::createSortedIndex(int a, int b, int c)
{
    if (a<=b && b<=c) return {a, b, c};
    else if (a<=c && c<=b) return {a, c, b};
    else if (b<=a && a<=c) return {b, a, c};
    else if (b<=c && c<=a) return {b, c, a};
    else if (c<=a && a<=b) return {c, a, b};
    else return {c, b, a};
}
}
