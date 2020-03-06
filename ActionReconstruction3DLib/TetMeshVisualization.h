#pragma once

#include <cinder/gl/gl.h>
#include <array>
#include "SoftBodyMesh3D.h"

namespace ar3d {

/**
 * \brief Visualization of the 3D Tet Simulation with SoftBodyMesh3D
 */
class TetMeshVisualization
{
public:

    enum class VisualizationMode
    {
        WIREFRAME,
        SOLID_SURFACE
    };

private:
    //Only use it as pointer
    CUMAT_DISALLOW_COPY_AND_ASSIGN(TetMeshVisualization);

public:

    /**
     * \brief Creates the visualization.
     * The input (indices + reference position) is assumed to be constant,
     * (or at least of the same size).
     * \param input 
     */
    explicit TetMeshVisualization();
    ~TetMeshVisualization();

	void setInput(const SoftBodyMesh3D::Input& input);

    /**
     * \brief Updates the visualization with the current state.
     * Can be called from any thread.
     * \param state 
     */
    void update(const SoftBodyMesh3D::State& state);

    /**
     * \brief Main draw call
     */
    void draw();

    void setVisualizationMode(VisualizationMode mode) {mode_ = mode;}
    VisualizationMode getVisualationMode() const {return mode_;}

	void reloadShaders();

private:
    std::array<int, 3> createSortedIndex(int a, int b, int c);

private:
    VisualizationMode mode_;
    SoftBodyMesh3D::Input input_;
	Vector3X displacements_;

	std::vector<unsigned int> surfaceIndices_;

	bool hasInput_;
	bool inputValid_;
    bool positionValid_;
	cinder::gl::GlslProgRef surfaceShader_;
    cinder::gl::VboRef vertexBuffer_;
	cinder::gl::VboMeshRef solidSurfaceVbo_;
    cinder::gl::BatchRef solidSurfaceBatch_;
	cinder::gl::VboMeshRef wireframeVbo_;
    cinder::gl::BatchRef wireframeBatch_;
};

}