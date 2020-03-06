#pragma once

#include <vector>
#include <Eigen/StdVector>
#include <cinder/gl/gl.h>

#include "Commons.h"
#include "GridUtils.h"
#include "TransferFunctionEditor.h"
#include "SoftBodyMesh2D.h"

namespace ar
{
	/**
	 * \brief Provides routines for rendering 2D grid and meshes.
	 */
	class GridVisualization
	{
	public:
		typedef std::vector<std::pair<Vector2, Vector2>, Eigen::aligned_allocator<std::pair<Vector2, Vector2>>> GridBoundaryLines;

	private:
		static const cinder::Color colors[3];
		//mesh
		SoftBodyMesh2D::Vector2List positions_;
		SoftBodyMesh2D::Vector3iList indices_;
		Eigen::ArrayXi nodeStates_;
		cinder::TriMeshRef triMesh_;
		bool triMeshValid_ = false;

		//grid
		GridUtils2D::grid_t referenceSdf_;
		GridUtils2D::grid_t currentSdf_;
		GridUtils2D::grid_t currentUx_;
		GridUtils2D::grid_t currentUy_;
		cinder::gl::Texture2dRef sdfTexture_;
		bool sdfTextureValid_ = false;
		bool sdfLinesValid_ = false;
		cinder::gl::GlslProgRef sdfShader_;
		cinder::gl::BatchRef sdfBatch_;
		GridBoundaryLines sdfLines_;
		std::unique_ptr<TransferFunctionEditor> tfe_;

	public:
		GridVisualization() = default;

		void setup();
		void setTfeVisible(bool visible);
		void update();

		/**
		 * \brief Applies transformations so that the grid can be drawn in the interval [-0.5, +0.5]^2.
		 */
		void applyTransformation();

		/**
		 * \brief Sets the grid to be rendered.
		 * Can be called from any thread.
		 */
		void setGrid(const GridUtils2D::grid_t& referenceSdf, const GridUtils2D::grid_t& currentSdf,
			const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy);

		/**
		 * \brief Draws the grid SDF
		 */
		void gridDrawSdf();
		/**
		 * \brief Draws the boundary conditions of the SDF
		 * \param dirichlet 
		 * \param neumannX 
		 * \param neumannY 
		 */
		void gridDrawBoundaryConditions(const GridUtils2D::bgrid_t& dirichlet, const GridUtils2D::grid_t& neumannX, const GridUtils2D::grid_t& neumannY);
		/**
		 * \brief Draws the exact boundary of the object represented by the SDF
		 */
		void gridDrawObjectBoundary();
		/**
		 * \brief Computes the line list that specifies the object boundary
		 * \return the line list
		 */
		static GridBoundaryLines getGridBoundaryLines(const GridUtils2D::grid_t& referenceSdf,
			const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy);
		/**
		 * \brief Draws the grid lines
		 */
		void gridDrawGridLines();
		/**
		 * \brief Draws the displacements
		 */
		void gridDrawDisplacements();

		/**
		 * \brief Sets the mesh to be rendered.
		 * \param positions 
		 * \param indices 
		 */
		void setMesh(const SoftBodyMesh2D::Vector2List& positions, const SoftBodyMesh2D::Vector3iList& indices, const Eigen::ArrayXi& nodeStates);

		/**
		 * \brief Draws the mesh
		 */
		void meshDraw();

		void drawGround(real height, real angle);

		void drawTransferFunctionEditor();
	};
}
