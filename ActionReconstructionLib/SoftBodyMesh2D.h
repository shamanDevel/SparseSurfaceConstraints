#pragma once

#include <vector>
#include <Eigen/StdVector>

#include "SoftBodySimulation.h"

namespace ar
{
    class SoftBodyMesh2D : public SoftBodySimulation
    {
    public:
        typedef std::vector<Vector2, Eigen::aligned_allocator<Vector2> > Vector2List;
        typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iList;
        enum Boundary
        {
            FREE = 0,
            NEUMANN = 1,
            DIRICHLET = 2
        };

    private:
        int numNodes_;
        int numFreeNodes_;
        int numTris_;

        Vector2List referencePositions_;
        Vector3iList triIndices_;
        Vector2List currentPositions_;
        Vector2List currentDisplacements_;
		Vector2List currentVelocities_;
        std::shared_ptr<TimeIntegrator> integrator_;

        Eigen::ArrayXi nodeStates_;
        Vector2List boundaries_;

        bool needsReordering_;
        std::vector<int> nodeToFree_;
        std::vector<int> freeToNode_;

    public:
        SoftBodyMesh2D() : SoftBodySimulation() {}
        virtual ~SoftBodyMesh2D() {}

        void setMesh(const Vector2List& positions, const Vector3iList& indices);
        const Vector3iList& getTriangles() const { return triIndices_; }
        const Vector2List& getReferencePositions() const { return referencePositions_; }
		Vector2List& getReferencePositions() { return referencePositions_; }
        int getNumNodes() const { return numNodes_; }
        int getNumElements() const { return numTris_; }
        int getNumFreeNodes() const { return numFreeNodes_; }

        void resetBoundaries();
        void addDirichletBoundary(int node, const Vector2& displacement);
        void addNeumannBoundary(int node, const Vector2& force);

        void resetSolution() override;
        void solve(bool dynamic, BackgroundWorker* worker) override;

        const Eigen::ArrayXi& getNodeStates() const { return nodeStates_; }
        const Vector2List& getBoundaries() const { return boundaries_; }
        const Vector2List& getCurrentPositions() const;
        const Vector2List& getCurrentDisplacements() const;
		const Vector2List& getCurrentVelocities() const;

		//Computes the elastic energy for the given displacements
		real computeElasticEnergy(const Vector2List& displacements);

    public: //for adjoint method
        bool isReadyForSimulation() const;
        void reorderNodes();
        void computeElementMatrix(int e, const Matrix3& C, Matrix6& Ke, Vector6& Fe, const Vector2List& currentDisplacements) const;
        void assembleForceVector(VectorX& F, const VectorX* additionalForces = nullptr) const;
        void assembleStiffnessMatrix(const Matrix3& C, MatrixX& K, VectorX* F, const Vector2List& currentDisplacements) const;
        void assembleMassMatrix(VectorX& Mvec) const;
        real computeBoundaryIntegral(int node) const; //Computes the boundary integral of the given node
        const std::vector<int>& getNodeToFreeMap() const { return nodeToFree_; }
        const std::vector<int>& getFreeToNodeMap() const { return freeToNode_; }

        /**
         * \brief Places the element matrix + load vector in the big stiffness matrix and load vector of the whole mesh.
         */
        void placeIntoMatrix(int element, const Matrix6* Ke, MatrixX* K, const Vector6* Fe, VectorX* F) const;

        VectorX linearize(const Vector2List& list) const;
        Vector2List delinearize(const VectorX& vector) const;

	    /**
		 * \brief Resolves a single collision using the "PostRepair" strategy.
		 * \param dist the signed distance to the ground, as returned by SoftBodySimulation::groundCollision
		 * \param normal the surface normal of the ground, as returned by SoftBodySimulation::groundCollision
		 * \param u the current displacement
		 * \param uDot the current velocities
		 * \param alpha the penetration parameter (1->no penetration, <1 -> more penetration)
		 * \param beta the velocity damping (=0, no damping, =1, full damping)
		 * \return first: updated displacements, second: updated velocities
		 */
        [[deprecated("Implicit Spring Forces should be used instead")]]
		static std::pair<Vector2, Vector2> resolveSingleCollision_PostRepair(
			real dist, const Vector2& normal,
			const Vector2& u, const Vector2& uDot,
			real alpha, real beta);

    public: //Factories
        static SoftBodyMesh2D CreateBar(
            const Vector2& rectCenter, const Vector2& rectHalfSize,
            int resolution, bool fitObjectToGrid,
            bool enableDirichlet, const Vector2& dirichletBoundaries, const Vector2& neumannBoundaries
        );

        static SoftBodyMesh2D CreateTorus(
            real torusOuterRadius, real torusInnerRadius,
            int resolution,
			bool enableDirichlet, const Vector2& dirichletBoundaries, const Vector2& neumannBoundaries
        );
    };
}