#pragma once

#include <vector>
#include <array>

#include "IInverseProblem.h"
#include "GridUtils.h"

namespace ar {

    /**
     * \brief Solve for the initial configuration using the Adjoint Method
     */
    class InverseProblem_Adjoint_InitialConfiguration :
        public IInverseProblem
    {
    public:
		struct GridPriors
		{
			real sdfPrior; //weight on the prior that forces the SDF property
			real sdfEpsilon; //the epsilon on the smoothed signum function
			real sdfMaxDistance; //band width of the SDF
		};
		using grid_t = GridUtils2D::grid_t;

    private:
        real meshPositionPrior;
		GridPriors gridPriors;
		real gridHyperEpsilon;
        int numIterations;

    public:
        InverseProblem_Adjoint_InitialConfiguration();
        virtual ~InverseProblem_Adjoint_InitialConfiguration() = default;

        InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
        InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
        void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) override;
        void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const override;

    public:
        /**
         * \brief Computes the gradient of the mesh with respect to the reference position using the adjoint method
         * \param targetPositions the target positions
         * \param simulation the simulation, used to obtain boundary conditions, gravity, material parameters and connectivity
         * \param initialPositions the initial / current positions
         * \param positionPrior the prior weight
         * \param outputCost will contain the cost of the loss function
         * \param worker background worker for status updates and interruptions
         * \return the gradient of all positions
         */
        static VectorX gradientMesh_AdjointMethod(const SoftBodyMesh2D::Vector2List& targetPositions, 
			SoftBodyMesh2D& simulation, const VectorX& initialPositions, real positionPrior, real& outputCost, 
			BackgroundWorker* worker);

        /**
        * \brief Computes the gradient of the mesh with respect to the reference position using forward derivatives
        * \param targetPositions the target positions
        * \param simulation the simulation, used to obtain boundary conditions, gravity, material parameters and connectivity
        * \param initialPositions the initial / current positions
        * \param positionPrior the prior weight
        * \param outputCost will contain the cost of the loss function
        * \param worker background worker for status updates and interruptions
        * \return the gradient of all positions
        */
        static VectorX gradientMesh_Forward(const SoftBodyMesh2D::Vector2List& targetPositions, 
			SoftBodyMesh2D& simulation, const VectorX& initialPositions, real positionPrior, real& outputCost, 
			BackgroundWorker* worker);

        /**
        * \brief Computes the gradient of the mesh with respect to the reference position using numerical differentiation
        * \param targetPositions the target positions
        * \param simulation the simulation, used to obtain boundary conditions, gravity, material parameters and connectivity
        * \param initialPositions the initial / current positions
        * \param positionPrior the prior weight
        * \param outputCost will contain the cost of the loss function
        * \param worker background worker for status updates and interruptions
        * \return the gradient of all positions
        */
        static VectorX gradientMesh_Numerical(const SoftBodyMesh2D::Vector2List& targetPositions, 
			SoftBodyMesh2D& simulation, const VectorX& initialPositions, real positionPrior, real& outputCost, 
			BackgroundWorker* worker);

	    /**
		 * \brief Computes the gradient of the grid simulation
		 *	with respect to the initial SDF
		 *	using the adjoint method
		 * \param targetSdf the target SDF
		 * \param simulation the simulation, for gravity, cell size, B's, material parameters and boundary conditions
		 *	The reference SDF of the simulation will be overwritten
		 * \param initialSdf the initial / current SDF. The gradient is evaluated at that position
		 * \param gridPriors the prior weights
		 * \param outputCost will contain the cost of the loss function
		 * \param worker background worker for status updates and interruptions
		 * \param onlyCost true -> only the cost is computed, not the gradient. (For testing)
		 * \return the gradient of the initial SDF
		 */
		static SoftBodyGrid2D::grid_t gradientGrid_AdjointMethod(const SoftBodyGrid2D::grid_t& targetSdf,
			SoftBodyGrid2D& simulation, const SoftBodyGrid2D::grid_t& initialSdf,
			const GridPriors& gridPriors,
			real& outputCost, BackgroundWorker* worker, bool onlyCost = false);

    public:
        static VectorX linearizePositions(const SoftBodyMesh2D::Vector2List& positions);
        static SoftBodyMesh2D::Vector2List delinearizePositions(const VectorX& linearizedPositions);

        typedef std::array<Vector2, 3> triangle_t;
        /**
         * \brief Computes the signed area of the triangle.
         * The area is positive if the triangle is counter-clockwise
         * \param triangle the triangle
         * \return the signed area
         */
        static real meshTriangleArea(const triangle_t& triangle);

		static int meshTriangleWindingOrder(const triangle_t& triangle);
        /**
         * \brief Returns the derivative of the signed area
         * \param triangle the triangle positions
         * \return the derivative with respect to x1,y1,x2,y2,x3,y3
         */
        static std::array<real, 6> meshTriangleAreaDerivative(const triangle_t& triangle);
        /**
         * \brief Returns the derivative matrix Be
         * \param triangle the triangle positions
         * \return the derivative matrix
         */
        static Matrix36 meshDerivativeMatrix(const triangle_t& triangle);
        /**
         * \brief Returns the derivative with respect to {x1,y1,x2,y2,x3,y3}[i] of the derivative matrix Be.
         * \param triangle the triangle positions
         * \param i the coordinate for which to take the derivate
         * \return the derivative of Be
         */
        static Matrix36 meshDerivativeMatrixDerivative(const triangle_t& triangle, int i);
        /**
         * \brief Returns the stiffness matrix Ke
         * \param triangle the triangle positions
         * \param C the material matrix C
         * \return the stiffness matrix Ke
         */
        static Matrix6 meshStiffnessMatrix(const triangle_t& triangle, const Matrix3& C);
        /**
         * \brief Returns the derivative of the stiffness matrix Ke
         *  with respect to {x1,y1,x2,y2,x3,y3}[i]
         * \param triangle the triangle positions
         * \param i the coordinate for which to take the derivate
         * \param C the material matrix C
         * \return the derivative of Ke
         */
        static Matrix6 meshStiffnessMatrixDerivative(const triangle_t& triangle, int i, const Matrix3& C, int windingOrder);
	    /**
		 * \brief Computes the per-triangle force vector induced by gravity
		 * \param triangle the triangle positions
		 * \param gravity the gravity vector
		 * \return the force vector
		 */
		static Vector6 meshForce(const triangle_t& triangle, const Vector2& gravity);
		/**
		 * \brief Computes the derivative of the force with respect to the position
		 */
		static Vector6 meshForceDerivative(const triangle_t& triangle, int i, const Vector2& gravity, int windingOrder);

		/**
		 * \brief The function gamma that transforms the SDF values into a better range for comparisons
		 */
		static real sdfTransformFun(real phi) { return std::atan(phi) * 2 / M_PI; }
		/**
		 * \brief The derivative of sdfTransformFun
		 */
		static real sdfTransformFunDerivative(real phi) { return 2 / ((1 + phi * phi) * M_PI); }
	    /**
		 * \brief Computes the similarity term of the cost function
		 * \param currentPhi the current SDF
		 * \param targetPhi the target SDF (the reference)
		 * \param weighting Optional: a weighting term to each cell
		 * \return the cost value
		 */
		static real gridCostSimilarity(const SoftBodyGrid2D::grid_t& currentPhi, const SoftBodyGrid2D::grid_t& targetPhi, const SoftBodyGrid2D::grid_t* weighting);
	    /**
		 * \brief Computes the derivative of the similarity term of the cost function
		 *	with respect to the current SDF
		 * \param currentPhi the current SDF
		 * \param targetPhi the target SDF (the reference)
		 * \param weighting Optional: a weighting term to each cell
		 * \return the derivative with respect to the current SDF
		 */
		static SoftBodyGrid2D::grid_t gridCostSimilarityDerivative(const SoftBodyGrid2D::grid_t& currentPhi, const SoftBodyGrid2D::grid_t& targetPhi, const SoftBodyGrid2D::grid_t* weighting);
	    /**
		 * \brief Computse the derivative of the force vector (gravity forces)
		 *	with respect to the current SDF
		 * \param sdfs the currentSDF
		 * \param gravity the gravity
		 * \param simulation the simulation, needed for the cell size
		 * \return the derivative of the force vector with respect to the 4 SDF values
		 */
		static std::array<Vector8, 4> gridGravityForceDerivative(const std::array<real, 4>& sdfs, const Vector2& gravity, const SoftBodyGrid2D& simulation);
		static Vector8 gridGravityForce(const std::array<real, 4>& sdfs, const Vector2& gravity, const SoftBodyGrid2D& simulation);
		/**
		* \brief Computse the derivative of the stiffness matrix Ke
		*	with respect to the current SDF
		* \param sdfs the currentSDF
		* \param C the material matrix C
		* \param simulation the simulation, needed for the cell size and B's
		* \return the derivative of the stiffness matrix Ke with respect to the 4 SDF values
		*/
		static std::array<Matrix8, 4> gridStiffnessMatrixDerivative(const std::array<real, 4>& sdfs, const Matrix3& C, const SoftBodyGrid2D& simulation);
		static Matrix8 gridStiffnessMatrix(const std::array<real, 4>& sdfs, const Matrix3& C, const SoftBodyGrid2D& simulation);

        /**
         * \brief Computes the product y'F with
         *  y being the adjoint solution of the elasticity simulation (lambdaU)
         *  and F being the negative derivative of the elasticity simulation (K(phi) u - f(phi))
         *  with respect to the input SDF phi.
         * \param lambdaU the adjoint solution of the elasticity simulation (linearized displacements)
         * \param u the current displacements from the forward solve
         * \param sdf the current input SDF
         * \param gravity the gravity
         * \param C the stiffness matrix
         * \param simulation the simulation (for boundary conditions)
         * \param posToIndex the mapping of position to degrees of freedom
         * \return the partial gradient of the input SDF on the whole grid.
         *  Only evaluated on cells inside the object
         */
        static SoftBodyGrid2D::grid_t gridElasticityAdjOpMult(
            const VectorX& lambdaU, const VectorX& u, const SoftBodyGrid2D::grid_t& sdf,
            const Vector2& gravity, const Matrix3& C, const SoftBodyGrid2D& simulation,
            const Eigen::MatrixXi& posToIndex
        );
    };
}
