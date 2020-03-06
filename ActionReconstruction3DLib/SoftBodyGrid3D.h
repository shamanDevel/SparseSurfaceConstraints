#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"
#include "SoftBodySimulation3D.h"
#include "BackgroundWorker2.h"
#include "WorldGrid.h"
#include "GeometryUtils3D.h"

#include <cuMat/src/ConjugateGradient.h>

namespace ar3d {

	class SoftBodyGrid3D : public SoftBodySimulation3D
	{
	private:
        static int3 offsets[8];

	public:

		/**
		* \brief Input specification of the mesh simulation
		*/
		struct Input
		{
			/**
			 * \brief The grid definition (resolution + location).
			 * This must be the same definition as in all following data grids.
			 */
			WorldGridPtr grid_;
			/**
			 * \brief Reference SDF
			 */
			WorldGridDataPtr<real> referenceSdf_;
			/**
			 * \brief Number of active grid cells.
			 */
			int numActiveCells_;
			/**
			 * \brief Number of active grid nodes
			 */
			int numActiveNodes_;
            /**
             * \brief Number of nodes that are diffused.
             * These are a subset of the inactive nodes.
             */
            int numDiffusedNodes_;
			/**
			* \brief Mapping from position in the grid to index. <br>
			* Index > 0: index+1 in the linear array of elasticity. (Counted in numActiveNodes_) <br>
			* Index < 0: -index-1 in the linear array of the displacement diffusion. (Counted in numDiffusedNodes_) <br>
			* Index = 0: node that is so far outside that it does not need to be diffused
			*/
			WorldGridDataPtr<int> posToIndex_;

			/**
			 * \brief the center of mass in the reference SDF in world coordinates
			 */
			real3 centerOfMass_;

		    /**
             * \brief The eight volume interpolation weights per cell
             * Size: numActiveCells_
             */
            Vector8X interpolationVolumeWeights_;
            /**
             * \brief The eight surface boundary interpolation weights per cell
             * Size: numActiveCells_
             */
            Vector8X interpolationBoundaryWeights_;
		    /**
             * \brief the surface normal per cell.
             * Size: numActiveCells_
             */
            Vector3X surfaceNormals_;
            /**
             * \brief True if this cell is a dirichlet boundary.
             * Size: numActiveCells_
             */
            VectorXc dirichlet_;
            /**
             * \brief True iff at least one cell is a dirichlet boundary
             */
            bool hasDirichlet_;

		    /**
			 * \brief Mapping from the current cell to the eight incident nodes.
			 * Because the four nodes with +x coordinate are always exactly one position after the 
			 * nodes with -x, I only store these four indices: (-x,-y,-z), (-x,+y,-z), (-x,-y,+z), (-x,+y,+z).
			 * 
			 * Size: numActiveCells_
			 */
			Vector4Xi mapping_;

			/**
			 * \brief per cell the eight SDF values at the incident nodes.
			 * This is only needed for the collision tests, everywhere else, the
			 * derived interpolation weights are used for speed.
			 * 
			 * Size: numActiveCells_
			 */
			Vector8X cellSdfs_;

            /**
             * \brief The reference position for each free node.
             * 
             * Size: numActiveNodes_
             */
            Vector3X referencePositions_;

			/**
			 * \brief The index of the free node that is closest to the center of mass.
			 */
			int centerOfMassIndex_;

            /**
		     * \brief The sparsity pattern of all elasticity parameters.
		     * Rows+Cols: numActiveNodes_
		     */
		    cuMat::SparsityPattern<cuMat::CSR> sparsityPattern_;

			cuMat::ConjugateGradient<SMatrix> diffusionCG_;

			//Debug: check if all vectors have correct sizes
			void assertSizes() const;
		};

		/**
		* \brief Precomputed variables that are shared across multiple timesteps,
		* Depending on the Input and the settings.
		*/
		struct Precomputed
		{
			/**
             * \brief The lumped mass of the free vertices, with the scaling by the mass in the settings.
             * Size: numActiveNodes_
             */
            VectorX lumpedMass_;
            /**
             * \brief body forces (gravity + neumann, without collision) applied to the free vertices.
             * Size: numActiveNodes_
             */
            Vector3X bodyForces_;
		};

		/**
		* \brief The current time step
		*/
		struct State
		{
			/**
			* \brief Displacements of the active nodes.
			*/
			Vector3X displacements_;
			/**
			* \brief velocities of the active nodes
			*/
			Vector3X velocities_;
			/**
			 * \brief A bounding box that contains the adveccted SDF
			 */
			ar::geom3d::AABBBox advectedBoundingBox_;
			/**
			 * Displacememnts mapped and diffused on the whole grid
			 */
			WorldGridData<real3>::DeviceArray_t gridDisplacements_;
			/**
			 * \brief The advected levelset
			 */
			WorldGridDataPtr<real> advectedSDF_;

			/**
			 * \brief Performs a deep clone of the state, so that it can be saved somewhere else
			 */
			State deepClone() const;
		};

		/**
		 * \brief Creates a world grid definition from the given bounding box and global resolution
		 * \param boundingBox the bounding box
		 * \param resolution the global resolution of the grid
		 * \param border additional border in grid cells
		 * \return 
		 */
		static WorldGridPtr createGridFromBoundingBox(const ar::geom3d::AABBBox boundingBox, int resolution, int border = 3);

        //Setup 1: grid sizes
        static void setupInput1Grid(Input& input, real3 center, real3 halfsize, int resolution, int border = 3);
        //Setup 2: SDF values
        static void setupInput2Sdf(Input& input, std::function<real(real3)> posToSdf);
        //Setup 3: mapping from cell to linear index for elasticity and diffusion
        static void setupInput3Mapping(Input& input, int diffusionDistance);
        //Setup 4: per cell data (boundary conditions, interpolation weights), and also computes the center of mass
        static void setupInput4CellData(Input& input, bool enableDirichlet, real3 dirichletCenter, real3 dirichletHalfsize, bool integralsSampled);
        //Setup 5: sparsity pattern for the elasticity
        static void setupInput5SparsityPattern(Input& input);
		//Setup 6: Diffusion matrix
		static void setupInput6DiffusionMatrix(Input& input);

		/**
		* \brief Allocates the precomputed values and initializes them with zero
		* \param input the input specification
		* \return the precomputed values
		*/
		static Precomputed allocatePrecomputed(const Input& input);
		/**
		* \brief Allocate a new state and initializes it with zero
		* \param input the input specifications
		* \return the state
		*/
		static State allocateState(const Input& input);

		static void computeInitialVelocity(const Input& input, const Settings& settings, Vector3X& velocities);

		/**
		* \brief Computes the mass matrix (lumped mass on the vertices).
		* Usually, \code lumpedMass=precomputed.lumpedMass_ \endcode
		* \param input the input specification
		* \param settings the soft body settings (\ref Settings::mass is read)
		* \param lumpedMass the lumped mass is added to this vector, has to be initialized with zero.
		*/
		static void computeMassMatrix(const Input& input, const Settings& settings, VectorX& lumpedMass);

		/**
		* \brief Computes the body forces (Neumann+gravity, no collision).
		* Usually, \code bodyForces=precomputed.bodyForces_ \endcode
		* \param input the input specification
		* \param settings the soft body settings (\ref Settings::gravity is read)
		* \param bodyForces the lumped mass is added to this vector, has to be initialized with zero.
		*/
		static void computeBodyForces(const Input& input, const Settings& settings, Vector3X& bodyForces);

		/**
		* \brief Computes the collision forces with the ground plane and adds the forces to the force vector
		* \param input the input configuration
		* \param settings the settings (time step, ground plane)
		* \param state the current state
		* \param bodyForces modified force vector
		*/
		static void applyCollisionForces(const Input& input, const Settings& settings, const State& state, Vector3X& bodyForces);

		/**
		* \brief Computes the elasticity matrix
		* \param input the input mesh specification
		* \param lastState the last state (displacements for the corotation)
		* \param settings the elasticity settings.
		*	The properties \c materialLambda, \c materialMu, \c enableCorotation_ are used.
		* \param outputMatrix the output matrix, should be initialized with zero
		* \param outputForce the output force, contributions are added
		*/
		static void computeStiffnessMatrix(
			const Input& input, const State& lastState, const Settings& settings,
			SMatrix3x3& outputMatrix, Vector3X& outputForce);

		/**
		 * \brief Computes a bounding box that encloses the deformed object specified by state.
		 * The resulting bounding box is used to specify the grid for the advected levelset.
		 * \param input the reference input
		 * \param state the current state
		 * \return aa bounding box enclosing the deformed object
		 */
		static ar::geom3d::AABBBox computeTransformedBoundingBox(const Input& input, const State& state);

	    /**
         * \brief In some cases, the solver diverges and the displacements explodes.
         * In these cases, this function is a safety measurement that detects when the bounding box suddenly grows extremely
         * and limits this grow, so that we don't run out of memory.
         * \param newBox the new box
         * \param oldBox the old box from the previous frame
         * \return the new box, but limited in divergent cases
         */
        static ar::geom3d::AABBBox limitBoundingBox(const ar::geom3d::AABBBox& newBox, const ar::geom3d::AABBBox& oldBox);

		typedef cuMat::Matrix<real, cuMat::Dynamic, 1, 3, cuMat::ColumnMajor> DiffusionRhs;
		/**
		 * \brief Diffuses the displacements of the object into the empty regions and maps all displacements onto the whole grid
		 * \param input the input specification
		 * \param state the current state
		 * \param gridDisp the displacements on the whole grid (size must match input.grid_.getSize() )
		 * \param tmp1 temporar array of length (input.grid_.getSize().prod() - input.numActiveNodes_)
		 * \param tmp2 temporar array of length (input.grid_.getSize().prod() - input.numActiveNodes_)
		 */
		static void diffuseDisplacements(const Input& input, const State& state, 
			WorldGridData<real3>::DeviceArray_t& gridDisp, DiffusionRhs& tmp1, DiffusionRhs& tmp2);

		struct AdvectionSettings
		{
			real kernelRadiusIn = 1.5f;
			real kernelRadiusOut = 4.0f;
			real outerKernelWeight = 1e-5f;
			real outerSdfThreshold = 1.01f;
			real outerSdfWeight = 1e-10f;
		};
		/**
		 * \brief Advects the reference levelset by the given displacements
		 * \param input the input specification (reference grid + SDF)
		 * \param gridDisp the diffused displacements, as computed by \ref diffuseDisplacements
		 * \param advectSdf the advected levelset, boundaries computed by \ref computeTransformedBoundingBox
		 */
		static void advectLevelset(const Input& input, const WorldGridData<real3>::DeviceArray_t& gridDisp,
			WorldGridDataPtr<real> advectSdf, const AdvectionSettings& settings);

		/**
		* \brief Create the "Bar" testcase
		* \param settings
		* \return the input parameters for the simulation
		*/
		static Input createBar(const InputBarSettings& settings);

		/**
		* \brief Create the "Torus" testcase
		* \param settings
		* \return the input parameters for the simulation
		*/
		static Input createTorus(const InputTorusSettings& settings);

		/**
		* \brief Creates the input from a config (ignoring the file) and the sdf
		* \param settings (ignores grid settings and sdf file)
		* \param referenceSdf the 
		* \return the input parameters for the simulation
		*/
		static Input createFromSdf(const InputSdfSettings& settings, WorldGridRealDataPtr referenceSdf);

        /**
        * \brief Load a SDF from file
        * \param settings
        * \return the input parameters for the simulation
        */
        static Input createFromFile(const InputSdfSettings& settings);

		//---------------------------------------------
		// The actual instances:
		// They only store the settings for simple access
		// No logic is implemented here
		//---------------------------------------------

	protected:
		const Input input_;
		Precomputed precomputed_;
		State state_;

		//temporary
		Vector3X forces_;
		SMatrix3x3 stiffness_;
		SMatrix3x3 newmarkA_;
		Vector3X newmarkB_;
		DiffusionRhs diffusionTmp1_;
		DiffusionRhs diffusionTmp2_;

	public:
		SoftBodyGrid3D(const Input& input);
		virtual ~SoftBodyGrid3D();

		const Input& getInput() const { return input_; }
		const Precomputed& getPrecomputed() const { return precomputed_; }
		const State& getState() const { return state_; }

		/**
		* \brief Resets the current solution.
		*/
		void reset();

		/**
		* \brief Solves for the current solution
		* \param dynamic true: dynamic, time dependent simulation; false: static solution
		* \param worker worker for status updates
		* \param advect 
		*	false -> only displacement and velocity are computed, 
		*	true -> also the advected bounding box and levelset are computed
		*/
		void solve(bool dynamic, BackgroundWorker2* worker, bool advect=true);

	protected:
		void updateSettings() override;
		void allocateTemporary(const Input& input);
		void resetTemporary();
	};

}
