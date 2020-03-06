#pragma once

#include "Commons3D.h"
#include "SoftBodySimulation3D.h"
#include "BackgroundWorker2.h"

namespace ar3d {

class SoftBodyMesh3D : public SoftBodySimulation3D
{
public:

    /**
     * \brief Input specification of the mesh simulation
     */
    struct Input
    {
        /**
         * \brief Index buffer of the tets (4 indices each), size is \ref numElements_.
         */
        Vector4Xi indices_;
        /**
         * \brief Reference positions of the vertices (size \ref numTotalNodes).
         * They must be ordered in the following way:
         * <pre> 
         * |--- free/neumann ---| |--- fixed/dirichlet ---|
         */
        Vector3X referencePositions_;
        /**
         * \brief These neumann forces are applied to the first N vertices, with N the size of this vector.
         * N must be smaller or equal than numFreeNodes_.
         */
        Vector3X neumannForces_;
        /**
		* \brief The sparsity pattern of all elasticity parameters
		*/
		cuMat::SparsityPattern<cuMat::CSR> sparsityPattern_;
		/**
		* \brief number of elements (size of \ref indices_)
		*/
		int numElements_;
		/**
		* number of total nodes (free + dirichlet)
		*/
		int numTotalNodes_;
		/**
		* \brief The number of free nodes.
		* These are the nodes in the beginning; the last (numTotalNodes-numFreeNodes) vertices are fixed dirichlet nodes.
		*/
		int numFreeNodes_;

        //Debug: check if all vectors have correct sizes
        void assertSizes() const;
    };

    /**
     * \brief Precomputed variables that are shared across multiple timesteps
     */
    struct Precomputed
    {
        /**
         * \brief The lumped mass of the free vertices, with the scaling by the mass in the settings.
         */
        VectorX lumpedMass_;
        /**
         * \brief body forces (gravity + neumann, without collision) applied to the free vertices.
         */
        Vector3X bodyForces_;
    };

    /**
     * \brief The current time step
     */
    struct State
    {
        /**
         * \brief Displacements of the free vertices.
         */
        Vector3X displacements_;
        /**
         * \brief velocities of the free vertices
         */
        Vector3X velocities_;
    };

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

	static CUMAT_STRONG_INLINE __host__ __device__ real tetSize(real3 a, real3 b, real3 c, real3 d)
	{
		//real3x3 refShape = real3x3(a - d, b - d, c - d);
		//return refShape.det() / 6;

		//The Linear Tetrahedron Course, eq. (9.4)
		return ( (b.x-a.x) * ((b.y-c.y)*(c.z-d.z) - (c.y-d.y)*(b.z-c.z)) 
			   + (c.x-b.x) * ((c.y-d.y)*(a.z-b.z) - (a.y-b.y)*(c.z-d.z))
			   + (d.x-c.x) * ((a.y-b.y)*(b.z-c.z) - (b.y-c.y)*(a.z-b.z))) / 6;
	}

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
	 * \brief Computes the sparsity pattern of the large elasticity matrix.
	 * This is implemented on the CPU and is therefore directly called when the input mesh
	 * is assembled, since the index buffer still resides in CPU memory then.
	 * \param indices the index buffer of the input (on the CPU)
	 * \param numFreeNodes the number of free nodes
	 * \return the sparsity pattern
	 */
	static cuMat::SparsityPattern<cuMat::CSR> computeSparsityPattern(const std::vector<int4>& indices, int numFreeNodes);

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
     * \brief Create the "Bar" testcase
     * \param settings 
     * \return the input parameters for the simulation
     */
    static Input createBar(const InputBarSettings& settings);

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

public:
	SoftBodyMesh3D(const Input& input);
	virtual ~SoftBodyMesh3D();

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
	 */
	void solve(bool dynamic, BackgroundWorker2* worker);

protected:
	void updateSettings() override;
	void allocateTemporary(const Input& input);
	void resetTemporary();
};

}