#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"
#include <ostream>
#include <cinder/params/Params.h>
#include <cinder/Json.h>
#include <cstring>

namespace ar3d {

class SoftBodySimulation3D
{
public:

    struct Settings
	{
	public:
		real3 gravity_;
		real youngsModulus_;
		real poissonsRatio_;
		real mass_;
		real dampingAlpha_;
		real dampingBeta_;
		real materialLambda_;
		real materialMu_;
		bool enableCorotation_;
        real timestep_;

        real3 initialLinearVelocity_;
        real3 initialAngularVelocity_;

		real4 groundPlane_; //nx, ny, nz, dist
		bool enableCollision_;
        real groundStiffness_; //the stiffness of the spring that models the ground collision
        real softmaxAlpha_; //The smootheness of the softmax. As it goes to infinity, the hard max is approximated
		bool stableCollision_;

        int solverIterations_;
        real solverTolerance_;

		real newmarkTheta_;

        bool debugSaveMatrices_ = false; //save the matrices to a matlab file

		Settings();

	    /**
		 * \brief Validates the settings and computes the Lamé coefficients
		 */
		void validate();

	    friend std::ostream& operator<<(std::ostream& os, const Settings& obj)
	    {
			return os
				<< "Settings {"
				<< "\n   gravity: (" << obj.gravity_.x << "," << obj.gravity_.y << "," << obj.gravity_.z << ")"
				<< "\n   youngsModulus: " << obj.youngsModulus_
				<< "\n   poissonsRatio: " << obj.poissonsRatio_
				<< "\n   mass: " << obj.mass_
				<< "\n   dampingAlpha: " << obj.dampingAlpha_
				<< "\n   dampingBeta: " << obj.dampingBeta_
				<< "\n   materialLambda: " << obj.materialLambda_
				<< "\n   materialMu: " << obj.materialMu_
				<< "\n   enableCorotation: " << obj.enableCorotation_
				<< "\n   timestep: " << obj.timestep_
				<< "\n   initialLinearVelocity: (" << obj.initialLinearVelocity_.x << "," << obj.initialLinearVelocity_.y << "," << obj.initialLinearVelocity_.z << ")"
				<< "\n   initialAngularVelocity: (" << obj.initialAngularVelocity_.x << "," << obj.initialAngularVelocity_.y << "," << obj.initialAngularVelocity_.z << ")"
				<< "\n   groundPlane: (" << obj.groundPlane_.x << "," << obj.groundPlane_.y << "," << obj.groundPlane_.z << "," << obj.groundPlane_.w << ")"
				<< "\n   enableCollision: " << obj.enableCollision_
				<< "\n   groundStiffness: " << obj.groundStiffness_
				<< "\n   softmaxAlpha: " << obj.softmaxAlpha_
				<< "\n   solverIterations: " << obj.solverIterations_
				<< "\n   solverTolerance: " << obj.solverTolerance_
				<< "\n   newmarkTheta: " << obj.newmarkTheta_
				<< "\n}";
	    }
		bool operator==(const Settings& rhs) const
		{
			return std::memcmp(this, &rhs, sizeof(Settings)) == 0;
		}
		bool operator!=(const Settings& rhs) const { return !(*this == rhs); }
    };
    class SettingsGui
    {
    private:
        Settings settings_;
    public:
        SettingsGui() : settings_({}) {}
        SettingsGui(const Settings& settings) : settings_(settings) {}
        const Settings& getSettings() const { return settings_; }
        void setSettings(const Settings& settings) { settings_ = settings; }

        void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
        void setVisible(cinder::params::InterfaceGlRef params, bool visible);
        void load(const cinder::JsonTree& parent);
        void save(cinder::JsonTree& parent) const;
    };

    /**
     * \brief Computes the Lamé coefficients mu and lambda based on the material properties 
     * (youngs modulus and poisson ratio)
     * \param youngsModulus 
     * \param poissonsRatio 
     * \param materialMu 
     * \param materialLambda 
     */
    static void computeMaterialParameters(real youngsModulus, real poissonsRatio,
                                          real& materialMu, real& materialLambda);

    /**
     * \brief Computes the distance and normal from the specified position to the ground.
     * \param groundPlane the ground plane as defined in the settings
     * \param position the current position
     * \return (x,y,z): collision normal, w: distance to the ground (positive outside, negative inside)
     */
    static __host__ __device__ CUMAT_STRONG_INLINE real4 groundDistance(const real4& groundPlane, const real3& position)
    {
        real4 ret = groundPlane;
        ret.w = dot(make_real3(groundPlane.x, groundPlane.y, groundPlane.z), position) - groundPlane.w;
        return ret;
    }

	/**
	 * \brief The derivative of the distance to the ground with respect to time
	 * \param groundPlane 
	 * \param velocity 
	 * \return 
	 */
	static __host__ __device__ CUMAT_STRONG_INLINE real groundDistanceDt(const real4& groundPlane, const real3& velocity)
    {
		return dot(make_real3(groundPlane.x, groundPlane.y, groundPlane.z), velocity);
    }

	/**
	 * \brief Computes the initial velocity of the vertices
	 * \param position the vertex to process
	 * \param centerOfMass the center of mass of the object
	 * \param linearVelocity the linear velocity
	 * \param angularVelocity the angular velocity
	 * \return the initial velocity of that vertex
	 */
	static __host__ __device__ CUMAT_STRONG_INLINE real3 computeInitialVelocity(
		const real3& position, const real3& centerOfMass, const real3& linearVelocity, const real3& angularVelocity)
    {
		return linearVelocity + cross(angularVelocity, position - centerOfMass);
    }

	struct IInputSettings
    {
		bool enableDirichlet; //enable dirichlet boundaries
		real3 centerDirichlet, halfsizeDirichlet; //points in these boundaries are dirichlet boundaries
        bool sampleIntegrals;
        int diffusionDistance;
        real zeroCutoff;

        IInputSettings();
        static void initParams(IInputSettings* settings, cinder::params::InterfaceGlRef params, const std::string& group, const std::string& prefix);
        static void setVisible(IInputSettings* settings, cinder::params::InterfaceGlRef params, const std::string& prefix, bool visible);
        static void load(IInputSettings* settings, const cinder::JsonTree& parent);
        static void save(const IInputSettings* settings, cinder::JsonTree& parent);

		friend std::ostream& operator<<(std::ostream& os, const IInputSettings& obj)
		{
			return os
				<< "enableDirichlet: " << obj.enableDirichlet
				<< " centerDirichlet: " << obj.centerDirichlet
				<< " halfsizeDirichlet: " << obj.halfsizeDirichlet
				<< " sampleIntegrals: " << obj.sampleIntegrals
				<< " diffusionDistance: " << obj.diffusionDistance
				<< " zeroCutoff: " << obj.zeroCutoff;
		}
    };

    /**
     * \brief Settings that specify the Bar-Testcase.
     * This is used by SoftBodyMesh3D::createBar and SoftBodyGrid3D::createBar
     */
    struct InputBarSettings : IInputSettings
    {
        //The size of each voxel as a fraction of 1 world unit
        //voxelResolution=4 -> voxel side length = 1/4 world units
        int resolution;
        real3 center, halfsize; //minimal corner and maximal corner of the bar

		bool operator==(const InputBarSettings& rhs)
		{
			return std::memcmp(this, &rhs, sizeof(InputBarSettings)) == 0;
		}
		bool operator!=(const InputBarSettings& rhs) { return !(*this == rhs); }

        friend std::ostream& operator<<(std::ostream& os, const InputBarSettings& obj)
        {
	        return os
		        << static_cast<const IInputSettings&>(obj)
		        << " resolution: " << obj.resolution
		        << " center: " << obj.center
		        << " halfsize: " << obj.halfsize;
        }
    };
    class InputBarSettingsGui
    {
    private:
        InputBarSettings settings_;
    public:
        InputBarSettingsGui() : settings_({}) {}
        InputBarSettingsGui(const InputBarSettings& settings) : settings_(settings) {}
        const InputBarSettings& getSettings() const { return settings_; }
        void setSettings(const InputBarSettings& settings) { settings_ = settings; }

        void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
        void setVisible(cinder::params::InterfaceGlRef params, bool visible);
        void load(const cinder::JsonTree& parent);
        void save(cinder::JsonTree& parent) const;
    };

	/**
	 * \brief Settings that specify the Torus-Testcase.
	 * This is used by SoftBodyGrid3D::createTorus
	 */
	struct InputTorusSettings : IInputSettings
	{
        //The size of each voxel as a fraction of 1 world unit
        //voxelResolution=4 -> voxel side length = 1/4 world units
        int resolution;
		real3 center;
		glm::vec3 orientation;
		real innerRadius, outerRadius;

		bool operator==(const InputTorusSettings& rhs)
		{
			return std::memcmp(this, &rhs, sizeof(InputTorusSettings)) == 0;
		}
		bool operator!=(const InputTorusSettings& rhs) { return !(*this == rhs); }

        friend std::ostream& operator<<(std::ostream& os, const InputTorusSettings& obj)
        {
	        return os
		        << static_cast<const IInputSettings&>(obj)
		        << " resolution: " << obj.resolution
		        << " center: " << obj.center
		        << " orientation: " << obj.orientation
		        << " innerRadius: " << obj.innerRadius
		        << " outerRadius: " << obj.outerRadius;
        }
	};
    class InputTorusSettingsGui
    {
    private:
        InputTorusSettings settings_;
    public:
        InputTorusSettingsGui() : settings_({}) {}
        InputTorusSettingsGui(const InputTorusSettings& settings) : settings_(settings) {}
        const InputTorusSettings& getSettings() const { return settings_; }
        void setSettings(const InputTorusSettings& settings) { settings_ = settings; }

        void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
        void setVisible(cinder::params::InterfaceGlRef params, bool visible);
        void load(const cinder::JsonTree& parent);
        void save(cinder::JsonTree& parent) const;
    };

    struct InputSdfSettings : IInputSettings
    {
        std::string file = "";
        int voxelResolution = 0;
        glm::ivec3 offset = {0,0,0};
        glm::ivec3 size = {0,0,0};
        bool filledCells = false;

		bool operator==(const InputSdfSettings& rhs)
		{
			return std::memcmp(this, &rhs, sizeof(InputSdfSettings)) == 0;
		}
		bool operator!=(const InputSdfSettings& rhs) { return !(*this == rhs); }

        friend std::ostream& operator<<(std::ostream& os, const InputSdfSettings& obj)
        {
	        return os
		        << static_cast<const IInputSettings&>(obj)
		        << " file: " << obj.file
		        << " voxelResolution: " << obj.voxelResolution
		        << " offset: " << obj.offset.x << "," << obj.offset.y << "," << obj.offset.z
		        << " size: " << obj.size.x << "," << obj.size.y << "," << obj.size.z
		        << " filledCells: " << obj.filledCells;
        }
    };
    class InputSdfSettingsGui
    {
    private:
        InputSdfSettings settings_;
        int counter_ = 0;
    public:
        InputSdfSettingsGui() : settings_({}) {}
        InputSdfSettingsGui(const InputSdfSettings& settings) : settings_(settings) {}
        const InputSdfSettings& getSettings() const { return settings_; }
        void setSettings(const InputSdfSettings& settings) { settings_ = settings; }

        bool isAvailable() const;
        bool hasChanged();

        void preload();
        void initParams(cinder::params::InterfaceGlRef params, const std::string& group, bool fixedInput=false);
        void setVisible(cinder::params::InterfaceGlRef params, bool visible);
        void load(const cinder::JsonTree& parent, bool fixedInput = false);
        void save(cinder::JsonTree& parent, bool fixedInput = false) const;
    };

	/**
	 * \brief Statistics about the execution of the simulation
	 * All times are in seconds.
	 */
	struct Statistics
	{
		int numFreeNodes;
		int numFixedNodes; //only for Tet simulation
		int numEmptyNodes; //only for Grid simulation
		int numElements;
		double avgEntriesPerRow; //average entries per row in the stiffness matrix
		std::vector<double> matrixAssembleTime;
		std::vector<double> collisionForcesTime;
		std::vector<int> cgIterations;
		std::vector<double> cgTime;
		std::vector<double> gridBoundingBoxTime;
		std::vector<double> gridDiffusionTime;
		std::vector<double> gridAdvectionTime;

		friend std::ostream& operator<<(std::ostream& os, const Statistics& obj)
		{
			return os
				<< "Statistics {"
				<< "\n   numFreeNodes: " << obj.numFreeNodes
				<< "\n   numFixedNodes: " << obj.numFixedNodes
				<< "\n   numEmptyNodes: " << obj.numEmptyNodes
				<< "\n   numElements: " << obj.numElements
				<< "\n   avgEntriesPerRow: " << obj.avgEntriesPerRow
				<< "\n   matrixAssembleTime: " << obj.matrixAssembleTime
				<< "\n   collisionForcesTime: " << obj.collisionForcesTime
				<< "\n   cgIterations: " << obj.cgIterations
				<< "\n   cgTime: " << obj.cgTime
				<< "\n   gridBoundingBoxTime: " << obj.gridBoundingBoxTime
				<< "\n   gridDiffusionTime: " << obj.gridDiffusionTime
				<< "\n   gridAdvectionTime: " << obj.gridAdvectionTime
				<< "\n}";
		}
	};

	//---------------------------------------------
	// The actual instances:
	// They only store the settings for simple access
	// No logic is implemented here
	//---------------------------------------------

protected:
	Settings settings_;
	Statistics statistics_ = {0};
	bool recordTimings_ = false;

	virtual void updateSettings() {}

public:

	SoftBodySimulation3D() {}
	virtual ~SoftBodySimulation3D() {}

	void setSettings(Settings settings)
	{
		settings.validate();
		settings_ = settings;
		updateSettings();
	}
	const Settings& getSettings() const { return settings_; }

	void setRecordTimings(bool enable) { recordTimings_ = enable; }
	bool isRecordTimings() const { return recordTimings_; }
	void resetTimings();
	const Statistics& getStatistics() const { return statistics_; }
};

}
