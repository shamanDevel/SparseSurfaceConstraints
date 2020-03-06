#pragma once

#include <cinder/params/Params.h>

#include <Eigen/StdVector>
#include <cuMat/Core>

#include "SoftBodyGrid3D.h"
#include "SimulationResults3D.h"

namespace ar3d
{
    class ICostFunction
    {
    public:
        /**
            * \brief Specifies which input is required by the cost function
            */
        enum RequiredInput
        {
            //only the linear diplacements on the active cells
            ActiveDisplacements = 0x1,
            //the diffused displacements on the whole grid
            GridDisplacements = 0x2,
            //the advected SDF
            AdvectedSdf = 0x4
        };

        /**
            * \brief The input to the cost function.
            */
        struct Input
        {
            /**
                * \brief Displacements of the active nodes.
                * Only available if \code getRequiredInput() & int(ActiveDisplacements)\endcode is true.
                */
            Vector3X displacements_;
            /**
                * \brief velocities of the active nodes.
                * Only available if \code getRequiredInput() & int(ActiveDisplacements)\endcode is true.
                */
            Vector3X velocities_;
            /**
                * Displacememnts mapped and diffused on the whole grid.
                * Only available if \code getRequiredInput() & int(GridDisplacements)\endcode is true.
                */
            WorldGridData<real3>::DeviceArray_t gridDisplacements_;
            /**
                * \brief The advected levelset.
                * Only available if \code getRequiredInput() & int(AdvectedSdf)\endcode is true.
                */
            WorldGridDataPtr<real> advectedSDF_;
            /**
             * \brief The reference SDF.
             * Only available/needed if \code getRequiredInput() & int(GridDisplacements)\endcode is true.
             */
            WorldGridDataPtr<real> referenceSDF_;
        };

        struct Output
        {
            /**
                * \brief The value of the cost function
                */
            real cost_ = 0;
            /**
                * \brief The adjoint/gradient with respect to the displacements.
                * Only used/allocated if \code getRequiredInput() & int(ActiveDisplacements)\endcode is true.
                */
            Vector3X adjDisplacements_;
            /**
                * \brief The adjoint/gradient with respect to the velocities.
                * Only used/allocated if \code getRequiredInput() & int(ActiveDisplacements)\endcode is true.
                */
            Vector3X adjVelocities_;
            /**
                * \brief The adjoint/gradient with respect to the displacmeents on the whole grid.
                * Only used/allocated if \code getRequiredInput() & int(GridDisplacements)\endcode is true.
                */
            WorldGridData<real3>::DeviceArray_t adjGridDisplacements_;
            /**
                * \brief The adjoint/gradient with respect to the SDF values.
                * Only used/allocated if \code getRequiredInput() & int(AdvectedSdf)\endcode is true.
                */
            WorldGridDataPtr<real> adjAdvectedSDF_;
        };

        /**
            * \brief returns a combination of RequiredInput specifying which input is needed
            */
        virtual int getRequiredInput() const  = 0;
        /**
            * \brief Returns the number of time steps that the algorithm needs to run
            */
        virtual int getNumSteps() const = 0;
        /**
        * \brief Tests if the cost function has entries at the specified timestep and should be evaluated.
        * \see evaluate(int timestep, const Input& input, Output& output)
        */
        virtual bool hasTimestep(int timestep) const = 0;
        /**
        * \brief Evaluates the cost function at the specified timestep.
        * \param timestep the current timestep, pre-check: \code hasTimestep(timestep)==true \endcode
        * \param input the input to the cost function
        * \param output the output of the cost function (preallocated)
        */
        virtual void evaluate(int timestep, const Input& input, Output& output) const = 0;
        /**
         * Preprocessing of the ground truth for a fast cost function evaluation.
         */
        virtual void preprocess(BackgroundWorker2* worker) {}

        virtual ~ICostFunction() = default;
    };
    typedef std::shared_ptr<ICostFunction> CostFunctionPtr;

    /**
     * \brief Cost function that directly compares the displacements and velocities per time step.
     */
    class CostFunctionActiveDisplacements : public ICostFunction
    {
    public:
		class GUI
		{
		public:
			std::string timestepWeights_;
			real displacementWeight_;
			real velocityWeight_;
			real noise_;
		public:
			GUI();
			void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
			void setVisible(cinder::params::InterfaceGlRef params, bool visible) const;
            void load(const cinder::JsonTree& parent);
            void save(cinder::JsonTree& parent) const;
		private:
			friend class CostFunctionActiveDisplacements;
		};

    private:
        SimulationResults3DPtr results_;
        std::vector<real> timestepWeights_;
        real displacementWeight_;
        real velocityWeight_;
		real noise_;

    public:
        CostFunctionActiveDisplacements(SimulationResults3DPtr results);
		CostFunctionActiveDisplacements(SimulationResults3DPtr results, const GUI* gui);
		void setFromGUI(const GUI* gui);

		CostFunctionActiveDisplacements& setTimestepWeights(const std::vector<real>& weights);
		const std::vector<real>& getTimestepWeights() const { return timestepWeights_; }
		CostFunctionActiveDisplacements& setDisplacementWeight(real weight);
		real getDisplacementWeight() const { return displacementWeight_; }
		CostFunctionActiveDisplacements& setVelocityWeight(real weight);
		real getNoise() const { return noise_; }
		CostFunctionActiveDisplacements& setNoise(real noise);

        int getRequiredInput() const override;
        int getNumSteps() const override;
        bool hasTimestep(int timestep) const override;
        void evaluate(int timestep, const Input& input, Output& output) const override;
    };

    /**
    * \brief Cost function that directly compares the displacements and velocities per time step.
    */
    class CostFunctionPartialObservations : public ICostFunction
    {
    public:
        typedef cuMat::Matrix<real, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor> Image;
        typedef std::vector<Image, Eigen::aligned_allocator<Image>> Observation;
        typedef std::vector<Observation, Eigen::aligned_allocator<Observation>> ObservationVector;
        typedef std::vector<DataCamera, Eigen::aligned_allocator<DataCamera>> CameraVector;
        struct Observations
        {
            real noise_;
            int resolution_;
            int numCameras_;
            CameraVector cameras_;
            ObservationVector observations_;
			bool gpuPreprocess_;
			bool gpuEvaluate_;
			real maxSdf_;
        };

        class GUI
        {
        public:
            std::string timestepWeights_;
            int numCameras_;
            real radius_;
            real centerHeight_;
            int resolution_;
            real noise_;
            CameraVector cameras_;
			bool gpuPreprocess_;
			bool gpuEvaluate_;
			real maxSdf_;
        public:
            GUI();
            void initParams(cinder::params::InterfaceGlRef params, const std::string& group);
            void setVisible(cinder::params::InterfaceGlRef params, bool visible) const;
            void load(const cinder::JsonTree& parent);
            void save(cinder::JsonTree& parent) const;
            void update(); //Update cameras
            void draw(); //Draw cameras
            Observations getObservationSettings() const;
        private:
            friend class CostFunctionPartialObservations;
        };

    public:
        SimulationResults3DPtr results_;
        std::vector<real> timestepWeights_;
        Observations observations_;

    public:
        CostFunctionPartialObservations(SimulationResults3DPtr results, const GUI* gui);
        CostFunctionPartialObservations(SimulationResults3DPtr results, 
            const std::string& timestepWeights, const Observations& observationSettings);
		CostFunctionPartialObservations(
			const std::vector<real>& timestepWeights, const Observations& observations);
        const std::vector<real>& getTimestepWeights() const { return timestepWeights_; }

        int getRequiredInput() const override;
        int getNumSteps() const override;
        bool hasTimestep(int timestep) const override;
        void evaluate(int timestep, const Input& input, Output& output) const override;
        void preprocess(BackgroundWorker2* worker) override;

		void exportObservations(const std::string& path, int everyNframe, BackgroundWorker2* worker);

		//For visualization: get depth image
		Eigen::MatrixXf getDepthImage(int frame, int camera);
		//For visualization: get observations in world space
		std::vector<glm::vec3> getObservations(int frame, int camera);

    public:
        static void preprocessSetSdfCPU(WorldGridDataPtr<real> advectedSdf);
        static void preprocessSetSdfGPU(WorldGridDataPtr<real> advectedSdf);
        //static Image preprocessCreateObservation(WorldGridDataPtr<real> advectedSdf, const DataCamera& camera, int resolution);
        static Image preprocessCreateObservationCPU(SimulationResults3DPtr results, int timestep, const DataCamera& camera, int resolution, real noise);
        static Image preprocessCreateObservationGPU(SimulationResults3DPtr results, int timestep, const DataCamera& camera, int resolution, cuMat::SimpleRandom& rnd, real noise);

        static std::pair<real, int> evaluateCameraCPU(
            WorldGridDataPtr<real> referenceSdf, WorldGridData<real3>::DeviceArray_t gridDisplacements,
            const DataCamera& camera, const Image& observedImage,
            WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut,
			const real maxSdf);
        static std::pair<real, int> evaluateCameraGPU(
            WorldGridDataPtr<real> referenceSdf, WorldGridData<real3>::DeviceArray_t gridDisplacements,
            const DataCamera& camera, const Image& observedImage,
            WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut, 
			const real maxSdf);
		static std::pair<real, int> evaluateCameraGPU_v2(
			WorldGridDataPtr<real> referenceSdf, WorldGridData<real3>::DeviceArray_t gridDisplacements,
			const DataCamera& camera, const Image& observedImage,
			WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut,
			const real maxSdf);
    };
}
