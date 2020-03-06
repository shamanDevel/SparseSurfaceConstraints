#include "CostFunctions.h"

#include <random>
#include "CoordinateTransformation.h"
#include "Utils.h"
#include <cinder/gl/gl.h>
#include "tinyformat.h"
#include <cinder/app/AppBase.h>

namespace ar3d
{
    static std::vector<real> parseTimestepWeights(int numSteps, const std::string& weights)
    {
        std::vector<real> timestepWeights;
        if (weights.empty())
            timestepWeights = std::vector<real>(numSteps, real(1));
        else
        {
            timestepWeights = std::vector<real>(numSteps, real(0));
            std::stringstream s(weights);
            while (s.good())
            {
                std::string token;
                s >> token;
                int index;
                real weight = 1;
                int i = static_cast<int>(token.find(':'));
                try {
                    if (i == std::string::npos)
                        index = std::stoi(token) - 1;
                    else {
                        index = std::stoi(token.substr(0, i)) - 1;
                        weight = std::stof(token.substr(i + 1));
                    }
                }
                catch (const std::invalid_argument& ex)
                {
                    CI_LOG_EXCEPTION("Unable to parse token " << token, ex);
                    continue;
                }
                if (index >= timestepWeights.size()) continue;
                if (weight < 0)
                {
                    CI_LOG_W("negative weight " << weight << " for timestep " << index);
                    weight = 1;
                }
                timestepWeights[index] = weight;
            }
        }
        return timestepWeights;
    }

	CostFunctionActiveDisplacements::GUI::GUI()
		: timestepWeights_("")
		, displacementWeight_(1)
		, velocityWeight_(0)
		, noise_(0)
	{
	}

	void CostFunctionActiveDisplacements::GUI::initParams(cinder::params::InterfaceGlRef params, const std::string& group)
	{
		params->addParam("CostFunctionActiveDisplacements-TimestepWeights", &timestepWeights_)
			.group(group).label("Timestep weights");
		params->addParam("CostFunctionActiveDisplacements-DisplacementWeight", &displacementWeight_)
			.group(group).label("Displacement weight").min(0).step(0.01);
		params->addParam("CostFunctionActiveDisplacements-VelocityWeight", &velocityWeight_)
			.group(group).label("Velocity weight").min(0).step(0.01);
		params->addParam("CostFunctionActiveDisplacements-Noise", &noise_)
			.group(group).label("Camera Noise").min(0).step(0.01);
	}

	void CostFunctionActiveDisplacements::GUI::setVisible(cinder::params::InterfaceGlRef params, bool visible) const
	{
		std::string s = visible ? "visible=true" : "visible=false";
		params->setOptions("CostFunctionActiveDisplacements-TimestepWeights", s);
		params->setOptions("CostFunctionActiveDisplacements-DisplacementWeight", s);
		params->setOptions("CostFunctionActiveDisplacements-VelocityWeight", s);
		params->setOptions("CostFunctionActiveDisplacements-Noise", s);
	}

    void CostFunctionActiveDisplacements::GUI::load(const cinder::JsonTree& parent)
    {
        timestepWeights_ = parent.getValueForKey("TimestepWeights");
        displacementWeight_ = parent.getValueForKey<real>("DisplacementWeight");
        velocityWeight_ = parent.getValueForKey<real>("VelocityWeight");
		if (parent.hasChild("Noise")) noise_ = parent.getValueForKey<real>("Noise");
    }

    void CostFunctionActiveDisplacements::GUI::save(cinder::JsonTree& parent) const
    {
        parent.addChild(cinder::JsonTree("TimestepWeights", timestepWeights_));
        parent.addChild(cinder::JsonTree("DisplacementWeight", displacementWeight_));
        parent.addChild(cinder::JsonTree("VelocityWeight", velocityWeight_));
		parent.addChild(cinder::JsonTree("Noise", noise_));
    }

    CostFunctionActiveDisplacements::CostFunctionActiveDisplacements(SimulationResults3DPtr results)
		: results_(results)
		, timestepWeights_(results->states_.size(), real(1))
		, displacementWeight_(1)
		, velocityWeight_(0)
		, noise_(0)
	{
	}

	CostFunctionActiveDisplacements::CostFunctionActiveDisplacements(SimulationResults3DPtr results, const GUI* gui)
		: CostFunctionActiveDisplacements(results)
	{
		setFromGUI(gui);
	}

	CostFunctionActiveDisplacements& CostFunctionActiveDisplacements::setTimestepWeights(const std::vector<real>& weights)
	{
		assert(weights.size() == getNumSteps());
		timestepWeights_ = weights;
		return *this;
	}

	CostFunctionActiveDisplacements& CostFunctionActiveDisplacements::setDisplacementWeight(real weight)
	{
		assert(weight >= 0);
		displacementWeight_ = weight;
		return *this;
	}

	CostFunctionActiveDisplacements& CostFunctionActiveDisplacements::setVelocityWeight(real weight)
	{
		assert(weight >= 0);
		velocityWeight_ = weight;
		return *this;
	}

	void CostFunctionActiveDisplacements::setFromGUI(const GUI* gui)
	{
		displacementWeight_ = gui->displacementWeight_;
		velocityWeight_ = gui->velocityWeight_;
		noise_ = gui->noise_;
        timestepWeights_ = parseTimestepWeights(getNumSteps(), gui->timestepWeights_);

		CI_LOG_I("Timestep weighting: " << timestepWeights_);
	}

    CostFunctionActiveDisplacements& CostFunctionActiveDisplacements::setNoise(real noise)
    {
		assert(noise >= 0);
		noise_ = noise;
		return *this;
    }

	int CostFunctionActiveDisplacements::getRequiredInput() const
	{
		return RequiredInput::ActiveDisplacements;
	}

	int CostFunctionActiveDisplacements::getNumSteps() const
	{
		return results_->states_.size();
	}

	bool CostFunctionActiveDisplacements::hasTimestep(int timestep) const
	{
		assert(timestepWeights_.size() == getNumSteps());
		assert(timestep < getNumSteps());
		return timestepWeights_[timestep] > 0;
	}

    CostFunctionPartialObservations::GUI::GUI()
        : timestepWeights_("")
        , numCameras_(4)
        , radius_(3)
        , centerHeight_(1)
        , resolution_(50)
        , noise_(0.5)
		, gpuPreprocess_(true)
		, gpuEvaluate_(true)
		, maxSdf_(1.0f)
    {
        update();
    }

    void CostFunctionPartialObservations::GUI::initParams(cinder::params::InterfaceGlRef params,
        const std::string& group)
    {
        const std::function<void()> updateFun = [this]() {update(); };
        params->addParam("CostFunctionPartialObservations-TimestepWeights", &timestepWeights_)
            .group(group).label("Timestep weights");
        params->addParam("CostFunctionPartialObservations-NumCameras", &numCameras_)
            .group(group).label("Num Cameras").min(1).updateFn(updateFun);
        params->addParam("CostFunctionPartialObservations-Radius", &radius_)
            .group(group).label("Camera Distance").min(1).step(0.1).updateFn(updateFun);
        params->addParam("CostFunctionPartialObservations-CenterHeight", &centerHeight_)
            .group(group).label("Focus Height").step(0.1).updateFn(updateFun);
        params->addParam("CostFunctionPartialObservations-Resolution", &resolution_)
            .group(group).label("Camera resolution").min(10);
        params->addParam("CostFunctionPartialObservations-Noise", &noise_)
            .group(group).label("Camera Noise").min(0).step(0.01);
		params->addParam("CostFunctionPartialObservations-GpuPreprocess", &gpuPreprocess_)
			.group(group).label("Preprocess").optionsStr("true='GPU' false='CPU'");
		params->addParam("CostFunctionPartialObservations-GpuEvaluate", &gpuEvaluate_)
			.group(group).label("Evaluate").optionsStr("true='GPU' false='CPU'");
		params->addParam("CostFunctionPartialObservations-MaxSdf", &maxSdf_)
			.group(group).label("Max SDF value").min(0.1).step(0.1);
        update();
    }

    void CostFunctionPartialObservations::GUI::setVisible(cinder::params::InterfaceGlRef params, bool visible) const
    {
        std::string s = visible ? "visible=true" : "visible=false";
        params->setOptions("CostFunctionPartialObservations-TimestepWeights", s);
        params->setOptions("CostFunctionPartialObservations-NumCameras", s);
        params->setOptions("CostFunctionPartialObservations-Radius", s);
        params->setOptions("CostFunctionPartialObservations-CenterHeight", s);
        params->setOptions("CostFunctionPartialObservations-Resolution", s);
        params->setOptions("CostFunctionPartialObservations-Noise", s);
		params->setOptions("CostFunctionPartialObservations-GpuPreprocess", s);
		params->setOptions("CostFunctionPartialObservations-GpuEvaluate", s);
		params->setOptions("CostFunctionPartialObservations-MaxSdf", s);
    }

    void CostFunctionPartialObservations::GUI::load(const cinder::JsonTree& parent)
    {
        timestepWeights_ = parent.getValueForKey("TimestepWeights");
        numCameras_ = parent.getValueForKey<int>("NumCameras");
        radius_ = parent.getValueForKey<real>("Radius");
        centerHeight_ = parent.getValueForKey<real>("CenterHeight");
        resolution_ = parent.getValueForKey<int>("Resolution");
        noise_ = parent.getValueForKey<real>("Noise");
		if (parent.hasChild("GpuPreprocess"))
			gpuPreprocess_ = parent.getValueForKey<bool>("GpuPreprocess");
		if (parent.hasChild("GpuEvaluate"))
			gpuEvaluate_ = parent.getValueForKey<bool>("GpuEvaluate");
		if (parent.hasChild("MaxSDF"))
			maxSdf_ = parent.getValueForKey<real>("MaxSDF");
        update();
    }

    void CostFunctionPartialObservations::GUI::save(cinder::JsonTree& parent) const
    {
        parent.addChild(cinder::JsonTree("TimestepWeights", timestepWeights_));
        parent.addChild(cinder::JsonTree("NumCameras", numCameras_));
        parent.addChild(cinder::JsonTree("Radius", radius_));
        parent.addChild(cinder::JsonTree("CenterHeight", centerHeight_));
        parent.addChild(cinder::JsonTree("Resolution", resolution_));
        parent.addChild(cinder::JsonTree("Noise", noise_));
		parent.addChild(cinder::JsonTree("GpuPreprocess", gpuPreprocess_));
		parent.addChild(cinder::JsonTree("GpuEvaluate", gpuEvaluate_));
		parent.addChild(cinder::JsonTree("MaxSDF", maxSdf_));
    }

    /// @brief Returns a view transformation matrix like the one from glu's lookAt
    /// @see http://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt.xml
    /// @see glm::lookAt
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 4, 4> lookAt(Derived const & eye, Derived const & center, Derived const & up) {
        typedef Eigen::Matrix<typename Derived::Scalar, 4, 4> Matrix4;
        typedef Eigen::Matrix<typename Derived::Scalar, 3, 1> Vector3;
        Vector3 f = (center - eye).normalized();
        Vector3 u = up.normalized();
        Vector3 s = f.cross(u).normalized();
        u = s.cross(f);
        Matrix4 mat = Matrix4::Zero();
        mat(0, 0) = s.x();
        mat(0, 1) = s.y();
        mat(0, 2) = s.z();
        mat(0, 3) = -s.dot(eye);
        mat(1, 0) = u.x();
        mat(1, 1) = u.y();
        mat(1, 2) = u.z();
        mat(1, 3) = -u.dot(eye);
        mat(2, 0) = -f.x();
        mat(2, 1) = -f.y();
        mat(2, 2) = -f.z();
        mat(2, 3) = f.dot(eye);
        mat.row(3) << 0, 0, 0, 1;
        return mat;
    }

    void CostFunctionPartialObservations::GUI::update()
    {
        static int SEED = 42;
        std::default_random_engine rnd(SEED);
        std::uniform_real_distribution<real> thetaDistr(M_PI/6, M_PI / 2), phiDistr(0, 2 * M_PI);

        glm::vec3 target(0, centerHeight_, 0);
        glm::vec3 up(0, 1, 0);
        double fov = glm::radians(45.f), aspect = 1, zNear = 0.5, zFar = 100;
        glm::mat4 perspective = glm::perspective(fov, aspect, zNear, zFar);

        cameras_.resize(numCameras_);
        for (int i=0; i<numCameras_; ++i)
        {
            glm::vec3 locSpherical(radius_, thetaDistr(rnd), phiDistr(rnd));
            glm::vec3 locCartesian = CoordinateTransformation::spherical2cartesian(locSpherical);
            glm::vec3 loc(locCartesian.x, locCartesian.z, locCartesian.y);
            glm::mat4 transform = glm::lookAt(loc, target, up);
            glm::mat4 mat = perspective * transform;
            cameras_[i] = DataCamera(loc, mat);
        }
    }

    void CostFunctionPartialObservations::GUI::draw()
    {
        for (const auto& c : cameras_)
        {
            using ar::utils::toGLM;
            cinder::gl::ScopedColor scopedColor;
            cinder::gl::color(0.8, 0.2, 0.2);
            double dist = 0; //Near plane
            cinder::gl::drawSphere(c.location, 0.03f, 16);
            cinder::gl::drawSphere(c.getWorldCoordinates(glm::vec3(0, 0, dist)), 0.02f, 16);
            cinder::gl::drawSphere(c.getWorldCoordinates(glm::vec3(1, 0, dist)), 0.02f, 16);
            cinder::gl::drawSphere(c.getWorldCoordinates(glm::vec3(0, 1, dist)), 0.02f, 16);
            cinder::gl::drawSphere(c.getWorldCoordinates(glm::vec3(1, 1, dist)), 0.02f, 16);
            //lines
            cinder::gl::color(0.8, 0.4, 0.4);
            cinder::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(0, 0, dist)));
            cinder::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(1, 0, dist)));
            cinder::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(0, 1, dist)));
            cinder::gl::drawLine(c.location, c.getWorldCoordinates(glm::vec3(1, 1, dist)));
            cinder::gl::drawLine(c.getWorldCoordinates(glm::vec3(0, 0, dist)), c.getWorldCoordinates(glm::vec3(1, 0, dist)));
            cinder::gl::drawLine(c.getWorldCoordinates(glm::vec3(0, 0, dist)), c.getWorldCoordinates(glm::vec3(0, 1, dist)));
            cinder::gl::drawLine(c.getWorldCoordinates(glm::vec3(1, 1, dist)), c.getWorldCoordinates(glm::vec3(1, 0, dist)));
            cinder::gl::drawLine(c.getWorldCoordinates(glm::vec3(1, 1, dist)), c.getWorldCoordinates(glm::vec3(0, 1, dist)));
        }
    }

    CostFunctionPartialObservations::Observations CostFunctionPartialObservations::GUI::getObservationSettings() const
    {
        Observations o;
        o.noise_ = noise_;
        o.resolution_ = resolution_;
        o.numCameras_ = numCameras_;
        o.cameras_ = cameras_;
		o.gpuPreprocess_ = gpuPreprocess_;
		o.gpuEvaluate_ = gpuEvaluate_;
		o.maxSdf_ = maxSdf_;
        return o;
    }

    CostFunctionPartialObservations::CostFunctionPartialObservations(SimulationResults3DPtr results, const GUI* gui)
        : results_(results)
        , timestepWeights_(parseTimestepWeights(results->states_.size(), gui->timestepWeights_))
        , observations_(gui->getObservationSettings())
    {
    }

    CostFunctionPartialObservations::CostFunctionPartialObservations(SimulationResults3DPtr results,
        const std::string& timestepWeights, const Observations& observationSettings)
        : results_(results)
        , timestepWeights_(parseTimestepWeights(results->states_.size(), timestepWeights))
        , observations_(observationSettings)
    {
    }

	CostFunctionPartialObservations::CostFunctionPartialObservations(const std::vector<real>& timestepWeights, const Observations & observations)
		: results_(nullptr)
		, timestepWeights_(timestepWeights)
		, observations_(observations)
	{
	}

    int CostFunctionPartialObservations::getRequiredInput() const
    {
        return RequiredInput::GridDisplacements;
    }

    int CostFunctionPartialObservations::getNumSteps() const
    {
		if (results_)
			return results_->states_.size();
		else
			return timestepWeights_.size();
    }

    bool CostFunctionPartialObservations::hasTimestep(int timestep) const
    {
        assert(timestepWeights_.size() == getNumSteps());
        assert(timestep < getNumSteps());
        return timestepWeights_[timestep] > 0;
    }

    void CostFunctionPartialObservations::evaluate(int timestep, const Input& input, Output& output) const
    {
        real cost = 0;
        int numPoints = 0;
        for (int i=0; i<observations_.numCameras_; ++i)
        {
			std::pair<real, int> costNum;
			if (observations_.gpuEvaluate_) {
				std::cout << "evaluate SSC for timestep " << timestep << " on the GPU" << std::endl;
				//costNum = evaluateCameraGPU(
				costNum = evaluateCameraGPU_v2(
					input.referenceSDF_, input.gridDisplacements_,
					observations_.cameras_[i], observations_.observations_[timestep][i],
					output.adjGridDisplacements_, observations_.maxSdf_);
			}
			else {
				std::cout << "evaluate SSC for timestep " << timestep << " on the CPU" << std::endl;
				costNum = evaluateCameraCPU(
					input.referenceSDF_, input.gridDisplacements_,
					observations_.cameras_[i], observations_.observations_[timestep][i],
					output.adjGridDisplacements_, observations_.maxSdf_);
			}

        	cost += costNum.first;
            numPoints += costNum.second;
            CI_LOG_D("timestep=" << timestep << ", camera=" << i << " -> cost=" << costNum.first << ", numPoints=" << costNum.second);
        }
        if (numPoints > 0) {
            output.cost_ += cost / numPoints;
            output.adjGridDisplacements_ *= make_real3(1.0f / numPoints);
            CI_LOG_I("timestep=" << timestep << " -> total cost: " << output.cost_ << ", total points: " << numPoints
                << ", adjoint norm: " << std::sqrt(static_cast<real>(output.adjGridDisplacements_.cwiseAbs2().sum<cuMat::Axis::All>())));
        } else
        {
            CI_LOG_W("timestep=" << timestep << " -> no points found, current solution to far away from the observations");
        }
    }

    void CostFunctionPartialObservations::preprocess(BackgroundWorker2* worker)
    {
		if (!results_) return;

        //allocations
        assert(observations_.numCameras_ == observations_.cameras_.size());
        observations_.observations_.resize(getNumSteps());
        for (int t=0; t<getNumSteps(); ++t)
            observations_.observations_[t].resize(observations_.numCameras_);

        cuMat::SimpleRandom rnd;

        //computation
        for (int t=0; t<getNumSteps(); ++t)
        {
            worker->setStatus(tinyformat::format("Cost Function: simulate observations for timestep %d with resolution %d", t + 1, observations_.resolution_));
            preprocessSetSdfGPU(results_->states_[t].advectedSDF_);
            for (int i = 0; i < observations_.numCameras_; ++i) {
				if (observations_.gpuPreprocess_)
					observations_.observations_[t][i] = preprocessCreateObservationGPU(results_, t, observations_.cameras_[i], observations_.resolution_, rnd, observations_.noise_);
				else
					observations_.observations_[t][i] = preprocessCreateObservationCPU(results_, t, observations_.cameras_[i], observations_.resolution_, observations_.noise_);

				//cinder::app::console() << "Observation GPU: \n" << observations_.observations_[t][i] << std::endl;
                //observations_.observations_[t][i] = preprocessCreateObservationCPU(results_, t, observations_.cameras_[i], observations_.resolution_, observations_.noise_);
				//cinder::app::console() << "Observation CPU: \n" << observations_.observations_[t][i] << std::endl;

                //Debug
//#ifndef _NDEBUG
                //cinder::app::console() << "Timestep " << (t + 1) << ", observation " << (i + 1) << ":\n" << observations_.observations_[t][i].toEigen() << std::endl;
//#endif
            }
            if (worker->isInterrupted()) break;
        }
        preprocessSetSdfGPU(nullptr); //cleanup
    }

	void CostFunctionPartialObservations::exportObservations(const std::string & path, int everyNframe, BackgroundWorker2 * worker)
	{
		for (int t = 0; t < getNumSteps(); t += everyNframe)
		{
			int tt = t / everyNframe;

			//open file
			std::string fileName = tinyformat::format("../screenshots/%s%05d.csv", path.c_str(), tt);
			std::ofstream out(fileName);
			for (int i = 0; i < observations_.numCameras_; ++i) {

				const auto& obs = observations_.observations_[t][i];
				const auto& cam = observations_.cameras_[i];
				int res = observations_.resolution_;
				Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> imageHost = obs.toEigen();

				for (int x = 0; x < res; ++x) for (int y = 0; y < res; ++y)
				{
					float depth = imageHost(x, y);
					if (depth <= 0) continue;
					glm::vec3 screenPos(x / float(res), y / float(res), depth);
					glm::vec3 worldPosGlm = cam.getWorldCoordinates(screenPos);
					out << worldPosGlm.x << ", " << worldPosGlm.y << ", " << worldPosGlm.z << ", " << i << "\n";
				}

				worker->setStatus(tinyformat::format("Export Observation %d:%d", tt, i));
				if (worker->isInterrupted()) break;
			}
			out.close();

			if (worker->isInterrupted()) break;
		}
	}

    Eigen::MatrixXf CostFunctionPartialObservations::getDepthImage(int frame, int camera)
    {
		if (frame < 0 || frame >= getNumSteps()) throw std::out_of_range("illegal frame index");
		if (camera < 0 || camera >= observations_.numCameras_) throw std::out_of_range("illegal camera index");
		const auto& obs = observations_.observations_[frame][camera];
		Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> imageHost = obs.toEigen();
		return imageHost.cast<float>();
    }

	std::vector<glm::vec3> CostFunctionPartialObservations::getObservations(int frame, int camera)
    {
		Eigen::MatrixXf image = getDepthImage(frame, camera);
		const auto& cam = observations_.cameras_[camera];
		std::vector<glm::vec3> points;
		int res = observations_.resolution_;
		for (int x = 0; x < res; ++x) for (int y = 0; y < res; ++y)
		{
			float depth = image(x, y);
			if (depth <= 0) continue;
			glm::vec3 screenPos(x / float(res), y / float(res), depth);
			glm::vec3 worldPosGlm = cam.getWorldCoordinates(screenPos);
			points.push_back(worldPosGlm);
		}
		return points;
    }
}
