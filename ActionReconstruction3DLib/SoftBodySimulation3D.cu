#include "SoftBodySimulation3D.h"
#include <cinder/app/AppBase.h>

#include "WorldGrid.h"

#include <windows.h>
#include "CommonKernels.h"

namespace ar3d {

	SoftBodySimulation3D::Settings::Settings()
		: gravity_(make_real3(0, -10, 0))
		, youngsModulus_(2000)
		, poissonsRatio_(0.4)
		, mass_(1)
		, dampingAlpha_(0.1)
		, dampingBeta_(0.01)
		, enableCorotation_(true)
		, timestep_(0.01)
		, initialLinearVelocity_(make_real3(0,0,0))
		, initialAngularVelocity_(make_real3(0,0,0))
		, groundPlane_(make_real4(0, 1, 0, 0))
		, enableCollision_(true)
		, groundStiffness_(10000)
		, softmaxAlpha_(100)
		, stableCollision_(true)
		, solverIterations_(100)
		, solverTolerance_(1e-5)
		, newmarkTheta_(CommonKernels::NewmarkTheta)
	{
		validate();
	}

void SoftBodySimulation3D::Settings::validate()
{
	assert(youngsModulus_ > 0);
	assert(poissonsRatio_ > 0 && poissonsRatio_ < 0.5);
	assert(mass_ > 0);
	assert(dampingAlpha_ >= 0);
	assert(dampingBeta_ >= 0);
	assert(timestep_ > 0);
	assert(solverIterations_ > 0);
	assert(solverTolerance_ >= 0);

	computeMaterialParameters(youngsModulus_, poissonsRatio_, materialMu_, materialLambda_);
}

    void SoftBodySimulation3D::SettingsGui::initParams(cinder::params::InterfaceGlRef params, const std::string& group)
    {
        params->addParam("Elasticity-Gravity", &settings_.gravity_.y).group(group).label("Gravity").step(0.01);
        params->addParam("Elasticity-YoungsModulus", &settings_.youngsModulus_).group(group).label("Young's Modulus").min(0.1).step(0.01);
        params->addParam("Elasticity-PoissonRatio", &settings_.poissonsRatio_).group(group).label("Poisson's Ratio").min(0.01).max(0.499).step(0.001);
        params->addParam("Elasticity-Mass", &settings_.mass_).group(group).label("Mass").min(0.1).step(0.01);
        params->addParam("Elasticity-DampingMass", &settings_.dampingAlpha_).group(group).label("Damping Mass").min(0).step(0.001);
        params->addParam("Elasticity-DampingStiffness", &settings_.dampingBeta_).group(group).label("Damping Stiffness").min(0).step(0.001);
        params->addParam("Elasticity-Corotation", &settings_.enableCorotation_).group(group).label("Corotation");
        params->addParam("Elasticity-Collision", &settings_.enableCollision_).group(group).label("Collision");
		params->addParam("Elasticity-InitialLinearVelocity", reinterpret_cast<glm::tvec3<real, glm::highp>*>(&settings_.initialLinearVelocity_)).group(group).label("Initial Linear Velocity");
		params->addParam("Elasticity-InitialAngularVelocity", reinterpret_cast<glm::tvec3<real, glm::highp>*>(&settings_.initialAngularVelocity_)).group(group).label("Initial Angular Velocity");
		params->addParam("Elasticity-GroundOrientation", reinterpret_cast<glm::tvec3<real, glm::highp>*>(&settings_.groundPlane_.x)).group(group).label("Ground Orientation").precision(5);
        params->addParam("Elasticity-GroundHeight", &settings_.groundPlane_.w).group(group).label("Ground Height").step(0.00001);
        params->addParam("Elasticity-GroundStiffness", &settings_.groundStiffness_).group(group).label("Ground Stiffness").min(1).step(1);
        params->addParam("Elasticity-Softmax", &settings_.softmaxAlpha_).group(group).label("Collision Softmax").min(1).step(1);
		params->addParam("Elasticity-StableCollision", &settings_.stableCollision_).group(group).label("Stable Collision");
        params->addParam("Elasticity-SolverIterations", &settings_.solverIterations_).group(group).label("Solver Iterations").min(1);
        params->addParam("Elasticity-SolverTolerance", &settings_.solverTolerance_).group(group).label("Solver Tolerance").min(0).step(1e-6);
        params->addParam("Elasticity-Timestep", &settings_.timestep_).group(group).label("Timestep").min(0.00001).step(0.00001);
		params->addParam("Elasticity-NewmarkTheta", &settings_.newmarkTheta_).group(group).label("Newmark Theta").min(0.5).max(1.0).step(0.001);
        params->addParam("Elasticity-DebugSaveMatrices", &settings_.debugSaveMatrices_).group(group).label("Debug: Save Matrices");
    }

    void SoftBodySimulation3D::SettingsGui::setVisible(cinder::params::InterfaceGlRef params, bool visible)
    {
        std::string s = visible ? "visible=true" : "visible=false";
        params->setOptions("Elasticity-Gravity", s);
        params->setOptions("Elasticity-YoungsModulus", s);
        params->setOptions("Elasticity-PoissonRatio", s);
        params->setOptions("Elasticity-Mass", s);
        params->setOptions("Elasticity-DampingMass", s);
        params->setOptions("Elasticity-DampingStiffness", s);
        params->setOptions("Elasticity-Corotation", s);
        params->setOptions("Elasticity-Collision", s);
        params->setOptions("Elasticity-GroundOrientation", s);
        params->setOptions("Elasticity-GroundHeight", s);
        params->setOptions("Elasticity-GroundStiffness", s);
        params->setOptions("Elasticity-Softmax", s);
		params->setOptions("Elasticity-StableCollision", s);
        params->setOptions("Elasticity-SolverIterations", s);
        params->setOptions("Elasticity-SolverTolerance", s);
        params->setOptions("Elasticity-Timestep", s);
		params->setOptions("Elasticity-NewmarkTheta", s);
        params->setOptions("Elasticity-DebugSaveMatrices", s);
    }

    void SoftBodySimulation3D::SettingsGui::load(const cinder::JsonTree& parent)
    {
        settings_.gravity_.x = parent.getChild("Gravity").getValueAtIndex<real>(0);
        settings_.gravity_.y = parent.getChild("Gravity").getValueAtIndex<real>(1);
        settings_.gravity_.z = parent.getChild("Gravity").getValueAtIndex<real>(2);
        settings_.youngsModulus_ = parent.getValueForKey<real>("YoungsModulus");
        settings_.poissonsRatio_ = parent.getValueForKey<real>("PoissonRatio");
        settings_.mass_ = parent.getValueForKey<real>("Mass");
        settings_.dampingAlpha_ = parent.getValueForKey<real>("DampingMass");
        settings_.dampingBeta_ = parent.getValueForKey<real>("DampingStiffness");
        settings_.enableCorotation_ = parent.getValueForKey<bool>("Corotation");
        settings_.enableCollision_ = parent.getValueForKey<bool>("Collision");
		if (parent.hasChild("InitialLinearVelocity"))
		{
			settings_.initialLinearVelocity_.x = parent.getChild("InitialLinearVelocity").getValueAtIndex<real>(0);
			settings_.initialLinearVelocity_.y = parent.getChild("InitialLinearVelocity").getValueAtIndex<real>(1);
			settings_.initialLinearVelocity_.z = parent.getChild("InitialLinearVelocity").getValueAtIndex<real>(2);
		}
		if (parent.hasChild("InitialAngularVelocity"))
		{
			settings_.initialAngularVelocity_.x = parent.getChild("InitialAngularVelocity").getValueAtIndex<real>(0);
			settings_.initialAngularVelocity_.y = parent.getChild("InitialAngularVelocity").getValueAtIndex<real>(1);
			settings_.initialAngularVelocity_.z = parent.getChild("InitialAngularVelocity").getValueAtIndex<real>(2);
		}
        settings_.groundPlane_.x = parent.getChild("GroundOrientation").getValueAtIndex<real>(0);
        settings_.groundPlane_.y = parent.getChild("GroundOrientation").getValueAtIndex<real>(1);
        settings_.groundPlane_.z = parent.getChild("GroundOrientation").getValueAtIndex<real>(2);
        settings_.groundPlane_.w = parent.getValueForKey<real>("GroundHeight");
        settings_.groundStiffness_ = parent.getValueForKey<real>("GroundStiffness");
        settings_.softmaxAlpha_ = parent.getValueForKey<real>("Softmax");
		if (parent.hasChild("StableCollision"))
			settings_.stableCollision_ = parent.getValueForKey<bool>("StableCollision");
		else
			settings_.stableCollision_ = true;
        settings_.solverIterations_ = parent.getValueForKey<int>("SolverIterations");
        settings_.solverTolerance_ = parent.getValueForKey<real>("SolverTolerance");
        settings_.timestep_ = parent.getValueForKey<real>("Timestep");
        if (parent.hasChild("DebugSaveMatrices")) settings_.debugSaveMatrices_ = parent.getValueForKey<bool>("DebugSaveMatrices");
		if (parent.hasChild("NewmarkTheta"))
			settings_.newmarkTheta_ = parent.getValueForKey<real>("NewmarkTheta");
    }

    void SoftBodySimulation3D::SettingsGui::save(cinder::JsonTree& parent) const
    {
        parent.addChild(cinder::JsonTree::makeArray("Gravity")
            .addChild(cinder::JsonTree("", settings_.gravity_.x))
            .addChild(cinder::JsonTree("", settings_.gravity_.y))
            .addChild(cinder::JsonTree("", settings_.gravity_.z)));
        parent.addChild(cinder::JsonTree("YoungsModulus", settings_.youngsModulus_));
        parent.addChild(cinder::JsonTree("PoissonRatio", settings_.poissonsRatio_));
        parent.addChild(cinder::JsonTree("Mass", settings_.mass_));
        parent.addChild(cinder::JsonTree("DampingMass", settings_.dampingAlpha_));
        parent.addChild(cinder::JsonTree("DampingStiffness", settings_.dampingBeta_));
        parent.addChild(cinder::JsonTree("Corotation", settings_.enableCorotation_));
		parent.addChild(cinder::JsonTree::makeArray("InitialLinearVelocity")
			.addChild(cinder::JsonTree("", settings_.initialLinearVelocity_.x))
			.addChild(cinder::JsonTree("", settings_.initialLinearVelocity_.y))
			.addChild(cinder::JsonTree("", settings_.initialLinearVelocity_.z)));
		parent.addChild(cinder::JsonTree::makeArray("InitialAngularVelocity")
			.addChild(cinder::JsonTree("", settings_.initialAngularVelocity_.x))
			.addChild(cinder::JsonTree("", settings_.initialAngularVelocity_.y))
			.addChild(cinder::JsonTree("", settings_.initialAngularVelocity_.z)));
        parent.addChild(cinder::JsonTree("Collision", settings_.enableCollision_));
        parent.addChild(cinder::JsonTree::makeArray("GroundOrientation")
            .addChild(cinder::JsonTree("", settings_.groundPlane_.x))
            .addChild(cinder::JsonTree("", settings_.groundPlane_.y))
            .addChild(cinder::JsonTree("", settings_.groundPlane_.z)));
        parent.addChild(cinder::JsonTree("GroundHeight", settings_.groundPlane_.w));
        parent.addChild(cinder::JsonTree("GroundStiffness", settings_.groundStiffness_));
        parent.addChild(cinder::JsonTree("Softmax", settings_.softmaxAlpha_));
		parent.addChild(cinder::JsonTree("StableCollision", settings_.stableCollision_));
        parent.addChild(cinder::JsonTree("SolverIterations", settings_.solverIterations_));
        parent.addChild(cinder::JsonTree("SolverTolerance", settings_.solverTolerance_));
        parent.addChild(cinder::JsonTree("Timestep", settings_.timestep_));
        parent.addChild(cinder::JsonTree("DebugSaveMatrices", settings_.debugSaveMatrices_));
		parent.addChild(cinder::JsonTree("NewmarkTheta", settings_.newmarkTheta_));
    }

    void SoftBodySimulation3D::computeMaterialParameters(real youngsModulus, real poissonsRatio, real& materialMu,
    real& materialLambda)
{
    materialMu = youngsModulus / (2 * (1 + poissonsRatio));
    materialLambda = (youngsModulus * poissonsRatio) / ((1 + poissonsRatio) * (1 - 2 * poissonsRatio));
}

    SoftBodySimulation3D::IInputSettings::IInputSettings(): enableDirichlet(false), centerDirichlet(),
                                                            halfsizeDirichlet(), sampleIntegrals(false),
                                                            diffusionDistance(10), zeroCutoff(0.01)
    {
    }

    void SoftBodySimulation3D::IInputSettings::initParams(IInputSettings* settings, 
        cinder::params::InterfaceGlRef params, const std::string& group, const std::string& prefix)
    {
        params->addParam("Input" + prefix+"-Dirichlet", &settings->enableDirichlet).group(group).label("Enable Dirichlet");
        params->addParam("Input" + prefix+"-Dirichlet-CenterX", &settings->centerDirichlet.x).group(group).label("Dirichlet Center X").step(0.01);
        params->addParam("Input" + prefix+"-Dirichlet-CenterY", &settings->centerDirichlet.y).group(group).label("Dirichlet Center Y").step(0.01);
        params->addParam("Input" + prefix+"-Dirichlet-CenterZ", &settings->centerDirichlet.z).group(group).label("Dirichlet Center Z").step(0.01);
        params->addParam("Input" + prefix+"-Dirichlet-SizeX", &settings->halfsizeDirichlet.x).group(group).label("Dirichlet Halfsize X").step(0.01).min(0.01);
        params->addParam("Input" + prefix+"-Dirichlet-SizeY", &settings->halfsizeDirichlet.y).group(group).label("Dirichlet Halfsize Y").step(0.01).min(0.01);
        params->addParam("Input" + prefix+"-Dirichlet-SizeZ", &settings->halfsizeDirichlet.z).group(group).label("Dirichlet Halfsize Z").step(0.01).min(0.01);
        params->addParam("Input" + prefix + "-IntegrationMode", &settings->sampleIntegrals).group(group).label("Cell Integration Mode").optionsStr("true=Sampled false=Analytic");
        params->addParam("Input" + prefix + "-DiffusionDistance", &settings->diffusionDistance).group(group).label("Diffusion Distance").min(0)
            .optionsStr("help='Maximal distance to the surface where diffusion happens. Set to zero to diffuse everything'");
        params->addParam("Input" + prefix + "-ZeroCutoff", &settings->zeroCutoff).group(group).label("Zero Cutoff").min(0).max(0.5).step(0.001)
            .optionsStr("help='Value at which levelset values are clamped to zero to improve the numerical stability'");
    }

    void SoftBodySimulation3D::IInputSettings::setVisible(IInputSettings* settings, 
        cinder::params::InterfaceGlRef params, const std::string& prefix, bool visible)
    {
        std::string o = visible ? "visible=true" : "visible=false";
        params->setOptions("Input" + prefix+"-Dirichlet", o);
        params->setOptions("Input" + prefix+"-Dirichlet-CenterX", o);
        params->setOptions("Input" + prefix+"-Dirichlet-CenterY", o);
        params->setOptions("Input" + prefix+"-Dirichlet-CenterZ", o);
        params->setOptions("Input" + prefix+"-Dirichlet-SizeX", o);
        params->setOptions("Input" + prefix+"-Dirichlet-SizeY", o);
        params->setOptions("Input" + prefix+"-Dirichlet-SizeZ", o);
        params->setOptions("Input" + prefix + "-IntegrationMode", o);
        params->setOptions("Input" + prefix + "-DiffusionDistance", o);
        params->setOptions("Input" + prefix + "-ZeroCutoff", o);
    }

    void SoftBodySimulation3D::IInputSettings::load(
        IInputSettings* settings, const cinder::JsonTree& parent)
    {
        if (parent.hasChild("EnableDirichlet")) settings->enableDirichlet = parent.getValueForKey<bool>("EnableDirichlet");
        if (parent.hasChild("CenterDirichlet")) settings->centerDirichlet.x = parent.getChild("CenterDirichlet").getValueAtIndex<real>(0);
        if (parent.hasChild("CenterDirichlet")) settings->centerDirichlet.y = parent.getChild("CenterDirichlet").getValueAtIndex<real>(1);
        if (parent.hasChild("CenterDirichlet")) settings->centerDirichlet.z = parent.getChild("CenterDirichlet").getValueAtIndex<real>(2);
        if (parent.hasChild("HalfsizeDirichlet")) settings->halfsizeDirichlet.x = parent.getChild("HalfsizeDirichlet").getValueAtIndex<real>(0);
        if (parent.hasChild("HalfsizeDirichlet")) settings->halfsizeDirichlet.y = parent.getChild("HalfsizeDirichlet").getValueAtIndex<real>(1);
        if (parent.hasChild("HalfsizeDirichlet")) settings->halfsizeDirichlet.z = parent.getChild("HalfsizeDirichlet").getValueAtIndex<real>(2);
        if (parent.hasChild("IntegrationMode")) settings->sampleIntegrals = parent.getValueForKey<bool>("IntegrationMode");
        if (parent.hasChild("DiffusionDistance")) settings->diffusionDistance = parent.getValueForKey<int>("DiffusionDistance");
        if (parent.hasChild("ZeroCutoff")) settings->zeroCutoff = parent.getValueForKey<real>("ZeroCutoff");
    }

    void SoftBodySimulation3D::IInputSettings::save(
        const IInputSettings* settings, cinder::JsonTree& parent)
    {
        parent.addChild(cinder::JsonTree("EnableDirichlet", settings->enableDirichlet));
        parent.addChild(cinder::JsonTree::makeArray("CenterDirichlet")
            .addChild(cinder::JsonTree("", settings->centerDirichlet.x))
            .addChild(cinder::JsonTree("", settings->centerDirichlet.y))
            .addChild(cinder::JsonTree("", settings->centerDirichlet.z)));
        parent.addChild(cinder::JsonTree::makeArray("HalfsizeDirichlet")
            .addChild(cinder::JsonTree("", settings->halfsizeDirichlet.x))
            .addChild(cinder::JsonTree("", settings->halfsizeDirichlet.y))
            .addChild(cinder::JsonTree("", settings->halfsizeDirichlet.z)));
        parent.addChild(cinder::JsonTree("IntegrationMode", settings->sampleIntegrals));
        parent.addChild(cinder::JsonTree("DiffusionDistance", settings->diffusionDistance));
        parent.addChild(cinder::JsonTree("ZeroCutoff", settings->zeroCutoff));
    }

    void SoftBodySimulation3D::InputBarSettingsGui::initParams(cinder::params::InterfaceGlRef params,
        const std::string& group)
    {
        params->addParam("InputBar-Resolution", &settings_.resolution).group(group).label("Bar Resolution").min(1);
        params->addParam("InputBar-CenterX", &settings_.center.x).group(group).label("Bar Center X").step(0.01);
        params->addParam("InputBar-CenterY", &settings_.center.y).group(group).label("Bar Center Y").step(0.01);
        params->addParam("InputBar-CenterZ", &settings_.center.z).group(group).label("Bar Center Z").step(0.01);
        params->addParam("InputBar-SizeX", &settings_.halfsize.x).group(group).label("Bar Halfsize X").step(0.01).min(0.01);
        params->addParam("InputBar-SizeY", &settings_.halfsize.y).group(group).label("Bar Halfsize Y").step(0.01).min(0.01);
        params->addParam("InputBar-SizeZ", &settings_.halfsize.z).group(group).label("Bar Halfsize Z").step(0.01).min(0.01);
        IInputSettings::initParams(&settings_, params, group, "Bar");
    }

    void SoftBodySimulation3D::InputBarSettingsGui::setVisible(cinder::params::InterfaceGlRef params, bool visible)
    {
        std::string barOption = visible ? "visible=true" : "visible=false";
        params->setOptions("InputBar-Resolution", barOption);
        params->setOptions("InputBar-CenterX", barOption);
        params->setOptions("InputBar-CenterY", barOption);
        params->setOptions("InputBar-CenterZ", barOption);
        params->setOptions("InputBar-SizeX", barOption);
        params->setOptions("InputBar-SizeY", barOption);
        params->setOptions("InputBar-SizeZ", barOption);
        IInputSettings::setVisible(&settings_, params, "Bar", visible);
    }

    void SoftBodySimulation3D::InputBarSettingsGui::load(const cinder::JsonTree& parent)
    {
        settings_.resolution = parent.getValueForKey<int>("Resolution");
        settings_.center.x = parent.getChild("Center").getValueAtIndex<real>(0);
        settings_.center.y = parent.getChild("Center").getValueAtIndex<real>(1);
        settings_.center.z = parent.getChild("Center").getValueAtIndex<real>(2);
        settings_.halfsize.x = parent.getChild("Halfsize").getValueAtIndex<real>(0);
        settings_.halfsize.y = parent.getChild("Halfsize").getValueAtIndex<real>(1);
        settings_.halfsize.z = parent.getChild("Halfsize").getValueAtIndex<real>(2);
        IInputSettings::load(&settings_, parent);
    }

    void SoftBodySimulation3D::InputBarSettingsGui::save(cinder::JsonTree& parent) const
    {
        parent.addChild(cinder::JsonTree("Resolution", settings_.resolution));
        parent.addChild(cinder::JsonTree::makeArray("Center")
            .addChild(cinder::JsonTree("", settings_.center.x))
            .addChild(cinder::JsonTree("", settings_.center.y))
            .addChild(cinder::JsonTree("", settings_.center.z)));
        parent.addChild(cinder::JsonTree::makeArray("Halfsize")
            .addChild(cinder::JsonTree("", settings_.halfsize.x))
            .addChild(cinder::JsonTree("", settings_.halfsize.y))
            .addChild(cinder::JsonTree("", settings_.halfsize.z)));
        IInputSettings::save(&settings_, parent);
    }

    void SoftBodySimulation3D::InputTorusSettingsGui::initParams(cinder::params::InterfaceGlRef params,
        const std::string& group)
    {
        params->addParam("InputTorus-Resolution", &settings_.resolution).group(group).label("Torus Resolution").min(1);
        params->addParam("InputTorus-CenterX", &settings_.center.x).group(group).label("Torus Center X").step(0.01);
        params->addParam("InputTorus-CenterY", &settings_.center.y).group(group).label("Torus Center Y").step(0.01);
        params->addParam("InputTorus-CenterZ", &settings_.center.z).group(group).label("Torus Center Z").step(0.01);
        params->addParam("InputTorus-Orientation", reinterpret_cast<glm::vec3*>(&settings_.orientation)).group(group).label("Orientation");
        params->addParam("InputTorus-InnerRadius", &settings_.innerRadius).group(group).label("Inner Radius").step(0.01).min(0);
        params->addParam("InputTorus-OuterRadius", &settings_.outerRadius).group(group).label("Outer Radius").step(0.01).min(0);
        IInputSettings::initParams(&settings_, params, group, "Torus");
    }

    void SoftBodySimulation3D::InputTorusSettingsGui::setVisible(cinder::params::InterfaceGlRef params, bool visible)
    {
        std::string torusOption = visible ? "visible=true" : "visible=false";
        params->setOptions("InputTorus-Resolution", torusOption);
        params->setOptions("InputTorus-CenterX", torusOption);
        params->setOptions("InputTorus-CenterY", torusOption);
        params->setOptions("InputTorus-CenterZ", torusOption);
        params->setOptions("InputTorus-Orientation", torusOption);
        params->setOptions("InputTorus-InnerRadius", torusOption);
        params->setOptions("InputTorus-OuterRadius", torusOption);
        IInputSettings::setVisible(&settings_, params, "Torus", visible);
    }

    void SoftBodySimulation3D::InputTorusSettingsGui::load(const cinder::JsonTree& parent)
    {
        settings_.resolution = parent.getValueForKey<int>("Resolution");
        settings_.center.x = parent.getChild("Center").getValueAtIndex<real>(0);
        settings_.center.y = parent.getChild("Center").getValueAtIndex<real>(1);
        settings_.center.z = parent.getChild("Center").getValueAtIndex<real>(2);
        settings_.orientation.x = parent.getChild("Orientation").getValueAtIndex<real>(0);
        settings_.orientation.y = parent.getChild("Orientation").getValueAtIndex<real>(1);
        settings_.orientation.z = parent.getChild("Orientation").getValueAtIndex<real>(2);
        settings_.innerRadius = parent.getValueForKey<real>("InnerRadius");
        settings_.outerRadius = parent.getValueForKey<real>("OuterRadius");
        IInputSettings::load(&settings_, parent);
    }

    void SoftBodySimulation3D::InputTorusSettingsGui::save(cinder::JsonTree& parent) const
    {
        parent.addChild(cinder::JsonTree("Resolution", settings_.resolution));
        parent.addChild(cinder::JsonTree("EnableDirichlet", settings_.enableDirichlet));
        parent.addChild(cinder::JsonTree::makeArray("CenterDirichlet")
            .addChild(cinder::JsonTree("", settings_.centerDirichlet.x))
            .addChild(cinder::JsonTree("", settings_.centerDirichlet.y))
            .addChild(cinder::JsonTree("", settings_.centerDirichlet.z)));
        parent.addChild(cinder::JsonTree::makeArray("HalfsizeDirichlet")
            .addChild(cinder::JsonTree("", settings_.halfsizeDirichlet.x))
            .addChild(cinder::JsonTree("", settings_.halfsizeDirichlet.y))
            .addChild(cinder::JsonTree("", settings_.halfsizeDirichlet.z)));
        parent.addChild(cinder::JsonTree::makeArray("Center")
            .addChild(cinder::JsonTree("", settings_.center.x))
            .addChild(cinder::JsonTree("", settings_.center.y))
            .addChild(cinder::JsonTree("", settings_.center.z)));
        parent.addChild(cinder::JsonTree::makeArray("Orientation")
            .addChild(cinder::JsonTree("", settings_.orientation.x))
            .addChild(cinder::JsonTree("", settings_.orientation.y))
            .addChild(cinder::JsonTree("", settings_.orientation.z)));
        parent.addChild(cinder::JsonTree("InnerRadius", settings_.innerRadius));
        parent.addChild(cinder::JsonTree("OuterRadius", settings_.outerRadius));
        IInputSettings::save(&settings_, parent);
    }

    bool SoftBodySimulation3D::InputSdfSettingsGui::isAvailable() const
    {
        return !settings_.file.empty();
    }

    bool SoftBodySimulation3D::InputSdfSettingsGui::hasChanged()
    {
        static int oldCounter = -1;
        static IInputSettings oldBasicSettings = {};
        static bool oldFilledCells = 2;
        if (counter_ != oldCounter)
        {
            oldCounter = counter_;
            return true;
        }
        if (settings_.filledCells != oldFilledCells)
        {
            oldFilledCells = settings_.filledCells;
            return true;
        }
        if (memcmp(this, &oldBasicSettings, sizeof(IInputSettings)) != 0)
        {
            oldBasicSettings = *reinterpret_cast<IInputSettings*>(this);
            return true;
        }
        return false;
    }

    void SoftBodySimulation3D::InputSdfSettingsGui::preload()
    {
        if (settings_.file.empty()) return;
        std::ifstream i(settings_.file, std::ios::in | std::ios::binary);
        if (!i)
        {
            CI_LOG_E("unable to open file " << settings_.file);
            
            //why does cinder not have a message box?
            MessageBox(NULL, ("Unable to open file\n" + settings_.file).c_str(), NULL, MB_OK | MB_ICONEXCLAMATION | MB_TASKMODAL);

            settings_.file = "";
            return;
        }
		WorldGridPtr ptr = WorldGrid::load(i);
		settings_.voxelResolution = ptr->getVoxelResolution();
		settings_.offset = glm::ivec3(ptr->getOffset().x(), ptr->getOffset().y(), ptr->getOffset().z());
		settings_.size = glm::ivec3(ptr->getSize().x(), ptr->getSize().y(), ptr->getSize().z());
        i.close();
        counter_++;
    }

    void SoftBodySimulation3D::InputSdfSettingsGui::initParams(cinder::params::InterfaceGlRef params,
        const std::string& group, bool fixedInput)
    {
		if (!fixedInput) {
			params->addParam("InputSdf-File", &settings_.file, true).group(group).label("File Name");
			params->addButton("InputSdf-Load", std::function<void()>([this]() {
				cinder::fs::path path = cinder::app::getOpenFilePath("", std::vector<std::string>({ "sdf" }));
				if (path.empty())
				{
					CI_LOG_I("loading cancelled");
					return;
				}
				settings_.file = path.string();
				preload();
			}), std::string("group=") + group + " label='Load'");
			params->addParam("InputSdf-VoxelResolution", &settings_.voxelResolution, true).group(group).label("Voxel Resolution");
			params->addParam("InputSdf-OffsetX", &settings_.offset.x, true).group(group).label("Offset X");
			params->addParam("InputSdf-OffsetY", &settings_.offset.y, true).group(group).label("Offset Y");
			params->addParam("InputSdf-OffsetZ", &settings_.offset.z, true).group(group).label("Offset Z");
			params->addParam("InputSdf-SizeX", &settings_.size.x, true).group(group).label("Size X");
			params->addParam("InputSdf-SizeY", &settings_.size.y, true).group(group).label("Size Y");
			params->addParam("InputSdf-SizeZ", &settings_.size.z, true).group(group).label("Size Z");
		}
        IInputSettings::initParams(&settings_, params, group, "Sdf");
        params->addParam("InputSdf-FilledCells", &settings_.filledCells).group(group).label("Filled cells")
            .optionsStr("help='Disable partially filled cells, approximate object by completely filled cells'");
    }

    void SoftBodySimulation3D::InputSdfSettingsGui::setVisible(cinder::params::InterfaceGlRef params, bool visible)
    {
        std::string o = visible ? "visible=true" : "visible=false";
        params->setOptions("InputSdf-File", o);
        params->setOptions("InputSdf-Load", o);
        params->setOptions("InputSdf-VoxelResolution", o);
        params->setOptions("InputSdf-OffsetX", o);
        params->setOptions("InputSdf-OffsetY", o);
        params->setOptions("InputSdf-OffsetZ", o);
        params->setOptions("InputSdf-SizeX", o);
        params->setOptions("InputSdf-SizeY", o);
        params->setOptions("InputSdf-SizeZ", o);
        IInputSettings::setVisible(&settings_, params, "Sdf", visible);
        params->setOptions("InputSdf-FilledCells", o);
    }

    void SoftBodySimulation3D::InputSdfSettingsGui::load(const cinder::JsonTree& parent, bool fixedInput)
    {
		if (!fixedInput) {
			settings_.file = parent.getValueForKey("File");
		}
        IInputSettings::load(&settings_, parent);
        if (parent.hasChild("FilledCells")) settings_.filledCells = parent.getValueForKey<bool>("FilledCells");
		if (!fixedInput) {
			preload();
		}
    }

    void SoftBodySimulation3D::InputSdfSettingsGui::save(cinder::JsonTree& parent, bool fixedInput) const
    {
		if (!fixedInput) {
			parent.addChild(cinder::JsonTree("File", settings_.file));
		}
        IInputSettings::save(&settings_, parent);
        parent.addChild(cinder::JsonTree("FilledCells", settings_.filledCells));
    }

    void SoftBodySimulation3D::resetTimings()
{
	statistics_.cgIterations.clear();
	statistics_.cgTime.clear();
	statistics_.matrixAssembleTime.clear();
	statistics_.collisionForcesTime.clear();
	statistics_.gridAdvectionTime.clear();
	statistics_.gridBoundingBoxTime.clear();
	statistics_.gridDiffusionTime.clear();
}
}
