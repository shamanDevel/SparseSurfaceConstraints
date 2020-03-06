#pragma once

#include <optional>
#include <functional>

#include <cinder/params/Params.h>

#include "SoftBody2DResults.h"
#include "BackgroundWorker.h"

namespace ar {

    struct InverseProblemOutput
    {
		real finalCost_ = 0;

        std::optional<real> youngsModulus_;
		std::optional<real> youngsModulusStdDev_;
        std::optional<real> poissonsRatio_;
		std::optional<real> poissonsRatioStdDev_;
        std::optional<real> mass_;
		std::optional<real> massStdDev_;
        std::optional<real> dampingAlpha_;
		std::optional<real> dampingAlphaStdDev_;
        std::optional<real> dampingBeta_;
		std::optional<real> dampingBetaStdDev_;

		std::optional<real> groundHeight;
		std::optional<real> groundAngle;
        
        std::optional<SoftBodyGrid2D::grid_t> initialGridSdf_;
        std::optional<SoftBodyMesh2D::Vector2List> initialMeshPositions_;

		std::optional<SoftBodyGrid2D::grid_t> resultGridSdf_;
		std::optional<SoftBodyMesh2D::Vector2List> resultMeshDisp_;
    };

    class IInverseProblem
    {
    protected:
        const SoftBody2DResults* input = nullptr;

    public:
        IInverseProblem() = default;;
        virtual ~IInverseProblem() = default;;

        void setInput(const SoftBody2DResults* results) { input = results; }

		typedef std::function<void(const InverseProblemOutput&)> IntermediateResultCallback_t;
	    inline static const IntermediateResultCallback_t dummyCallback = IntermediateResultCallback_t([](const InverseProblemOutput&) {});

		virtual InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) = 0;
		virtual InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) = 0;

		virtual void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) = 0;
		virtual void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const = 0;

		virtual void testPlot(int deformedTimestep, BackgroundWorker* worker) {}
    };

}