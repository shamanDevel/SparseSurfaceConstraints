#pragma once

#include <vector>

#include "IInverseProblem.h"
#include "GridUtils.h"

namespace ar {

    /**
     * \brief Solve for young's modulus using the Adjoint Method
     */
    class InverseProblem_Adjoint_YoungsModulus :
        public IInverseProblem
    {
    private:
        real initialYoungsModulus;
        real youngsModulusPrior;
        int numIterations;

    public:
        InverseProblem_Adjoint_YoungsModulus();
        virtual ~InverseProblem_Adjoint_YoungsModulus() = default;

        InverseProblemOutput solveGrid(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
        InverseProblemOutput solveMesh(int deformedTimestep, BackgroundWorker* worker, IntermediateResultCallback_t callback) override;
        void setupParams(cinder::params::InterfaceGlRef params, const std::string& group) override;
        void setParamsVisibility(cinder::params::InterfaceGlRef params, bool visible) const override;


	    real getInitialYoungsModulus() const
	    {
		    return initialYoungsModulus;
	    }

	    void setInitialYoungsModulus(real initial_youngs_modulus)
	    {
		    initialYoungsModulus = initial_youngs_modulus;
	    }

	    real getYoungsModulusPrior() const
	    {
		    return youngsModulusPrior;
	    }

	    void setYoungsModulusPrior(real youngs_modulus_prior)
	    {
		    youngsModulusPrior = youngs_modulus_prior;
	    }

	    int getNumIterations() const
	    {
		    return numIterations;
	    }

	    void setNumIterations(int num_iterations)
	    {
		    numIterations = num_iterations;
	    }

        struct DataPoint
        {
            real youngsModulus;
            real costGrid;
            real gradientGrid;
            real costMesh;
            real gradientMesh;
        };
        /**
         * \brief Test function: plots the energy graph with the gradients.
         * Nothing is plotted, only the data points are created
         * \param deformedTimestep 
         * \param worker 
         * \return 
         */
        std::vector<DataPoint> plotEnergy(int deformedTimestep, BackgroundWorker* worker) const;
		void testPlot(int deformedTimestep, BackgroundWorker* worker) override;

    public:
        real gradientMesh(int deformedTimestep, const SoftBodyMesh2D& simulation, real youngsModulus, VectorX& outputU, real& outputCost, BackgroundWorker* worker) const;
        real gradientGrid(int deformedTimestep, const SoftBodyGrid2D& simulation, real youngsModulus, GridUtils2D::grid_t& outputSdf, real& outputCost, BackgroundWorker* worker) const;
    };
}
