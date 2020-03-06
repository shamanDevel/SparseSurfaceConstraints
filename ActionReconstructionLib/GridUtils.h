#pragma once

#include <Eigen/Core>
#include <cinder/Log.h>
#include <algorithm>

#include "Commons.h"
#include "Utils.h"

namespace ar
{
    struct GridUtils2D
    {
        typedef Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic> grid_t;
        typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> igrid_t;
        typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> bgrid_t;
        typedef Eigen::Matrix<real, Eigen::Dynamic, 1> linear_t;
        typedef Eigen::Matrix<real, 2, 1> grad_t;
        typedef Eigen::Matrix<real, 2, 1> vec2_t;

        static Eigen::Map<const linear_t> linearize(const grid_t& grid) {
            return Eigen::Map<const linear_t>(grid.data(), grid.rows()*grid.cols());
        }
        static Eigen::Map<const grid_t> delinearize(const linear_t& linear, int width, int height) {
            return Eigen::Map<const grid_t>(linear.data(), width, height);
        }

		static const int getLinearCoords(int x, int y, int width) { return x + width * y; }

        static const real& getLinear(const real* linear, int x, int y, int width) {
            return linear[x + width*y];
        }
        static real& atLinear(real* linear, int x, int y, int width) {
            return linear[x + width*y];
        }

        static void clampCoord(int& x, int& y, int width, int height) {
            x = std::max(0, std::min(width - 1, x));
            y = std::max(0, std::min(height - 1, y));
        }

        static const real& getLinearClamped(const real* linear, int x, int y, int width, int height) {
            clampCoord(x, y, width, height);
            return linear[x + width*y];
        }
        static real& atLinearClamped(real* linear, int x, int y, int width, int height) {
            clampCoord(x, y, width, height);
            return linear[x + width*y];
        }

        static real getLinearClampedF(const real* linear, real x, real y, int width, int height);

        static const real& getClamped(const grid_t& grid, int x, int y);

        static real& atClamped(grid_t& grid, int x, int y);

        static real getClampedF(const grid_t& grid, real x, real y);

        static grad_t getGradientLinear(const real* linear, int x, int y, int width, int height);

		static grad_t getGradient(const grid_t& grid, Eigen::Index i, Eigen::Index j);

        //The positive gradient for the Upwind scheme
        //Eq. (4.11) from NUMERICAL RECOVERY OF THE SIGNED DISTANCE FUNCTION, Tomas Oberhuber
        static real getGradientPosLinear(const real* linear, int x, int y, int width, int height);

        //The negative gradient for the Upwind scheme
        //Eq. (4.12) from NUMERICAL RECOVERY OF THE SIGNED DISTANCE FUNCTION, Tomas Oberhuber
        static real getGradientNegLinear(const real* linear, int x, int y, int width, int height);

        static grad_t getGradientLinearF(const real* linear, real x, real y, int width, int height);

        static real getLaplacianLinear(const real* linear, int x, int y, int width, int height);

		static real getLaplacian(const grid_t& grid, Eigen::Index i, Eigen::Index j);

	    /**
		 * \brief Computes the gradient for each grid cell
		 * \param v the field
		 * \return the gradient <dx, dy>
		 */
		static std::pair<grid_t, grid_t> getFullGradient(const grid_t& v);

	    /**
		 * \brief Computes the laplacian for each grid cell
		 * \param v the field
		 * \return the laplacian per element
		 */
		static grid_t getFullLaplacian(const grid_t& v);

        /**
         * \brief Heuristically Diffuses the values into invalid cells.
         * It performs a simple sweeping algorithm that fills every invalid cells with the average of the valid neighboring cells
         * \param input
         * \param valid 
         * \return 
         */
        static grid_t fillGrid(const grid_t& input, const bgrid_t& valid);

        /**
        * \brief Diffuses the values into invalid cells.
        * It uses a full diffusion process
        * \param input
        * \param valid
        * \return
        */
        static grid_t fillGridDiffusion(const grid_t& input, const bgrid_t& valid);

        /**
         * \brief Computes the adjoint of the grid diffusion
         * \param adjInput Output: the adjoint of the input values
         * \param adjOutput Input: the adjoint of the output values
         * \param input the input values from the forward simulation
         * \param valid the valid cells from the forward simulation
         */
        static void fillGridDiffusionAdjoint(grid_t& adjInput,
            const grid_t& adjOutput, const grid_t& input, const bgrid_t& valid);

        //Inverts the displacement map u=(inputX,inputY) and places it into the output map uInv=(outputX,outputY)
        //Input properties: x_new = x + u(x)
        //Output properties: x = x_new - uInv(x_new)
        //Input h: the size of each cell
        //This method uses a simple direct method of interpolating scattered points
        static void invertDisplacementDirectShepard(const grid_t& inputX, const grid_t& inputY, grid_t& outputX,
                                                    grid_t& outputY, real h);

        //Computes the adjoint of the displacement map inversion using Shepard interpolation.
        //Input: adjoint of the output displacement
        //Input: input to the forward shepard algorithm
        //Input: Valid cells and cell size
        //Output: adjoint of the input displacement
        // the output has to be properly resized and might already contain some data (new gradient is added)
        static void invertDisplacementDirectShepardAdjoint(
            grid_t& adjInputX, grid_t& adjInputY,
            const grid_t& adjOutputX, const grid_t& adjOutputY,
            const grid_t& inputX, const grid_t& inputY, const grid_t& outputX, const grid_t& outputY,
            real h
        );

        //Inverts the displacement map u=(inputX,inputY) and places it into the output map uInv=(outputX,outputY)
        //Input properties: x_new = x + u(x)
        //Input h: the size of each cell
        //Output properties: x = x_new - uInv(x_new)
        static void invertDisplacement(const grid_t& inputX, const grid_t& inputY, grid_t& outputX, grid_t& outputY, real h);

        //Advects the grid using Semi-Lagrange advection
        //For each x in Omega: output(x) = input(x + step*[dispX(x), dispY(x)])
        static grid_t advectGridSemiLagrange(const grid_t& input, const grid_t& dispX, const grid_t& dispY, real step);

        //Computes the adjoint of the Semi-Lagrange advection 
        //(the transposed, inversed derivative of the output with respect to the input variables input, dispX, dispY)
        //Input: constant step size
        //Input: adjoint of the output
        //Input: computed displacement and input from the forward pass
        //Output: adjoint of the input and x+y displacement
        // the output has to be properly resized and might already contain some data (new gradient is added)
        static void advectGridSemiLagrangeAdjoint(grid_t& adjInput, grid_t& adjDispX, grid_t& adjDispY, 
            real step, const grid_t& adjOutput, 
            const grid_t& dispX, const grid_t& dispY, const grid_t& input);

    private:
		static const real directForwardKernelRadiusIn;
		static const real directForwardKernelRadiusOut;
		static const real directForwardOuterKernelWeight;
		static const real directForwardOuterSdfThreshold;
		static const real directForwardOuterSdfWeight;

    public:
        /**
         * \brief Advects the grid using a direct forward interpolation.
         * The values of the current cell are blured into the target cells using a gaussian kernel of the specified radius.
         * For each x in Omega: output(x - step*[dispX(x), dispY(x)]) = input(x)
         * \param input the input values
         * \param dispX x displacements
         * \param dispY y displacements
         * \param step the (negative) step size
         * \param outputWeights Optional output: the weight matrix is placed into this
         * \return the advected input
         */
        static grid_t advectGridDirectForward(const grid_t& input, const grid_t& dispX, const grid_t& dispY, 
			real step, grid_t* outputWeights = nullptr);

        //Computes the adjoint of the advection using a direct forward interpolation. 
        //(the transposed, inversed derivative of the output with respect to the input variables input, dispX, dispY)
        //Input: constant step size
        //Input: adjoint of the output
        //Input: computed displacement and input from the forward pass
		//Input: the weights from the forward pass
        //Output: adjoint of the input and x+y displacement
        // the output has to be properly resized and might already contain some data (new gradient is added)
        static void advectGridDirectForwardAdjoint(grid_t& adjInput, grid_t& adjDispX, grid_t& adjDispY,
            const grid_t& adjOutput, const grid_t& dispX, const grid_t& dispY, const grid_t& input, const grid_t& output,
            real step, const grid_t& outputWeights);

	    /**
		 * \brief Computes the product y'F with
		 *   y being the adjoint variable of the input field (adjInput)
		 *   and F being the negative of the derivative of the advection with respect to the input field.
		 *  The matrix F would be of size (resolution^2 x resolution^2) but very sparse,
		 *  this function performs the multiplication matrix-free.
		 * \param adjInput the adjoint input field (called y or lambda)
		 * \param dispX the x displacements from the forward pass
		 * \param dispY the y displacements from the forward pass
		 * \param step the step size
		 * \param weights the accumulated weights from the forward pass
		 * \return the output of the multiplication
		 */
		static grid_t advectGridDirectForwardAdjOpMult(const grid_t& adjInput,
			const grid_t& dispX, const grid_t& dispY, real step, const grid_t& weights);

        //Recover signed distance function using the viscosity solution. Unstable!
        static grid_t recoverSDFViscosity(const grid_t& input, real viscosity = real(0.1), int iterations = 50);

        //Recover signed distance function using an upwind scheme. Unstable!
        static grid_t recoverSDFUpwind(const grid_t& input, int iterations = 50);

	    /**
		 * \brief Storage for the adjoint of \ref recoverSDFSussmann
		 */
		struct RecoverSDFSussmannAdjointStorage
		{
			real epsilon = 0;
			int iterations = 0;
            real stepsize = 0;
            bgrid_t canChange;
            std::vector<grid_t> inputs;
		};

		//Reconver the signed distance function using the method presented in
		//"A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow", Sussman et al. 1994
		static grid_t recoverSDFSussmann(const grid_t& input, 
			real epsilon = real(0.01), int iterations = 50, 
			RecoverSDFSussmannAdjointStorage* adjointStorage = nullptr);

		static void recoverSDFSussmannAdjoint(
			const grid_t& input, const grid_t& output,
			const grid_t& adjOutput, grid_t& adjInput,
			const RecoverSDFSussmannAdjointStorage& adjointStorage);

	    /**
		 * \brief Computes the gradient of the SDF reconstruction using the discretization described in 
		 * "A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow", Sussman et al. 1994.
		 * This is the gradient that is used in \ref recoverSDFSussmann(const grid_t& input, real epsilon, int iterations)
		 * \param input the current signed distance function
		 * \param input0 the function that is used as reference for the sign computations
		 * \param canChange true iff the value is allowed to change at that position
		 * \param cost the value of the cost that is minimized by this gradient
		 * \param epsilon smoothing epsilon factor
		 * \return the gradient
		 */
		static grid_t recoverSDFSussmannGradient(const grid_t& input, const grid_t& input0, const bgrid_t& canChange, real& cost, real epsilon = real(0.01));
        
        static void recoverSDFSussmannGradientAdjoint(const grid_t& input, const grid_t& input0, const bgrid_t& canChange, real epsilon, const grid_t& adjOutput, grid_t& adjInput);

        /**
         * \brief Recovers the SDF using the Fast Marching method
         * as described in "The Fast Construction of Extension Velocities in Level Set Methods",
         * Adalsteinsson, Sethian (1998)
         * \param input 
         * \return 
         */
        static grid_t recoverSDFFastMarching(const grid_t& input);
    
    };

}