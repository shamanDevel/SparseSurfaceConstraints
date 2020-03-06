#pragma once

#include <functional>
#include <cassert>

namespace ar
{
	/**
	 * \brief Computes the gradient descent x_{n+1} = x_{n} - t * \nabla F(x_{n}).
	 * It uses the Resilient Back Propagation method to scale each dimension seperately.
	 *
	 * See "Lecture Notes: Some notes on gradient descent" by Marc Toussaint; May 3, 2012
	 * AND
	 * Riedmiller, M., & Braun, H. (1993). A direct adaptive method for faster backpropagation learning: The RPROP algorithm. In Neural Networks, 1993., IEEE International Conference on (pp. 586-591). IEEE.
	 * 
	 * For interactivity, each iteration has to be triggered individually.
	 *
	 * \tparam X the type of x, must be an Eigen vector
	 */
	template <class X>
	class RpropGradientDescent
	{
	public:
		using X_t = X;
		using GF = std::function<X(const X&)>;

	private:
		//x[k%2] = current solution
		//x[1-k%2] = old / new solution
		const int n;
		X x[2];
		GF gf;
		X gPrevious;
		X gCurrent;
		X stepsize;

		//iteration counter
		int k;

		//stopping criterion, if the norm of the gradient falls below that
		double epsilon;
		X minStepsize, maxStepsize;
		X minValues, maxValues;
		bool hasMaxStepsize = false, hasMinStepsize = false;
		bool hasMinValues = false, hasMaxValues = false;

	public:
		RpropGradientDescent(const Eigen::MatrixBase<X>& initialX, const GF& gf, double epsilon = 0.0001, double initialStepsize = 0.1)
			: n(initialX.size()), x{ initialX.derived(), X() }, gf(gf), k(0), epsilon(epsilon)
		{
			assert(n > 0);
			stepsize.resize(n);
			stepsize.setConstant(initialStepsize);
			gPrevious.resize(n);
			gPrevious.setConstant(0.0);
		};


		double getEpsilon() const
		{
			return epsilon;
		}

		void setEpsilon(double epsilon)
		{
			this->epsilon = epsilon;
		}

		X getStepsize() const
		{
			return stepsize;
		}

		void setStepsize(const Eigen::MatrixBase<X>& stepsize)
		{
			assert(stepsize.size() == n);
			this->stepsize = stepsize;
		}

		void setInitialStepsize(double stepsize)
		{
			this->stepsize.setConstant(stepsize);
		}

		const X& getMaxStepsize() const
		{
			return maxStepsize;
		}

		void setMaxStepsize(const Eigen::MatrixBase<X>& max_stepsize)
		{
			assert(max_stepsize.size() == n);
			hasMaxStepsize = true;
			maxStepsize = max_stepsize;
		}

		const X& getMinStepsize() const
		{
			return minStepsize;
		}

		void setMinStepsize(const Eigen::MatrixBase<X>& min_stepsize)
		{
			assert(min_stepsize.size() == n);
			hasMinStepsize = true;
			minStepsize = min_stepsize;
		}

		const X& getMinValues() const
		{
			return minValues;
		}

		void setMinValues(const X& min_values)
		{
			assert(min_values.size() == n);
			hasMinValues = true;
			minValues = min_values;
		}

		const X& getMaxValues() const
		{
			return maxValues;
		}

		void setMaxValues(const X& max_values)
		{
			assert(max_values.size() == n);
			hasMaxValues = true;
			maxValues = max_values;
		}

	private:

		void checkBounds(X& x, const X& prev)
		{
			if (hasMinValues) {
				//x = x.cwiseMax(minValues);
				for (size_t i=0; i<x.size(); ++i)
				{ //if lower bound is hit, set value to half the distance to the bound
					x[i] = x[i] >= minValues[i] ? x[i] : 0.5f * (prev[i] + minValues[i]);
				}
			}
			if (hasMaxValues) {
				//x = x.cwiseMin(maxValues);
				for (size_t i = 0; i < x.size(); ++i)
				{ //if upper bound is hit, set value to half the distance to the bound
					x[i] = x[i] <= maxValues[i] ? x[i] : 0.5f * (prev[i] + maxValues[i]);
				}
			}
		}

		void capStepsize(X& s)
		{
			if (hasMinStepsize) s = s.cwiseMax(minStepsize);
			if (hasMaxStepsize) s = s.cwiseMin(maxStepsize);
		}

	public:
		/**
		 * Performs a step of the gradient descent.
		 * \return true if the stopping condition |\nabla F| < epsilon is fullfilled
		 */
		bool step() {
			//compute gradient
			gCurrent = gf(x[k % 2]);
			X gCurrentCopy = gCurrent;
			assert(gCurrent.size() == n);
			if (gCurrentCopy.squaredNorm() <= epsilon * epsilon) {
				x[1 - (k % 2)] = x[k % 2];
				return true; //stopping
			}
			X gSign = gCurrentCopy.array().sign().matrix();
			//compute stepsize component-wise
			for (int i = 0; i < n; ++i) {
				if (gCurrentCopy[i] * gPrevious[i] > 0) {
					stepsize[i] *= 1.2; //same direction as last time
				}
				else if (gCurrentCopy[i] * gPrevious[i] < 0) {
					stepsize[i] *= 0.5; //change of direction
					gCurrentCopy[i] = 0;
				}
			}
			capStepsize(stepsize);
			//do the step
			x[1 - (k % 2)] = x[k % 2] - stepsize.cwiseProduct(gSign);
			checkBounds(x[1 - (k % 2)], x[k % 2]);
			gPrevious = gCurrentCopy;
			k++;
			return false;
		}

		//Returns the current solution
		const X& getCurrentSolution() const {
			return x[k % 2];
		}

		//Returns the gradient from the last evaluation
		const X& getCurrentGradient() const {
			return gCurrent;
		}

		const X& getLastStepSize() const {
			return stepsize;
		}

		//Returns true iff an error occured: e.g. negative stepsize due to the Barzillai Borwein Step
		bool hasError() const { return error; }
	};
}