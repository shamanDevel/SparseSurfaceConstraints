#pragma once

#include <functional>
#include <cassert>

namespace ar
{
    /**
     * \brief Computes the gradient descent x_{n+1} = x_{n} - t * \nabla F(x_{n}).
     * This method computes the time step with the Barzilai-Borwein method.
     *
     * For interactivity, each iteration has to be triggered individually.
     *
     * \tparam X the type of x, must be an Eigen vector
     */
    template <class X>
    class GradientDescent
    {
    public:
        using X_t = X;
        using GF = std::function<X(const X&)>;

    private:
        //x[k%2] = current solution
        //x[1-k%2] = old / new solution
        X x[2];
        GF gf;
        X gPrevious;

        //iteration counter
        int k;

        //stopping criterion, if the norm of the gradient falls below that
        double epsilon;
        double linearStepsize;
		double maxStepsize;
		double minStepsize;
		X minValues, maxValues;
		bool hasMaxStepsize = false, hasMinStepsize = false;
		bool hasMinValues = false, hasMaxValues = false;

        double stepsize;
		bool error;

    public:
        GradientDescent(const Eigen::MatrixBase<X>& initialX, const GF& gf, double epsilon = 0.0001, double linearStepsize = 0.1)
            : x{ initialX.derived(), X() }, gf(gf), k(0), epsilon(epsilon), linearStepsize(linearStepsize), stepsize(linearStepsize), error(false)
        {};


	    double getEpsilon() const
	    {
		    return epsilon;
	    }

	    void setEpsilon(double epsilon)
	    {
		    this->epsilon = epsilon;
	    }

	    double getLinearStepsize() const
	    {
		    return linearStepsize;
	    }

	    void setLinearStepsize(double linear_stepsize)
	    {
		    linearStepsize = linear_stepsize;
	    }

	    double getMaxStepsize() const
	    {
		    return maxStepsize;
	    }

	    void setMaxStepsize(double max_stepsize)
	    {
			hasMaxStepsize = true;
		    maxStepsize = max_stepsize;
	    }

		double getMinStepsize() const
		{
			return minStepsize;
		}

		void setMinStepsize(double min_stepsize)
		{
			hasMinStepsize = true;
			minStepsize = min_stepsize;
		}

	    X getMinValues() const
	    {
		    return minValues;
	    }

	    void setMinValues(const X& min_values)
	    {
			hasMinValues = true;
		    minValues = min_values;
	    }

	    X getMaxValues() const
	    {
		    return maxValues;
	    }

	    void setMaxValues(const X& max_values)
	    {
			hasMaxValues = true;
		    maxValues = max_values;
	    }

    private:

		void checkBounds(X& x)
		{
			if (hasMinValues) x = x.cwiseMax(minValues);
			if (hasMaxValues) x = x.cwiseMin(maxValues);
		}

        bool lineSearchStep() {
            X gCurrent = gf(x[k % 2]);
            if (gCurrent.squaredNorm() <= epsilon*epsilon) {
                x[1 - (k % 2)] = x[k % 2];
                return true; //stopping
            }

            //perform line search, set x[1 - (k%2)]
            //for now, just step in a single direction a bit
            stepsize = linearStepsize / gCurrent.norm();
            x[1 - (k % 2)] = x[k % 2] - stepsize * gCurrent;
			checkBounds(x[1 - (k % 2)]);

            //set old gradient
            gPrevious = gCurrent;
            return false;
        }
        bool barzilaiBorweinStep() {
            X gCurrent = gf(x[k % 2]);
            if (gCurrent.squaredNorm() <= epsilon*epsilon) {
                x[1 - (k % 2)] = x[k % 2];
                return true; //stopping
            }
            //reuse gPrevious from the previous iteration
            //and compute the step size
            X s = x[k % 2] - x[1 - (k % 2)];
            X y = gCurrent - gPrevious;
            if (s.isZero(1e-20) || y.isZero(1e-20)) return true;
            //now alternate between the two BB step sizes
            if (k%2==0) {
                stepsize = s.dot(s) / s.dot(y);
            }
            else {
                stepsize = s.dot(y) / y.dot(y);
            }
            if (stepsize < 0)
            {
                CI_LOG_D("Negative step size of " << stepsize << "!\n"
                    << "prevX=" << x[1 - (k % 2)] << ", prevGrad=" << gPrevious
                    << ", curX=" << x[k % 2] << ", curGrad=" << gCurrent);
				CI_LOG_E("Negative step size of " << stepsize << "!");
                //stepsize = linearStepsize;
				stepsize = -stepsize;
				error = true;
            }
			if (hasMaxStepsize)
				stepsize = std::min(stepsize, maxStepsize);
			if (hasMinStepsize)
				stepsize = std::max(stepsize, minStepsize);
            //do the step
            x[1 - (k % 2)] = x[k % 2] - stepsize * gCurrent;
			checkBounds(x[1 - (k % 2)]);
            gPrevious = gCurrent;
            return false;
        }

    public:
        /**
         * Performs a step of the gradient descent.
         * \return true if the stopping condition |\nabla F| < epsilon is fullfilled
         */
        bool step() {
            if (k == 0) {
                bool s = lineSearchStep();
                k++;
                return s;
            }
            else {
                bool s = barzilaiBorweinStep();
                k++;
                return s;
            }
        }

        //Returns the current solution
        const X& getCurrentSolution() const {
            return x[k % 2];
        }

		//Returns the gradient from the last evaluation
		const X& getCurrentGradient() const {
			return gPrevious;
		}

        double getLastStepSize() const {
            return stepsize;
        }

		//Returns true iff an error occured: e.g. negative stepsize due to the Barzillai Borwein Step
		bool hasError() const { return error; }
    };
}
