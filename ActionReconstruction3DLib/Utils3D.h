#pragma once

#include "Commons3D.h"

//Device version of ar::utils

namespace ar3d {
	namespace utils
	{
#define DEVICE_CALL __host__ __device__ __inline__

		template <typename T> static DEVICE_CALL int sgn(T val) {
			return (T(0) < val) - (val < T(0));
		}
		template <typename T> static DEVICE_CALL int outside(T val) {
			return (val >= T(0)) ? 1 : 0;;
		}
		template <typename T> static DEVICE_CALL int inside(T val) {
			return (val < T(0)) ? 1 : 0;;
		}
		template <typename T> static DEVICE_CALL int insideEq(T val) {
			return (val <= T(0)) ? 1 : 0;;
		}
		template <typename T> static DEVICE_CALL T square(T val) { return val * val; }

		/**
		* \brief Numerical accurate computation of log(1 + exp(x)).
		* See "Accurately Computing log(...)" by Martin Mächler, 2012
		* \tparam T
		* \param x
		* \return
		*/
		template <typename T>
		DEVICE_CALL T log1pexp(T x)
		{
			if (x <= -37) return std::exp(x);
			if (x <= 18) return std::log1p(std::exp(x));
			if (x <= 33.3) return x + std::exp(-x);
			return x;
		}

		/**
		* \brief Returns a soft approximation of min(x, 0), with the smootheness parameter alpha.
		* \tparam T
		* \param x the value
		* \param alpha the smootheness, as alpha->infinity, softmin becomes the hard mininum.
		* \return
		*/
		template <typename T>
		DEVICE_CALL T softmin(T x, T alpha)
		{
			//softmin
			return -log1pexp(-x * alpha) / alpha;
		}

        /**
        * \brief Adjoint code of \code softmin<T>(x, alpha) \endcode.<br>
        * \param x the value
        * \param alpha the smooetheness
        * \param adjResult the adjoint of the result of the forward pass
        * \return the adjoint of the input value
        */
        template <typename T>
        DEVICE_CALL T softminAdjoint(T x, T alpha, T adjResult)
		{
            return adjResult / (std::exp(alpha*x) + 1);
		}

		/**
		* \brief The derivative of softmin with respect to x
		* \tparam T
		* \param x
		* \param alpha
		* \return
		*/
		template<typename T>
		DEVICE_CALL T softminDx(T x, T alpha)
		{
			//d/dx softmin
			return T(1) / (exp(alpha*x) + T(1));
		}

	    /**
         * \brief Adjoint of \ref softminDx()
         * \tparam T 
         * \param x 
         * \param alpha 
         * \param adjResult 
         * \return the adjoint of x
         */
        template<typename T>
        DEVICE_CALL T softminDxAdjoint(T x, T alpha, T adjResult)
		{
            if (x <= 100 / alpha && x >= -100 / alpha)
            {
                real e = exp(alpha*x);
                return adjResult * -(alpha * e) / square(1 + e);
            }
            return 0;
		}

#undef DEVICE_CALL
	}
}