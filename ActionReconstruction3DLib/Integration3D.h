#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"

namespace ar3d {

	/**
	 * \brief This class contains routines for volume and surface integrals over partial cells.
	 * 
	 * Layout:
	 * <pre>
	 *      v6 ---------- v7
	 *  z  /|            /|
	 *  | / |           / |
	 *  v4 ---------- v5  |  y
	 *  |   |         |   | /  
	 *  |   v2 ------ |-- v3
	 *  |  /          |  /
	 *  | /           | /
	 *  v0 ---------- v1 - x
	 * </pre>
	 */
	class Integration3D
	{
	public:

		typedef real8 InterpolationWeight_t;
		typedef real8 InputPhi_t;

		/**
		 * \brief Computes the volume integral over \f$ \{ x\in[0,1]^3 : \phi(x)<=0 \} \f$
		 *	for each of the eight basis functions.
		 * \param phi SDF values
		 * \param h cell size
		 * \return interpolation weights
		 */
		static InterpolationWeight_t volumeIntegral(const InputPhi_t& phi, real h);

		/**
		* \brief Computes the surface integral over \f$ \{ x\in[0,1]^3 : \phi(x)=0 \} \f$
		*	for each of the eight basis functions.
		* \param phi SDF values
		* \param h cell size
		* \return interpolation weights
		*/
		static InterpolationWeight_t surfaceIntegral(const InputPhi_t& phi, real h);

		Integration3D() = delete;
		~Integration3D() = default;

		static real interpolate(const InputPhi_t& phi, real x, real y, real z);

        // volume is sampled
		static InterpolationWeight_t volumeIntegralSampled(const InputPhi_t& phi, real h, int samples = 1000);

        // volume is approximated linearly and then analytically integrated
		static InterpolationWeight_t volumeIntegralLinear(const InputPhi_t& phi, real h);

        // approximated by marching cube triangles
		static InterpolationWeight_t surfaceIntegralMC(const InputPhi_t& phi, real h);
	};

}
