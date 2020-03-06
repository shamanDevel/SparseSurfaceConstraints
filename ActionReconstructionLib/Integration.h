#pragma once

#include <algorithm>
#include <array>
#if _HAS_CXX17
#include <optional>
#endif

#include <Eigen/Core>
#include <cinder/Log.h>

#include "Commons.h"
#include "Utils.h"

namespace ar
{
    template<typename T>
    struct Integration2D
    {
    public:
		typedef Eigen::Matrix<T, 2, 1> Vec;

        //Integrates over the volume in the triangle that is inside the surface represented by the SDF
		template<typename X>
        static X integrateTriangle(const std::array<Vec, 3>& pos, const std::array<T, 3>& sdf, const std::array<X, 3>& value)
        {
            using namespace ar::utils;
            T area3 = abs((pos[1].x() - pos[0].x())*(pos[2].y() - pos[0].y()) - (pos[2].x() - pos[0].x())*(pos[1].y() - pos[0].y())) / 6; //already divided by 3
            T alpha = sdf[0] / (sdf[0] - sdf[1]);
            T beta = sdf[1] / (sdf[1] - sdf[2]);
            T gamma = sdf[0] / (sdf[0] - sdf[2]);
            int c = outside(sdf[0]) | (outside(sdf[1]) << 1) | (outside(sdf[2]) << 2);
            switch (c)
            {
            case 0b000: return area3 * (value[0] + value[1] + value[2]); //completely inside
            case 0b111: return 0 * value[0]; //completely outside
            case 0b110: return area3 * alpha*gamma*((T(3) - alpha - gamma)*value[0] + alpha*value[1] + gamma*value[2]); //only node 0 inside
            case 0b001: return area3 * ((value[0] + value[1] + value[2]) - (alpha*gamma*((T(3) - alpha - gamma)*value[0] + alpha*value[1] + gamma*value[2]))); //only node 0 outside
            case 0b101: return area3 * (T(1) - alpha) * beta * ((T(2) + alpha - beta)*value[1] + (T(1) - alpha)*value[0] + beta * value[2]); //only node 1 inside
            case 0b010: return area3 * ((value[0] + value[1] + value[2]) - ((T(1) - alpha) * beta * ((T(2) + alpha - beta)*value[1] + (T(1) - alpha)*value[0] + beta * value[2]))); //only node 1 outside
            case 0b011: return area3 * (T(1) - beta) * (T(1) - gamma) * ((T(1) + beta + gamma)*value[2] + (T(1) - beta)*value[1] + (T(1) - gamma)*value[0]); //only node 2 inside
            case 0b100: return area3 * ((value[0] + value[1] + value[2]) - ((T(1) - beta) * (1 - gamma) * ((T(1) + beta + gamma)*value[2] + (T(1) - beta)*value[1] + (T(1) - gamma)*value[0]))); //only node 2 outside
            default: return X(); //this should never happen
            }
        }

    public:
		typedef Eigen::Matrix<T, 4, 1> IntegrationWeights;

    private:
		static IntegrationWeights shuffleWeights(const IntegrationWeights& in, const std::array<int, 4> idx)
		{
			IntegrationWeights v;
			for (int i = 0; i < 4; ++i) v[idx[i]] = in[i];
			return v;
		}

        static IntegrationWeights integrateQuadCase3a(const std::array<T, 4>& sdf)
        {
            //only node 0 inside
            T a = sdf[0] / (sdf[0] - sdf[1]);
            T b = sdf[0] / (sdf[0] - sdf[2]);
            T c0 = (a*(a*(-4 + b) - 4 * (-3 + b))*b) / 24;
            T c1 = -(a*a*(-4 + b)*b) / 24;
            T c2 = -((-4 + a)*a*b*b) / 24;
            T c3 = (a*a*b*b) / 24.;
			return { c0, c1, c2, c3 };
            //return c0 * value[0] + c1 * value[1] + c2 * value[2] + c3 * value[3];
        }
        static IntegrationWeights integrateQuadCase3b(const std::array<T, 4>& sdf)
        {
            //only node 0 outside
            T a = sdf[0] / (sdf[0] - sdf[1]);
            T b = sdf[0] / (sdf[0] - sdf[2]);
            T c0 = T(0.25) - (a*(a*(-4 + b) - 4 * (-3 + b))*b) / 24;
            T c1 = T(0.25) + (a*a*(-4 + b)*b) / 24;
            T c2 = T(0.25) + ((-4 + a)*a*b*b) / 24;
            T c3 = T(0.25) - (a*a*b*b) / 24.;
			return { c0, c1, c2, c3 };
            //return c0 * value[0] + c1 * value[1] + c2 * value[2] + c3 * value[3];
        }
        static IntegrationWeights integrateQuadCase4(const std::array<T, 4>& sdf)
        {
            //node 0 and node 1 inside
            T a = sdf[0] / (sdf[0] - sdf[2]);
            T b = sdf[1] / (sdf[1] - sdf[3]);
            T c0 = (-3 * a*a - 2 * a*(-4 + b) - (-4 + b)*b) / 24;
            T c1 = (-(a*a) - 2 * a*(-2 + b) + (8 - 3 * b)*b) / 24;
            T c2 = (3 * a*a + 2 * a*b + b*b) / 24;
            T c3 = (a*a + 2 * a*b + 3 * b*b) / 24;
			return { c0, c1, c2, c3 };
            //return c0 * value[0] + c1 * value[1] + c2 * value[2] + c3 * value[3];
        }
        
    public:

		/**
		* \brief Computes the weights for the integration over the the area that is inside the surface
		*  represented by the SDF.
		* Layout:
		* <pre>
		* x -- size
		* |     |
		* 0 --  x
		*
		* v2 -- v3
		* |     |
		* v0 -- v1
		* </pre>
		* \param size the size of the cells
		* \param sdf the sdf values
		* \return the integration weights as used by \ref integrateQuad(const IntegrationWeights& weights, const std::array<X, 4>& value)
		*/
		static IntegrationWeights getIntegrateQuadWeights(const Vec& size, const std::array<T, 4>& sdf)
		{
			using namespace ar::utils;
			T area = size.prod();
			int c = insideEq(sdf[0]) | (insideEq(sdf[1]) << 1) | (insideEq(sdf[2]) << 2) | (insideEq(sdf[3]) << 3);
			switch (c)
			{
			case 0b1111: return IntegrationWeights::Constant(0.25 * area); //case 2: completely inside
			case 0b0000: return IntegrationWeights::Zero(); //case 1: completely outside
			case 0b0001: return area * shuffleWeights(integrateQuadCase3a({ sdf[0], sdf[1], sdf[2], sdf[3] }), {0,1,2,3}); //node 0 inside
			case 0b0010: return area * shuffleWeights(integrateQuadCase3a({ sdf[1], sdf[3], sdf[0], sdf[2] }), {1,3,0,2});
			case 0b0100: return area * shuffleWeights(integrateQuadCase3a({ sdf[2], sdf[0], sdf[3], sdf[1] }), {2,0,3,1});
			case 0b1000: return area * shuffleWeights(integrateQuadCase3a({ sdf[3], sdf[2], sdf[1], sdf[0] }), {3,2,1,0});
			case 0b1110: return area * shuffleWeights(integrateQuadCase3b({ sdf[0], sdf[1], sdf[2], sdf[3] }), {0,1,2,3}); //node 0 outside
			case 0b1101: return area * shuffleWeights(integrateQuadCase3b({ sdf[1], sdf[3], sdf[0], sdf[2] }), {1,3,0,2});
			case 0b1011: return area * shuffleWeights(integrateQuadCase3b({ sdf[2], sdf[0], sdf[3], sdf[1] }), {2,0,3,1});
			case 0b0111: return area * shuffleWeights(integrateQuadCase3b({ sdf[3], sdf[2], sdf[1], sdf[0] }), {3,2,1,0});
			case 0b0011: return area * shuffleWeights(integrateQuadCase4 ({ sdf[0], sdf[1], sdf[2], sdf[3] }), {0,1,2,3}); //node 0 and 1 inside
			case 0b1010: return area * shuffleWeights(integrateQuadCase4 ({ sdf[1], sdf[3], sdf[0], sdf[2] }), {1,3,0,2});
			case 0b1100: return area * shuffleWeights(integrateQuadCase4 ({ sdf[3], sdf[2], sdf[1], sdf[0] }), {3,2,1,0});
			case 0b0101: return area * shuffleWeights(integrateQuadCase4 ({ sdf[2], sdf[0], sdf[3], sdf[1] }), {2,0,3,1});
			case 0b1001: CI_LOG_E("case 1001: not implemented"); return IntegrationWeights::Zero(); //throw std::exception("Not implemented yet");
			case 0b0110: CI_LOG_E("case 0110: not implemented"); return IntegrationWeights::Zero(); //throw std::exception("Not implemented yet");
			default: throw std::exception("This should not happen");
			}
		}

	    /**
		 * \brief Integrates over the volume in the quad that is inside the surface represented by the SDF.
		 * Layout:
		 * <pre>
		 * x -- size
		 * |     |
		 * 0 --  x
		 * 
		 * v2 -- v3
		 * |     |
		 * v0 -- v1
		 * </pre>
		 * \param weights the weights as computed by \ref getIntegrateQuadWeights
		 * \param value the values at the corners
		 * \return the integrated value
		 */
		template<typename X>
		static X integrateQuad(const IntegrationWeights& weights, const std::array<X, 4>& value)
		{
			return weights[0] * value[0] + weights[1] * value[1] + weights[2] * value[2] + weights[3] * value[3];
		}

        //Integrates over the volume in the quad that is inside the surface represented by the SDF
        // x -- size
        // |     |
        // 0 --  x
        //
        // v2 -- v3
        // |     |
        // v0 -- v1
		template<typename X>
        static X integrateQuad(const Vec& size, const std::array<T, 4>& sdf, const std::array<X, 4>& value)
        {
			return integrateQuad(getIntegrateQuadWeights(size, sdf), value);
        }

        //Integrates over the boundary of the surface represented by the SDF, inside the triangle
		template<typename X>
        static X integrateTriangleBoundary(const std::array<Vec, 3>& pos, const std::array<T, 3>& sdf, const std::array<X, 3>& value)
        {
            using namespace ar::utils;
            T area3 = abs((pos[1].x() - pos[0].x())*(pos[2].y() - pos[0].y()) - (pos[2].x() - pos[0].x())*(pos[1].y() - pos[0].y())) / 6; //already divided by 3
            T alpha = sdf[0] / (sdf[0] - sdf[1]);
            T beta = sdf[1] / (sdf[1] - sdf[2]);
            T gamma = sdf[0] / (sdf[0] - sdf[2]);
            int c = outside(sdf[0]) | (outside(sdf[1]) << 1) | (outside(sdf[2]) << 2);
            switch (c)
            {
            case 0b110:
            case 0b001: //boundary at node 0
                return ((2 - alpha - gamma) * value[0] + alpha * value[1] + gamma * value[2]) * T(0.5) * ((gamma - alpha) * pos[0] + alpha * pos[1] - gamma * pos[2]).norm();
            case 0b010:
            case 0b101: //boundary at node 1
                return ((1 + alpha - beta) * value[1] + (1 - alpha)*value[0] + beta *value[2]) * T(0.5) * ((alpha + beta - 1)*pos[1] + (1 - alpha)*pos[0] - beta*pos[2]).norm();
            case 0b100:
            case 0b011: //boundary at node 2
                return ((gamma + beta)*value[2] + (1 - gamma)*value[0] + (1 - beta)*value[1]) * T(0.5) * ((gamma - beta)*pos[2] + (1 - gamma)*pos[0] - (1 - beta)*pos[1]).norm();
            default: return 0 * value[0];
            }
        }

    private:
        static IntegrationWeights integrateQuadBoundaryCase3(const std::array<T, 4>& sdf)
        {
            //only node 0 inside, or only node 0 outside
            T a = sdf[0] / (sdf[0] - sdf[1]);
            T b = sdf[0] / (sdf[0] - sdf[2]);
            T l = std::sqrt(a*a + b * b);
            T c0 = (6 + a * (-3 + b) - 3 * b) / 6;
            T c1 = -(a*(-3 + b)) / 6;
            T c2 = -((-3 + a)*b) / 6;
            T c3 = (a*b) / 6;
			return l * IntegrationWeights(c0, c1, c2, c3);
            //return l * (c0 * value[0] + c1 * value[1] + c2 * value[2] + c3 * value[3]);
        }
        static IntegrationWeights integrateQuadBoundaryCase4(const std::array<T, 4>& sdf)
        {
            //only node 0 and node 1 inside or outside
            T a = sdf[0] / (sdf[0] - sdf[2]);
            T b = sdf[1] / (sdf[1] - sdf[3]);
            T l = std::sqrt(1 + a * a - 2*a*b + b * b);
            T c0 = (3 - 2 * a - b) / 6;
            T c1 = (3 - a - 2 * b) / 6;
            T c2 = (2 * a + b) / 6;
            T c3 = (a + 2 * b) / 6;
			return l * IntegrationWeights(c0, c1, c2, c3);
            //return l * (c0 * value[0] + c1 * value[1] + c2 * value[2] + c3 * value[3]);
        }
        
    public:

		/**
		* \brief Computes the weights for the integration over the boundary of the surface
		*  represented by the SDF.
		* Layout:
		* <pre>
		* x -- size
		* |     |
		* 0 --  x
		*
		* v2 -- v3
		* |     |
		* v0 -- v1
		* </pre>
		* \param size the size of the cells
		* \param sdf the sdf values
		* \return the integration weights as used by \ref integrateQuadBoundary(const IntegrationWeights& weights, const std::array<X, 4>& value)
		*/
		static IntegrationWeights getIntegrateQuadBoundaryWeights(const Vec& size, const std::array<T, 4>& sdf)
		{
			using namespace ar::utils;
			T len = sqrt(size.x()*size.y());
			int c = inside(sdf[0]) | (inside(sdf[1]) << 1) | (inside(sdf[2]) << 2) | (inside(sdf[3]) << 3);
			switch (c)
			{
			case 0b1111: return IntegrationWeights::Zero(); //case 2: completely inside
			case 0b0000: return IntegrationWeights::Zero(); //case 1: completely outside
			case 0b0001: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[0], sdf[1], sdf[2], sdf[3] }), { 0,1,2,3 }); //node 0 inside
			case 0b0010: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[1], sdf[3], sdf[0], sdf[2] }), { 1,3,0,2 });
			case 0b0100: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[2], sdf[0], sdf[3], sdf[1] }), { 2,0,3,1 });
			case 0b1000: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[3], sdf[2], sdf[1], sdf[0] }), { 3,2,1,0 });
			case 0b1110: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[0], sdf[1], sdf[2], sdf[3] }), { 0,1,2,3 }); //node 0 outside
			case 0b1101: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[1], sdf[3], sdf[0], sdf[2] }), { 1,3,0,2 });
			case 0b1011: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[2], sdf[0], sdf[3], sdf[1] }), { 2,0,3,1 });
			case 0b0111: return len * shuffleWeights(integrateQuadBoundaryCase3({ sdf[3], sdf[2], sdf[1], sdf[0] }), { 3,2,1,0 });
			case 0b0011: return len * shuffleWeights(integrateQuadBoundaryCase4({ sdf[0], sdf[1], sdf[2], sdf[3] }), { 0,1,2,3 }); //node 0 and 1 inside
			case 0b1010: return len * shuffleWeights(integrateQuadBoundaryCase4({ sdf[1], sdf[3], sdf[0], sdf[2] }), { 1,3,0,2 });
			case 0b1100: return len * shuffleWeights(integrateQuadBoundaryCase4({ sdf[3], sdf[2], sdf[1], sdf[0] }), { 3,2,1,0 });
			case 0b0101: return len * shuffleWeights(integrateQuadBoundaryCase4({ sdf[2], sdf[0], sdf[3], sdf[1] }), { 2,0,3,1 });
			case 0b1001: CI_LOG_E("case 1001: not implemented"); return IntegrationWeights::Zero(); //throw std::exception("Not implemented yet");
			case 0b0110: CI_LOG_E("case 0110: not implemented"); return IntegrationWeights::Zero(); //throw std::exception("Not implemented yet");
			default: throw std::exception("This should not happen");
			}
		}

		/**
		* \brief Integrates over the boundary of the surface represented by the SDF.
		* Layout:
		* <pre>
		* x -- size
		* |     |
		* 0 --  x
		*
		* v2 -- v3
		* |     |
		* v0 -- v1
		* </pre>
		* \param weights the weights as computed by \ref getIntegrateQuadBoundaryWeights(const Vec& size, const std::array<T, 4>& sdf)
		* \param value the values at the corners
		* \return the integrated value
		*/
		template<typename X>
		static X integrateQuadBoundary(const IntegrationWeights& weights, const std::array<X, 4>& value)
		{
			return weights[0] * value[0] + weights[1] * value[1] + weights[2] * value[2] + weights[3] * value[3];
		}

        //Integrates over the boundary of the surface represented by the SDF, inside the quad
        // x -- size
        // |     |
        // 0 --  x
        //
        // v2 -- v3
        // |     |
        // v0 -- v1
		template<typename X>
        static X integrateQuadBoundary(const Vec& size, const std::array<T, 4>& sdf, const std::array<X, 4>& value)
        {
			return integrateQuadBoundary(getIntegrateQuadBoundaryWeights(size, sdf), value);
        }

    private:
		//Derivatives with respect to the SDF values
		template<typename X>
		static std::array<X, 4> integrateQuadCase3aDsdf(const std::array<T, 4>& sdf, const std::array<X, 4>& value, T area)
		{
			T sdf0 = sdf[0];
			T sdf1 = sdf[1];
			T sdf2 = sdf[2];
			T sdf3 = sdf[3];
			const X& val0 = value[0];
			const X& val1 = value[1];
			const X& val2 = value[2];
			const X& val3 = value[3];

			return {
			
				area * (
				-(sdf0*(-12 * ar::square(sdf1)*ar::square(sdf2)*val0 +
				6 * sdf0*sdf1*sdf2*(sdf2*(2 * val0 + val1) + sdf1 * (2 * val0 + val2)) +
					ar::cube(sdf0)*(sdf1*(val0 + 3 * val1 + val2 + val3) +
						sdf2 * (val0 + val1 + 3 * val2 + val3)) -
					2 * ar::square(sdf0)*(ar::square(sdf2)*(2 * val0 + val1) +
						ar::square(sdf1)*(2 * val0 + val2) +
						sdf1 * sdf2*(3 * val0 + 4 * val1 + 4 * val2 + val3)))) /
						(12*ar::cube(sdf0 - sdf1)*ar::cube(sdf0 - sdf2))),

				area * (
				(ar::square(sdf0)*(6 * sdf1*sdf2*val0 -
					2 * sdf0*(sdf2*(val0 + 2 * val1) + sdf1 * (2 * val0 + val2)) +
					ar::square(sdf0)*(val0 + 3 * val1 + val2 + val3))) /
					(12*ar::cube(sdf0 - sdf1)*ar::square(sdf0 - sdf2))),

				area * (
				(ar::square(sdf0)*(6 * sdf1*sdf2*val0 -
					2 * sdf0*(sdf2*(2 * val0 + val1) + sdf1 * (val0 + 2 * val2)) +
					ar::square(sdf0)*(val0 + val1 + 3 * val2 + val3))) /
					(12*ar::square(sdf0 - sdf1)*ar::cube(sdf0 - sdf2))),

				0 * val3
			};
		}
		template<typename X>
		static std::array<X, 4> integrateQuadCase3bDsdf(const std::array<T, 4>& sdf, const std::array<X, 4>& value, T area)
		{
			std::array<X, 4> grad = integrateQuadCase3aDsdf(sdf, value, area);
			return {
				-grad[0],
				-grad[1],
				-grad[2],
				-grad[3]
			};
		}
		template<typename X>
		static std::array<X, 4> integrateQuadCase4Dsdf(const std::array<T, 4>& sdf, const std::array<X, 4>& value, T area)
		{
			T sdf0 = sdf[0];
			T sdf1 = sdf[1];
			T sdf2 = sdf[2];
			T sdf3 = sdf[3];
			const X& val0 = value[0];
			const X& val1 = value[1];
			const X& val2 = value[2];
			const X& val3 = value[3];

			return {

				area * (
				(sdf2*(sdf1*(sdf2*(3 * val0 + val1 + val2 + val3) - 2 * sdf0*(2 * val2 + val3)) +
				sdf3 * (-2 * sdf2*(2 * val0 + val1) + sdf0 * (val0 + val1 + 3 * val2 + val3)))) /
					(12*ar::cube(sdf0 - sdf2)*(sdf1 - sdf3))),

				area * (
				(sdf3*(sdf3*(-2 * sdf2*(val0 + 2 * val1) + sdf0 * (val0 + 3 * val1 + val2 + val3)) +
					sdf1 * (-2 * sdf0*(val2 + 2 * val3) + sdf2 * (val0 + val1 + val2 + 3 * val3)))) /
					(12*(sdf0 - sdf2)*ar::cube(sdf1 - sdf3))),

				area * (
				(sdf0*(-(sdf1*(sdf2*(3 * val0 + val1 + val2 + val3) - 2 * sdf0*(2 * val2 + val3))) -
					sdf3 * (-2 * sdf2*(2 * val0 + val1) + sdf0 * (val0 + val1 + 3 * val2 + val3)))) /
					(12*ar::cube(sdf0 - sdf2)*(sdf1 - sdf3))),

				area * (
				-(sdf1*(sdf3*(-2 * sdf2*(val0 + 2 * val1) + sdf0 * (val0 + 3 * val1 + val2 + val3)) +
					sdf1 * (-2 * sdf0*(val2 + 2 * val3) + sdf2 * (val0 + val1 + val2 + 3 * val3)))) /
					(12*(sdf0 - sdf2)*ar::cube(sdf1 - sdf3))),

			};
		}
		template<typename X>
		static std::array<X, 4> shuffleArray(const std::array<X, 4>& in, const std::array<int, 4> idx)
		{
			std::array<X, 4> v;
			for (int i = 0; i < 4; ++i) v[idx[i]] = in[i];
			return v;
		}

    public:
		//Computes the derivatives of the integration over the volume in the quad that is inside the surface represented by the SDF
		// with respect to the SDF values
		// x -- size
		// |     |
		// 0 --  x
		//
		// v2 -- v3
		// |     |
		// v0 -- v1
		template<typename X>
		static std::array<X, 4> integrateQuadDsdf(const Vec& size, const std::array<T, 4>& sdf, const std::array<X, 4>& value)
		{
			using namespace ar::utils;
			T area = size.prod();
			int c = insideEq(sdf[0]) | (insideEq(sdf[1]) << 1) | (insideEq(sdf[2]) << 2) | (insideEq(sdf[3]) << 3);
			switch (c)
			{
			case 0b1111: return { 0 * value[0], 0 * value[1], 0 * value[2], 0 * value[3] }; //case 2: completely inside
			case 0b0000: return { 0 * value[0], 0 * value[1], 0 * value[2], 0 * value[3] }; //case 1: completely outside
			case 0b0001: return shuffleArray<X>(integrateQuadCase3aDsdf<X>({ sdf[0], sdf[1], sdf[2], sdf[3] }, { value[0], value[1], value[2], value[3] }, area), {0,1,2,3}); //node 0 inside
			case 0b0010: return shuffleArray<X>(integrateQuadCase3aDsdf<X>({ sdf[1], sdf[3], sdf[0], sdf[2] }, { value[1], value[3], value[0], value[2] }, area), {1,3,0,2});
			case 0b0100: return shuffleArray<X>(integrateQuadCase3aDsdf<X>({ sdf[2], sdf[0], sdf[3], sdf[1] }, { value[2], value[0], value[3], value[1] }, area), {2,0,3,1});
			case 0b1000: return shuffleArray<X>(integrateQuadCase3aDsdf<X>({ sdf[3], sdf[2], sdf[1], sdf[0] }, { value[3], value[2], value[1], value[0] }, area), {3,2,1,0});
			case 0b1110: return shuffleArray<X>(integrateQuadCase3bDsdf<X>({ sdf[0], sdf[1], sdf[2], sdf[3] }, { value[0], value[1], value[2], value[3] }, area), {0,1,2,3}); //node 0 outside
			case 0b1101: return shuffleArray<X>(integrateQuadCase3bDsdf<X>({ sdf[1], sdf[3], sdf[0], sdf[2] }, { value[1], value[3], value[0], value[2] }, area), {1,3,0,2});
			case 0b1011: return shuffleArray<X>(integrateQuadCase3bDsdf<X>({ sdf[2], sdf[0], sdf[3], sdf[1] }, { value[2], value[0], value[3], value[1] }, area), {2,0,3,1});
			case 0b0111: return shuffleArray<X>(integrateQuadCase3bDsdf<X>({ sdf[3], sdf[2], sdf[1], sdf[0] }, { value[3], value[2], value[1], value[0] }, area), {3,2,1,0});
			case 0b0011: return shuffleArray<X>(integrateQuadCase4Dsdf<X>( { sdf[0], sdf[1], sdf[2], sdf[3] }, { value[0], value[1], value[2], value[3] }, area), {0,1,2,3}); //node 0 and 1 inside
			case 0b1010: return shuffleArray<X>(integrateQuadCase4Dsdf<X>( { sdf[1], sdf[3], sdf[0], sdf[2] }, { value[1], value[3], value[0], value[2] }, area), {1,3,0,2});
			case 0b1100: return shuffleArray<X>(integrateQuadCase4Dsdf<X>( { sdf[3], sdf[2], sdf[1], sdf[0] }, { value[3], value[2], value[1], value[0] }, area), {3,2,1,0});
			case 0b0101: return shuffleArray<X>(integrateQuadCase4Dsdf<X>( { sdf[2], sdf[0], sdf[3], sdf[1] }, { value[2], value[0], value[3], value[1] }, area), {2,0,3,1});
			case 0b1001: CI_LOG_E("case 1001: not implemented"); return { 0 * value[0], 0 * value[1], 0 * value[2], 0 * value[3] }; //throw std::exception("Not implemented yet");
			case 0b0110: CI_LOG_E("case 0110: not implemented"); return { 0 * value[0], 0 * value[1], 0 * value[2], 0 * value[3] }; //throw std::exception("Not implemented yet");
			default: throw std::exception("This should not happen");
			}
		}

    public:
#if _HAS_CXX17
	    /**
		 * \brief Returns the two intersection points of the boundary with the cell
		 * (if there is a boundary)
		 * \param sdf 
		 * \param corners the locations of the four corners
		 * \return 
		 */
		static std::optional<std::pair<Vec, Vec> > getIntersectionPoints(const std::array<T, 4>& sdf, const std::array<Vec, 4>& corners)
		{
			using namespace ar::utils;
			int c = inside(sdf[0]) | (inside(sdf[1]) << 1) | (inside(sdf[2]) << 2) | (inside(sdf[3]) << 3);
			switch (c)
			{
			case 0b1111: return {}; //case 2: completely inside
			case 0b0000: return {}; //case 1: completely outside

			case 0b0001:
			case 0b1110:
				return std::make_pair(mix(sdf[0] / (sdf[0] - sdf[1]), corners[0], corners[1]), mix(sdf[0] / (sdf[0] - sdf[2]), corners[0], corners[2]));
			case 0b0010:
			case 0b1101:
				return std::make_pair(mix(sdf[0] / (sdf[0] - sdf[1]), corners[0], corners[1]), mix(sdf[1] / (sdf[1] - sdf[3]), corners[1], corners[3]));
			case 0b0100:
			case 0b1011:
				return std::make_pair(mix(sdf[2] / (sdf[2] - sdf[3]), corners[2], corners[3]), mix(sdf[0] / (sdf[0] - sdf[2]), corners[0], corners[2]));
			case 0b1000:
			case 0b0111:
				return std::make_pair(mix(sdf[2] / (sdf[2] - sdf[3]), corners[2], corners[3]), mix(sdf[1] / (sdf[1] - sdf[3]), corners[1], corners[3]));

			case 0b0011:
			case 0b1100:
				return std::make_pair(mix(sdf[0] / (sdf[0] - sdf[2]), corners[0], corners[2]), mix(sdf[1] / (sdf[1] - sdf[3]), corners[1], corners[3]));
			case 0b1010:
			case 0b0101:
				return std::make_pair(mix(sdf[0] / (sdf[0] - sdf[1]), corners[0], corners[1]), mix(sdf[2] / (sdf[2] - sdf[3]), corners[2], corners[3]));
			
			case 0b1001: CI_LOG_E("case 1001: not implemented"); return {}; //throw std::exception("Not implemented yet");
			case 0b0110: CI_LOG_E("case 0110: not implemented"); return {}; //throw std::exception("Not implemented yet");
			default: throw std::exception("This should not happen");
			}
		}
#endif
    };

}