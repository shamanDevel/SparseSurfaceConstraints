#pragma once
#include <cinder/Vector.h>
#include <cinder/Matrix.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include "Commons.h"

namespace ar
{
    namespace utils
    {
		inline glm::vec2 toGLM(const Eigen::Vector2d& v) { return glm::vec2(v.x(), v.y()); }
        inline glm::vec3 toGLM(const Eigen::Vector3d& v) { return glm::vec3(v.x(), v.y(), v.z()); }
        inline glm::vec4 toGLM(const Eigen::Vector4d& v) { return glm::vec4(v.x(), v.y(), v.z(), v.w()); }
        inline glm::quat toGLM(const Eigen::Quaterniond& v) { return glm::quat(float(v.w()), float(v.x()), float(v.y()), float(v.z())); }
        inline glm::mat4 toGLM(const Eigen::Matrix4d& mat)
        {
            cinder::dmat4 m;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] = mat(j, i);
            return m;
        }

        inline Eigen::Vector3d toEigen(const glm::vec3& v) { return Eigen::Vector3d(v.x, v.y, v.z); }
        inline Eigen::Vector4d toEigen(const glm::vec4& v) { return Eigen::Vector4d(v.x, v.y, v.z, v.w); }

        template <typename T> static int sgn(T val) {
            return (T(0) < val) - (val < T(0));
        }
        template <typename T> static int outside(T val) {
            return (val >= T(0)) ? 1 : 0;;
        }
        template <typename T> static int inside(T val) {
            return (val < T(0)) ? 1 : 0;;
        }
        template <typename T> static int insideEq(T val) {
            return (val <= T(0)) ? 1 : 0;;
        }
        template <typename T> static T sqr(T val) { return val * val; }

		template<typename X, typename S>
		X mix(S alpha, const X& a, const X& b) { return (1 - alpha)*a + alpha * b; }

        //Interpolates into a cell:
        //
        //  y
        // /\
        // v2 -- n3
        //  |    |
        // v0 -- v1 -> x
        template <typename T, typename S>
        T bilinearInterpolate(const std::array<T, 4>& values, S fx, S fy)
        {
            //T v0 = values[0] + fx * (values[1] - values[0]);
            //T v1 = values[2] + fx * (values[3] - values[2]);
            //return v0 + fy * (v1 - v0);
            T v0 = mix(fx, values[0], values[1]);
            T v1 = mix(fx, values[2], values[3]);
            return mix(fy, v0, v1);
        }

        template <typename T, typename S>
        T trilinearInterpolate(const std::array<T, 8>& values, S fx, S fy, S fz)
		{
            T v0 = bilinearInterpolate({ values[0], values[1], values[2], values[3] }, fx, fy);
            T v1 = bilinearInterpolate({ values[4], values[5], values[6], values[7] }, fx, fy);
            return mix(fz, v0, v1);
		}

        static constexpr int interpolateInvNewtonSteps = 10;
        /**
         * \brief Solves \code x = bilinearInterpolate(values, alpha, beta)\endcode for alpha and beta.
         *  It is only guaranteered to converge if alpha,beta is in [0,1].
         *  You have to check for yourselve if the returned values are valid or not
         * \param values the corner values
         * \param x the expected interpolated value
         * \return the alpha, beta interpolation weights
         */
	    Vector2 bilinearInterpolateInv(const std::array<Vector2, 4>& values, const Vector2& x);

		std::pair<Vector2, Vector2> bilinearInterpolateInvAnalytic(const std::array<Vector2, 4>& values, const Vector2& x);

	    /**
		 * \brief Numerical accurate computation of log(1 + exp(x)).
		 * See "Accurately Computing log(...)" by Martin Mächler, 2012
		 * \tparam T 
		 * \param x 
		 * \return 
		 */
		template <typename T>
		T log1pexp(T x)
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
        T softmin(T x, T alpha)
        {
            //softmin
			return -log1pexp(-x * alpha) / alpha;
        }

        /**
         * \brief The derivative of softmin with respect to x
         * \tparam T 
         * \param x 
         * \param alpha 
         * \return 
         */
        template<typename T>
        T softminDx(T x, T alpha)
        {
            //d/dx softmin
            return T(1) / (exp(alpha*x) + T(1));
        }


    }
}
