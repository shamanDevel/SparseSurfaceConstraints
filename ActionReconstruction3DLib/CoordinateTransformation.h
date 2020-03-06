#pragma once

#include <cmath>

namespace ar3d
{
    /**
     * \brief Utility class for coordinate transformations.
     * 
     * The vector types as template parameters must have
     * public elements \c x, \c y and \c z.
     */
    struct CoordinateTransformation
    {
        CoordinateTransformation() = delete;

        /**
         * \brief Converts the cartesian coordinates into spherical coordinates
         * \param cartesian cartesian coordinates X, Y, Z
         * \return spherical coordinates: x=radius, Y=inclination theta, Z=azimuth phi
         */
        template <typename V>
        static V cartesian2spherical(const V& cartesian)
        {
            V spherical;
            spherical.x = sqrt(cartesian.x*cartesian.x + cartesian.y*cartesian.y + cartesian.z*cartesian.z);
            spherical.y = std::acos(cartesian.z / spherical.x);
            spherical.z = std::atan2(cartesian.y, cartesian.x);
            return spherical;
        }

        /**
        * \brief Converts the spherical coordinates into cartesian coordinates
        * \param spherical spherical coordinates: x=radius, Y=inclination theta, Z=azimuth phi
        * \return cartesian cartesian coordinates X, Y, Z
        */
        template <typename V>
        static V spherical2cartesian(const V& spherical)
        {
            V cartesian;
            cartesian.x = spherical.x * std::sin(spherical.y) * std::cos(spherical.z);
            cartesian.y = spherical.x * std::sin(spherical.y) * std::sin(spherical.z);
            cartesian.z = spherical.x * std::cos(spherical.y);
            return cartesian;
        }

        /**
        * \brief Adjoint: Converts the spherical coordinates into cartesian coordinates
        * \param spherical spherical coordinates: x=radius, Y=inclination theta, Z=azimuth phi
        * \param adjCartesian adjoint of the cartesian coordinates
        * \return adjoint of the sperical coordinates
        */
        template <typename V>
        static V spherical2cartesianAdjoint(const V& spherical, const V& adjCartesian)
        {
            V adjSpherical;

            adjSpherical.x =
                adjCartesian.x * std::sin(spherical.y) * std::cos(spherical.z) +
                adjCartesian.y * std::sin(spherical.y) * std::sin(spherical.z) +
                adjCartesian.z * std::cos(spherical.y);

            adjSpherical.y =
                adjCartesian.x * spherical.x * std::cos(spherical.y) * std::cos(spherical.z) +
                adjCartesian.y * spherical.x * std::cos(spherical.y) * std::sin(spherical.z) +
                -adjCartesian.z * spherical.x * std::sin(spherical.y);

            adjSpherical.z =
                -adjCartesian.x * spherical.x * std::sin(spherical.y) * std::sin(spherical.z) +
                adjCartesian.y * spherical.x * std::sin(spherical.y) * std::cos(spherical.z);

            return adjSpherical;
        }
    };
}