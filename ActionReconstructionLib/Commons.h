#pragma once

#include <Eigen/Core>
#include <fstream>

namespace ar
{
    // The main typedef, specifies the floating point precision
    typedef double real;

    //Helper for Eigen

    typedef Eigen::Matrix<real, 3, 6> Matrix36;
    typedef Eigen::Matrix<real, 2, 3> Matrix23;
    typedef Eigen::Matrix<real, 2, 8> Matrix28;
    typedef Eigen::Matrix<real, 3, 8> Matrix38;
    typedef Eigen::Matrix<real, 8, 8> Matrix8;
    typedef Eigen::Matrix<real, 3, 3> Matrix3;
    typedef Eigen::Matrix<real, 2, 2> Matrix2;
    typedef Eigen::Matrix<real, 6, 6> Matrix6;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<real, 2, Eigen::Dynamic> Matrix2X;

    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> VectorX;
	typedef Eigen::Matrix<real, 1, 1> Vector1;
    typedef Eigen::Matrix<real, 2, 1> Vector2;
	typedef Eigen::Matrix<real, 3, 1> Vector3;
    typedef Eigen::Matrix<real, 4, 1> Vector4;
    typedef Eigen::Matrix<real, 6, 1> Vector6;
	typedef Eigen::Matrix<real, 8, 1> Vector8;
	typedef Eigen::Matrix<real, 1, 2> RowVector2;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVectorX;

    template<typename Derived>
    void saveAsCSV(const Eigen::MatrixBase<Derived>& m, const std::string& filename)
    {
        static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::fstream o (filename, std::fstream::out, std::fstream::trunc);
        o << m.format(CSVFormat);
        o.close();
    }

    template <typename T, typename R = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    R square(T t) { return t * t; }

	template <typename T, typename R = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	R cube(T t) { return t * t * t; }
}

//Serialization
namespace boost {
	namespace serialization {
		template<class Archive>
		void serialize(Archive & ar, ar::Vector2 & v, const unsigned int version)
		{
			ar & v.x();
			ar & v.y();
		}
		template<class Archive>
		void serialize(Archive & ar, Eigen::Vector2i & v, const unsigned int version)
		{
			ar & v.x();
			ar & v.y();
		}
		template<class Archive>
		void serialize(Archive & ar, Eigen::Vector3i & v, const unsigned int version)
		{
			ar & v.x();
			ar & v.y();
			ar & v.z();
		}
		template<class Archive>
		void serialize(Archive & ar, Eigen::Array<ar::real, Eigen::Dynamic, Eigen::Dynamic> & v, const unsigned int version)
		{
			if (Archive::is_saving()) {
				//save
                Eigen::Index rows = v.rows(), cols = v.cols();
				ar & rows;
				ar & cols;
				for (int x = 0; x < v.rows(); ++x)
					for (int y = 0; y < v.cols(); ++y)
						ar & v(x, y);
			}
			else {
				//load
				Eigen::Index rows, cols;
				ar & rows;
				ar & cols;
				v.resize(rows, cols);
				for (int x = 0; x < v.rows(); ++x)
					for (int y = 0; y < v.cols(); ++y)
						ar & v(x, y);
			}
		}
		template<class Archive>
		void serialize(Archive & ar, ar::VectorX & v, const unsigned int version)
		{
			if (Archive::is_saving()) {
				//save
				Eigen::Index rows = v.rows();
				ar & rows;
				for (int x = 0; x < v.rows(); ++x)
						ar & v[x];
			}
			else {
				//load
				Eigen::Index rows;
				ar & rows;
				v.resize(rows);
				for (int x = 0; x < v.rows(); ++x)
						ar & v[x];
			}
		}
	} // namespace serialization
} // namespace boost

//Enable/disable sparse matrix support
//Disabling that greatly improves compilation times
#define SOFT_BODY_SUPPORT_SPARSE_MATRICES 1