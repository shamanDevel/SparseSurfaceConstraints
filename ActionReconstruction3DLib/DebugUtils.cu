#include "DebugUtils.h"

#include <fstream>

Eigen::MatrixXf ar3d::DebugUtils::matrixToEigen(const SMatrix3x3& mat)
{
	std::vector<SMatrix3x3::StorageIndex> IA(mat.nnz());
	std::vector<SMatrix3x3::StorageIndex> JA(mat.rows() + 1);
	std::vector<real3x3> data(mat.nnz());

	mat.getSparsityPattern().JA.copyToHost(&JA[0]);
	mat.getSparsityPattern().IA.copyToHost(&IA[0]);
	mat.getData().copyToHost(&data[0]);

	Eigen::MatrixXf dense = Eigen::MatrixXf::Zero(3 * mat.rows(), 3 * mat.cols());
	for (int row = 0; row<mat.rows(); ++row)
	{
		for (int k = JA[row]; k<JA[row + 1]; ++k)
		{
			int col = IA[k];
			real3x3 e = data[k];
			dense(3 * row + 0, 3 * col + 0) = e.r1.x;
			dense(3 * row + 0, 3 * col + 1) = e.r1.y;
			dense(3 * row + 0, 3 * col + 2) = e.r1.z;
			dense(3 * row + 1, 3 * col + 0) = e.r2.x;
			dense(3 * row + 1, 3 * col + 1) = e.r2.y;
			dense(3 * row + 1, 3 * col + 2) = e.r2.z;
			dense(3 * row + 2, 3 * col + 0) = e.r3.x;
			dense(3 * row + 2, 3 * col + 1) = e.r3.y;
			dense(3 * row + 2, 3 * col + 2) = e.r3.z;
		}
	}
	return dense;
}

Eigen::VectorXf ar3d::DebugUtils::vectorToEigen(const Vector3X& vec)
{
	std::vector<real3> data(vec.rows());
	vec.copyToHost(&data[0]);
	Eigen::VectorXf dense(3 * vec.rows());
	for (int i = 0; i<vec.rows(); ++i)
	{
		dense[3 * i + 0] = data[i].x;
		dense[3 * i + 1] = data[i].y;
		dense[3 * i + 2] = data[i].z;
	}
	return dense;
}

void ar3d::DebugUtils::eigenToMatrix(const Eigen::MatrixXf& src, SMatrix3x3& target)
{
    assert(target.rows() * 3 == src.rows());
    assert(target.cols() * 3 == src.cols());
    std::vector<SMatrix3x3::StorageIndex> IA(target.nnz());
    std::vector<SMatrix3x3::StorageIndex> JA(target.rows() + 1);
    std::vector<real3x3> data(target.nnz(), real3x3(0));

    target.getSparsityPattern().JA.copyToHost(&JA[0]);
    target.getSparsityPattern().IA.copyToHost(&IA[0]);

    for (int row = 0; row<target.rows(); ++row)
    {
        for (int k = JA[row]; k<JA[row + 1]; ++k)
        {
            int col = IA[k];
            real3x3 e;
            e.r1.x = src(3 * row + 0, 3 * col + 0);
            e.r1.y = src(3 * row + 0, 3 * col + 1);
            e.r1.z = src(3 * row + 0, 3 * col + 2);
            e.r2.x = src(3 * row + 1, 3 * col + 0);
            e.r2.y = src(3 * row + 1, 3 * col + 1);
            e.r2.z = src(3 * row + 1, 3 * col + 2);
            e.r3.x = src(3 * row + 2, 3 * col + 0);
            e.r3.y = src(3 * row + 2, 3 * col + 1);
            e.r3.z = src(3 * row + 2, 3 * col + 2);
            data[k] = e;
        }
    }

    target.getData().copyFromHost(data.data());
}

ar3d::SMatrix3x3 ar3d::DebugUtils::eigenToMatrix(const Eigen::MatrixXf& src)
{
    assert(src.rows() % 3 == 0);
    assert(src.cols() % 3 == 0);

	typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
    SPattern pattern;
    pattern.rows = src.rows() / 3;
    pattern.cols = src.cols() / 3;
    pattern.nnz = pattern.rows * pattern.cols;
    std::vector<SPattern::StorageIndex> IA(pattern.nnz);
    std::vector<SPattern::StorageIndex> JA(pattern.rows + 1);
    for (int i = 0; i <= pattern.rows; ++i) JA[i] = pattern.cols * i;
    for (int i = 0; i < pattern.nnz; ++i) IA[i] = i%pattern.cols;
    pattern.IA = SPattern::IndexVector(pattern.nnz);
    pattern.IA.copyFromHost(IA.data());
    pattern.JA = SPattern::IndexVector(pattern.rows + 1);
    pattern.JA.copyFromHost(JA.data());
    SMatrix3x3 target(pattern);

    std::vector<real3x3> data(pattern.nnz, real3x3(0));

    for (int row = 0; row<target.rows(); ++row)
    {
        for (int k = JA[row]; k<JA[row + 1]; ++k)
        {
            int col = IA[k];
            real3x3 e;
            e.r1.x = src(3 * row + 0, 3 * col + 0);
            e.r1.y = src(3 * row + 0, 3 * col + 1);
            e.r1.z = src(3 * row + 0, 3 * col + 2);
            e.r2.x = src(3 * row + 1, 3 * col + 0);
            e.r2.y = src(3 * row + 1, 3 * col + 1);
            e.r2.z = src(3 * row + 1, 3 * col + 2);
            e.r3.x = src(3 * row + 2, 3 * col + 0);
            e.r3.y = src(3 * row + 2, 3 * col + 1);
            e.r3.z = src(3 * row + 2, 3 * col + 2);
            data[k] = e;
        }
    }

    target.getData().copyFromHost(&data[0]);
    return target;
}

void ar3d::DebugUtils::saveToMatlab(const SMatrix3x3& mat, const std::string& filename)
{
    std::vector<SMatrix3x3::StorageIndex> IA(mat.nnz());
    std::vector<SMatrix3x3::StorageIndex> JA(mat.rows() + 1);
    std::vector<real3x3> data(mat.nnz());

    mat.getSparsityPattern().JA.copyToHost(&JA[0]);
    mat.getSparsityPattern().IA.copyToHost(&IA[0]);
    mat.getData().copyToHost(&data[0]);

    std::ofstream o(filename);
    for (int row = 0; row < mat.rows(); ++row)
    {
        for (int k = JA[row]; k < JA[row + 1]; ++k)
        {
            int col = IA[k];
            const real3x3& e = data[k];
            o << (3 * row + 1) << "    " << (3 * col + 1) << "    " << e.r1.x << "\n";
            o << (3 * row + 1) << "    " << (3 * col + 2) << "    " << e.r1.y << "\n";
            o << (3 * row + 1) << "    " << (3 * col + 3) << "    " << e.r1.z << "\n";
            o << (3 * row + 2) << "    " << (3 * col + 1) << "    " << e.r2.x << "\n";
            o << (3 * row + 2) << "    " << (3 * col + 2) << "    " << e.r2.y << "\n";
            o << (3 * row + 2) << "    " << (3 * col + 3) << "    " << e.r2.z << "\n";
            o << (3 * row + 3) << "    " << (3 * col + 1) << "    " << e.r3.x << "\n";
            o << (3 * row + 3) << "    " << (3 * col + 2) << "    " << e.r3.y << "\n";
            o << (3 * row + 3) << "    " << (3 * col + 3) << "    " << e.r3.z << "\n";
        }
    }
    o << (3 * mat.rows()) << "    " << (3 * mat.cols()) << "    0.0" << std::endl;
    o.close();
}

void ar3d::DebugUtils::saveToMatlab(const Vector3X& vec, const std::string& filename)
{
    std::vector<real3> data(vec.rows());
    vec.copyToHost(&data[0]);

    std::ofstream o(filename);
    for (int row = 0; row < vec.rows(); ++row)
    {
        const real3 e = data[row];
        o << e.x << "\n" << e.y << "\n" << e.z << "\n";
    }
    o.close();
}

namespace cuMat
{
    namespace internal
    {
        template<>
        struct TransposeFunctor<ar3d::real3x3>
        {
            /**
            * \Brief Transposes the scalar type.
            * The default implementation does nothing.
            */
            __device__ ar3d::real3x3 operator()(const ar3d::real3x3& val) const { return val.transpose(); }
        };
    }
}

ar3d::SMatrix3x3 ar3d::DebugUtils::makeSymmetric(const SMatrix3x3& A)
{
    SMatrix3x3 ret(A.getSparsityPattern());
    ret = real3x3(0.5f) * (A + A.transpose());
    return ret;
}

void ar3d::DebugUtils::eigenToVector(const Eigen::VectorXf& src, Vector3X& target)
{
    assert(target.rows() * 3 == src.rows());
    std::vector<real3> data(target.rows());
    for (int i = 0; i<target.rows(); ++i)
    {
        data[i].x = src[3 * i + 0];
        data[i].y = src[3 * i + 1];
        data[i].z = src[3 * i + 2];
    }
    target.copyFromHost(data.data());
}

ar3d::Vector3X ar3d::DebugUtils::eigenToVector(const Eigen::VectorXf& src)
{
    assert(src.rows() % 3 == 0);
    Vector3X target(src.rows() / 3);
    eigenToVector(src, target);
    return target;
}
