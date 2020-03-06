//#ifndef NDEBUG
//#define CUMAT_EIGEN_SUPPORT 1
//#endif
#include "../SoftBodyGrid3D.h"

#include <cuMat/src/ConjugateGradient.h>

namespace ar3d
{

	typedef WorldGridData<int>::DeviceArray_t PosToIndex_t;

	void SoftBodyGrid3D::setupInput6DiffusionMatrix(Input& input)
	{
		const Eigen::Vector3i& size = input.grid_->getSize();
		const int numDiffused = input.numDiffusedNodes_;

		//validity check
		input.posToIndex_->requireHostMemory();
		const auto valid = [&input](int x, int y, int z)
		{
			return input.posToIndex_->atHost(x, y, z) >= 0;
		};

		//build matrix in coordinate form
		typedef std::pair<std::pair<SMatrix3x3::StorageIndex, SMatrix3x3::StorageIndex>, real> entry_t;
		std::set<entry_t> entries;
		VectorX rhs = VectorX::Zero(numDiffused);
		static const int neighborsX[] = { -1, 1, 0, 0, 0, 0 };
		static const int neighborsY[] = { 0, 0, -1, 1, 0, 0 };
		static const int neighborsZ[] = { 0, 0, 0, 0, -1, 1 };
		for (int z = 0; z < size.z(); ++z) for (int y = 0; y < size.y(); ++y) for (int x = 0; x < size.x(); ++x)
		{
			if (!valid(x, y, z)) {
				int row = -input.posToIndex_->atHost(x, y, z) - 1;
				int c = 0;
				for (int i = 0; i < 6; ++i) {
					int ix = x + neighborsX[i];
					int iy = y + neighborsY[i];
					int iz = z + neighborsZ[i];
					if (ix < 0 || ix >= size.x() || iy < 0 || iy >= size.y() || iz < 0 || iz >= size.z()) continue; //outside, Neumann boundary
					if (!valid(ix, iy, iz)) {
						//inside
						int col = -input.posToIndex_->atHost(ix, iy, iz) - 1;
						entries.insert(std::make_pair(std::make_pair(row, col), -1));
					}
					c++;
				}
				entries.insert(std::make_pair(std::make_pair(row, row), c));
			}
		}

		//allocate indices on the host
		SMatrix3x3::StorageIndex rows = numDiffused;
		SMatrix3x3::StorageIndex nnz = entries.size();
		std::vector<SMatrix3x3::StorageIndex> JA(rows + 1, 0); //outer
		std::vector<SMatrix3x3::StorageIndex> IA; IA.reserve(nnz); //inner
		std::vector<real> Data; Data.reserve(nnz);

		//loop through all sorted entries and build indices
		entry_t lastEntry(std::make_pair(-1, -1), 0);
		for (const entry_t& e : entries)
		{
			//assert sorted
			assert(lastEntry.first.first < e.first.first || (lastEntry.first.first == e.first.first && lastEntry.first.second<e.first.second));
			lastEntry = e;
			//increment outer index, add inner index
			JA[lastEntry.first.first + 1]++;
			IA.push_back(lastEntry.first.second);
			Data.push_back(lastEntry.second);
		}
		assert(IA.size() == nnz);
		for (int i = 0; i<rows; ++i) //prefix sum
			JA[i + 1] += JA[i]; 

		//copy to device
		typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
		SPattern pattern;
		pattern.rows = rows;
		pattern.cols = rows;
		pattern.nnz = nnz;
		pattern.JA = SPattern::IndexVector(rows + 1);
		pattern.JA.copyFromHost(JA.data());
		pattern.IA = SPattern::IndexVector(nnz);
		pattern.IA.copyFromHost(IA.data());
        pattern.assertValid();
		SMatrix diffusionMatrix = SMatrix(pattern);
		diffusionMatrix.getData().copyFromHost(Data.data());

//#ifndef NDEBUG
//		MatrixX denseMat = diffusionMatrix;
//		Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> emat = denseMat.toEigen();
//		try
//		{
//			Eigen::IOFormat CsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "", "", "");
//			std::ofstream f("DiffusionMatrix.dat", std::ofstream::out | std::ofstream::trunc);
//			f << emat.format(CsvFmt) << std::endl;
//			f.close();
//		}
//		catch (std::exception ex)
//		{
//			CI_LOG_EXCEPTION("Unable to save matrix", ex);
//		}
//#endif

		//init CG
		input.diffusionCG_ = cuMat::ConjugateGradient<SMatrix>(diffusionMatrix);
		input.diffusionCG_.setMaxIterations(100);

		CI_LOG_D("Diffusion matrix created, rows=cols=" << rows << ", nnz=" << nnz);
	}

	__global__ void GridDiffusionFillRhsKernel(dim3 size, const PosToIndex_t posToIndex, const Vector3X displacements, SoftBodyGrid3D::DiffusionRhs rhs)
	{
		static const int neighborsX[] = { -1, 1, 0, 0, 0, 0 };
		static const int neighborsY[] = { 0, 0, -1, 1, 0, 0 };
		static const int neighborsZ[] = { 0, 0, 0, 0, -1, 1 };
		CUMAT_KERNEL_3D_LOOP(x, y, z, size)
			int row = - posToIndex.coeff(x, y, z, -1) - 1;
			if (row <= 0) continue;
			real3 val = make_real3(0);
			for (int i = 0; i < 6; ++i) {
				int ix = x + neighborsX[i];
				int iy = y + neighborsY[i];
				int iz = z + neighborsZ[i];
				if (ix < 0 || ix >= size.x || iy < 0 || iy >= size.y || iz < 0 || iz >= size.z) continue;
				int col = posToIndex.coeff(ix, iy, iz, -1) - 1;
				if (col >= 0)
					val += displacements.coeff(col, 0, 0, -1);
			}
			rhs.coeff(row, 0, 0, -1) = val.x;
			rhs.coeff(row, 0, 1, -1) = val.y;
			rhs.coeff(row, 0, 2, -1) = val.z;
		CUMAT_KERNEL_3D_LOOP_END
	}

	__global__ void GridDiffusionMappingKernel(dim3 size, const PosToIndex_t posToIndex, const Vector3X displacements, SoftBodyGrid3D::DiffusionRhs diffusedDisp, WorldGridData<real3>::DeviceArray_t gridDisp)
	{
		CUMAT_KERNEL_3D_LOOP(x, y, z, size)
			int i = posToIndex.coeff(x, y, z, -1);
			if (i > 0) {
				gridDisp.coeff(x, y, z, -1) = displacements.getRawCoeff(i - 1);
			} 
			else if (i < 0)
			{
				gridDisp.coeff(x, y, z, -1) = make_real3(
					diffusedDisp.coeff(-i - 1, 0, 0, -1),
					diffusedDisp.coeff(-i - 1, 0, 1, -1),
					diffusedDisp.coeff(-i - 1, 0, 2, -1)
				);
			}
            else
            {
                gridDisp.coeff(x, y, z, -1) = make_real3(0, 0, 0); //outside
            }
		CUMAT_KERNEL_3D_LOOP_END
	}

	void SoftBodyGrid3D::diffuseDisplacements(const Input& input, const State& state,
		WorldGridData<real3>::DeviceArray_t& gridDisp, DiffusionRhs& tmp1, DiffusionRhs& tmp2)
	{
		//Build RHS
		cuMat::Context& ctx = cuMat::Context::current();
		const Eigen::Vector3i& size = input.grid_->getSize();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(size.x(), size.y(), size.z(), GridDiffusionFillRhsKernel);
		GridDiffusionFillRhsKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.posToIndex_->getDeviceMemory(), state.displacements_, tmp1);
		CUMAT_CHECK_ERROR();
		CI_LOG_D("Diffusion: RHS assembled");

		//Perform diffusion

#if 1
		//Solve batched CG
		tmp2 = input.diffusionCG_.solveWithGuess(tmp1, tmp2);
		CI_LOG_D("Diffusion CG finished after " << input.diffusionCG_.iterations() << " iterations with an error of " << input.diffusionCG_.error());
#else
		//Solve CG for every dimension
		VectorX tmp(tmp1.rows());
		for (int i=0; i<3; ++i)
		{
			tmp = input.diffusionCG_.solveWithGuess(tmp1.slice(i), tmp2.slice(i));
			tmp2.slice(i) = tmp;
			CI_LOG_I("Diffusion: Dimension " << i << ", CG finished after " << input.diffusionCG_.iterations() << " iterations with an error of " << input.diffusionCG_.error());
		}
#endif

		//Map back to the whole grid
		cfg = ctx.createLaunchConfig3D(size.x(), size.y(), size.z(), GridDiffusionMappingKernel);
		GridDiffusionMappingKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.posToIndex_->getDeviceMemory(), state.displacements_, tmp2, gridDisp);
		CUMAT_CHECK_ERROR();
		CI_LOG_D("Diffusion: Displacements mapped to the whole grid");
	}

}
