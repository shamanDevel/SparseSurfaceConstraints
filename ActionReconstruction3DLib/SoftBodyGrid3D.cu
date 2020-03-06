#include "SoftBodyGrid3D.h"

#include <cinder/app/AppBase.h>
#include <cinder/CinderGlm.h>
#include <Eigen/Dense>

#include "Utils.h"
#include "Integration3D.h"
#include "CommonKernels.h"
#include "DebugUtils.h"
#include "CudaTimer.h"

//For testing: set to 1 to enforce a symmetric matrix in the CG
//If 0, small unsymmetries of a few ulps are in the matrix due to the ordering of the operations
//If 1, the upper and lower triangular parts are averaged to create a numerically exact symmetric matrix
#define MAKE_NEWMARK_SYMMETRIC 0

namespace ar3d
{
    int3 SoftBodyGrid3D::offsets[8] = {
            make_int3(-1, -1, -1),
            make_int3(0, -1, -1),
            make_int3(-1, 0, -1),
            make_int3(0, 0, -1),
            make_int3(-1, -1, 0),
            make_int3(0, -1, 0),
            make_int3(-1, 0, 0),
            make_int3(0, 0, 0)
        };

	void SoftBodyGrid3D::Input::assertSizes() const
	{
		assert(referenceSdf_->getGrid() == grid_);
		assert(posToIndex_->getGrid() == grid_);
        assert(numActiveCells_ > 0);
        assert(numActiveNodes_ > 0);
        assert(interpolationVolumeWeights_.rows() == numActiveCells_);
        assert(interpolationBoundaryWeights_.rows() == numActiveCells_);
        assert(surfaceNormals_.rows() == numActiveCells_);
        assert(dirichlet_.rows() == numActiveCells_);
        assert(mapping_.rows() == numActiveCells_);
        assert(referencePositions_.rows() == numActiveNodes_);
        //assert(sparsityPattern_.cols == numActiveNodes_);
        //assert(sparsityPattern_.rows == numActiveNodes_);
	}

	SoftBodyGrid3D::State SoftBodyGrid3D::State::deepClone() const
	{
		State s;
		s.displacements_ = displacements_.deepClone();
		s.velocities_ = velocities_.deepClone();
		s.advectedBoundingBox_ = advectedBoundingBox_;
		s.gridDisplacements_ = gridDisplacements_.deepClone();
        if (advectedSDF_) {
            s.advectedSDF_ = std::make_shared<WorldGridData<real>>(advectedSDF_->getGrid());
            s.advectedSDF_->setDeviceMemory(advectedSDF_->getDeviceMemory().deepClone());
        }
		return s;
	}

	WorldGridPtr SoftBodyGrid3D::createGridFromBoundingBox(
		const ar::geom3d::AABBBox boundingBox, int resolution, int border)
	{
		Eigen::Vector3i offset = (boundingBox.min.array() * resolution).floor().cast<int>() - border;
		Eigen::Vector3i size = ((boundingBox.max - boundingBox.min).array() * resolution).ceil().cast<int>() + (2 * border);
		return std::make_shared<WorldGrid>(resolution, offset, size);
	}

    void SoftBodyGrid3D::setupInput1Grid(Input& input, real3 center, real3 halfsize, int resolution, int border)
    {
        Eigen::Vector3d bbmin(center.x - halfsize.x, center.y - halfsize.y, center.z - halfsize.z);
		Eigen::Vector3d bbmax(center.x + halfsize.x, center.y + halfsize.y, center.z + halfsize.z);
		input.grid_ = createGridFromBoundingBox(ar::geom3d::AABBBox(bbmin, bbmax), resolution, border);
    }

    void SoftBodyGrid3D::setupInput2Sdf(Input& input, std::function<real(real3)> posToSdf)
    {
        input.referenceSdf_ = std::make_shared<WorldGridData<real>>(input.grid_);
		input.referenceSdf_->allocateHostMemory();
        const auto& size = input.grid_->getSize();
        const auto& offset = input.grid_->getOffset();
		for (int z=0; z<size.z(); ++z) for (int y=0; y<size.y(); ++y) for (int x=0; x<size.x(); ++x)
		{
			real3 pos = make_real3(
				(offset.x() + x) / real(input.grid_->getVoxelResolution()),
				(offset.y() + y) / real(input.grid_->getVoxelResolution()),
				(offset.z() + z) / real(input.grid_->getVoxelResolution())
			);
            input.referenceSdf_->atHost(x, y, z) = posToSdf(pos);
        }
    }

    void SoftBodyGrid3D::setupInput3Mapping(Input& input, int diffusionDistance)
    {
        bool noDiffusionLimit = diffusionDistance <= 0;

        const auto& size = input.grid_->getSize();
        const auto& offset = input.grid_->getOffset();
        const int resolution = input.grid_->getVoxelResolution();
        int numActiveNodes = 0;
        int numDiffusedNodes = 0;
        int numInactiveNodes = 0;
        input.posToIndex_ = std::make_shared<WorldGridData<int>>(input.grid_);
		input.posToIndex_->allocateHostMemory();
        input.posToIndex_->getHostMemory().setConstant(noDiffusionLimit ? -1 : -diffusionDistance-1);
        //Loop: active cells -> active nodes
        for (int z=1; z<size.z(); ++z) for (int y=1; y<size.y(); ++y) for (int x=1; x<size.x(); ++x)
		{
            real values[8];
            bool active = false;
            for (int i=0; i<8; ++i) {
                values[i] = input.referenceSdf_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z);
                if (ar::utils::inside(values[i])) active = true;
            }
            if (active)
                for (int i=0; i<8; ++i)
                    input.posToIndex_->atHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z) = 1;
        }
        //Loop: for the inactive nodes, compute the minimal distance to the next active node in the Manhattan norm
        if (!noDiffusionLimit) {
            for (int i = 0; i <= diffusionDistance; ++i)
            {
                for (int z = 0; z < size.z(); ++z) for (int y = 0; y < size.y(); ++y) for (int x = 0; x < size.x(); ++x)
                {
                    int idx = input.posToIndex_->atHost(x, y, z);
                    if (idx > 0) continue; //active
                    if (x > 0) idx = std::max(idx, input.posToIndex_->atHost(x - 1, y, z) - 1);
                    if (y > 0) idx = std::max(idx, input.posToIndex_->atHost(x, y - 1, z) - 1);
                    if (z > 0) idx = std::max(idx, input.posToIndex_->atHost(x, y, z - 1) - 1);
                    if (x < size.x() - 1) idx = std::max(idx, input.posToIndex_->atHost(x + 1, y, z) - 1);
                    if (y < size.y() - 1) idx = std::max(idx, input.posToIndex_->atHost(x, y + 1, z) - 1);
                    if (z < size.z() - 1) idx = std::max(idx, input.posToIndex_->atHost(x, y, z + 1) - 1);
                    if (idx == 0) idx = -1;  //neighbor to active
                    input.posToIndex_->atHost(x, y, z) = idx;
                }
            }
        }
        //Loop: mapping
        std::vector<real3> referencePositions;
        for (int z=0; z<size.z(); ++z) for (int y=0; y<size.y(); ++y) for (int x=0; x<size.x(); ++x)
		{
            int idx = input.posToIndex_->getHost(x, y, z);
            if (idx > 0)
            {
                //active nodes, participates in the elasticity simulation
                numActiveNodes++;
                input.posToIndex_->atHost(x, y, z) = numActiveNodes;
                referencePositions.push_back(
                    make_real3(
                    (offset.x() + x) / real(resolution),
                        (offset.y() + y) / real(resolution),
                        (offset.z() + z) / real(resolution)
                    ));
            } else if (idx <= -diffusionDistance - 1 && !noDiffusionLimit)
            {
                //too far outside, not needed in the diffusion
                numInactiveNodes++;
                input.posToIndex_->atHost(x, y, z) = 0;
                //input.referenceSdf_->atHost(x, y, z) = std::numeric_limits<real>::infinity();
            } else
            {
                //participates in the diffusion
                numDiffusedNodes++;
                input.posToIndex_->atHost(x, y, z) = -numDiffusedNodes;
            }
        }
        input.posToIndex_->copyHostToDevice();
        input.numActiveNodes_ = numActiveNodes;
        input.numDiffusedNodes_ = numDiffusedNodes;
        input.referencePositions_ = Vector3X(numActiveNodes);
        input.referencePositions_.copyFromHost(referencePositions.data());
        input.referenceSdf_->copyHostToDevice();
        CI_LOG_I("num active nodes: " << numActiveNodes << ", num diffused nodes: " << numDiffusedNodes << ", num inactive nodes: " << numInactiveNodes);
    }

    void SoftBodyGrid3D::setupInput4CellData(Input& input, bool enableDirichlet, real3 dirichletCenter,
        real3 dirichletHalfsize, bool integralsSampled)
    {
        const auto& size = input.grid_->getSize();
        const auto& offset = input.grid_->getOffset();
        const int resolution = input.grid_->getVoxelResolution();
        int numActiveCells = 0;
        std::vector<char> dirichlet;
        std::vector<int4> mapping;
        std::vector<real8> volumeWeights;
        std::vector<real8> surfaceWeights;
        std::vector<real3> surfaceNormals;
		std::vector<real8> cellSdfs;
        real3 dirichletMin = dirichletCenter - dirichletHalfsize;
        real3 dirichletMax = dirichletCenter + dirichletHalfsize;
        input.hasDirichlet_ = false;
		real3 centerOfMass = make_real3(0, 0, 0);
		real totalVolume = 0;
        for (int z=1; z<size.z(); ++z) for (int y=1; y<size.y(); ++y) for (int x=1; x<size.x(); ++x)
		{
            real values[8];
            bool active = false;
            for (int i=0; i<8; ++i) {
                values[i] = input.referenceSdf_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z);
                if (ar::utils::inside(values[i])) active = true;
            }
            if (!active) continue;
            numActiveCells++;

            //mapping
            int4 map = make_int4(
                input.posToIndex_->getHost(x + offsets[0].x, y + offsets[0].y, z + offsets[0].z) - 1,
                input.posToIndex_->getHost(x + offsets[2].x, y + offsets[2].y, z + offsets[2].z) - 1,
                input.posToIndex_->getHost(x + offsets[4].x, y + offsets[4].y, z + offsets[4].z) - 1,
                input.posToIndex_->getHost(x + offsets[6].x, y + offsets[6].y, z + offsets[6].z) - 1
            );
            mapping.push_back(map);
            assert(map.x + 2 == input.posToIndex_->getHost(x + offsets[1].x, y + offsets[1].y, z + offsets[1].z) && "Invariant '+x follows -x' violated");
            assert(map.y + 2 == input.posToIndex_->getHost(x + offsets[3].x, y + offsets[3].y, z + offsets[3].z) && "Invariant '+x follows -x' violated");
            assert(map.z + 2 == input.posToIndex_->getHost(x + offsets[5].x, y + offsets[5].y, z + offsets[5].z) && "Invariant '+x follows -x' violated");
            assert(map.w + 2 == input.posToIndex_->getHost(x + offsets[7].x, y + offsets[7].y, z + offsets[7].z) && "Invariant '+x follows -x' violated");

            //interpolation weights
            Integration3D::InputPhi_t interpPhi;
            interpPhi.first.x =  values[0];
            interpPhi.first.y =  values[1];
            interpPhi.first.z =  values[2];
            interpPhi.first.w =  values[3];
            interpPhi.second.x = values[4];
            interpPhi.second.y = values[5];
            interpPhi.second.z = values[6];
            interpPhi.second.w = values[7];
            real h = static_cast<real>(input.grid_->getVoxelSize());
            Integration3D::InterpolationWeight_t wx = integralsSampled
                ? Integration3D::volumeIntegralSampled(interpPhi, h, 1000)
                : Integration3D::volumeIntegralLinear(interpPhi, h);
            real volume = wx.first.x + wx.first.y + wx.first.z + wx.first.w + wx.second.x + wx.second.y + wx.second.z + wx.second.w;
			totalVolume += volume;
        	assert(volume > 0);
            volumeWeights.push_back(wx);
            auto sx = Integration3D::surfaceIntegral(interpPhi, h);
            surfaceWeights.push_back(sx);
            assert(isfinite(wx.first.x)); assert(isfinite(wx.first.y)); assert(isfinite(wx.first.z)); assert(isfinite(wx.first.w));
            assert(isfinite(wx.second.x)); assert(isfinite(wx.second.y)); assert(isfinite(wx.second.z)); assert(isfinite(wx.second.w));
            assert(isfinite(sx.first.x)); assert(isfinite(sx.first.y)); assert(isfinite(sx.first.z)); assert(isfinite(sx.first.w));
            assert(isfinite(sx.second.x)); assert(isfinite(sx.second.y)); assert(isfinite(sx.second.z)); assert(isfinite(sx.second.w));

			//cell SDF values
			cellSdfs.push_back(interpPhi);

            //normal
            float3 normal = make_float3(
                (values[1]+values[3]+values[5]+values[7])/4 - (values[0]+values[2]+values[4]+values[6])/4,
                (values[2]+values[3]+values[6]+values[7])/4 - (values[0]+values[1]+values[4]+values[5])/4,
                (values[4]+values[5]+values[6]+values[7])/4 - (values[0]+values[1]+values[2]+values[3])/4
            );
            normal = safeNormalize(normal);
            surfaceNormals.push_back(make_real4(normal.x, normal.y, normal.z, 0));

            //dirichlet
            if (enableDirichlet) {
                real3 cellMin = make_real3(
				    (offset.x() + x - 1) / real(resolution),
				    (offset.y() + y - 1) / real(resolution),
				    (offset.z() + z - 1) / real(resolution)
			    );
                real3 cellMax = make_real3(
				    (offset.x() + x) / real(resolution),
				    (offset.y() + y) / real(resolution),
				    (offset.z() + z) / real(resolution)
			    );
                bool isDirichlet = all(cellMax >= dirichletMin) && all(dirichletMax >= cellMin);
                dirichlet.push_back(isDirichlet);
                if (isDirichlet) input.hasDirichlet_ = true;
            }
            else
                dirichlet.push_back(false);

			//center of mass
			real weights[] = { wx.first.x, wx.first.y, wx.first.z, wx.first.w, wx.second.x, wx.second.y, wx.second.z, wx.second.w };
			for (int i = 0; i < 8; ++i)
				centerOfMass += weights[i] * make_real3(
					(offset.x() + x + offsets[i].x) / real(resolution),
					(offset.y() + y + offsets[i].y) / real(resolution),
					(offset.z() + z + offsets[i].z) / real(resolution));
        }
        input.numActiveCells_ = numActiveCells;
        input.mapping_ = Vector4Xi(numActiveCells);
        input.mapping_.copyFromHost(mapping.data());
        input.dirichlet_ = VectorXc(numActiveCells);
        input.dirichlet_.copyFromHost(dirichlet.data());
        input.interpolationVolumeWeights_ = Vector8X(numActiveCells);
        input.interpolationVolumeWeights_.copyFromHost(volumeWeights.data());
        input.interpolationBoundaryWeights_ = Vector8X(numActiveCells);
        input.interpolationBoundaryWeights_.copyFromHost(surfaceWeights.data());
        input.surfaceNormals_ = Vector3X(numActiveCells);
        input.surfaceNormals_.copyFromHost(surfaceNormals.data());
		input.cellSdfs_ = Vector8X(numActiveCells);
		input.cellSdfs_.copyFromHost(cellSdfs.data());
		input.centerOfMass_ = centerOfMass / totalVolume;

		//find index of free nodes that is closest to the center of mass
		std::vector<real3> referencePositions(input.referencePositions_.size());
		input.referencePositions_.copyToHost(&referencePositions[0]);
		input.centerOfMassIndex_ = 0;
		float minCoMDistance = FLT_MAX;
		for (int i=0; i<referencePositions.size(); ++i)
		{
			float d = lengthSquared3(referencePositions[i] - input.centerOfMass_);
			if (d < minCoMDistance)
			{
				minCoMDistance = d;
				input.centerOfMassIndex_ = i;
			}
		}

        CI_LOG_I("num active cells: " << numActiveCells);
		CI_LOG_I("center of mass: (" << input.centerOfMass_.x << "," << input.centerOfMass_.y << "," << input.centerOfMass_.z << ")");
		CI_LOG_I("total volume: " << totalVolume);
    }

    void SoftBodyGrid3D::setupInput5SparsityPattern(Input& input)
    {
		typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
        SPattern pattern;
		pattern.rows = input.numActiveNodes_;
		pattern.cols = input.numActiveNodes_;

        //create entry set
        typedef std::pair<int, int> entry_t;
        std::set<entry_t> entries;

        const auto& size = input.grid_->getSize();
        for (int z=1; z<size.z(); ++z) for (int y=1; y<size.y(); ++y) for (int x=1; x<size.x(); ++x)
		{
            real values[8];
            bool active = false;
            for (int i=0; i<8; ++i) {
                values[i] = input.referenceSdf_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z);
                if (ar::utils::inside(values[i])) active = true;
            }
            if (!active) continue;

            int indices[8];
            for (int i=0; i<8; ++i) {
                indices[i] = input.posToIndex_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z) - 1;
                assert(indices[i] >= 0);
            }

            for (int i=0; i<8; ++i)
                for (int j=0; j<8; ++j)
                    entries.insert(std::make_pair(indices[i], indices[j]));
        }

        SMatrix3x3::StorageIndex nnz = static_cast<SMatrix3x3::StorageIndex>(entries.size());
        pattern.nnz = nnz;

        //allocate indices on the host
        std::vector<SMatrix3x3::StorageIndex> JA(pattern.rows + 1, 0); //outer
        std::vector<SMatrix3x3::StorageIndex> IA; IA.reserve(nnz); //inner

        //loop through all sorted entries and build indices
        entry_t lastEntry(-1,-1);
        for (const entry_t& e : entries)
        {
            //assert sorted
            assert(lastEntry.first < e.first || (lastEntry.first==e.first && lastEntry.second<e.second));
            lastEntry = e;
            //increment outer index, add inner index
            JA[lastEntry.first + 1]++;
            IA.push_back(lastEntry.second);
        }
		assert(IA.size() == nnz);
        for (int i=0; i<pattern.rows; ++i)
            JA[i+1] += JA[i]; //prefix sum

        //copy to device
		pattern.JA = SPattern::IndexVector(pattern.rows + 1);
        pattern.JA.copyFromHost(JA.data());
		pattern.IA = SPattern::IndexVector(nnz);
		pattern.IA.copyFromHost(IA.data());

		CI_LOG_I("Sparsity pattern created, matrix size: " << pattern.rows << ", non-zeros: " << nnz
			<< " (" << (100.0*nnz / pattern.rows / pattern.rows) << "%, avg " << (real(nnz)/pattern.rows) << " per row)");
		pattern.assertValid();
		input.sparsityPattern_ = pattern;
    }

	SoftBodyGrid3D::Precomputed SoftBodyGrid3D::allocatePrecomputed(const Input& input)
	{
		Precomputed p;
		p.lumpedMass_ = VectorX(input.numActiveNodes_); 
	    p.lumpedMass_.setZero();
        p.bodyForces_ = Vector3X(input.numActiveNodes_); 
	    p.bodyForces_.setZero();
		return p;
	}

	SoftBodyGrid3D::State SoftBodyGrid3D::allocateState(const Input& input)
	{
		State s;
		s.displacements_ = Vector3X(input.numActiveNodes_);
		s.displacements_.setZero();
		s.velocities_ = Vector3X(input.numActiveNodes_);
		s.velocities_.setZero();
		const auto& size = input.grid_->getSize();
		s.gridDisplacements_ = WorldGridData<real3>::DeviceArray_t(size.x(), size.y(), size.z());
		return s;
	}

    SoftBodyGrid3D::Input SoftBodyGrid3D::createBar(const InputBarSettings& settings)
	{
		Input input;
		
        setupInput1Grid(input, settings.center, settings.halfsize, settings.resolution);

        std::function<real(real3)> posToSdf = [&settings](real3 pos)
        {
            real val = 0;
            if (all(pos >= settings.center - settings.halfsize) && all(pos <= settings.center + settings.halfsize))
			{
				val = -1e10;
				val = std::max(val, settings.center.x - settings.halfsize.x - pos.x);
				val = std::max(val, pos.x - settings.center.x - settings.halfsize.x);
				val = std::max(val, settings.center.y - settings.halfsize.y - pos.y);
				val = std::max(val, pos.y - settings.center.y - settings.halfsize.y);
				val = std::max(val, settings.center.z - settings.halfsize.z - pos.z);
				val = std::max(val, pos.z - settings.center.z - settings.halfsize.z);
			} else
			{
				real3 closestPoint = clamp(pos, settings.center - settings.halfsize, settings.center + settings.halfsize);
				val = length(closestPoint - pos);
			}
			if (abs(val) < settings.zeroCutoff) val = 0;
            return val;
        };
        setupInput2Sdf(input, posToSdf);

        setupInput3Mapping(input, settings.diffusionDistance);

        setupInput4CellData(input,
            settings.enableDirichlet, settings.centerDirichlet, settings.halfsizeDirichlet, settings.sampleIntegrals);

        setupInput5SparsityPattern(input);

		setupInput6DiffusionMatrix(input);

		input.assertSizes();
		return input;
	}

	SoftBodyGrid3D::Input SoftBodyGrid3D::createTorus(const InputTorusSettings& settings)
	{
		Input input;

		real3 halfsize;
		real3 orientation = make_real3(settings.orientation.x, settings.orientation.y, settings.orientation.z);
		halfsize.x = sqrt(1 - ar::square(dot3(orientation, make_real3(1, 0, 0)))) * settings.outerRadius + settings.innerRadius;
		halfsize.y = sqrt(1 - ar::square(dot3(orientation, make_real3(0, 1, 0)))) * settings.outerRadius + settings.innerRadius;
		halfsize.z = sqrt(1 - ar::square(dot3(orientation, make_real3(0, 0, 1)))) * settings.outerRadius + settings.innerRadius;
        halfsize.w = 0;
		setupInput1Grid(input, settings.center, halfsize, settings.resolution);

		glm::vec3 a(0, 0, 1);
		glm::vec3 b(settings.orientation.x, settings.orientation.y, settings.orientation.z);
		glm::vec3 v = glm::cross(b, a);
		float angle = acos(glm::dot(b, a) / (glm::length(b) * glm::length(a)));
		glm::mat4 rotmat = glm::rotate(angle, v);
		std::function<real(real3)> posToSdf = [&settings, rotmat](real3 pos)
		{
			real val = 0;

			pos -= settings.center; //move towards the center
			glm::vec4 pos4(pos.x, pos.y, pos.z, 1);
			pos4 = rotmat * pos4; //rotate

			val = ar::square(sqrt(pos4.x*pos4.x + pos4.y*pos4.y) - settings.outerRadius) + pos4.z*pos4.z - ar::square(settings.innerRadius);

            val *= settings.resolution;
			if (abs(val) < settings.zeroCutoff) val = 0;
			return val;
		};
		setupInput2Sdf(input, posToSdf);

		setupInput3Mapping(input, settings.diffusionDistance);

		setupInput4CellData(input,
			settings.enableDirichlet, settings.centerDirichlet, settings.halfsizeDirichlet, settings.sampleIntegrals);

		setupInput5SparsityPattern(input);

		setupInput6DiffusionMatrix(input);

		input.assertSizes();
		return input;
	}

	SoftBodyGrid3D::Input SoftBodyGrid3D::createFromSdf(const InputSdfSettings & settings, WorldGridRealDataPtr referenceSdf)
	{
		Input input;
		input.grid_ = referenceSdf->getGrid();
		input.referenceSdf_ = referenceSdf;

		for (size_t j = 0; j < input.referenceSdf_->getHostMemory().size(); ++j) {
			if (abs(input.referenceSdf_->getHostMemory()[j]) < settings.zeroCutoff)
				input.referenceSdf_->getHostMemory()[j] = 0; //This step is essential for stability
		}

		if (settings.filledCells)
		{
			WorldGridData<real>::HostArray_t copy = input.referenceSdf_->getHostMemory();
			copy.setConstant(real(+1));
			const auto size = input.referenceSdf_->getGrid()->getSize();
			//mark complete grid cells
			for (int z = 1; z < size.z(); ++z) for (int y = 1; y < size.y(); ++y) for (int x = 1; x < size.x(); ++x)
			{
				real values[8];
				bool active = false;
				for (int i = 0; i < 8; ++i) {
					values[i] = input.referenceSdf_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z);
					if (ar::utils::inside(values[i])) active = true;
				}
				if (!active) continue;

				Integration3D::InputPhi_t interpPhi;
				interpPhi.first.x = values[0];
				interpPhi.first.y = values[1];
				interpPhi.first.z = values[2];
				interpPhi.first.w = values[3];
				interpPhi.second.x = values[4];
				interpPhi.second.y = values[5];
				interpPhi.second.z = values[6];
				interpPhi.second.w = values[7];
				real h = real(1);
				Integration3D::InterpolationWeight_t wx = settings.sampleIntegrals
					? Integration3D::volumeIntegralSampled(interpPhi, h, 1000)
					: Integration3D::volumeIntegral(interpPhi, h);
				real volume = wx.first.x + wx.first.y + wx.first.z + wx.first.w + wx.second.x + wx.second.y + wx.second.z + wx.second.w;

				if (volume > 0.5)
				{
					for (int i = 0; i < 8; ++i)
						copy[input.referenceSdf_->toLinear(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z)] = 0;
				}
			}
			//repair inner values
			for (int z = 1; z < size.z() - 1; ++z) for (int y = 1; y < size.y() - 1; ++y) for (int x = 1; x < size.x() - 1; ++x)
			{
				bool inner = true;
				for (int iz = z - 1; iz <= z + 1; ++iz) for (int iy = y - 1; iy <= y + 1; ++iy) for (int ix = x - 1; ix <= x + 1; ix++)
				{
					if (copy[input.referenceSdf_->toLinear(ix, iy, iz)] == +1) inner = false;
				}
				if (inner)
				{
					copy[input.referenceSdf_->toLinear(x, y, z)] = -1;
				}
			}
			input.referenceSdf_->getHostMemory() = copy;
			//Test / Validation
			for (int z = 1; z < size.z(); ++z) for (int y = 1; y < size.y(); ++y) for (int x = 1; x < size.x(); ++x)
			{
				int numInside = 0;
				bool active = false;
				for (int i = 0; i < 8; ++i) {
					real val = input.referenceSdf_->getHost(x + offsets[i].x, y + offsets[i].y, z + offsets[i].z);
					if (ar::utils::insideEq(val)) numInside++;
					if (ar::utils::inside(val)) active = true;
				}
				if (numInside != 0 && numInside != 8 && active)
				{
					CI_LOG_E("If filledCells==true, each cell should be completely filled or completely empty, but the values are:"
						<< " " << input.referenceSdf_->getHost(x + offsets[0].x, y + offsets[0].y, z + offsets[0].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[1].x, y + offsets[1].y, z + offsets[1].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[2].x, y + offsets[2].y, z + offsets[2].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[3].x, y + offsets[3].y, z + offsets[3].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[4].x, y + offsets[4].y, z + offsets[4].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[5].x, y + offsets[5].y, z + offsets[5].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[6].x, y + offsets[6].y, z + offsets[6].z)
						<< " " << input.referenceSdf_->getHost(x + offsets[7].x, y + offsets[7].y, z + offsets[7].z));
				}
			}
		}
		input.referenceSdf_->copyHostToDevice();

		setupInput3Mapping(input, settings.diffusionDistance);

		setupInput4CellData(input,
			settings.enableDirichlet, settings.centerDirichlet, settings.halfsizeDirichlet, settings.sampleIntegrals);

		setupInput5SparsityPattern(input);

		setupInput6DiffusionMatrix(input);

		input.assertSizes();
		return input;
	}

	SoftBodyGrid3D::Input SoftBodyGrid3D::createFromFile(const InputSdfSettings & settings)
    {
		/*
		WorldGridPtr grid = std::make_shared<WorldGrid>(settings.voxelResolution,
			Eigen::Vector3i(settings.offset.x, settings.offset.y, settings.offset.z),
			Eigen::Vector3i(settings.size.x, settings.size.y, settings.size.z));
		WorldGridRealDataPtr sdf = std::make_shared<WorldGridData<real>>(grid);
		sdf->allocateHostMemory();

		std::ifstream i(settings.file, std::ios::in | std::ios::binary);
		i.seekg(sizeof(int) + 6 * sizeof(float));
		std::vector<float> data(grid->getSize().prod());
		i.read(reinterpret_cast<char*>(&data[0]), sizeof(float) * grid->getSize().prod());
		i.close();
		for (size_t j = 0; j < sdf->getHostMemory().size(); ++j) {
			sdf->getHostMemory()[j] = static_cast<real>(data[j]);
		}
		*/
		std::ifstream i(settings.file, std::ios::in | std::ios::binary);
		if (!i.is_open())
			throw std::exception("Unable to open SDF file");
		WorldGridRealDataPtr sdf = WorldGridData<real>::load(i);
		i.close();

		return SoftBodyGrid3D::createFromSdf(settings, sdf);
    }

	//---------------------------------------------
	// The actual instances:
	// They only store the settings for simple access
	// No logic is implemented here
	//---------------------------------------------

    static int debugTimer = 0;

	SoftBodyGrid3D::SoftBodyGrid3D(const Input& input)
		: input_(input)
        , precomputed_(allocatePrecomputed(input))
		, state_(allocateState(input))
	{
		allocateTemporary(input);

		//fill statistics
		statistics_.numElements = input_.numActiveCells_;
		statistics_.numFreeNodes = input_.numActiveNodes_;
		statistics_.numEmptyNodes = input_.grid_->getSize().prod() - input_.numActiveNodes_;
		statistics_.avgEntriesPerRow = input_.sparsityPattern_.nnz / double(input_.sparsityPattern_.rows);

        reset();
	}

	SoftBodyGrid3D::~SoftBodyGrid3D()
	{
	}

	void SoftBodyGrid3D::reset()
	{
        state_.displacements_.setZero();
		state_.velocities_.setZero();
		computeInitialVelocity(input_, settings_, state_.velocities_); CI_LOG_I("Initial velocities applied");
        state_.advectedSDF_ = input_.referenceSdf_;
        state_.advectedBoundingBox_ = computeTransformedBoundingBox(input_, state_);
		state_.gridDisplacements_.setZero();
		diffusionTmp1_.setZero();
		diffusionTmp2_.setZero();
		resetTimings();
        debugTimer = 0;
	}

	void SoftBodyGrid3D::solve(bool dynamic, BackgroundWorker2* worker, bool advect)
	{
        resetTemporary();
		CudaTimer timer;

		//1. Forces
		worker->setStatus("Grid: compute forces");
		if (isRecordTimings()) timer.start();
		forces_.inplace() = precomputed_.bodyForces_;
		if (settings_.enableCollision_)
		{
			applyCollisionForces(input_, settings_, state_, forces_);
		}
		if (isRecordTimings()) { timer.stop(); statistics_.collisionForcesTime.push_back(timer.duration()); }
		if (worker->isInterrupted()) return;

		//2. stiffness matrix
		worker->setStatus("Grid: compute stiffness matrix");
		if (isRecordTimings()) timer.start();
		computeStiffnessMatrix(input_, state_, settings_, stiffness_, forces_);
		if (isRecordTimings()) { timer.stop(); statistics_.matrixAssembleTime.push_back(timer.duration()); }
		if (worker->isInterrupted()) return;

#if 0
		//DEBUG
		Eigen::VectorXf forcesEigen = DebugUtils::vectorToEigen(forces_);
		Eigen::MatrixXf stiffnessEigen = DebugUtils::matrixToEigen(stiffness_);
#if 1
		cinder::app::console() << "Grid Force Vector:\n" << forcesEigen.transpose() << std::endl;
		//cinder::app::console() << "Grid Stiffness matrix:\n" << stiffnessEigen << std::endl;
		try
		{
			Eigen::IOFormat CsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "", "", "");
			std::ofstream f("StiffnessMatrix.dat", std::ofstream::out | std::ofstream::trunc);
			f << stiffnessEigen.format(CsvFmt) << std::endl;
			f.close();
		} catch (std::exception ex)
		{
			CI_LOG_EXCEPTION("Unable to save matrix", ex);
		}
#endif
		CI_LOG_I("Is stiffness matrix symmetric? " << (stiffnessEigen.isApprox(stiffnessEigen.transpose(), 1e-5)));
		int rank = stiffnessEigen.colPivHouseholderQr().rank();
		CI_LOG_I("Rank of the stiffness matrix: " << rank << " of " << stiffnessEigen.rows());
		if (rank < stiffnessEigen.rows())
		{
			//check if there are empty rows / columns
			for (int i=0; i<stiffnessEigen.rows(); ++i)
			{
				if (stiffnessEigen.row(i).isZero(1e-5))
				{
					CI_LOG_I("Row " << i << " is zero");
				}
			}
		}
#endif

		//3. Solve
		if (dynamic)
		{
			worker->setStatus("Grid: Newmark compute matrices");
			CommonKernels::newmarkTimeIntegration(
				stiffness_, forces_, precomputed_.lumpedMass_,
				state_.displacements_, state_.velocities_,
				settings_.dampingAlpha_, settings_.dampingBeta_, settings_.timestep_,
				newmarkA_, newmarkB_, settings_.newmarkTheta_);

#if 0
			//DEBUG
			Eigen::VectorXf massEigen(input_.numActiveNodes_); precomputed_.lumpedMass_.copyToHost(massEigen.data());
			Eigen::MatrixXf newmarkAEigen = DebugUtils::matrixToEigen(newmarkA_);
			Eigen::VectorXf newmarkBEigen = DebugUtils::vectorToEigen(newmarkB_);
#if 1
			try
			{
				Eigen::IOFormat CsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "", "", "");
				std::ofstream f("MassVector.dat", std::ofstream::out | std::ofstream::trunc);
				f << massEigen.format(CsvFmt) << std::endl;
				f.close();
			}
			catch (std::exception ex)
			{
				CI_LOG_EXCEPTION("Unable to save matrix", ex);
			}
			try
			{
				Eigen::IOFormat CsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "", "", "");
				std::ofstream f("NewmarkMatrix.dat", std::ofstream::out | std::ofstream::trunc);
				f << newmarkAEigen.format(CsvFmt) << std::endl;
				f.close();
			}
			catch (std::exception ex)
			{
				CI_LOG_EXCEPTION("Unable to save matrix", ex);
			}
			try
			{
				Eigen::IOFormat CsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "", "", "");
				std::ofstream f("NewmarkVector.dat", std::ofstream::out | std::ofstream::trunc);
				f << newmarkBEigen.format(CsvFmt) << std::endl;
				f.close();
			}
			catch (std::exception ex)
			{
				CI_LOG_EXCEPTION("Unable to save matrix", ex);
			}
#endif
			CI_LOG_I("Is the Newmark matrix symmetric? " << (newmarkAEigen.isApprox(newmarkAEigen.transpose(), 1e-5)));
			int rank = newmarkAEigen.colPivHouseholderQr().rank();
			CI_LOG_I("Rank of the Newmark matrix: " << rank << " of " << newmarkAEigen.rows());
#endif

            //Testing
#if MAKE_NEWMARK_SYMMETRIC==1
            newmarkA_ = DebugUtils::makeSymmetric(newmarkA_);
#endif

			worker->setStatus("Grid: CG solve");
			Vector3X currentDisplacement = state_.displacements_ + make_real3(settings_.timestep_) * state_.velocities_; //initial guess
			int iterations = settings_.solverIterations_;
			real tolError = settings_.solverTolerance_;
			if (isRecordTimings()) timer.start();
			CommonKernels::solveCG(newmarkA_, newmarkB_, currentDisplacement, iterations, tolError);
			if (isRecordTimings()) { timer.stop(); statistics_.cgTime.push_back(timer.duration()); statistics_.cgIterations.push_back(iterations); }

            //Testing
            if (settings_.debugSaveMatrices_)
            {
                CI_LOG_E("save matrices");
                DebugUtils::saveToMatlab(newmarkA_, "NewmarkA_" + std::to_string(debugTimer) + ".dat");
                DebugUtils::saveToMatlab(newmarkB_, "NewmarkB_" + std::to_string(debugTimer) + ".dat");
            }
            debugTimer++;

			worker->setStatus("Grid: Newmark compute velocity");
			Vector3X currentVelocity(input_.numActiveNodes_);
			CommonKernels::newmarkComputeVelocity(
				state_.displacements_, state_.velocities_,
				currentDisplacement, currentVelocity,
				settings_.timestep_, settings_.newmarkTheta_);

			state_.displacements_.inplace() = currentDisplacement;
			state_.velocities_.inplace() = currentVelocity;

		} else
		{
			worker->setStatus("Grid: CG solve");
			state_.displacements_.setZero();
			int iterations = settings_.solverIterations_;
			real tolError = settings_.solverTolerance_;
			if (isRecordTimings()) timer.start();
			CommonKernels::solveCG(stiffness_, forces_, state_.displacements_, iterations, tolError);
			if (isRecordTimings()) { timer.stop(); statistics_.cgTime.push_back(timer.duration()); statistics_.cgIterations.push_back(iterations); }
		}

#if 0
		//DEBUG
		Eigen::VectorXf displacementsEigen = DebugUtils::vectorToEigen(state_.displacements_);
		cinder::app::console() << "Grid Solution Displacements:\n" << displacementsEigen.transpose() << std::endl;
		Eigen::VectorXf velocitiesEigen = DebugUtils::vectorToEigen(state_.velocities_);
		cinder::app::console() << "Grid Solution Velocities:\n" << velocitiesEigen.transpose() << std::endl;
#endif

		if (advect)
		{
			//4. Advect the levelset

			//a) compute new bounding box and grid
			worker->setStatus("Grid: Compute new bounding box");
			if (isRecordTimings()) timer.start();
			auto newBox = computeTransformedBoundingBox(input_, state_);
            state_.advectedBoundingBox_ = limitBoundingBox(newBox, state_.advectedBoundingBox_);
			WorldGridPtr grid = createGridFromBoundingBox(state_.advectedBoundingBox_, input_.grid_->getVoxelResolution(), 3);
			if (isRecordTimings()) { timer.stop(); statistics_.gridBoundingBoxTime.push_back(timer.duration()); }

			//b) Diffuse displacements
			worker->setStatus("Grid: Diffuse Displacements");
			if (isRecordTimings()) timer.start();
			diffuseDisplacements(input_, state_, state_.gridDisplacements_, diffusionTmp1_, diffusionTmp2_);
			if (isRecordTimings()) { timer.stop(); statistics_.gridDiffusionTime.push_back(timer.duration()); }

			//c) Advect levelset
			worker->setStatus("Grid: Advect Levelset");
			auto advectedSdf = std::make_shared<WorldGridData<real>>(grid);
			advectedSdf->allocateDeviceMemory();
			if (isRecordTimings()) timer.start();
			advectLevelset(input_, state_.gridDisplacements_, advectedSdf, AdvectionSettings());
			if (isRecordTimings()) { timer.stop(); statistics_.gridAdvectionTime.push_back(timer.duration()); }
			state_.advectedSDF_ = advectedSdf;
		}

		worker->setStatus("Grid: done");
	}

	void SoftBodyGrid3D::updateSettings()
	{
        precomputed_.bodyForces_.setZero();
		precomputed_.lumpedMass_.setZero();

		computeMassMatrix(input_, settings_, precomputed_.lumpedMass_);
		computeBodyForces(input_, settings_, precomputed_.bodyForces_);
		computeInitialVelocity(input_, settings_, state_.velocities_); CI_LOG_I("Initial velocities applied");

		CI_LOG_I("Settings updated, mass matrix and body forces recomputed");
	}

    void SoftBodyGrid3D::allocateTemporary(const Input& input)
    {
        forces_ = Vector3X(input.numActiveNodes_);
		stiffness_ = SMatrix3x3(input.sparsityPattern_);
		newmarkA_ = SMatrix3x3(input.sparsityPattern_);
		newmarkB_ = Vector3X(input.numActiveNodes_);
		const Eigen::Vector3i& size = input.grid_->getSize();
		diffusionTmp1_ = DiffusionRhs(input.numDiffusedNodes_, 1, 3);
		diffusionTmp2_ = DiffusionRhs(input.numDiffusedNodes_, 1, 3);
		diffusionTmp1_.setZero();
		diffusionTmp2_.setZero();
    }

    void SoftBodyGrid3D::resetTemporary()
    {
        forces_.setZero();
		stiffness_.setZero();
    }

}
