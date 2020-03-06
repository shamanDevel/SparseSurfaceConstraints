#pragma once

#include <Eigen/StdVector>
#include <vector>
#include <map>

#include <boost/serialization/access.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/version.hpp>

#include "SoftBodySimulation.h"
#include "SoftBodyMesh2D.h"
#include "SoftBodyGrid2D.h"

namespace ar {

	//This struct contains the results of a 2D soft body simulation, as used as input to the reconstruction
	struct SoftBody2DResults
	{
	private:
		friend class boost::serialization::access;
	public:
		//general
		SoftBodySimulation::Settings settings_;
		int numSteps_ = 0;

		//mesh
		SoftBodyMesh2D::Vector2List meshReferencePositions_;
		SoftBodyMesh2D::Vector3iList meshReferenceIndices_;
		std::map<int, Vector2, std::less<>, Eigen::aligned_allocator<std::pair<const int, Vector2> > > meshDirichletBoundaries_;
		std::map<int, Vector2, std::less<>, Eigen::aligned_allocator<std::pair<const int, Vector2> > > meshNeumannBoundaries_;
		std::vector<SoftBodyMesh2D::Vector2List> meshResultsDisplacement_;
		std::vector<SoftBodyMesh2D::Vector2List> meshResultsVelocities_;

		//grid
		int gridResolution_ = 0;
		SoftBodyGrid2D::grid_t gridReferenceSdf_;
		SoftBodyGrid2D::GridSettings gridSettings_;
		std::map<std::pair<int, int>, Vector2, std::less<>, Eigen::aligned_allocator<std::pair<const std::pair<int, int>, Vector2> > > gridDirichletBoundaries_;
		std::map<std::pair<int, int>, Vector2, std::less<>, Eigen::aligned_allocator<std::pair<const std::pair<int, int>, Vector2> > > gridNeumannBoundaries_;
		std::vector<SoftBodyGrid2D::grid_t, Eigen::aligned_allocator<SoftBodyGrid2D::grid_t> > gridResultsSdf_;
		std::vector<SoftBodyGrid2D::grid_t, Eigen::aligned_allocator<SoftBodyGrid2D::grid_t> > gridResultsUx_;
		std::vector<SoftBodyGrid2D::grid_t, Eigen::aligned_allocator<SoftBodyGrid2D::grid_t> > gridResultsUy_;
		std::vector<VectorX, Eigen::aligned_allocator<VectorX>> gridResultsUxy_;
        std::vector<Matrix2X> gridPartialObservations_;

        void initMeshReference(const SoftBodyMesh2D& simulation);
        void initGridReference(const SoftBodyGrid2D& simulation);

	private:
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & settings_;
			ar & numSteps_;
			
			ar & meshReferencePositions_;
			ar & meshReferenceIndices_;
			ar & meshDirichletBoundaries_;
			ar & meshNeumannBoundaries_;
			ar & meshResultsDisplacement_;
			if (version >= 1) ar & meshResultsVelocities_;

			ar & gridResolution_;
			ar & gridReferenceSdf_;
			ar & gridNeumannBoundaries_;
            ar & gridDirichletBoundaries_;
			ar & gridResultsSdf_;
			ar & gridResultsUxy_;
		}
	};

}
BOOST_CLASS_VERSION(ar::SoftBody2DResults, 1)