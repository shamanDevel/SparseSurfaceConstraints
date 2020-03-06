#pragma once

#include <cuda_runtime.h>
#include "../Commons3D.h"

namespace ar3d 
{
    extern __constant__ real cGridB[8][8][3]; //Vertex(interpolation), Basis, Coordinate
    constexpr real dirichletRho = 1e8;

    constexpr int polarDecompositionIterations = 5;

	constexpr real POLAR_DECOMPOSITION_THRESHOLD = 1e-5; //If det(R)<threshold, no rotation is applied
	// (was originally 1e-15)
}