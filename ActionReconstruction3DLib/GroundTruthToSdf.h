#pragma once

#include "WorldGrid.h"
#include "InputConfig.h"

namespace ar3d
{
	WorldGridRealDataPtr groundTruthToSdf(ar3d::InputConfigPtr config, int frame, int resolution);
}