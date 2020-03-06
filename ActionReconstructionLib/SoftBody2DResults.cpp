#include "SoftBody2DResults.h"

void ar::SoftBody2DResults::initMeshReference(const SoftBodyMesh2D & simulation)
{
    meshReferencePositions_ = simulation.getReferencePositions();
    meshReferenceIndices_ = simulation.getTriangles();
    for (int i=0; i<simulation.getNumNodes(); ++i)
    {
        int state = simulation.getNodeStates()[i];
        if (state == SoftBodyMesh2D::DIRICHLET)
        {
            meshDirichletBoundaries_.emplace(i, simulation.getBoundaries()[i]);
        } else if (state == SoftBodyMesh2D::NEUMANN)
        {
            meshNeumannBoundaries_.emplace(i, simulation.getBoundaries()[i]);
        }
    }
}

void ar::SoftBody2DResults::initGridReference(const SoftBodyGrid2D & simulation)
{
    gridResolution_ = simulation.getGridResolution();
    gridReferenceSdf_ = simulation.getSdfReference();
	gridSettings_ = simulation.getGridSettings();
    for (int y=0; y<gridResolution_; ++y)
        for (int x=0; x<gridResolution_; ++x)
        {
            if (simulation.getGridDirichlet()(x, y))
                gridDirichletBoundaries_.emplace(std::make_pair(x, y), Vector2(0, 0));
            else if (simulation.getGridNeumannX()(x, y) != 0
                && simulation.getGridNeumannY()(x, y) != 0)
            {
                gridNeumannBoundaries_.emplace(std::make_pair(x, y),
                    Vector2(simulation.getGridNeumannX()(x, y), simulation.getGridNeumannY()(x, y)));
            }
        }
}
