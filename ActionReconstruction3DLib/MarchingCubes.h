#pragma once

#include <vector>
#include "Commons3D.h"
#include "WorldGrid.h"

namespace ar3d 
{

class MarchingCubes
{
public:
    /**
     * \brief The information of a vertex that is generated
     * as the tesselation of a grid cell.
     * Each vertex is located on a grid edge and computed by
     * interpolating the two incident cell nodes.
     * -> The real position is given by (1-weight)*position[indexA] + weight*position[indexB]
     */
    struct Vertex
    {
        //The indices of the cell nodes
        int indexA, indexB;
        //The interpolation weight.
        real weight;
    };

    //A single triangle, three Vertices
    struct Triangle
    {
        Vertex v[3];
    };

    //Describes a grid cell: the eight SDF values
    struct Cell
    {
        real val[8];
    };

    /**
     * \brief Polygonizes a single cell.
     * \param cell The eight SDF values of the cell
     * \param isovalue the isovalues
     * \param triangles [Out] at most 5 triangles are written in here
     * \return the number of triangles
     */
    static int polygonize(const Cell& cell, real isovalue, Triangle* triangles);

    /**
     * \brief Polygonizes a whole grid and builds the vertex and index buffer.
     * The vertex positions are not directly written into the vertex buffer,
     * instead they are defined as a pair of indices of the nodes in the grid
     * plus an interpolation weight.
     * Hence, when the grid deforms, only the positions of the grid nodes 
     * have to be rewritten and not the whole marching cubes vertex buffer.
     * \param sdf 
     * \param posToIndex  mapping from grid position to linear index+1
     * \param indexBuffer 
     * \param vertexBuffer 
     */
    static void polygonizeGrid(
        WorldGridDataPtr<real> sdf, WorldGridDataPtr<int> posToIndex,
        std::vector<int>& indexBuffer, std::vector<Vertex>& vertexBuffer);

private:

    static real vertexInterpolate(real isovalue, real val1, real val2);

    static const int edgeTable[256];
    static const int triTable[256][16];

    MarchingCubes() = delete;
    ~MarchingCubes() = default;
};
}
