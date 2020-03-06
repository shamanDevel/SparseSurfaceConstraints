#include "GroundTruthToSdf.h"

#include "SoftBodyGrid3D.h"
#include "helper_math.h"
#include <ObjectLoader.h>

static ar3d::WorldGridRealDataPtr sphereToSdf(const Eigen::Vector3d& center, double radius, int resolution)
{
	//create grid
	Eigen::Vector3d bbmin(center.x() - radius, center.y() - radius, center.z() - radius);
	Eigen::Vector3d bbmax(center.x() + radius, center.y() + radius, center.z() + radius);
	ar3d::WorldGridPtr grid = ar3d::SoftBodyGrid3D::createGridFromBoundingBox(ar::geom3d::AABBBox(bbmin, bbmax), resolution);

	//fill data
	ar3d::WorldGridRealDataPtr sdf = std::make_shared<ar3d::WorldGridData<ar3d::real>>(grid);
	sdf->allocateHostMemory();
	const auto& size = grid->getSize();
	const auto& offset = grid->getOffset();
	for (int z = 0; z < size.z(); ++z) for (int y = 0; y < size.y(); ++y) for (int x = 0; x < size.x(); ++x)
	{
		ar3d::real3 pos = ar3d::make_real3(
			(offset.x() + x) / ar3d::real(grid->getVoxelResolution()),
			(offset.y() + y) / ar3d::real(grid->getVoxelResolution()),
			(offset.z() + z) / ar3d::real(grid->getVoxelResolution())
		);
		sdf->atHost(x, y, z) = resolution * (length3(pos - ar3d::make_real3(center.x(), center.y(), center.z())) - radius);
	}

	return sdf;
}

static float3 closesPointOnTriangle(const std::array<float3, 3> triangle, const float3 &sourcePosition);
static float pointTriangleDistance(float3 a, float3 b, float3 c, float3 pos)
{
	float3 p = closesPointOnTriangle({ a, b, c }, pos);
	float dist = length(p - pos);
	return dist;
}

static int intersect_triangle(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3], double *t, double *u, double *v);
static bool pointTriangleIntersection(float3 a, float3 b, float3 c, float3 pos)
{
	double orig[3] = { pos.x, pos.y, pos.z };
	double dir[3] = { 1, 0,0 };
	double vert0[3] = { a.x, a.y, a.z };
	double vert1[3] = { b.x, b.y, b.z };
	double vert2[3] = { c.x, c.y, c.z };
	double u, v, t;
	if (intersect_triangle(orig, dir, vert0, vert1, vert2, &t, &u, &v))
	{
		if (t >= 0) return true;
	}
	return false;
}

static ar3d::WorldGridRealDataPtr meshToSdf(const std::string& file, int resolution)
{
	ci::TriMeshRef triMesh = ObjectLoader::loadCustomObj(file);

	//create grid
	float3 minPos = make_float3(0, 0, 0);
	float3 maxPos = make_float3(0, 0, 0);
	for (size_t i = 0; i < triMesh->getNumVertices(); ++i)
	{
		float3 pos = make_float3(triMesh->getPositions<3>()[i].x, triMesh->getPositions<3>()[i].y, triMesh->getPositions<3>()[i].z);
		minPos = fminf(minPos, pos);
		maxPos = fmaxf(maxPos, pos);
	}
	int border = 3;

	int3 size = make_int3(roundf(make_float3(resolution) * (maxPos - minPos))) + 2 * make_int3(border);
	int3 offset = make_int3(roundf(make_float3(resolution) * minPos)) - make_int3(border);
	ar3d::WorldGridPtr grid = std::make_shared<ar3d::WorldGrid>(resolution, Eigen::Vector3i(offset.x, offset.y, offset.z), Eigen::Vector3i(size.x, size.y, size.z));
	CI_LOG_I("size=(" << size.x << "," << size.y << "," << size.z << "), offset=(" << offset.x << "," << offset.y << "," << offset.z << ")");

	//fill grid
	const int numTris = triMesh->getNumTriangles();
	ar3d::WorldGridRealDataPtr sdf = std::make_shared<ar3d::WorldGridData<ar3d::real>>(grid);
	sdf->allocateHostMemory();
	int inside = 0, outside = 0;
#pragma omp parallel for
	for (int z = 0; z < size.z; ++z) for (int y = 0; y < size.y; ++y) for (int x = 0; x < size.x; ++x)
	{
		float3 pos = make_float3(
			(offset.x + x) / ar3d::real(grid->getVoxelResolution()),
			(offset.y + y) / ar3d::real(grid->getVoxelResolution()),
			(offset.z + z) / ar3d::real(grid->getVoxelResolution())
		);
		float dist = std::numeric_limits<float>::max();
		int numIntersections = 0;
		for (int i = 0; i < numTris; ++i)
		{
			glm::vec3 ag, bg, cg;
			triMesh->getTriangleVertices(i, &ag, &bg, &cg);
			float3 a = make_float3(ag.x, ag.y, ag.z); float3 b = make_float3(bg.x, bg.y, bg.z); float3 c = make_float3(cg.x, cg.y, cg.z);
			float d = pointTriangleDistance(a, b, c, pos);
			if (d < dist) dist = d;
			if (pointTriangleIntersection(a, b, c, pos)) numIntersections++;
		}
		if (numIntersections % 2 == 1) dist = -dist;
		sdf->atHost(x, y, z) = resolution * dist;
		if (dist < 0) inside++; else outside++;
	}
	CI_LOG_I("voxels inside=" << inside << ", outside=" << outside);

	return sdf;
}

ar3d::WorldGridRealDataPtr ar3d::groundTruthToSdf(ar3d::InputConfigPtr config, int frame, int resolution)
{
	if (frame >= config->duration) throw std::exception("frame out of bounds");

	if (config->groundTruth) {
		//ground truth is a sphere
		return sphereToSdf(config->groundTruth->locations[frame], config->groundTruth->radius, resolution);
	}
	else if (config->groundTruthMeshes)
	{
		//ground truth is saved in meshes
		std::string file = config->getPathToGroundTruth(frame);
		return meshToSdf(file, resolution);
	}

	throw std::exception("no ground truth found");
}


static float3 closesPointOnTriangle(const std::array<float3, 3> triangle, const float3 &sourcePosition)
{
	float3 edge0 = triangle[1] - triangle[0];
	float3 edge1 = triangle[2] - triangle[0];
	float3 v0 = triangle[0] - sourcePosition;

	float a = dot(edge0, edge0);
	float b = dot(edge0, edge1);
	float c = dot(edge1, edge1);
	float d = dot(edge0, v0);
	float e = dot(edge1, v0);

	float det = a * c - b * b;
	float s = b * e - c * d;
	float t = b * d - a * e;

	if (s + t < det)
	{
		if (s < 0.f)
		{
			if (t < 0.f)
			{
				if (d < 0.f)
				{
					s = clamp(-d / a, 0.f, 1.f);
					t = 0.f;
				}
				else
				{
					s = 0.f;
					t = clamp(-e / c, 0.f, 1.f);
				}
			}
			else
			{
				s = 0.f;
				t = clamp(-e / c, 0.f, 1.f);
			}
		}
		else if (t < 0.f)
		{
			s = clamp(-d / a, 0.f, 1.f);
			t = 0.f;
		}
		else
		{
			float invDet = 1.f / det;
			s *= invDet;
			t *= invDet;
		}
	}
	else
	{
		if (s < 0.f)
		{
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1 - s;
			}
			else
			{
				t = clamp(-e / c, 0.f, 1.f);
				s = 0.f;
			}
		}
		else if (t < 0.f)
		{
			if (a + d > b + e)
			{
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1 - s;
			}
			else
			{
				s = clamp(-e / c, 0.f, 1.f);
				t = 0.f;
			}
		}
		else
		{
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = clamp(numer / denom, 0.f, 1.f);
			t = 1.f - s;
		}
	}

	return triangle[0] + s * edge0 + t * edge1;
}

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

/* the original jgt code */
static int intersect_triangle(double orig[3], double dir[3],
	double vert0[3], double vert1[3], double vert2[3],
	double *t, double *u, double *v)
{
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det, inv_det;

	/* find vectors for two edges sharing vert0 */
	SUB(edge1, vert1, vert0);
	SUB(edge2, vert2, vert0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);

	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1.0 / det;

	/* calculate distance from vert0 to ray origin */
	SUB(tvec, orig, vert0);

	/* calculate U parameter and test bounds */
	*u = DOT(tvec, pvec) * inv_det;
	if (*u < 0.0 || *u > 1.0)
		return 0;

	/* prepare to test V parameter */
	CROSS(qvec, tvec, edge1);

	/* calculate V parameter and test bounds */
	*v = DOT(dir, qvec) * inv_det;
	if (*v < 0.0 || *u + *v > 1.0)
		return 0;

	/* calculate t, ray intersects triangle */
	*t = DOT(edge2, qvec) * inv_det;

	return 1;
}