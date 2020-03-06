#include "Utils.h"

using namespace ar;

Vector2 utils::bilinearInterpolateInv(const std::array<Vector2, 4>& values, const Vector2& x)
{
	Vector2 ab(0.5, 0.5); //initial values for alpha and beta
	const Vector2 z1 = values[0] - x;
	const Vector2 z2 = values[1] - values[0];
	const Vector2 z3 = values[2] - values[0];
	const Vector2 z4 = values[0] - values[1] - values[2] + values[3];
	for (int i = 0; i < interpolateInvNewtonSteps; ++i)
	{
		Vector2 F = z1 + ab.x() * z2 + ab.y() * z3 + ab.x() * ab.y() * z4; //function value
		Matrix2 J;
		J << z2 + ab.y() * z4, z3 + ab.x() * z4; //Jacobian
		ab -= J.inverse() * F; //Newton step
	}
	return ab;
}

std::pair<Vector2, Vector2> utils::bilinearInterpolateInvAnalytic(const std::array<Vector2, 4>& values,
	const Vector2& xp)
{
	const Vector2 z1 = values[0];
	const Vector2 z2 = values[1] - values[0];
	const Vector2 z3 = values[2] - values[0];
	const Vector2 z4 = values[0] - values[1] - values[2] + values[3];
	real x1 = z1.x(), y1 = z1.y();
	real x2 = z2.x(), y2 = z2.y();
	real x3 = z3.x(), y3 = z3.y();
	real x4 = z4.x(), y4 = z4.y();
	real x = xp.x(), y = xp.y();

	real alpha1 = (x4*y - x4 * y1 + x3 * y2 - x2 * y3 - x * y4 + x1 * y4 + 
		sqrt(square(x4*y - x4 * y1 + x3 * y2 - x2 * y3 - x * y4 + x1 * y4) + 
			4 * (x2*(y - y1) + (-x + x1)*y2)*(x4*y3 - x3 * y4))) / (2 * x4*y3 - 2 * x3*y4);

	real beta1 = -((-(x4*y) + x4 * y1 + x3 * y2 - x2 * y3 + x * y4 - x1 * y4 +
		sqrt(square(x4*y - x4 * y1 + x3 * y2 - x2 * y3 - x * y4 + x1 * y4) +
			4 * (x2*(y - y1) + (-x + x1)*y2)*(x4*y3 - x3 * y4))) / (2 * x4*y2 - 2 * x2*y4));

	real alpha2 = (-(x4*y) + x4 * y1 - x3 * y2 + x2 * y3 + x * y4 - x1 * y4 + 
		sqrt(square(x4*y - x4 * y1 + x3 * y2 - x2 * y3 - x * y4 + x1 * y4) + 
			4 * (x2*(y - y1) + (-x + x1)*y2)*(x4*y3 - x3 * y4))) / (2.*(-(x4*y3) + x3 * y4));

	real beta2 = (x4*y - x4 * y1 - x3 * y2 + x2 * y3 - x * y4 + x1 * y4 + 
		sqrt(square(x4*y - x4 * y1 + x3 * y2 - x2 * y3 - x * y4 + x1 * y4) + 
			4 * (x2*(y - y1) + (-x + x1)*y2)*(x4*y3 - x3 * y4))) / (2 * x4*y2 - 2 * x2*y4);

	return std::make_pair(Vector2(alpha1, beta1), Vector2(alpha2, beta2));
}
