#include "AdjointUtilities.h"

#include "Utils.h"
#include <cinder/Log.h>

namespace ar
{
    ar::Matrix2 ar::AdjointUtilities::polarDecompositionAdjoint(const Matrix2 & F, const Matrix2 & adjR2)
    {
        //Forward code:
        Matrix2 m;
        m << F(1, 1), -F(1, 0),
            -F(0, 1), F(0, 0);
        real s = ar::utils::sgn(F.determinant());
        Matrix2 R = F + s * m;
        real scale = 1 / R.col(0).norm();
        //Matrix2 R2 = R * scale;
        //return R2; //this would be exactly the forward code

        Matrix2 adjR = adjR2 * scale;
        real adjScale = adjR2.cwiseProduct(R).sum();

        real tmp = -1 / std::sqrt(ar::cube(R(0, 0)*R(0, 0) + R(1, 0)*R(1, 0)));
        adjR(0, 0) += R(0, 0)*tmp*adjScale;
        adjR(1, 0) += R(1, 0)*tmp*adjScale;

        return adjR + s * (Matrix2() << adjR(1, 1), -adjR(1, 0), -adjR(0, 1), adjR(0, 0)).finished();
    }

    void ar::AdjointUtilities::resolveSingleCollision_PostRepair_Adjoint(real dist, const Vector2& normal,
        const Vector2& u, const Vector2& uDot, real alpha, real beta, const Vector2& uNew, const Vector2& uDotNew,
        const Vector2& adjUNew, const Vector2& adjUDotNew, real& adjDist, Vector2& adjNormal, Vector2& adjU,
        Vector2& adjUDot, real& adjAlpha, real& adjBeta)
    {
        /*
        * Forward code:
        Vector2 uPost = u - alpha * dist * normal;
        Vector2 uDotPos = (1 - beta) * (uDot - 2 * (uDot.dot(normal))*normal);
        */

        assert(dist < 0); //should have been only called in that branch
        real dot = uDot.dot(normal);

        adjBeta += (2 * normal.x()*dot - uDot.x())*adjUDotNew.x();
        adjUDot.x() += (1 - beta) * (1 - 2 * square(normal.x())) * adjUDotNew.x();
        adjUDot.y() += (1 - beta) * (-2 * normal.x()*normal.y()) * adjUDotNew.x();
        adjNormal.x() += (1 - beta) * (-2 * normal.x()*uDot.x() - 2 * dot) * adjUDotNew.x();
        adjNormal.y() += (1 - beta) * (-2 * normal.x()*uDot.y()) * adjUDotNew.x();

        adjBeta += (2 * normal.y() * dot - uDot.y()) * adjUDotNew.y();
        adjUDot.x() += (1 - beta) * (-2 * normal.x()*normal.y()) * adjUDotNew.y();
        adjUDot.y() += (1 - beta) * (1 - 2 * square(normal.y())) * adjUDotNew.y();
        adjNormal.x() += (1 - beta) * (-2 * normal.y()*uDot.y()) * adjUDotNew.y();
        adjNormal.y() += (1 - beta) * (-2 * normal.y()*uDot.y() - 2 * dot) * adjUDotNew.y();

        adjAlpha += -(dist*normal).dot(adjUNew);
        adjU += adjUNew;
        adjDist += -(alpha*normal).dot(adjUNew);
        adjNormal += -alpha * dist*adjUNew;
    }

    void ar::AdjointUtilities::groundCollisionAdjoint(const Vector2 & p, real groundHeight, real groundAngle, real adjDist, const Vector2 & adjNormal, Vector2 & adjP, real & adjGroundHeight, real & adjGroundAngle)
    {
        /*
        * Forward code:
        real dx = cos(groundAngle);
        real dy = sin(groundAngle);
        real distance = -(dy*p.x() - dx * p.y() - (dy*0.5 - dx * groundHeight));
        Vector2 normal(-dy, dx);
        */

        real dx = cos(groundAngle);
        real dy = sin(groundAngle);

        real adjDx = 0, adjDy = 0;

        adjDy += -adjNormal.x();
        adjDx += adjNormal.y();

        adjDx += (-groundHeight + p.y()) * adjDist;
        adjDy += (0.5 - p.x()) * adjDist;
        adjP.x() += (-dy) * adjDist;
        adjP.y() += dx * adjDist;
        adjGroundHeight += -dx * adjDist;

        adjGroundAngle += (-sin(groundAngle)) * adjDx;
        adjGroundAngle += (cos(groundAngle)) * adjDy;
    }

    void ar::AdjointUtilities::groundCollisionDtAdjoint(const Vector2 & pDot, real groundHeight, real groundAngle, real adjDistDt, Vector2 & adjPDot, real & adjGroundHeight, real & adjGroundAngle)
    {
        /*
        * Forward code:
        real dx = cos(groundAngle);
        real dy = sin(groundAngle);
        real distanceDt = -dy * pDot.x() + dx * pDot.y();
        return distanceDt;
        */

        real dx = cos(groundAngle);
        real dy = sin(groundAngle);

        real adjDy = -pDot.x()*adjDistDt;
        adjPDot.x() += -dy * adjDistDt;
        real adjDx = pDot.y()*adjDistDt;
        adjPDot.y() += dx * adjDistDt;

        adjGroundAngle += (-sin(groundAngle)) * adjDx;
        adjGroundAngle += (cos(groundAngle)) * adjDy;
    }

    void AdjointUtilities::bilinearInterpolateAdjoint2(const std::array<ar::real, 4>& values, real fx, real fy,
                                                       const real& adjResult, real& adjFx, real& adjFy)
    {
        adjFx += ((values[1] - values[0]) + fy * (values[0] - values[1] - values[2] + values[3])) * adjResult;
        adjFy += ((values[2] - values[0]) + fx * (values[0] - values[1] - values[2] + values[3])) * adjResult;
    }

    Matrix2 AdjointUtilities::inverseAdjoint(const Matrix2& m, const Matrix2& adjMInv)
    {
        /*
         * Forward: 
         * a = m(0,0), b = m(0,1), c=m(1,0), d=m(1,1)
         * mInv = (1 / (ad-bc)) * [d,-b;-c,a]
         */
        Matrix2 adjM = Matrix2::Zero();
        real detInv = 1 / (m(0,0)*m(1,1)-m(1,0)*m(0,1));

        adjM(1, 1) += detInv * adjMInv(0, 0);
        adjM(0, 1) -= detInv * adjMInv(0, 1);
        adjM(1, 0) -= detInv * adjMInv(1, 0);
        adjM(0, 0) += detInv * adjMInv(1, 1);
        real adjDetInv = m(1, 1)*adjMInv(0, 0) - m(1, 0)*adjMInv(1, 0) - m(0, 1)*adjMInv(0, 1) + m(0, 0)*adjMInv(1, 1);
        adjM(0, 0) -= m(1, 1) * square(detInv) * adjDetInv;
        adjM(1, 0) += m(0, 1) * square(detInv) * adjDetInv;
        adjM(0, 1) += m(1, 0) * square(detInv) * adjDetInv;
        adjM(1, 1) -= m(0, 0) * square(detInv) * adjDetInv;
        return adjM;
    }

    void AdjointUtilities::bilinearInterpolateInvAdjoint(const std::array<Vector2, 4>& values, const Vector2& x, const Vector2& alphaBeta,
        const Vector2& adjAlphaBeta, std::array<Vector2, 4>& adjValues)
    {
#if 0
		//ALGORITHM 1: Adjoint of Newton
        //Allocate memory for the forward step
        Vector2 ab[utils::interpolateInvNewtonSteps+1];
        Vector2 F[utils::interpolateInvNewtonSteps];
        Matrix2 J[utils::interpolateInvNewtonSteps];
        const Vector2 z1 = values[0] - x;
        const Vector2 z2 = values[1] - values[0];
        const Vector2 z3 = values[2] - values[0];
        const Vector2 z4 = values[0] - values[1] - values[2] + values[3];
        //Forward simulation
        ab[0] = Vector2(0.5, 0.5);
        for (int i = 0; i < utils::interpolateInvNewtonSteps; ++i)
        {
            F[i] = z1 + ab[i].x()*z2 + ab[i].y()*z3 + ab[i].x()*ab[i].y()*z4; //function value
            J[i] << z2 + ab[i].y()*z4, z3 + ab[i].x()*z4; //Jacobian
            ab[i + 1] = ab[i] - J[i].inverse() * F[i];
        }

        //adjoint
        Vector2 adjZ1(0, 0), adjZ2(0, 0), adjZ3(0, 0), adjZ4(0, 0);
        for (int i=utils::interpolateInvNewtonSteps-1; i>=0; --i)
        {
            Vector2 adjF = -J[i] * adjAlphaBeta; //-J[i].inverse().transpose() * adjAlphaBeta;
            //Matrix2 adjJinv = -adjAlphaBeta * F[i].transpose();
            //Matrix2 adjJ = inverseAdjoint(J[i], adjJinv);
            Matrix2 adjJ = -J[i].inverse() * adjAlphaBeta * (J[i].inverse() * F[i]).transpose();
            adjZ1 += adjF;
            adjZ2 += ab[i].x() * adjF + adjJ.col(0);
            adjZ3 += ab[i].y() * adjF + adjJ.col(1);
            adjZ4 += ab[i].x() * ab[i].y() * adjF + ab[i].y() * adjJ.col(0) + ab[i].x() * adjJ.col(1);
        }
        adjValues[0] += adjZ1 - adjZ2 - adjZ3 + adjZ4;
        adjValues[1] += adjZ2 - adjZ4;
        adjValues[2] += adjZ3 - adjZ4;
        adjValues[3] += adjZ4;

#else
		//ALGORITHM 2: direct adjoint
		const Vector2 z1 = values[0] - x;
		const Vector2 z2 = values[1] - values[0];
		const Vector2 z3 = values[2] - values[0];
		const Vector2 z4 = values[0] - values[1] - values[2] + values[3];

		const real alpha = alphaBeta.x();
		const real beta = alphaBeta.y();
		Matrix2 m1; m1 << (z2 + beta * z4), (z3 + alpha * z4);
		Vector2 adjAlphaBetaPrime = m1.transpose().inverse() * adjAlphaBeta;
		Vector2 adjZ1 = -adjAlphaBetaPrime;
		Vector2 adjZ2 = -alpha * adjAlphaBetaPrime;
		Vector2 adjZ3 = -beta * adjAlphaBetaPrime;
		Vector2 adjZ4 = -alpha * beta * adjAlphaBetaPrime;

		adjValues[0] += adjZ1 - adjZ2 - adjZ3 + adjZ4;
		adjValues[1] += adjZ2 - adjZ4;
		adjValues[2] += adjZ3 - adjZ4;
		adjValues[3] += adjZ4;
#endif
    }

    void AdjointUtilities::softminAdjoint(real x, real alpha, real adjResult, real& adjX)
    {
        /**
         * Forward code:
        //to avoid numerical overflow
        if (x > 5/alpha) return 0;
        if (x < -5/alpha) return x;
        //softmin
        return -std::log(T(1) + std::exp(-x * alpha)) / alpha;
         */

		adjX += adjResult / (std::exp(alpha*x) + 1);
    }

    void AdjointUtilities::softminDxAdjoint(real x, real alpha, real adjResult, real& adjX)
    {
        /*
        //to avoid numerical overflow
        if (x > 5/alpha) return 0;
        if (x < -5/alpha) return 1;
        //d/dx softmin
        return T(1) / (exp(alpha*x) + T(1));
         */

        if (x <= 100/alpha && x >= -100/alpha)
        {
            real e = exp(alpha*x);
            adjX += adjResult * -(alpha * e) / square(1 + e);
        }
    }

	void AdjointUtilities::computeMaterialParameters_D_YoungsModulus(real youngsModulus, real poissonRatio,
		real& muDyoung, real& lambdaDyoung)
	{
		muDyoung = 1 / (2 * (1 + poissonRatio));
		lambdaDyoung = poissonRatio / ((1 - 2 * poissonRatio) * (1 + poissonRatio));
	}

	void AdjointUtilities::computeMaterialParameters_D_PoissonRatio(real youngsModulus, real poissonRatio,
		real& muDpoisson, real& lambdaDpoisson)
	{
		muDpoisson = -youngsModulus / (2 * square(poissonRatio + 1));
		lambdaDpoisson = (2 * youngsModulus * square(poissonRatio) + youngsModulus) /
			square(2 * square(poissonRatio) + poissonRatio - 1);
	}

	void AdjointUtilities::getIntersectionPointsAdjoint(const std::array<real, 4>& sdf,
		const std::array<Vector2, 4>& corners, const Vector2& adjPoint1, const Vector2& adjPoint2,
		std::array<real, 4>& adjSdf, std::array<Vector2, 4>& adjCorners)
    {
		using namespace ar::utils;
		int c = inside(sdf[0]) | (inside(sdf[1]) << 1) | (inside(sdf[2]) << 2) | (inside(sdf[3]) << 3);

#define MIX_POINT(sdf0, sdf1, corner0, corner1, adjPoint)																\
	adjCorners[corner0] += (-sdf[sdf1] / (sdf[sdf0] - sdf[sdf1])) * adjPoint;											\
	adjCorners[corner1] += (+sdf[sdf0] / (sdf[sdf0] - sdf[sdf1])) * adjPoint;											\
	adjSdf[sdf0] += (sdf[sdf1] * (corners[corner0] - corners[corner1]) / square(sdf[sdf0] - sdf[sdf1])).dot(adjPoint);	\
	adjSdf[sdf1] += (sdf[sdf0] * (corners[corner1] - corners[corner0]) / square(sdf[sdf0] - sdf[sdf1])).dot(adjPoint);

		switch (c)
		{
		case 0b1111: break; //case 2: completely inside
		case 0b0000: break; //case 1: completely outside
		case 0b0001:
		case 0b1110:
		{
			MIX_POINT(0, 1, 0, 1, adjPoint1);
			MIX_POINT(0, 2, 0, 2, adjPoint2);
		} break;
		case 0b0010:
		case 0b1101:
		{
			MIX_POINT(0, 1, 0, 1, adjPoint1);
			MIX_POINT(1, 3, 1, 3, adjPoint2);
		} break;
		case 0b0100:
		case 0b1011:
		{
			MIX_POINT(2, 3, 2, 3, adjPoint1);
			MIX_POINT(0, 2, 0, 2, adjPoint2);
		} break;
		case 0b1000:
		case 0b0111:
		{
			MIX_POINT(2, 3, 2, 3, adjPoint1);
			MIX_POINT(1, 3, 1, 3, adjPoint2);
		} break;
		case 0b0011:
		case 0b1100:
		{
			MIX_POINT(0, 2, 0, 2, adjPoint1);
			MIX_POINT(1, 3, 1, 3, adjPoint2);
		} break;
		case 0b1010:
		case 0b0101:
		{
			MIX_POINT(0, 1, 0, 1, adjPoint1);
			MIX_POINT(2, 3, 2, 3, adjPoint2);
		} break;
		case 0b1001: CI_LOG_E("case 1001: not implemented"); break;
		case 0b0110: CI_LOG_E("case 0110: not implemented"); break;
		default: throw std::exception("This should not happen");
		}

#undef MIX_POINT
    }
}
