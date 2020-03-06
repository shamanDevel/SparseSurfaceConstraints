#include "TimeIntegrator.h"

#include <exception>
#include <Eigen/Dense>
#include <cinder/Log.h>

namespace ar
{
    
    void TimeIntegrator_Newmark1::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevU = currentU;
        prevUDot = currentUDot;

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = getMatrixPartA(stiffness, damping, mass, load, deltaT, theta);
        VectorX b 
            = getMatrixPartB(stiffness, damping, mass, load, deltaT, theta) * prevU
            + getMatrixPartC(stiffness, damping, mass, load, deltaT, theta) * prevUDot
            + getMatrixPartD(stiffness, damping, mass, load, deltaT, theta);
        currentU = solveDense(A, b);

        currentUDot = computeUdot(currentU, prevU, prevUDot, deltaT, theta);
    }
    
    void TimeIntegrator_Newmark2::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevU = currentU;
        prevUDot = currentUDot;
        prevUDotDot = currentUDotDot;

        CI_LOG_V("prevU: " << prevU.transpose());
        CI_LOG_V("prevUDot: " << prevUDot.transpose());
        CI_LOG_V("prevUDotDot: " << prevUDotDot.transpose());

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = (4 / (deltaT*deltaT)) * massMatrix + (2 / deltaT) * damping + stiffness;
        VectorX b = load
            + massMatrix * ((4 / (deltaT*deltaT)) * (prevU + prevUDot) + prevUDotDot)
            + damping * ((2 / deltaT) * prevU + prevUDot);
        currentU = solveDense(A, b);
        CI_LOG_V("A:\n" << A);
        CI_LOG_V("b:\n" << b.transpose());
        CI_LOG_V("solution:\n" << currentU.transpose());

        currentUDot = (2 / deltaT) * (currentU - prevU) - prevUDot;
        currentUDotDot = (4 / (deltaT*deltaT)) * (currentU - prevU - deltaT * prevUDot) - prevUDotDot;
    }
    
    void TimeIntegrator_ExplicitCentralDifferences::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevPrevU = prevU;
        prevU = currentU;

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = (1 / (deltaT*deltaT)) * massMatrix + (1 / (2 * deltaT)) * damping;
        VectorX b = load
            + ((2 / (deltaT*deltaT)) * massMatrix - stiffness) * prevU
            + ((-1 / (deltaT*deltaT)) * massMatrix + (1 / (2 * deltaT)) * damping) * prevPrevU;
        currentU = solveDense(A, b);
    }

    void TimeIntegrator_ImplicitLinearAcceleration::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevU = currentU;
        prevUDot = currentUDot;
        prevUDotDot = currentUDotDot;

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = (massMatrix + (deltaT/2)*damping + (deltaT*deltaT/6)*stiffness);
        VectorX b = ((-deltaT / 2) * damping - (deltaT*deltaT / 3) *  stiffness) * prevUDotDot
            - stiffness * prevU
            - damping * prevUDot
            - deltaT * stiffness * prevUDot
            + load;
        currentUDotDot = solveDense(A, b);

        currentUDot = prevUDot + (deltaT / 2) * (currentUDotDot + prevUDotDot);
        currentU = prevU + deltaT * prevUDot + (deltaT*deltaT / 6)*(currentUDotDot + 2 * prevUDotDot);
    }

    void TimeIntegrator_Newmark3::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevU = currentU;
        prevUDot = currentUDot;
        prevUDotDot = currentUDotDot;

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = ((6 / (deltaT*deltaT))*massMatrix + (3 / deltaT)*damping + stiffness);
        VectorX b = load
            + (3 * massMatrix + (deltaT / 2) * damping) * prevUDotDot
            + ((6 / deltaT) * massMatrix + 3 * damping) * prevUDot;
        VectorX diffX = solveDense(A, b);

        currentU = prevU + diffX;
        currentUDot = -2 * prevUDot - (deltaT / 2)*prevUDotDot + (3 / deltaT) * diffX;
        MatrixX invMassMatrix = mass.cwiseInverse().asDiagonal();
        currentUDotDot = -invMassMatrix * (damping * currentUDot + stiffness * currentU - load);
    }

    void TimeIntegrator_HHTalpha::performStep(real deltaT, const VectorX & mass, const MatrixX & damping, const MatrixX & stiffness, const VectorX & load)
    {
        assertSize(mass);
        assertSize(damping);
        assertSize(stiffness);
        assertSize(load);

        prevU = currentU;
        prevUDot = currentUDot;
        prevUDotDot = currentUDotDot;

        MatrixX massMatrix = mass.asDiagonal();
        MatrixX A = massMatrix + (deltaT * (1 - alpha) * gamma) * damping + (deltaT * deltaT * (1 - alpha) * beta) * stiffness;
        VectorX b = load
            - ((deltaT * (1 - alpha) * (1 - gamma)) * damping + ((deltaT * deltaT) * (1 - alpha) * (0.5f - gamma)) * stiffness) * prevUDotDot
            - (damping + (deltaT * (1 - alpha)) * stiffness) * prevUDot
            - stiffness * prevU;
        currentUDotDot = solveDense(A, b);

        currentU = prevU + deltaT * prevUDot + deltaT * deltaT*((0.5f - beta)*prevUDotDot + beta * currentUDotDot);
        currentUDot = prevUDot + deltaT * ((1 - gamma)*prevUDotDot + gamma * currentUDotDot);
    }

}
