#include "GridUtils.h"

#include <queue>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <cassert>
#include <LBFGS.h>

#include "GradientDescent.h"

ar::real ar::GridUtils2D::getLinearClampedF(const real* linear, real x, real y, int width, int height)
{
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    real fx = x - ix;
    real fy = y - iy;
    real v00 = getLinearClamped(linear, ix, iy, width, height);
    real v10 = getLinearClamped(linear, ix + 1, iy, width, height);
    real v01 = getLinearClamped(linear, ix, iy + 1, width, height);
    real v11 = getLinearClamped(linear, ix + 1, iy + 1, width, height);
    real v0 = v00 + fx * (v01 - v00); //correct???
    real v1 = v10 + fx * (v11 - v10);
    return v0 + fy * (v1 - v0);
}

const ar::real& ar::GridUtils2D::getClamped(const grid_t& grid, int x, int y)
{
    int width = (int)grid.rows();
    int height = (int)grid.cols();
    clampCoord(x, y, width, height);
    return grid(x, y);
}

ar::real& ar::GridUtils2D::atClamped(grid_t& grid, int x, int y)
{
    int width = (int)grid.rows();
    int height = (int)grid.cols();
    clampCoord(x, y, width, height);
    return grid(x, y);
}

ar::real ar::GridUtils2D::getClampedF(const grid_t& grid, real x, real y)
{
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    real fx = x - ix;
    real fy = y - iy;
    real v00 = getClamped(grid, ix, iy);
    real v10 = getClamped(grid, ix + 1, iy);
    real v01 = getClamped(grid, ix, iy + 1);
    real v11 = getClamped(grid, ix + 1, iy + 1);
    real v0 = v00 + fy * (v01 - v00);
    real v1 = v10 + fy * (v11 - v10);
    return v0 + fx * (v1 - v0);
}

ar::GridUtils2D::grad_t ar::GridUtils2D::getGradientLinear(const real* linear, int x, int y, int width, int height)
{
    grad_t grad;
    //x
    if (x == 0)
        grad.x() = getLinear(linear, x + 1, y, width) - getLinear(linear, x, y, width); //forward
    else if (x == width - 1)
        grad.x() = getLinear(linear, x, y, width) - getLinear(linear, x - 1, y, width); //backward
    else
        grad.x() = 0.5 * (getLinear(linear, x + 1, y, width) - getLinear(linear, x - 1, y, width)); //central

    //y
    if (y == 0)
        grad.y() = getLinear(linear, x, y + 1, width) - getLinear(linear, x, y, width); //forward
    else if (y == height - 1)
        grad.y() = getLinear(linear, x, y, width) - getLinear(linear, x, y - 1, width); //backward
    else
        grad.y() = 0.5 * (getLinear(linear, x, y + 1, width) - getLinear(linear, x, y - 1, width)); //central

    return grad;
}

ar::GridUtils2D::grad_t ar::GridUtils2D::getGradient(const grid_t& grid, Eigen::Index i, Eigen::Index j)
{
	grad_t grad;
	//x
	if (i == 0)
		grad.x() = grid(i + 1, j) - grid(i, j); //forward
	else if (i == grid.rows() - 1)
		grad.x() = grid(i, j) - grid(i - 1, j); //backward
	else
		grad.x() = 0.5 * (grid(i + 1, j) - grid(i - 1, j)); //central

	//y
	if (j == 0)
		grad.y() = grid(i, j + 1) - grid(i, j); //forward
	else if (j == grid.cols() - 1)
		grad.y() = grid(i, j) - grid(i, j - 1); //backward
	else
		grad.y() = 0.5 * (grid(i, j + 1) - grid(i, j - 1)); //central

	return grad;
}

ar::real ar::GridUtils2D::getGradientPosLinear(const real* linear, int x, int y, int width, int height)
{
    real val = real(0);
    //x
    if (x == 0)
        val += ar::utils::sqr(getLinear(linear, x + 1, y, width) - getLinear(linear, x, y, width));
    else if (x == width - 1)
        val += ar::utils::sqr(getLinear(linear, x, y, width) - getLinear(linear, x - 1, y, width));
    else
        val += ar::utils::sqr(std::max(real(0), getLinear(linear, x, y, width) - getLinear(linear, x - 1, y, width)))
            + ar::utils::sqr(std::min(real(0), getLinear(linear, x + 1, y, width) - getLinear(linear, x, y, width)));

    //y
    if (y == 0)
        val += ar::utils::sqr(getLinear(linear, x, y + 1, width) - getLinear(linear, x, y, width));
    else if (y == height - 1)
        val += ar::utils::sqr(getLinear(linear, x, y, width) - getLinear(linear, x, y - 1, width));
    else
        val += ar::utils::sqr(std::max(real(0), getLinear(linear, x, y, width) - getLinear(linear, x, y - 1, width)))
            + ar::utils::sqr(std::min(real(0), getLinear(linear, x, y + 1, width) - getLinear(linear, x, y, width)));

    return std::sqrt(val);
}

ar::real ar::GridUtils2D::getGradientNegLinear(const real* linear, int x, int y, int width, int height)
{
    real val = real(0);
    //x
    if (x == 0)
        val += ar::utils::sqr(getLinear(linear, x + 1, y, width) - getLinear(linear, x, y, width));
    else if (x == width - 1)
        val += ar::utils::sqr(getLinear(linear, x, y, width) - getLinear(linear, x - 1, y, width));
    else
        val += ar::utils::sqr(std::min(real(0), getLinear(linear, x, y, width) - getLinear(linear, x - 1, y, width)))
            + ar::utils::sqr(std::max(real(0), getLinear(linear, x + 1, y, width) - getLinear(linear, x, y, width)));

    //y
    if (y == 0)
        val += ar::utils::sqr(getLinear(linear, x, y + 1, width) - getLinear(linear, x, y, width));
    else if (y == height - 1)
        val += ar::utils::sqr(getLinear(linear, x, y, width) - getLinear(linear, x, y - 1, width));
    else
        val += ar::utils::sqr(std::min(real(0), getLinear(linear, x, y, width) - getLinear(linear, x, y - 1, width)))
            + ar::utils::sqr(std::max(real(0), getLinear(linear, x, y + 1, width) - getLinear(linear, x, y, width)));

    return std::sqrt(val);
}

ar::GridUtils2D::grad_t ar::GridUtils2D::getGradientLinearF(const real* linear, real x, real y, int width, int height)
{
    grad_t grad;
    grad.x() = getLinearClampedF(linear, x + 0.5f, y, width, height) - getLinearClampedF(
        linear, x - 0.5f, y, width, height);
    grad.y() = getLinearClampedF(linear, x, y + 0.5f, width, height) - getLinearClampedF(
        linear, x, y - 0.5f, width, height);
    return grad;
}

ar::real ar::GridUtils2D::getLaplacianLinear(const real* linear, int x, int y, int width, int height)
{
    real laplacian = real(0);
    //x
    if (x == 0)
        laplacian += getLinear(linear, x, y, width) - 2 * getLinear(linear, x + 1, y, width) + getLinear(
            linear, x + 2, y, width);
    else if (x == width - 1)
        laplacian += getLinear(linear, x - 2, y, width) - 2 * getLinear(linear, x - 1, y, width) + getLinear(
            linear, x, y, width);
    else
        laplacian += getLinear(linear, x - 1, y, width) - 2 * getLinear(linear, x, y, width) + getLinear(
            linear, x + 1, y, width);

    //y
    if (y == 0)
        laplacian += getLinear(linear, x, y, width) - 2 * getLinear(linear, x, y + 1, width) + getLinear(
            linear, x, y + 2, width);
    else if (y == height - 1)
        laplacian += getLinear(linear, x, y - 2, width) - 2 * getLinear(linear, x, y - 1, width) + getLinear(
            linear, x, y, width);
    else
        laplacian += getLinear(linear, x, y - 1, width) - 2 * getLinear(linear, x, y, width) + getLinear(
            linear, x, y + 1, width);

    return laplacian;
}

ar::real ar::GridUtils2D::getLaplacian(const grid_t & grid, Eigen::Index i, Eigen::Index j)
{
	real laplacian = real(0);

	//x
	if (i == 0)
		laplacian += grid(i, j) - 2 * grid(i + 1, j) + grid(i + 2, j);
	else if (i == grid.rows() - 1)
		laplacian += grid(i - 2, j) - 2 * grid(i - 1, j) + grid(i, j);
	else
		laplacian += grid(i - 1, j) - 2 * grid(i, j) + grid(i + 1, j);

	//j
	if (j == 0)
		laplacian += grid(i, j) - 2 * grid(i, j + 1) + grid(i, j + 2);
	else if (j == grid.cols() - 1)
		laplacian += grid(i, j - 2) - 2 * grid(i, j - 1) + grid(i, j);
	else
		laplacian += grid(i, j - 1) - 2 * grid(i, j) + grid(i, j + 1);

	return laplacian;
}

std::pair<ar::GridUtils2D::grid_t, ar::GridUtils2D::grid_t> ar::GridUtils2D::getFullGradient(const grid_t & v)
{
	Eigen::Index rows = v.rows();
	Eigen::Index cols = v.cols();
	grid_t dx(rows, cols);
	grid_t dy(rows, cols);

	for (Eigen::Index j = 0; j<cols; ++j)
	{
		for (Eigen::Index i = 0; i<rows; ++i)
		{
			grad_t g = getGradient(v, i, j);
			dx(i, j) = g.x();
			dy(i, j) = g.y();
		}
	}

	return std::make_pair(dx, dy);
}

ar::GridUtils2D::grid_t ar::GridUtils2D::getFullLaplacian(const grid_t & v)
{
	Eigen::Index rows = v.rows();
	Eigen::Index cols = v.cols();
	grid_t laplacian(rows, cols);

	for (Eigen::Index j = 0; j<cols; ++j)
	{
		for (Eigen::Index i = 0; i<rows; ++i)
		{
			laplacian(i, j) = getLaplacian(v, i, j);
		}
	}

	return laplacian;
}

ar::GridUtils2D::grid_t ar::GridUtils2D::fillGrid(const grid_t& input, const bgrid_t& valid)
{
    int width = (int)input.rows();
    int height = (int)input.cols();
    grid_t grids[2] = {input, grid_t(width, height)};
    bgrid_t filled[2] = {valid, bgrid_t(width, height)};
    for (int i = 0; ; ++i)
    {
        bool hasChanges = false;
        int ci = i % 2;
        int ni = 1 - ci;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (filled[ci](x, y))
                {
                    filled[ni](x, y) = true;
                    grids[ni](x, y) = grids[ci](x, y);
                    continue;
                }
                real avg = real(0);
                int count = 0;
                if (x - 1 >= 0 && filled[ci](x - 1, y))
                {
                    avg += grids[ci](x - 1, y);
                    count++;
                }
                if (x + 1 < width && filled[ci](x + 1, y))
                {
                    avg += grids[ci](x + 1, y);
                    count++;
                }
                if (y - 1 >= 0 && filled[ci](x, y - 1))
                {
                    avg += grids[ci](x, y - 1);
                    count++;
                }
                if (y + 1 < height && filled[ci](x, y + 1))
                {
                    avg += grids[ci](x, y + 1);
                    count++;
                }
                if (count > 0)
                {
                    filled[ni](x, y) = true;
                    grids[ni](x, y) = avg / count;
                    hasChanges = true;
                }
                else
                    filled[ni](x, y) = false;
            }
        }
        if (!hasChanges)
        {
            return grids[ni];
        }
    }
}

ar::GridUtils2D::grid_t ar::GridUtils2D::fillGridDiffusion(const grid_t& input, const bgrid_t& valid)
{
    //find number of unknowns
    int width = (int)input.rows();
    int height = (int)input.cols();
	Eigen::Index numEmpty = 0;
	igrid_t indices = igrid_t::Constant(width, height, -1);
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (!valid(x, y)) {
				indices(x, y) = numEmpty;
				numEmpty++;
			}
		}
	}

    //build matrix
    std::vector<Eigen::Triplet<real>> entries;
	entries.reserve(5 * numEmpty);
	VectorX rhs = VectorX::Zero(numEmpty);
	const int neighborsX[] = { -1, 1, 0, 0 };
	const int neighborsY[] = { 0, 0, -1, 1 };
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (!valid(x, y)) {
				int row = indices(x, y);
				int c = 0;
				for (int i = 0; i < 4; ++i) {
					int ix = x + neighborsX[i];
					int iy = y + neighborsY[i];
					if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue; //outside, Neumann boundary
					if (valid(ix, iy)) {
						//dirichlet boundary
						rhs[indices(x, y)] += input(ix, iy);
					}
					else {
						//inside
						entries.emplace_back(row, indices(ix, iy), -1);
					}
					c++;
				}
				entries.emplace_back(row, row, c);
			}
		}
	}
	Eigen::SparseMatrix<real> M(numEmpty, numEmpty);
	M.setFromTriplets(entries.begin(), entries.end());

    //Solve it
	Eigen::ConjugateGradient<Eigen::SparseMatrix<real>> cg(M);
	VectorX r = cg.solve(rhs);

	//Map back
	grid_t result(width, height);
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (!valid(x, y))
				result(x, y) = r[indices(x, y)];
			else
				result(x, y) = input(x, y);
		}
	}
	return result;
}

void ar::GridUtils2D::fillGridDiffusionAdjoint(grid_t& adjInput, const grid_t& adjOutput, const grid_t& input,
    const bgrid_t& valid)
{
    //find number of unknowns
    int width = (int)input.rows();
    int height = (int)input.cols();
    Eigen::Index numEmpty = 0;
    igrid_t indices = igrid_t::Constant(width, height, -1);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (!valid(x, y)) {
                indices(x, y) = numEmpty;
                numEmpty++;
            }
        }
    }

    //build matrix
    std::vector<Eigen::Triplet<real>> entries;
    entries.reserve(5 * numEmpty);
    VectorX rhs = VectorX::Zero(numEmpty);
    const int neighborsX[] = { -1, 1, 0, 0 };
    const int neighborsY[] = { 0, 0, -1, 1 };
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (!valid(x, y)) {
                int row = indices(x, y);
                int c = 0;
                for (int i = 0; i < 4; ++i) {
                    int ix = x + neighborsX[i];
                    int iy = y + neighborsY[i];
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue; //outside, Neumann boundary
                    if (valid(ix, iy)) {
                        //dirichlet boundary
                        rhs[indices(x, y)] += adjOutput(ix, iy);
                    }
                    else {
                        //inside
                        entries.emplace_back(indices(ix, iy), row, -1);
                    }
                    c++;
                }
                entries.emplace_back(row, row, c);
            }
        }
    }
    Eigen::SparseMatrix<real> M(numEmpty, numEmpty);
    M.setFromTriplets(entries.begin(), entries.end());

    //Solve it
    Eigen::ConjugateGradient<Eigen::SparseMatrix<real>> cg(M);
    VectorX r = cg.solve(rhs);

    //Map back
    grid_t result(width, height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (!valid(x, y))
                adjInput(x, y) += r[indices(x, y)];
        }
    }
}

void ar::GridUtils2D::invertDisplacementDirectShepard(const grid_t& inputX, const grid_t& inputY, grid_t& outputX,
                                                      grid_t& outputY, real h)
{
    int width = (int)inputX.rows();
    int height = (int)inputX.cols();
    //find max displacement -> radius
    real maxDisp = sqrt((inputX * inputX + inputY * inputY).maxCoeff()) + 2 * h;
    int radius = static_cast<int>(ceil(maxDisp / h));
    //run the shepard interpolation
    //#pragma omp parallel for schedule(dynamic,8)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            vec2_t v(0, 0);
            real w = 0;
            vec2_t p1(x * h, y * h);
            for (int iy = std::max(0, y - radius); iy <= std::min(height - 1, y + radius); iy++)
            {
                for (int ix = std::max(0, x - radius); ix <= std::min(width - 1, x + radius); ix++)
                {
                    vec2_t u(inputX(ix, iy), inputY(ix, iy));
                    vec2_t p2 = vec2_t(ix * h, iy * h) + u;
                    real d = (p1 - p2).norm() + 0.000000001;
                    if (d <= maxDisp)
                    {
                        real wi = (1 / d - 1 / maxDisp);
                        wi *= wi;
                        v += wi * u;
                        w += wi;
                    }
                }
            }
            if (w > 0)
            {
                v /= w;
            }
            outputX(x, y) = v.x();
            outputY(x, y) = v.y();
        }
    }
}

void ar::GridUtils2D::invertDisplacementDirectShepardAdjoint(
    grid_t& adjInputX, grid_t& adjInputY,
    const grid_t& adjOutputX, const grid_t& adjOutputY, 
    const grid_t& inputX, const grid_t& inputY, const grid_t& outputX, const grid_t& outputY,
    real h)
{
    int width = (int)inputX.rows();
    int height = (int)inputX.cols();
    //find max displacement -> radius
    real maxDisp = sqrt((inputX * inputX + inputY * inputY).maxCoeff()) + 2 * h;
    int radius = static_cast<int>(ceil(maxDisp / h));
    //compute the adjoint of the shepard interpolation
    for (int y = height - 1; y >= 0; --y)
    {
        for (int x = width - 1; x>=0; --x)
        {
            //compute terms that are needed from the forward pass
            real w = 0;
            vec2_t p1(x * h, y * h);
            for (int iy = std::max(0, y - radius); iy <= std::min(height - 1, y + radius); iy++)
            {
                for (int ix = std::max(0, x - radius); ix <= std::min(width - 1, x + radius); ix++)
                {
                    vec2_t u(inputX(ix, iy), inputY(ix, iy));
                    vec2_t p2 = vec2_t(ix * h, iy * h) + u;
                    real d = (p1 - p2).norm() + 0.000000001;
                    if (d <= maxDisp)
                    {
                        real wi = square(1 / d - 1 / maxDisp);
                        w += wi;
                    }
                }
            }
            assert(w > 0);
            
            vec2_t adjOutput(adjOutputX(x, y), adjOutputY(x, y));
            vec2_t adjV = adjOutput / w;
            real adjW = -(vec2_t(outputX(x, y), outputY(x, y)) / (w*w)).dot(adjV);

            //compute adjoint, add it to the gradient of each component
            for (int iy = std::min(height - 1, y + radius); iy >= std::max(0, y - radius); --iy)
            {
                for (int ix = std::min(width - 1, x + radius); ix >= std::max(0, x - radius); --ix)
                {
                    vec2_t u(inputX(ix, iy), inputY(ix, iy));
                    vec2_t p2 = vec2_t(ix * h, iy * h) + u;
                    real d = (p1 - p2).norm() + 0.000000001;
                    if (d <= maxDisp)
                    {
                        real wi = square(1 / d - 1 / maxDisp);
                        real adjWi = adjW;
                        vec2_t adjU = wi * adjV;
                        adjWi += u.dot(adjV);
                        real adjD = -(2 / square(d)) * (1 / d - 1 / maxDisp) * adjWi;

                        adjU += (p2 - p1) / d * adjD;
                        adjInputX(ix, iy) += adjU.x();
                        adjInputY(ix, iy) += adjU.y();
                    }
                }
            }
        }
    }
}

void ar::GridUtils2D::invertDisplacement(const grid_t& inputX, const grid_t& inputY, grid_t& outputX, grid_t& outputY,
                                         real h)
{
    invertDisplacementDirectShepard(inputX, inputY, outputX, outputY, h);
    //TODO: global optimization
}

ar::GridUtils2D::grid_t ar::GridUtils2D::advectGridSemiLagrange(const grid_t& input, const grid_t& dispX, const grid_t& dispY,
                                                    real step)
{
    int width = (int)input.rows();
    int height = (int)input.cols();
    grid_t output(width, height);
    //#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            vec2_t v(dispX(x, y), dispY(x, y));
            output(x, y) = getClampedF(input, x + step * v.x(), y + step * v.y());
        }
    }
    return output;
}

void ar::GridUtils2D::advectGridSemiLagrangeAdjoint(grid_t& adjInput, grid_t& adjDispX, grid_t& adjDispY, 
    real step, const grid_t& adjOutput, const grid_t& dispX, const grid_t& dispY, const grid_t& input)
{
    int width = (int)adjOutput.rows();
    int height = (int)adjOutput.cols();
    assert(adjInput.rows() == width); assert(adjInput.cols() == height);
    assert(adjDispX.rows() == width); assert(adjDispX.cols() == height);
    assert(adjDispY.rows() == width); assert(adjDispY.cols() == height);

    for (int ky = height-1; ky >= 0; --ky)
    {
        for (int kx = width-1; kx >= 0; --kx)
        {
            //variables from the forward pass
            vec2_t v(dispX(kx, ky), dispY(kx, ky));
            real x = kx + step * v.x();
            real y = ky + step * v.y();
            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);
            real fx = x - ix;
            real fy = y - iy;
            real v00 = getClamped(input, ix, iy);
            real v10 = getClamped(input, ix + 1, iy);
            real v01 = getClamped(input, ix, iy + 1);
            real v11 = getClamped(input, ix + 1, iy + 1);
            real v0 = v00 + fy * (v01 - v00);
            real v1 = v10 + fy * (v11 - v10);
            //adjoint of the input
            real adjB = adjOutput(kx, ky);
            real adjV0 = (1 - fx) * adjB;
            real adjV1 = fx * adjB;
            real adjFx = (v1 - v0) * adjB;
            real adjV10 = (1 - fy) * adjV1;
            real adjV11 = fy * adjV1;
            real adjFy = (v11 - v10) * adjV1;
            real adjV00 = (1 - fy) * adjV0;
            real adjV01 = fy * adjV0;
            adjFy += (v01 - v00) * adjV0;
            atClamped(adjInput, ix, iy) += adjV00;
            atClamped(adjInput, ix + 1, iy) += adjV10;
            atClamped(adjInput, ix, iy + 1) += adjV01;
            atClamped(adjInput, ix + 1, iy + 1) += adjV11;
            real adjIx = -adjFx; real adjX = adjFx;
            real adjIy = -adjFy; real adjY = adjFy;
            real adjUx = step * adjX;
            real adjUy = step * adjY;
            adjDispX(kx, ky) += adjUx;
            adjDispY(kx, ky) += adjUy;
        }
    }
}

const ar::real ar::GridUtils2D::directForwardKernelRadiusIn = 1.5;
const ar::real ar::GridUtils2D::directForwardKernelRadiusOut = 4.0;
const ar::real ar::GridUtils2D::directForwardOuterKernelWeight = 1e-5;
const ar::real ar::GridUtils2D::directForwardOuterSdfThreshold = 1.01;
const ar::real ar::GridUtils2D::directForwardOuterSdfWeight = 1e-10;

ar::GridUtils2D::grid_t ar::GridUtils2D::advectGridDirectForward(const grid_t& input, const grid_t& dispX,
    const grid_t& dispY, real step, grid_t* outputWeights)
{
    int width = (int)input.rows();
    int height = (int)input.cols();
    grid_t weights = grid_t::Constant(width, height, 0);
    grid_t output = grid_t::Zero(width, height);
    real kernelDenom = 1.0 / (2 * square(directForwardKernelRadiusIn / 3)); //98% of the Gaussian kernel is within kernelRadius
    //blend into output grid
    for (int y=0; y<height; ++y)
    {
        for (int x=0; x<width; ++x)
        {
            real value = input(x, y);
			real extraWeight = 1;
			if (value <= -directForwardOuterSdfThreshold || value >= directForwardOuterSdfThreshold)
				extraWeight = std::max(directForwardOuterSdfWeight, -std::abs(value)+ directForwardOuterSdfThreshold);
            vec2_t v(dispX(x, y), dispY(x, y));
            vec2_t p = vec2_t(x, y) - step * v;
            for (int ix = std::max(0, (int)std::floor(p.x() - directForwardKernelRadiusOut)); ix <= std::min(width-1, (int)std::ceil(p.x() + directForwardKernelRadiusOut)); ++ix)
            {
                for (int iy = std::max(0, (int)std::floor(p.y() - directForwardKernelRadiusOut)); iy <= std::min(height - 1, (int)std::ceil(p.y() + directForwardKernelRadiusOut)); ++iy)
                {
                    real d = (vec2_t(ix, iy) - p).squaredNorm();
                    if (d <= square(directForwardKernelRadiusIn))
                    {
                        real w = exp(-d * kernelDenom) * extraWeight;
                        output(ix, iy) += w * value;
                        weights(ix, iy) += w;
                    } else if (d <= square(directForwardKernelRadiusOut))
                    {
						real w = exp(-d * kernelDenom) * directForwardOuterKernelWeight * extraWeight;
						output(ix, iy) += w * value;
						weights(ix, iy) += w;
                    }
                }
            }
        }
    }
    //normalize
    //output /= weights;
	output = grid_t::NullaryExpr(width, height, [&output, &input, &weights](Eigen::Index row, Eigen::Index col)->real
    {
		const real weight = weights(row, col);
		//return weight == 0 ? input(row, col) : output(row, col) / weight;
		return weight == 0 ? (weights.rows()+weights.cols()) : output(row, col) / weight; //well outside if weight=0
	});
	//TODO: change this also in the adjoint versions

	//place output weights
	if (outputWeights)
		*outputWeights = weights;
    //done
    return output;
}

//TODO: implement changes to the forward problem (different blur kernels and weights) also in the adjoint

void ar::GridUtils2D::advectGridDirectForwardAdjoint(grid_t& adjInput, grid_t& adjDispX, grid_t& adjDispY,
    const grid_t& adjOutput, const grid_t& dispX, const grid_t& dispY, const grid_t& input, const grid_t& output,
    real step, const grid_t& weights)
{
    int width = (int)input.rows();
    int height = (int)input.cols();
	real kernelDenom = 1.0 / (2 * square(directForwardKernelRadiusIn / 3)); //98% of the Gaussian kernel is within kernelRadius

    /*
    output = grid_t::NullaryExpr(width, height, [&output, &input, &weights](Eigen::Index row, Eigen::Index col)->real
    {
		const real weight = weights(row, col);
		//return weight == 0 ? input(row, col) : output(row, col) / weight;
		return weight == 0 ? (weights.rows()+weights.cols()) : output(row, col) / weight; //well outside if weight=0
	});
     */
    //adjoint: normalize
    grid_t adjOut = grid_t::NullaryExpr(width, height, [&adjOutput, &weights](Eigen::Index row, Eigen::Index col)->real
    {
        const real weight = weights(row, col);
        return weight == 0 ? 0 : adjOutput(row, col) / weight; //well outside if weight=0
    });
    grid_t adjWeights = grid_t::NullaryExpr(width, height, [&adjOutput, &output, &weights](Eigen::Index row, Eigen::Index col)->real
    {
        const real weight = weights(row, col);
        return weight == 0 ? 0 : -(adjOutput(row, col) * output(row, col)) / square(weight); //well outside if weight=0
    });

    //adjoint: blend into output grid
    for (int y=height-1; y>=0; --y)
    {
        for (int x=width-1; x>=0; --x)
        {
            real value = input(x, y);
			real extraWeight = 1;
			if (value <= -directForwardOuterSdfThreshold || value >= directForwardOuterSdfThreshold)
				extraWeight = std::max(directForwardOuterSdfWeight, -std::abs(value) + directForwardOuterSdfThreshold);
            vec2_t v(dispX(x, y), dispY(x, y));
            vec2_t p = vec2_t(x, y) - step * v;
            real adjValue = 0;
            vec2_t adjP(0, 0);
            for (int ix = std::min(width - 1, (int)std::ceil(p.x() + directForwardKernelRadiusOut)); ix >= std::max(0, (int)std::floor(p.x() - directForwardKernelRadiusOut)); --ix)
            {
                for (int iy = std::min(height - 1, (int)std::ceil(p.y() + directForwardKernelRadiusOut)); iy >= std::max(0, (int)std::floor(p.y() - directForwardKernelRadiusOut)); --iy)
                {
                    real d = (vec2_t(ix, iy) - p).squaredNorm();
                    if (d <= square(directForwardKernelRadiusIn))
                    {
                        real w = exp(-d * kernelDenom) * extraWeight;
                        real adjW = adjWeights(ix, iy);
                        adjValue += w * adjOut(ix, iy);
                        adjW += value * adjOut(ix, iy);
                        real adjD = -exp(-d * kernelDenom)*kernelDenom*extraWeight*adjW;
                        adjP += 2 * (p - vec2_t(ix, iy)) * adjD;
                    } 
                	else if (d <= square(directForwardKernelRadiusOut))
                    {
						real w = exp(-d * kernelDenom) * extraWeight * directForwardOuterKernelWeight;
						real adjW = adjWeights(ix, iy);
						adjValue += w * adjOut(ix, iy);
						adjW += value * adjOut(ix, iy);
						real adjD = -exp(-d * kernelDenom) * kernelDenom * extraWeight * directForwardOuterKernelWeight * adjW;
						adjP += 2 * (p - vec2_t(ix, iy)) * adjD;
                    }
                }
            }
            vec2_t adjV = -step * adjP;
            adjDispX(x, y) += adjV.x();
            adjDispY(x, y) += adjV.y();
            adjInput(x, y) += adjValue;
        }
    }
}

ar::GridUtils2D::grid_t ar::GridUtils2D::advectGridDirectForwardAdjOpMult(const grid_t & adjInput, 
	const grid_t & dispX, const grid_t & dispY, real step, const grid_t & weights)
{
	int width = (int)adjInput.rows();
	int height = (int)adjInput.cols();
	grid_t output(width, height);

	real kernelDenom = 1.0 / (2 * square(directForwardKernelRadiusIn / 3)); //98% of the Gaussian kernel is within kernelRadius
															 //blend into output grid
	for (int y = 0; y<height; ++y)
	{
		for (int x = 0; x<width; ++x)
		{
			//TODO: Do I need the correction by directForwardOuterSdfThreshold and directForwardOuterSdfWeight here?
			real out = 0;
			vec2_t v(dispX(x, y), dispY(x, y));
			vec2_t p = vec2_t(x, y) - step * v;
			for (int ix = std::max(0, (int)std::floor(p.x() - directForwardKernelRadiusOut)); ix <= std::min(width - 1, (int)std::ceil(p.x() + directForwardKernelRadiusOut)); ++ix)
			{
				for (int iy = std::max(0, (int)std::floor(p.y() - directForwardKernelRadiusOut)); iy <= std::min(height - 1, (int)std::ceil(p.y() + directForwardKernelRadiusOut)); ++iy)
				{
					real d = (vec2_t(ix, iy) - p).squaredNorm();
					if (d <= square(directForwardKernelRadiusIn))
					{
						real w = exp(-d * kernelDenom);
						out -= adjInput(ix, iy) * w / weights(ix, iy);
					}
					else if (d <= square(directForwardKernelRadiusOut))
					{
						real w = exp(-d * kernelDenom) * directForwardOuterKernelWeight;
						out -= adjInput(ix, iy) * w / weights(ix, iy);
					}
				}
			}
			output(x, y) = -out;
		}
	}

	return output;
}

ar::GridUtils2D::grid_t ar::GridUtils2D::recoverSDFViscosity(const grid_t& input, real viscosity, int iterations)
{
    int width = input.rows();
    int height = input.cols();
    grid_t output(width, height);
    grid_t inputOrigin = input;

    //use ar::GradientDescent
    auto gradFun = [&width, &height, &viscosity, &inputOrigin](const Eigen::Matrix<real, Eigen::Dynamic, 1>& a)
    {
        //allocate result
        Eigen::Matrix<real, Eigen::Dynamic, 1> grad(width * height);
        //compute new values
        //#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                grad_t g = getGradientLinear(a.data(), x, y, width, height);
                double l = getLaplacianLinear(a.data(), x, y, width, height);
                double update = ar::utils::sgn(getLinear(inputOrigin.data(), x, y, width)) * (1 - g.norm()) + viscosity
                    * l;
                atLinear(grad.data(), x, y, width) = update;
            }
        }
        //return array
        return grad;
    };

    VectorX inputLinear = linearize(input);
    GradientDescent<Eigen::Matrix<real, Eigen::Dynamic, 1>> gd(inputLinear, gradFun);
    for (int i = 0; i < iterations; ++i)
    {
        bool end = gd.step();
        CI_LOG_I("Iteration " << i << ", step size: " << gd.getLastStepSize());
        if (end) break;
    }

    return delinearize(gd.getCurrentSolution(), width, height);
}

ar::GridUtils2D::grid_t ar::GridUtils2D::recoverSDFUpwind(const grid_t& input, int iterations)
{
    int width = input.rows();
    int height = input.cols();
    grid_t output(width, height);
    grid_t inputOrigin = input;

    //use ar::GradientDescent
    auto gradFun = [&width, &height, &inputOrigin](const Eigen::Matrix<real, Eigen::Dynamic, 1>& a)
    {
        //allocate result
        Eigen::Matrix<real, Eigen::Dynamic, 1> grad(width * height);
        //compute new values
        //#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                real gradPos = getGradientPosLinear(a.data(), x, y, width, height);
                real gradNeg = getGradientNegLinear(a.data(), x, y, width, height);
                int sgn = ar::utils::sgn(getLinear(inputOrigin.data(), x, y, width));
                double update = std::max(0, sgn) * gradPos + std::min(0, sgn) * gradNeg - sgn;
                atLinear(grad.data(), x, y, width) = update;
            }
        }
        //return array
        return grad;
    };

    Eigen::Matrix<real, Eigen::Dynamic, 1> inputLinear = linearize(input);
    GradientDescent<Eigen::Matrix<real, Eigen::Dynamic, 1>> gd(inputLinear, gradFun);
    for (int i = 0; i < iterations; ++i)
    {
        bool end = gd.step();
        CI_LOG_I("Iteration " << i << ", step size: " << gd.getLastStepSize());
        if (end) break;
    }

    return delinearize(gd.getCurrentSolution(), width, height);
}

ar::GridUtils2D::grid_t ar::GridUtils2D::recoverSDFSussmann(
	const grid_t& input, real epsilon, int iterations, RecoverSDFSussmannAdjointStorage* adjointStorage)
{
	int width = input.rows();
	int height = input.cols();
	grid_t inputOrigin = input;

	bgrid_t canChange = bgrid_t::Constant(width, height, true);
	for (int x=1; x<width; ++x)
		for (int y=1; y<height; ++y)
		{
			int c = utils::insideEq(input(x-1, y-1)) 
				| (utils::insideEq(input(x-1, y)) << 1) 
				| (utils::insideEq(input(x, y-1)) << 2) 
				| (utils::insideEq(input(x, y)) << 3);
			if (!(c==0b0000 || c==0b111))
			{
				canChange(x - 1, y - 1) = false;
				canChange(x - 1, y) = false;
				canChange(x, y - 1) = false;
				canChange(x, y) = false;
			}
		}

    if (adjointStorage) {
        adjointStorage->epsilon = epsilon;
        adjointStorage->inputs.push_back(inputOrigin);
        adjointStorage->canChange = canChange;
    }

#if 0
    real finalCost = 0;
    const auto gradient = [width, height, &inputOrigin, &canChange, epsilon, &finalCost](const VectorX& x) -> VectorX
    {
        grid_t grad = recoverSDFSussmannGradient(delinearize(x, width, height), inputOrigin, canChange, finalCost, epsilon);
        return linearize(grad);
    };
    //start value
    VectorX value = linearize(input);
    //run optimization
    GradientDescent<VectorX> gd(value, gradient);
    gd.setEpsilon(1e-15);
    gd.setLinearStepsize(0.001);
    gd.setMaxStepsize(0.5);
    int oi;
    for (oi = 0; oi < iterations; ++oi) {
        bool done = gd.step();
        CI_LOG_I("Iteration " << oi << "/" << iterations << ", cost = " << finalCost << ", stepsize = " << gd.getLastStepSize());
        if (done) break;
    }
    value = gd.getCurrentSolution();
    return delinearize(value, width, height);
#elif 1
    //fixed step size
    real stepsize = 0.5;
    real cost = 0;
    grid_t value = input;
    for (int i=0; i<iterations; ++i)
    {
        grid_t grad = recoverSDFSussmannGradient(value, inputOrigin, canChange, cost, epsilon);
        value -= stepsize * grad;
        if (adjointStorage) adjointStorage->inputs.push_back(value);
    }
    if (adjointStorage) {
        adjointStorage->iterations = iterations;
        adjointStorage->stepsize = stepsize;
    }
    return value;
#else
	LBFGSpp::LBFGSParam<real> params;
	params.epsilon = 1e-10;
	params.max_iterations = iterations;
	LBFGSpp::LBFGSSolver<real> lbfgs(params);

	LBFGSpp::LBFGSSolver<real>::ObjectiveFunction_t gradFun = [width, height, &inputOrigin, &canChange, epsilon](const VectorX& va, VectorX& gradient) -> real
	{
		real cost = 0;
		grid_t grad = recoverSDFSussmannGradient(delinearize(va, width, height), inputOrigin, canChange, cost, epsilon);
		gradient = linearize(grad);
		return cost;
	};
	LBFGSpp::LBFGSSolver<real>::CallbackFunction_t callback([iterations](const VectorX& x, const real& v, int k) -> bool {
		CI_LOG_I("Iteration " << k << "/" << iterations << ", cost = " << v);
		return true;
	});

	real finalCost = 0;
	VectorX value = linearize(input);
	int finalIterations = lbfgs.minimize(gradFun, value, finalCost, callback);
    return delinearize(value, width, height);
#endif
}

void ar::GridUtils2D::recoverSDFSussmannAdjoint(const grid_t& input, const grid_t& output, const grid_t& adjOutput,
	grid_t& adjInput, const RecoverSDFSussmannAdjointStorage& adjointStorage)
{
	//Adjoint of recoverSDFSussmann
	
    const grid_t& input0 = adjointStorage.inputs[0];
    grid_t adjValue = adjOutput;
    for (int i=adjointStorage.iterations-1; i>=0; --i)
    {
        grid_t adjGrad = -adjointStorage.stepsize * adjValue;
        recoverSDFSussmannGradientAdjoint(adjointStorage.inputs[i], input0, adjointStorage.canChange, adjointStorage.epsilon, adjGrad, adjValue);
    }
    adjInput += adjValue;
}

ar::GridUtils2D::grid_t ar::GridUtils2D::recoverSDFSussmannGradient(
	const grid_t& input, const grid_t& input0, const bgrid_t& canChange,
	real& cost, real epsilon)
{
	int width = input.rows();
	int height = input.cols();

	VectorX inputLinear = linearize(input);
	const real* a = inputLinear.data();

	grid_t gradient(width, height);

	cost = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (!canChange(x, y))
			{
				gradient(x, y) = 0;
				continue;
			}

			real phi0 = input0(x, y);
			real s = phi0 / sqrt(phi0*phi0 + epsilon);
			real s01 = utils::sgn(s);

			real phi = getLinear(a, x, y, width);
			real ta = x > 0
				? (phi - getLinear(a, x - 1, y, width))
				: -s01;// (getLinear(a, x + 1, y, width) - phi);
			real tb = x < width - 1
				? (getLinear(a, x + 1, y, width) - phi)
				: s01;// (phi - getLinear(a, x - 1, y, width));
			real tc = y > 0
				? (phi - getLinear(a, x, y - 1, width))
				: -s01;// (getLinear(a, x, y - 1, width) - phi);
			real td = y < height - 1
				? (getLinear(a, x, y + 1, width) - phi)
				: s01;// (phi - getLinear(a, x, y - 1, width));
			real aPos = std::max(real(0), ta), aNeg = std::min(real(0), ta);
			real bPos = std::max(real(0), tb), bNeg = std::min(real(0), tb);
			real cPos = std::max(real(0), tc), cNeg = std::min(real(0), tc);
			real dPos = std::max(real(0), td), dNeg = std::min(real(0), td);

			real g = 0;
			if (phi0 > 0)
				g = sqrt(std::max(aPos*aPos, bNeg*bNeg) + std::max(cPos*cPos, dNeg*dNeg)) - 1;
			else if (phi0 < 0)
				g = sqrt(std::max(aNeg*aNeg, bPos*bPos) + std::max(cNeg*cNeg, dPos*dPos)) - 1;

			cost += 0.5 * square(g);

			gradient(x, y) = s * g;
		}
	}

	return gradient;
}

void ar::GridUtils2D::recoverSDFSussmannGradientAdjoint(const grid_t& input, const grid_t& input0,
    const bgrid_t& canChange, real epsilon, const grid_t& adjOutput, grid_t& adjInput)
{
    int width = input.rows();
    int height = input.cols();
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (!canChange(x, y))
                continue;

            real phi0 = input0(x, y);
            real s = phi0 / sqrt(phi0*phi0 + epsilon);
            real s01 = utils::sgn(s);

            real phi = input(x, y);
            real ta = x > 0
                ? (phi - input(x - 1, y))
                : -s01;// (getLinear(a, x + 1, y, width) - phi);
            real tb = x < width - 1
                ? (input(x + 1, y) - phi)
                : s01;// (phi - getLinear(a, x - 1, y, width));
            real tc = y > 0
                ? (phi - input(x, y - 1))
                : -s01;// (getLinear(a, x, y - 1, width) - phi);
            real td = y < height - 1
                ? (input(x, y + 1) - phi)
                : s01;// (phi - getLinear(a, x, y - 1, width));

            if (phi0 > 0)
            {
                real ap = std::max(ta, real(0)), bm = std::min(tb, real(0)), cp = std::max(tc, real(0)), dm = std::min(td, real(0));
                real ab = std::max(ap*ap, bm*bm), cd = std::max(cp*cp, dm*dm);
                real adjAB = (s / (2 * sqrt(ab + cd))) * adjOutput(x, y), adjCD = adjAB;
                real adjAp = ap * ap > bm*bm ? 0.5*ap*adjAB : 0;
                if (x > 0 && ta > 0) { adjInput(x, y) += adjAp; adjInput(x - 1, y) -= adjAp; }
                real adjBm = ap * ap < bm*bm ? 0.5*bm*adjAB : 0;
                if (x < width - 1 && tb < 0) { adjInput(x, y) -= adjBm; adjInput(x + 1, y) += adjBm; }
                real adjCp = cp * cp > dm*dm ? 0.5*cp*adjCD : 0;
                if (y > 0 && tc > 0) { adjInput(x, y) += adjCp; adjInput(x, y - 1) -= adjCp; }
                real adjDm = cp * cp < dm*dm ? 0.5*dm*adjCD : 0;
                if (y < height - 1 && td < 0) { adjInput(x, y) -= adjDm; adjInput(x, y + 1) += adjDm; }
            } else
            {
                real am = std::min(ta, real(0)), bp = std::max(tb, real(0)), cm = std::min(tc, real(0)), dp = std::max(td, real(0));
                real ab = std::max(am*am, bp*bp), cd = std::max(cm*cm, dp*dp);
                real adjAB = (s / (2 * sqrt(ab + cd))) * adjOutput(x, y), adjCD = adjAB;
                real adjAm = am * am > bp*bp ? 0.5*am*adjAB : 0;
                if (x > 0 && ta < 0) { adjInput(x, y) += adjAm; adjInput(x - 1, y) -= adjAm; }
                real adjBp = am * am < bp*bp ? 0.5*bp*adjAB : 0;
                if (x < width - 1 && tb > 0) { adjInput(x, y) -= adjBp; adjInput(x + 1, y) += adjBp; }
                real adjCm = cm * cm > dp*dp ? 0.5*cm*adjCD : 0;
                if (y > 0 && tc < 0) { adjInput(x, y) += adjCm; adjInput(x, y - 1) -= adjCm; }
                real adjDp = cm * cm < dp*dp ? 0.5*dp*adjCD : 0;
                if (y < height - 1 && td > 0) { adjInput(x, y) -= adjDp; adjInput(x, y + 1) += adjDp; }
            }
        }
    }
}

ar::GridUtils2D::grid_t ar::GridUtils2D::recoverSDFFastMarching(const grid_t& input)
{
    int width = input.rows();
    int height = input.cols();
    grid_t output = input;//(width, height);
    bgrid_t accepted = bgrid_t::Constant(width, height, false);

    //TODO: inside

    //outside loop
    typedef std::tuple<real, int, int, real> et;
    auto comp = [](const et& a, const et& b) {return std::get<0>(a) > std::get<0>(b); };
    std::priority_queue<et, std::vector<et>, decltype(comp)> close(comp);
    //initial poits
    for (int x=0; x<width; ++x) for (int y=0; y<height; ++y)
    {
        if (!utils::outside(input(x, y))) continue; //we are inside
        //find boundary case
        int c = (x > 0 && utils::insideEq(input(x - 1, y))) ? 1 : 0
            | ((x < width - 1 && utils::insideEq(input(x + 1, y))) ? 2 : 0)
            | ((y > 0 && utils::insideEq(input(x, y - 1))) ? 4 : 0)
            | ((y > height - 1 && utils::insideEq(input(x, y + 1))) ? 8 : 0);
        if (c == 0) continue; //completely outside, no boundary
        
        //process boundary case
        real vOld = input(x, y);
        real vNew = 0;
        switch (c) // +y -y +x -x
        {
        case 0b0001:
        case 0b0010:
        case 0b0100:
        case 0b1000:
        { //case a)
            real v = c == 0b0001 ? input(x - 1, y)
                : c == 0b0010 ? input(x + 1, y)
                : c == 0b0100 ? input(x, y - 1)
                : /*c == 0b1000 ?*/ input(x, y + 1);
            real s = v / (v - vOld); assert(s > 0);
            vNew = s;
        }break;
        case 0b0101:
        case 0b1010:
        case 0b0110:
        case 0b1001:
        { //case b)
            real v1 = c == 0b0101 ? input(x - 1, y)
                : c == 0b1010 ? input(x + 1, y)
                : c == 0b0110 ? input(x + 1, y)
                : /*c == 0b1001 ?*/ input(x - 1, y);
            real v2 = c == 0b0101 ? input(x, y - 1)
                : c == 0b1010 ? input(x, y + 1)
                : c == 0b0110 ? input(x, y - 1)
                : /*c == 0b1001 ?*/ input(x, y + 1);
            real s = v1 / (v1 - vOld); assert(s > 0);
            real t = v2 / (v2 - vOld); assert(t > 0);
            vNew = s * t / sqrt(s*s + t * t);
        } break;
        case 0b1110:
        case 0b1101:
        case 0b1011:
        case 0b0111:
        { //case c)
            real vs1 = c == 0b1110 ? input(x, y - 1)
                : c == 0b1101 ? input(x, y + 1)
                : c == 0b1011 ? input(x + 1, y)
                : /*c == 0b0111 ?*/ input(x - 1, y);
            real vs2 = c == 0b1110 ? input(x, y + 1)
                : c == 0b1101 ? input(x, y - 1)
                : c == 0b1011 ? input(x - 1, y)
                : /*c == 0b0111 ?*/ input(x + 1, y);
            real vt = c == 0b1110 ? input(x - 1, y)
                : c == 0b1101 ? input(x + 1, y)
                : c == 0b1011 ? input(x, y + 1)
                : /*c == 0b0111 ?*/ input(x, y - 1);
            real s1 = vs1 / (vs1 - vOld); assert(s1 > 0);
            real s2 = vs2 / (vs2 - vOld); assert(s2 > 0);
            real s = std::min(s1, s2);
            real t = vt / (vt - vOld); assert(t > 0);
            vNew = s * t / sqrt(s*s + t * t);
        } break;
        case 0b0011:
        case 0b1100:
        { //case d)
            real vs1 = c == 0b0011 ? input(x - 1, y)
                : /*c == 0b1100 ?*/ input(x, y - 1);
            real vs2 = c == 0b0011 ? input(x + 1, y)
                : /*c == 0b1100 ?*/ input(x, y + 1);
            real s1 = vs1 / (vs1 - vOld); assert(s1 > 0);
            real s2 = vs2 / (vs2 - vOld); assert(s2 > 0);
            vNew = std::min(s1, s2);
        } break;
        case 0b1111:
        { //case e);
            real vs1 = input(x - 1, y);
            real vs2 = input(x + 1, y);
            real vt1 = input(x, y - 1);
            real vt2 = input(x, y + 1);
            real s1 = vs1 / (vs1 - vOld); assert(s1 > 0);
            real s2 = vs2 / (vs2 - vOld); assert(s2 > 0);
            real t1 = vt1 / (vt1 - vOld); assert(t1 > 0);
            real t2 = vt2 / (vt2 - vOld); assert(t2 > 0);
            real s = std::min(s1, s2);
            real t = std::min(t1, t2);
            vNew = s * t / sqrt(s*s + t * t);
        } break;
        }
        assert(vNew > 0);

        //write output and mark as done
        output(x, y) = vNew;
        
        //add to the queue
        close.emplace(real(0), x, y, vNew);
    }
    //sweep
    while(!close.empty())
    {
        et e = close.top(); close.pop();
        real v = std::get<3>(e);
        int x = std::get<1>(e);
        int y = std::get<2>(e);
        
        if (accepted(x, y)) continue; //already processed

        //mark current as processed
        output(x, y) = v;
        accepted(x, y) = true;

        //add neighbors to the list
        int neighbors[4][2] = { {-1,0}, {+1,0}, {0,-1}, {0,+1} };
        const static real BIG = 1e10;
        for (int i=0; i<4; ++i)
        {
            int ix = x + neighbors[i][0];
            int iy = y + neighbors[i][1];
            if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;
            if (!utils::outside(input(ix, iy))) continue;
            if (accepted(ix, iy)) continue;
            real a = (ix + 1) >= width ? BIG
                : !accepted(ix + 1, iy) ? BIG
                : output(ix + 1, iy);
            real b = (ix - 1) < 0 ? BIG
                : !accepted(ix - 1, iy) ? BIG
                : output(ix - 1, iy);
            real c = (iy + 1) >= height ? BIG
                : !accepted(ix, iy + 1) ? BIG
                : output(ix, iy + 1);
            real d = (iy - 1) < 0 ? BIG
                : !accepted(ix, iy - 1) ? BIG
                : output(ix, iy - 1);
            real ab = std::min(a, b);
            real cd = std::min(c, d);
            real vNew = NAN;
            if (ab < BIG && cd < BIG)
                vNew = 0.5 * sqrt(-square(ab) + 2 * ab*cd - square(cd) + 2) + (ab + cd) / 2;
            else if (ab < BIG)
                vNew = ab + 1;
            else if (cd < BIG)
                vNew = cd + 1;
            close.emplace(vNew, ix, iy, vNew);
        }
    }

    return output;
}
