#include "MeshReconstruction.h"

#include <cinder/Log.h>
#include <numeric>

#include "GeometryUtils3D.h"
#include <GradientDescent.h>

using namespace std;
using namespace Eigen;

void ar3d::MeshReconstruction::SettingsGui::initParams(cinder::params::InterfaceGlRef params, const std::string& group)
{
	params->addParam("MeshReconstructionResolution", &settings_.gridResolution)
		.group(group).min(4).max(512).label("Grid Resolution");
	params->addParam("MeshReconstructionBorder", &settings_.gridBorder)
		.group(group).min(2).max(10).label("Grid Border");
	params->addParam("MeshReconstructionTruncation", &settings_.truncationDistance)
		.group(group).min(1).max(10).label("delta (Truncation Dist)");
	params->addParam("MeshReconstructionSDFIterations", &settings_.sdfIterations)
		.group(group).min(0).max(100).label("SDF iterations");
	params->addParam("MeshReconstructionSDFViscosity", &settings_.sdfViscosity)
		.group(group).min(0.0).max(10).step(0.0001).label("SDF viscosity");
}

void ar3d::MeshReconstruction::SettingsGui::load(const cinder::JsonTree& parent)
{
	settings_.gridResolution = parent.getValueForKey<int>("Resolution");
	settings_.gridBorder = parent.getValueForKey<int>("Border");
	settings_.truncationDistance = parent.getValueForKey<int>("TruncationDistance");
	settings_.sdfIterations = parent.getValueForKey<int>("SdfIterations");
	settings_.sdfViscosity = parent.getValueForKey<double>("SdfViscosity");
}

void ar3d::MeshReconstruction::SettingsGui::save(cinder::JsonTree& parent) const
{
	parent.addChild(cinder::JsonTree("Resolution", settings_.gridResolution));
	parent.addChild(cinder::JsonTree("Border", settings_.gridBorder));
	parent.addChild(cinder::JsonTree("TruncationDistance", settings_.truncationDistance));
	parent.addChild(cinder::JsonTree("SdfIterations", settings_.sdfIterations));
	parent.addChild(cinder::JsonTree("SdfViscosity", settings_.sdfViscosity));
}

ar3d::MeshReconstruction::MeshReconstruction(InputConfigPtr inputConfig, InputDataLoader::FramePtr frame)
    : inputConfig(inputConfig)
    , frame(frame)
{
}

bool ar3d::MeshReconstruction::runSegmentation(BackgroundWorker2* worker)
{
    int n = inputConfig->cameras.size();
    SegmentationMatrices images(n);
    for (int i = 0; i < n; ++i) {
        if (worker->isInterrupted()) return false;
        //first version of the segmentation
        //trivial algorithm that works only on the test data
        images[i] = frame->cameraImages[i].depthMatrix > 0.0;
        //CI_LOG_D("Depth " << i << ":\n" << frame->cameraImages[i].depthMatrix << "\n");
        //CI_LOG_D("Segmentation " << i << ":\n" << images[i] << "\n");
    }
    segmentedImages = images;
    return true;
}

cinder::gl::Texture2dRef ar3d::MeshReconstruction::getSegmentationTexture(int cam)
{
    if (segmentedImages.size() <= cam)
        return cinder::gl::Texture2dRef(); //not processed yet

    if (segmentedTextures.empty())
        segmentedTextures.resize(segmentedImages.size()); //allocate memory

    if (!segmentedTextures[cam]) {
        //create texture
        const SegmentationMatrix& m = segmentedImages[cam];
        cinder::Surface8u surface(inputConfig->width, inputConfig->height, true);
        cinder::Surface8u::Iter iter = surface.getIter();
        while (iter.line()) {
            while (iter.pixel()) {
                uint8_t c = m(iter.x(), inputConfig->height - iter.y() - 1) ? 250 : 30;
                iter.r() = c;
                iter.g() = c;
                iter.b() = c;
                iter.a() = 150;
            }
        }
        segmentedTextures[cam] = cinder::gl::Texture::create(surface);
    }
    return segmentedTextures[cam];
}

static inline Eigen::Vector3d toEigen(const glm::vec3& vec)
{
    return Eigen::Vector3d(vec.x, vec.y, vec.z);
}

bool ar3d::MeshReconstruction::runBoundingBoxExtraction(BackgroundWorker2* worker)
{
    int n = inputConfig->cameras.size();

    //First bounding box
    //Use the box that contains the camera locations and the origin
    // (in the case there is no camera from below)
    std::vector<Vector3d> points(n + 1);
    for (int i = 0; i < n; ++i) {
        points[i] = toEigen(inputConfig->cameras[i].camera.location);
    }
    points[n] = Vector3d(0, 0, 0); //origin
	ar::geom3d::AABBBox box = ar::geom3d::encloses(points);
    CI_LOG_D("Initial bounding box: (" << box.min.x() << "," << box.min.y() << "," << box.min.z()
        << ") - (" << box.max.x() << "," << box.max.y() << "," << box.max.z() << ")");

    //For each camera
    int w = inputConfig->width;
    int h = inputConfig->height;
    for (int i = 0; i < n; ++i) {
        if (worker->isInterrupted()) return false;

        //Find min and max pixel coordinates of the segmentation mask
        //x
        int minX = w - 1;
        int maxX = 0;
        Eigen::Array<bool, 1, Eigen::Dynamic> maskRow = segmentedImages[i].rowwise().any();
        for (int x = 0; x < w; ++x) {
            if (maskRow[x]) {
                minX = min(minX, x);
                maxX = x;
            }
        }
        //y
        int minY = h - 1;
        int maxY = 0;
        Eigen::Array<bool, Eigen::Dynamic, 1> maskCol = segmentedImages[i].colwise().any();
        for (int y = 0; y < h; ++y) {
            if (maskCol[y]) {
                minY = min(minY, y);
                maxY = y;
            }
        }
        //check result
        if (minX > maxX || minY > maxY) {
            CI_LOG_W("Camera " << i << " does not show the object (segmentation is empty)");
            continue;
        }
        if (worker->isInterrupted()) return false;

        //Project the rect of the mask above into 3d -> pyramid
	    ar::geom3d::Pyramid p;
        const DataCamera& cam = inputConfig->cameras[i].camera;
        p.center = toEigen(cam.location);
        p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(minX / double(w), minY / double(h), 0)) - cam.location).normalized());
        p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(maxX / double(w), minY / double(h), 0)) - cam.location).normalized());
        p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(maxX / double(w), maxY / double(h), 0)) - cam.location).normalized());
        p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(minX / double(w), maxY / double(h), 0)) - cam.location).normalized());
        maskPyramids.push_back(p);

        //Intersect with the box
        box = intersect(box, p);
        CI_LOG_D("Bounding box after camera " << i << ": (" << box.min.x() << "," << box.min.y() << "," << box.min.z()
            << ") - (" << box.max.x() << "," << box.max.y() << "," << box.max.z() << ")");
    }

    //done
    boundingBox = box;
    return true;
}

bool ar3d::MeshReconstruction::runInitReconstruction(BackgroundWorker2* worker)
{
    //Create world grid
    Eigen::Vector3i offset = (boundingBox.min.array() * settings.gridResolution).floor().cast<int>() - settings.gridBorder;
    Eigen::Vector3i size = ((boundingBox.max - boundingBox.min).array() * settings.gridResolution).ceil().cast<int>() + (2 * settings.gridBorder);
    CI_LOG_D("Create world grid: resolution = 1/" << settings.gridResolution << ", offset = ("
        << offset.x() << "," << offset.y() << "," << offset.z() << "), size = ("
        << size.x() << "," << size.y() << "," << size.z() << ")");
    worldGrid = std::make_shared<WorldGrid>(settings.gridResolution, offset, size);
    if (worker->isInterrupted()) return false;

    //Initialize the sdf with a first estimate of the surface
    WorldGridRealDataPtr sdf = std::make_shared<WorldGridData<double> >(worldGrid);
    sdf->allocateHostMemory();
    double voxelSize = worldGrid->getVoxelSize();
    int n = inputConfig->cameras.size();
    int numInsideVoxels = 0;
    for (int z = 0; z < size.z(); ++z) {
        if (worker->isInterrupted()) return false;
#pragma omp parallel for reduction(+:numInsideVoxels)
        for (int y = 0; y < size.y(); ++y) {
            int numInsideVoxelsLocal = 0;
            for (int x = 0; x < size.x(); ++x) {
                Eigen::Vector3d worldPos = (offset.cast<double>() + Eigen::Vector3d(x, y, z)) * voxelSize;
                bool isInside = true;
                for (int i = 0; i < n && isInside; ++i) {
                    const DataCamera& cam = inputConfig->cameras[i].camera;
                    Eigen::Vector3d screenPos = toEigen(cam.getScreenCoordinates(glm::vec3(worldPos.x(), worldPos.y(), worldPos.z())));
                    int screenX = static_cast<int>(screenPos.x() * inputConfig->width);
                    int screenY = static_cast<int>(screenPos.y() * inputConfig->height);
                    if (screenX < 0 || screenY < 0 || screenX >= inputConfig->width || screenY >= inputConfig->height) {
                        isInside = false;
                    }
                    else if (!segmentedImages[i](screenX, screenY)) {
                        isInside = false;
                    }
                }
                //Note: this initialization may be improved by also deleting voxels that are closer than the projected depth
                //I don't do this for two reasons:
                //1. To see improvements to the sdf during the actual reconstruction
                //2. To not overshoot the initialization if the depth channel is noisy
                if (isInside) {
                    sdf->atHost(x, y, z) = -1.0;
                    numInsideVoxelsLocal++;
                }
                else {
                    sdf->atHost(x, y, z) = +1.0;
                }
            }
            numInsideVoxels += numInsideVoxelsLocal;
        }
    }
    CI_LOG_I("SDF initialized, number of inner voxels: " << numInsideVoxels << ", number of outside voxels: " << (size.prod() - numInsideVoxels));
    sdfData = sdf;

    //create projected sdf per camera
    std::vector<WorldGridRealDataPtr> grids(n);
    std::vector<WorldGridRealDataPtr> certainties(n);
    for (int i = 0; i < n; ++i)
    {
        grids[i] = std::make_shared<WorldGridData<real> >(worldGrid); grids[i]->allocateHostMemory();
        certainties[i] = std::make_shared<WorldGridData<real> >(worldGrid); certainties[i]->allocateHostMemory();
    }
    for (int z = 0; z < size.z(); ++z) {
        if (worker->isInterrupted()) return false;
#pragma omp parallel for
        for (int y = 0; y < size.y(); ++y) {
            for (int x = 0; x < size.x(); ++x) {
                Eigen::Vector3d worldPos = (offset.cast<double>() + Eigen::Vector3d(x, y, z)) * voxelSize;
                for (int i = 0; i < n; ++i) {
                    const DataCamera& cam = inputConfig->cameras[i].camera;
                    WorldGridRealDataPtr camSdf = grids[i];
                    WorldGridRealDataPtr certainty = certainties[i];
                    Eigen::Vector3d screenPos = toEigen(cam.getScreenCoordinates(glm::vec3(worldPos.x(), worldPos.y(), worldPos.z())));
                    int screenX = static_cast<int>(round(screenPos.x() * inputConfig->width));
                    int screenY = static_cast<int>(round(screenPos.y() * inputConfig->height));
                    if (screenX < 0 || screenY < 0 || screenX >= inputConfig->width || screenY >= inputConfig->height) {
                        //outside
                        camSdf->atHost(x, y, z) = static_cast<float>(settings.truncationDistance);
                        certainty->atHost(x, y, z) = 1;
                    }
                    else
                    {
                        //inside
                        //This can be optimized
                        //Also, on the GPU, the texel lookup could be done with interpolation
                        double camDepthProj = frame->cameraImages[i].depthMatrix(screenX, screenY);
                        if (camDepthProj > 0.0)
                        {
                            //we have data
                            double camDepth = toEigen(cam.getWorldCoordinates(glm::vec3(screenPos.x(), screenPos.y(), camDepthProj))
                                - cam.location).norm();
                            double voxelDepth = (worldPos - toEigen(cam.location)).norm();
                            double distance = (camDepth - voxelDepth) * settings.gridResolution;
                            //distance contains now the SDF from the optimal surface in voxel units
                            //truncate it
                            double truncatedDistance = std::min(settings.truncationDistance, std::max(-settings.truncationDistance, distance));
                            //compute certainty
                            double c = std::min(1.0, std::max(0.0, 1 + distance / settings.certaintyDistance));
                            if (screenX > 0 && frame->cameraImages[i].depthMatrix(screenX - 1, screenY) == 0) c *= 0.5; //less certain on object boundaries?
                            if (screenX < inputConfig->width-1 && frame->cameraImages[i].depthMatrix(screenX + 1, screenY) == 0) c *= 0.5;
                            if (screenY > 0 && frame->cameraImages[i].depthMatrix(screenX, screenY - 1) == 0) c *= 0.5;
                            if (screenY < inputConfig->height-1 && frame->cameraImages[i].depthMatrix(screenX, screenY + 1) == 0) c *= 0.5;
                            //write into the grid
                            camSdf->atHost(x, y, z) = static_cast<real>(truncatedDistance);
                            certainty->atHost(x, y, z) = static_cast<real>(c);
                        }
                        else
                        {
                            //no data
                            camSdf->atHost(x, y, z) = static_cast<real>(settings.truncationDistance);
                            certainty->atHost(x, y, z) = 1;
                        }
                    }
                }
            }
        }
    }
    projectedSdfPerCamera = grids;
    certaintyPerCamera = certainties;
    CI_LOG_D("projected SDFs per camera created");

    return true;
}

bool ar3d::MeshReconstruction::runReconstruction(BackgroundWorker2 * worker)
{
    return runReconstructionKinectFusion(worker);
}
bool ar3d::MeshReconstruction::runReconstructionTV1(BackgroundWorker2 * worker)
{
    const Eigen::Vector3i size = worldGrid->getSize();
    const size_t numVoxels = size.prod();
    int n = inputConfig->cameras.size();
    //primal variable u
    WorldGridRealDataPtr u = sdfData;

    //create dual variable v
    WorldGridRealDataPtr v = std::make_shared<WorldGridData<double> >(worldGrid);
	v->setHostMemory(u->getHostMemory());

    //helper vectorfield p
    WorldGridRealDataPtr pX[2] = {
        std::make_shared<WorldGridData<double> >(worldGrid),
        std::make_shared<WorldGridData<double> >(worldGrid)
    };
    WorldGridRealDataPtr pY[2] = {
        std::make_shared<WorldGridData<real> >(worldGrid),
        std::make_shared<WorldGridData<real> >(worldGrid)
    };
    WorldGridRealDataPtr pZ[2] = {
        std::make_shared<WorldGridData<real> >(worldGrid),
        std::make_shared<WorldGridData<real> >(worldGrid)
    };

    //run primal dual iteration
    for (int i1 = 0; i1 < settings.maxPrimalDualIterations; ++i1)
    {
        if (worker->isInterrupted()) return false;
        CI_LOG_D("Primal-dual iteration " << (i1 + 1));

        //Optimize dual (fixed u)
        double changeV = 0;
        typedef std::pair<double, double> dd;
#pragma omp parallel for reduction(+:changeV)
        for (int z = 0; z < size.z(); ++z) {
            std::vector<dd> fx;
            double changeVLocal = 0;
            for (int y = 0; y < size.y(); ++y) {
                for (int x = 0; x < size.x(); ++x) {
                    double oldV = v->getHost(x, y, z);
                    double oldU = u->getHost(x, y, z);

                    //collect and sort fx
                    fx.clear();
                    for (int i3 = 0; i3 < n; ++i3)
                    {
                        if (certaintyPerCamera[i3]->getHost(x, y, z) > 0)
                        {
                            fx.emplace_back(projectedSdfPerCamera[i3]->getHost(x, y, z), certaintyPerCamera[i3]->getHost(x, y, z));
                        }
                    }
                    if (fx.empty())
                    {
                        //shortcut solution if no valid datapoints are available
                        v->atHost(x, y, z) = oldU;
                        continue;
                    }
                    std::sort(fx.begin(), fx.end(), [](const dd& a, const dd& b)
                    {
                        return a.first < b.first;
                    });

                    //find minimizer between the f_i's
                    bool found = false;
                    double newV = NAN;
                    double lambdaTheta = settings.dataFidelity * settings.primalDualConnection;
                    double wSumLeft = 0;
                    double wSumRight = std::accumulate(fx.begin(), fx.end(), double(0), [](double b, const dd& a) {return a.second + b; });
                    if (!found) //trivially true, interval [-inf, f1]
                    {
                        double v1 = oldU - lambdaTheta * (wSumLeft - wSumRight);
                        if (v1 <= fx[0].first)
                        {
                            found = true;
                            newV = v1;
                        } else
                        {
                            wSumLeft += fx[0].second;
                            wSumRight += fx[0].second;
                        }
                    }
                    for (int i3=1; i3<fx.size() && !found; ++i3) //interval [f_i-1, f_i]
                    {
                        double v1 = oldU - lambdaTheta * (wSumLeft - wSumRight);
                        if (v1 > fx[i3-1].first && v1 <= fx[i3].first)
                        {
                            found = true;
                            newV = v1;
                        }
                        else
                        {
                            wSumLeft += fx[i3].second;
                            wSumRight += fx[i3].second;
                        }
                    }
                    if (!found) //interval [fn, +inf]
                    {
                        double v1 = oldU - lambdaTheta * (wSumLeft - wSumRight);
                        assert(v1 > fx[fx.size() - 1].first); //we must find it now
                        newV = v1;
                    }

                    v->atHost(x, y, z) = newV;
                    changeVLocal += abs(oldV - newV);
                }
            }
            changeV += changeVLocal;
        }
        CI_LOG_D("Total change to v: " << changeV << " (L1-norm)");

        //Optimize primal (fixed v)
        int currentP = 0;
        memset(pX[currentP]->getHostMemory().data(), 0, sizeof(double) * numVoxels);
        memset(pY[currentP]->getHostMemory().data(), 0, sizeof(double) * numVoxels);
        memset(pZ[currentP]->getHostMemory().data(), 0, sizeof(double) * numVoxels);
        //compute p
        double timeStep = 1 / 6.5; //time step thau <= 1/6
        for (int i2 = 0; i2 < settings.maxROFIteratons; ++i2)
        {
            if (worker->isInterrupted()) return false;
            double change = 0;
#pragma omp parallel for reduction(+:change)
            for (int z = 0; z < size.z(); ++z) {
                double changeLocal = 0;
                for (int y = 0; y < size.y(); ++y) {
                    for (int x = 0; x < size.x(); ++x) {
                        const Vector3i pos(x, y, z);
                        Vector3d g = getGradientOfDivergence(pos, pX[currentP].get(), pY[currentP].get(), pZ[currentP].get(), size)
                            - (1 / settings.primalDualConnection) * getGradient(pos, v.get(), size);
                        Vector3d pOld = Vector3d(pX[currentP]->getHost(x, y, z), pZ[currentP]->getHost(x, y, z), pZ[currentP]->getHost(x, y, z));
                        Vector3d pNew = (pOld + timeStep * g) / (1 + timeStep * g.norm());
                        pX[1 - currentP]->atHost(x, y, z) = pNew.x();
                        pY[1 - currentP]->atHost(x, y, z) = pNew.y();
                        pZ[1 - currentP]->atHost(x, y, z) = pNew.z();
                        changeLocal += (pOld - pNew).norm();
                    }
                }
                change += changeLocal;
            }
            currentP = 1 - currentP;
            CI_LOG_D("P-Iteration " << (i2 + 1) << ": change = " << change);
        }
        //apply p to solve u
        double changeU = 0;
#pragma omp parallel for reduction(+:changeU)
        for (int z = 0; z < size.z(); ++z) {
            double changeULocal = 0;
            for (int y = 0; y < size.y(); ++y) {
                for (int x = 0; x < size.x(); ++x) {
                    double oldU = u->getHost(x, y, z);
                    u->atHost(x, y, z) = v->getHost(x, y, z) - settings.primalDualConnection * getDivergence(Eigen::Vector3i(x, y, z), pX[currentP].get(), pY[currentP].get(), pZ[currentP].get(), size);
                    changeULocal += abs(oldU - u->getHost(x, y, z));
                }
            }
            changeU += changeULocal;
        }
        u->invalidateTexture();
        CI_LOG_D("Total change to u: " << changeU << " (L1-norm)");
    }

    return true;
}

bool ar3d::MeshReconstruction::runReconstructionKinectFusion(BackgroundWorker2 * worker)
{
    const Eigen::Vector3i size = worldGrid->getSize();
    const size_t numVoxels = size.prod();
    int n = inputConfig->cameras.size();
    //the truncated sdf
    WorldGridRealDataPtr u = sdfData;

    for (int z = 0; z < size.z(); ++z) {
        if (worker->isInterrupted()) return false;
#pragma omp parallel for
        for (int y = 0; y < size.y(); ++y) {
            for (int x = 0; x < size.x(); ++x) {
                //compute a weighted average of the input
                double weight = 0;
                double sdf = 0;
                for (int i = 0; i < n; ++i) {
                    double w = certaintyPerCamera[i]->getHost(x, y, z);
                    sdf += w * projectedSdfPerCamera[i]->getHost(x, y, z);
                    weight += w;
                }
                if (weight > 0) {
                    u->atHost(x, y, z) = sdf / weight;
                }
            }
        }
    }
    u->invalidateTexture();
    return true;
}

bool ar3d::MeshReconstruction::runSignedDistanceReconstruction(BackgroundWorker2 * worker)
{
    if (settings.sdfIterations == 0) return true;

    const Eigen::Vector3i size = worldGrid->getSize();
    const size_t numVoxels = size.prod();
    int n = inputConfig->cameras.size();
    //the truncated sdf
    WorldGridRealDataPtr u = sdfData; if (!u->hasHostMemory()) u->allocateHostMemory();
    WorldGridRealDataPtr v = std::make_shared<WorldGridData<real> >(worldGrid); v->allocateHostMemory();
    WorldGridRealDataPtr d[2] = { u, v };
    int currentD = 0;
    //save original for signum
    WorldGridRealDataPtr uOrigin = std::make_shared<WorldGridData<real> >(worldGrid);
	uOrigin->setHostMemory(u->getHostMemory());

    //run optimization
    //NUMERICAL RECOVERY OF THE SIGNED DISTANCE FUNCTION, Tomas Oberhuber, Eq. 4.8 adopted
#if 1
    double epsilon = settings.sdfViscosity;
    double deltaT = 1 / 5.0;
    for (int i = 0; i < 2 * settings.sdfIterations; ++i) {
        for (int z = 0; z < size.z(); ++z) {
            if (worker->isInterrupted()) return false;
#pragma omp parallel for
            for (int y = 0; y < size.y(); ++y) {
                for (int x = 0; x < size.x(); ++x) {
                    Vector3d g = getGradient(Vector3i(x, y, z), d[currentD].get(), size);
                    double l = getLaplacian(Vector3i(x, y, z), d[currentD].get(), size);
                    double update = sgn(uOrigin->getHost(x, y, z)) * (1 - g.norm()) + epsilon * l;
                    d[1 - currentD]->atHost(x, y, z) = d[currentD]->getHost(x, y, z) + deltaT * update;
                }
            }
        }
        currentD = 1 - currentD;
        if (currentD == 1) u->invalidateTexture();
    }
#else
    //use ar::GradientDescent
    auto gradFun = [this, &size, uOrigin](const WorldGridData<real>::HostArray_t& a) {
        //convert array back to the world grid
        WorldGridData<real> currentD(this->worldGrid);
		currentD.setHostMemory(a);
        //allocate result
        WorldGridData<real> grad(this->worldGrid);
        grad.allocateHostMemory();
        //compute new values
        for (int z = 0; z < size.z(); ++z) {
#pragma omp parallel for
            for (int y = 0; y < size.y(); ++y) {
                for (int x = 0; x < size.x(); ++x) {
                    Vector3d g = getGradient(Vector3i(x, y, z), &currentD, size);
                    double l = getLaplacian(Vector3i(x, y, z), &currentD, size);
                    double update = sgn(uOrigin->getHost(x, y, z)) * (1 - g.norm()) + this->settings.sdfViscosity * l;
                    grad.atHost(x, y, z) = -update;
                }
            }
        }
        //return array
        return grad.getHostMemory();
    };
    ar::GradientDescent<WorldGridData<real>::HostArray_t> gd(u->getHostMemory(), gradFun);
    for (int i = 0; i < settings.sdfIterations; ++i) {
        if (worker->isInterrupted()) return false;
        bool end = gd.step();
        const auto& current = gd.getCurrentSolution();
		u->setHostMemory(current);
        u->invalidateTexture();
        CI_LOG_I("Iteration " << i << ", step size: " << gd.getLastStepSize());
        if (end) break;
    }
#endif

    return true;
}

std::vector<ci::vec3> ar3d::MeshReconstruction::getProjectedPoints(InputConfigPtr inputConfig, InputDataLoader::FramePtr frame)
{
	std::vector<ci::vec3> v;
	int w = inputConfig->width;
	int h = inputConfig->height;
	for (int n = 0; n < inputConfig->cameras.size(); ++n) {
		const auto& cam = inputConfig->cameras[n].camera;
		const auto& img = frame->cameraImages[n];
		for (int x = 0; x < w; ++x) for (int y = 0; y < h; ++y)
		{
			float d = img.depthMatrix(x, y);
			if (d <= 0.001 || d >= 0.999) continue;
			glm::vec3 screenPos(x / float(w), y / float(h), d);
			glm::vec3 worldPosGlm = cam.getWorldCoordinates(screenPos);
			v.push_back(worldPosGlm);
		}
	}
	return v;
}

double ar3d::MeshReconstruction::getDivergence(const Eigen::Vector3i& pos, const WorldGridData<real>* pX, const WorldGridData<real>* pY, const WorldGridData<real>* pZ, const Eigen::Vector3i & size)
{
    double div = 0;

    //x
    if (pos.x() == 0)
        div += pX->getHost(pos.x() + 1, pos.y(), pos.z()) - pX->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.x() == size.x() - 1)
        div += pX->getHost(pos.x(), pos.y(), pos.z()) - pX->getHost(pos.x() - 1, pos.y(), pos.z()); //backward
    else
        div += 0.5 * (pX->getHost(pos.x() + 1, pos.y(), pos.z()) - pX->getHost(pos.x() - 1, pos.y(), pos.z())); //central

    //y
    if (pos.y() == 0)
        div += pY->getHost(pos.x(), pos.y() + 1, pos.z()) - pY->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.y() == size.y() - 1)
        div += pY->getHost(pos.x(), pos.y(), pos.z()) - pY->getHost(pos.x(), pos.y() - 1, pos.z()); //backward
    else
        div += 0.5 * (pY->getHost(pos.x(), pos.y() + 1, pos.z()) - pY->getHost(pos.x(), pos.y() - 1, pos.z())); //central

    //z
    if (pos.z() == 0)
        div += pZ->getHost(pos.x(), pos.y(), pos.z() + 1) - pZ->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.z() == size.z() - 1)
        div += pZ->getHost(pos.x(), pos.y(), pos.z()) - pZ->getHost(pos.x(), pos.y(), pos.z() - 1); //backward
    else
        div += 0.5 * (pZ->getHost(pos.x(), pos.y(), pos.z() + 1) - pZ->getHost(pos.x(), pos.y(), pos.z() - 1)); //central

    return div;
}

Eigen::Vector3d ar3d::MeshReconstruction::getGradient(const Eigen::Vector3i & pos, const WorldGridData<real>* v, const Eigen::Vector3i & size)
{
    Eigen::Vector3d grad;

    //x
    if (pos.x() == 0)
        grad.x() = v->getHost(pos.x() + 1, pos.y(), pos.z()) - v->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.x() == size.x() - 1)
        grad.x() = v->getHost(pos.x(), pos.y(), pos.z()) - v->getHost(pos.x() - 1, pos.y(), pos.z()); //backward
    else
        grad.x() = 0.5 * (v->getHost(pos.x() + 1, pos.y(), pos.z()) - v->getHost(pos.x() - 1, pos.y(), pos.z())); //central

    //y
    if (pos.y() == 0)
        grad.y() = v->getHost(pos.x(), pos.y() + 1, pos.z()) - v->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.y() == size.y() - 1)
        grad.y() = v->getHost(pos.x(), pos.y(), pos.z()) - v->getHost(pos.x(), pos.y() - 1, pos.z()); //backward
    else
        grad.y() = 0.5 * (v->getHost(pos.x(), pos.y() + 1, pos.z()) - v->getHost(pos.x(), pos.y() - 1, pos.z())); //central

    //z
    if (pos.z() == 0)
        grad.z() = v->getHost(pos.x(), pos.y(), pos.z() + 1) - v->getHost(pos.x(), pos.y(), pos.z()); //forward
    else if (pos.z() == size.z() - 1)
        grad.z() = v->getHost(pos.x(), pos.y(), pos.z()) - v->getHost(pos.x(), pos.y(), pos.z() - 1); //backward
    else
        grad.z() = 0.5 * (v->getHost(pos.x(), pos.y(), pos.z() + 1) - v->getHost(pos.x(), pos.y(), pos.z() - 1)); //central

    return grad;
}

double ar3d::MeshReconstruction::getLaplacian(const Eigen::Vector3i & pos, const WorldGridData<real>* v, const Eigen::Vector3i & size)
{
    double laplacian = 0;

    //x
    if (pos.x() == 0)
        laplacian += v->getHost(pos.x(), pos.y(), pos.z()) - 2 * v->getHost(pos.x() + 1, pos.y(), pos.z()) + v->getHost(pos.x() + 2, pos.y(), pos.z());
    else if (pos.x() == size.x() - 1)
        laplacian += v->getHost(pos.x() - 2, pos.y(), pos.z()) - 2 * v->getHost(pos.x() - 1, pos.y(), pos.z()) + v->getHost(pos.x(), pos.y(), pos.z());
    else
        laplacian += v->getHost(pos.x() - 1, pos.y(), pos.z()) - 2 * v->getHost(pos.x(), pos.y(), pos.z()) + v->getHost(pos.x() + 1, pos.y(), pos.z());

    //y
    if (pos.y() == 0)
        laplacian += v->getHost(pos.x(), pos.y(), pos.z()) - 2 * v->getHost(pos.x(), pos.y() + 1, pos.z()) + v->getHost(pos.x(), pos.y() + 2, pos.z());
    else if (pos.y() == size.y() - 1)
        laplacian += v->getHost(pos.x(), pos.y() - 2, pos.z()) - 2 * v->getHost(pos.x(), pos.y() - 1, pos.z()) + v->getHost(pos.x(), pos.y(), pos.z());
    else
        laplacian += v->getHost(pos.x(), pos.y() - 1, pos.z()) - 2 * v->getHost(pos.x(), pos.y(), pos.z()) + v->getHost(pos.x(), pos.y() + 1, pos.z());

    //z
    if (pos.z() == 0)
        laplacian += v->getHost(pos.x(), pos.y(), pos.z()) - 2 * v->getHost(pos.x(), pos.y(), pos.z() + 1) + v->getHost(pos.x(), pos.y(), pos.z() + 2);
    else if (pos.z() == size.z() - 1)
        laplacian += v->getHost(pos.x(), pos.y(), pos.z() - 2) - 2 * v->getHost(pos.x(), pos.y(), pos.z() - 1) + v->getHost(pos.x(), pos.y(), pos.z());
    else
        laplacian += v->getHost(pos.x(), pos.y(), pos.z() - 1) - 2 * v->getHost(pos.x(), pos.y(), pos.z()) + v->getHost(pos.x(), pos.y(), pos.z() + 1);

    return laplacian;
}

Eigen::Vector3d ar3d::MeshReconstruction::getGradientOfDivergence(const Eigen::Vector3i & pos,
    const WorldGridData<real>* pX, const WorldGridData<real>* pY, const WorldGridData<real>* pZ, const Eigen::Vector3i & size)
{
    Eigen::Vector3d g;
    int x = pos.x();
    int y = pos.y();
    int z = pos.z();

    auto clamp = [&size](int& x, int& y, int& z) mutable
    {
        x = std::max(0, std::min(size.x() - 1, x));
        y = std::max(0, std::min(size.y() - 1, x));
        z = std::max(0, std::min(size.z() - 1, x));
    };
    auto getX = [pX, &clamp](int x, int y, int z)
    {
        clamp(x, y, z);
        return pX->getHost(x, y, z);
    };
    auto getY = [pY, &clamp](int x, int y, int z)
    {
        clamp(x, y, z);
        return pY->getHost(x, y, z);
    };
    auto getZ = [pZ, &clamp](int x, int y, int z)
    {
        clamp(x, y, z);
        return pZ->getHost(x, y, z);
    };

    //x
    g.x() = (getX(x - 1, y, z) - 2 * getX(x, y, z) + getX(x + 1, y, z))
        + 0.25 * (-getY(x - 1, y - 1, z) + getY(x + 1, y - 1, z) + getY(x - 1, y + 1, z) - getY(x + 1, y + 1, z))
        + 0.25 * (-getZ(x - 1, y, z - 1) + getZ(x + 1, y, z - 1) + getZ(x - 1, y, z + 1) - getZ(x + 1, y, z + 1));

    //y
    g.y() = 0.25 * (-getX(x - 1, y - 1, z) + getX(x + 1, y - 1, z) + getX(x - 1, y + 1, z) - getX(x + 1, y + 1, z))
        + (getY(x, y - 1, z) - 2 * getY(x, y, z) + getY(x, y + 1, z))
        + 0.25 * (-getZ(x, y - 1, z - 1) + getZ(x, y + 1, z - 1) + getZ(x, y - 1, z + 1) - getZ(x, y + 1, z + 1));

    //z
    g.z() = 0.25 * (-getX(x - 1, y, z - 1) + getX(x + 1, y, z - 1) + getX(x - 1, y, z + 1) + getX(x + 1, y, z + 1))
        + 0.25 * (-getY(x, y - 1, z - 1) + getY(x, y + 1, z - 1) + getY(x, y - 1, z + 1) - getY(x, y + 1, z + 1))
        + (getZ(x, y, z - 1) - 2 * getZ(x, y, z) + getZ(x, y, z + 1));

    return g;
}

