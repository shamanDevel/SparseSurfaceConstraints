#include "BackgroundWorker.h"

#include <cinder/Log.h>

void ar::BackgroundWorker::taskFn(const std::function<void(BackgroundWorker*)>& task, BackgroundWorker * worker)
{
    task(worker);
    worker->done.store(true);
}

ar::BackgroundWorker::BackgroundWorker()
    : done(false)
    , interrupted(false)
    , thread()
{
}

ar::BackgroundWorker::BackgroundWorker(const std::function<void(BackgroundWorker*)>& task)
    : done(false)
    , interrupted(false)
{
    thread = std::thread(taskFn, task, this);
}

ar::BackgroundWorker::~BackgroundWorker()
{
    if (!isDone()) {
        CI_LOG_D("cancel background worker");
    }
    interrupt();
    if (thread.joinable()) {
        thread.join();
    }
}

void ar::BackgroundWorker::setStatus(const std::string & message)
{
    std::lock_guard<std::mutex> lock(statusMutex);
	if (enableStatusLogging > 0) return;
    status = message;
    CI_LOG_D("new status: " << message);
}

const std::string & ar::BackgroundWorker::getStatus()
{
    std::lock_guard<std::mutex> lock(statusMutex);
    return status;
}

void ar::BackgroundWorker::interrupt()
{
    interrupted.store(true);
}

bool ar::BackgroundWorker::isInterrupted() const
{
    return interrupted;
}

bool ar::BackgroundWorker::isDone() const
{
    return done;
}

void ar::BackgroundWorker::join()
{
    thread.join();
}
