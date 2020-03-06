#pragma once

#include <thread>
#include <functional>
#include <string>
#include <atomic>
#include <mutex>
#include <memory>

namespace ar
{
    /**
     * \brief A simple background worker
     */
    class BackgroundWorker
    {
    private:
        std::atomic<bool> done;
        std::atomic<bool> interrupted;
        std::thread thread;

        std::mutex statusMutex;
        std::string status;
		int enableStatusLogging = 0;

        static void taskFn(const std::function<void(BackgroundWorker*)>& task, BackgroundWorker* worker);

    public:
        //Default constructor that does nothing
        //isDone() will always return false
        BackgroundWorker();

        //Constructs a new background worker that executes the given task.
        //The task is given a reference to *this, to allow the passing of messages and checking the interupted-flag
        BackgroundWorker(const std::function<void(BackgroundWorker*)>& task);
        ~BackgroundWorker();

        //Sets the status message, called from inside the background task
        void setStatus(const std::string& message);
        //Reads the status message, called from the main thread
        const std::string& getStatus();

		//Disables status logging
		void pushDisableStatusLogging() { enableStatusLogging++; }
		//Enables status logging
		void popDisableStatusLogging() { enableStatusLogging--; }

        //Interrupts the background task, called from the main thread
        void interrupt();
        //Checks if the task is interrupted, called from the background task
        bool isInterrupted() const;

        //Tests if the background task is done
        bool isDone() const;

        //Joins the background task
        void join();

    };
    typedef std::shared_ptr<BackgroundWorker> BackgroundWorkerPtr;

}
