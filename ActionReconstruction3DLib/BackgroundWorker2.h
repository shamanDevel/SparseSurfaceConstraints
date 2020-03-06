#pragma once

#include <ThreadPool.h>
#include <assert.h>
#include <cinder/Log.h>

namespace ar3d
{
	/**
	 * \brief Improved background worker that reuses the thread.
	 * Can only be allocated on the heap
	 */
	class BackgroundWorker2
	{
	public:
		typedef std::function<void(BackgroundWorker2*)> task;

	private:
		ThreadPool pool_;
		mutable std::future<void> future_;
		std::atomic<bool> interrupted_;
		std::mutex statusMutex_;
		std::string status_;
		int enableStatusLogging_ = 0;
		bool logToConsole_ = false;

	public:
		BackgroundWorker2(const BackgroundWorker2&) = delete; 
		void operator=(const BackgroundWorker2&) = delete;

		BackgroundWorker2()
			: pool_(1)
		{
			pool_.init();
			interrupted_.store(false);
		}
		~BackgroundWorker2()
		{
			pool_.shutdown();
		}

		void launch(const task& t)
		{
			assert(isDone());
			interrupted_.store(false);
			status_ = "";
			future_ = pool_.submit(t, this);
		}

		//Sets the status message, called from inside the background task
		void setStatus(const std::string& message)
		{
			std::lock_guard<std::mutex> lock(statusMutex_);
			if (enableStatusLogging_ > 0) return;
			if (isLogToConsole()) CI_LOG_I(message);
			status_ = message;
		}
		//Reads the status message, called from the main thread
		const std::string& getStatus()
		{
			std::lock_guard<std::mutex> lock(statusMutex_);
			return status_;
		}

		//Disables status logging
		void pushDisableStatusLogging() { enableStatusLogging_++; }
		//Enables status logging
		void popDisableStatusLogging() { enableStatusLogging_--; }

		void setLogToConsole(bool enabled) { logToConsole_ = enabled; }
		bool isLogToConsole() const { return logToConsole_; }

		//Interrupts the background task, called from the main thread
		void interrupt() { interrupted_.store(true); }
		//Checks if the task is interrupted, called from the background task
		bool isInterrupted() const { return interrupted_; }

		//Tests if the background task is done
		bool isDone() const
		{
			if (!future_.valid()) return true;
			if (future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
			{
				try {
					future_.get(); //done
				}
				catch (std::exception& ex) {
					CI_LOG_EXCEPTION("Exception in the background thread!", ex);
				}
				catch (...) {
					CI_LOG_E("Unknown exception in the background thread!");
				}
				return true;
			}
			return false;
		}
		/**
		 * \brief Waits until the task is done
		 */
		void wait()
		{
			if (!future_.valid()) return;
			future_.get();
		}
	};
	typedef std::unique_ptr<BackgroundWorker2> BackgroundWorker2Ptr;
}
