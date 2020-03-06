#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <cuMat/src/Errors.h>

namespace ar3d
{
	class CudaTimer
	{
		std::chrono::time_point<std::chrono::high_resolution_clock> start_;
		std::chrono::time_point<std::chrono::high_resolution_clock> end_;

	public:
		CudaTimer() {}

		//Resets and starts the timer
		void start()
		{
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			start_ = std::chrono::high_resolution_clock::now();
		}
		//Stops the timer
		void stop()
		{
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			end_ = std::chrono::high_resolution_clock::now();
		}
		//Returns the elapsed time in seconds.
		double duration() const
		{
			std::chrono::duration<double> elapsed = end_ - start_;
			return elapsed.count();
		}
	};
}