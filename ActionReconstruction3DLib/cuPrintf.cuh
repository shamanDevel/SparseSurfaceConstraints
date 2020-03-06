/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#ifndef CUPRINTF_H
#define CUPRINTF_H

#include <cuda_runtime.h>
#include <ostream>
#include <boost/preprocessor/repetition.hpp>

/*
*      This is the header file supporting cuPrintf.cu and defining both
*      the host and device-side interfaces. See that file for some more
*      explanation and sample use code. See also below for details of the
*      host-side interfaces.
*
*  Quick sample code:
*
#include "cuPrintf.cuh"

__global__ void testKernel(int val)
{
cuPrintf("Value is: %d\n", val);
}
int main()
{
cudaPrintfInit();
testKernel<<< 2, 3 >>>(10);
cudaPrintfDisplay(stdout, true);
cudaPrintfEnd();
return 0;
}
*/

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// function definitions for device-side code

#ifndef CUPRINTF_DEBUG_PRINT
#if defined(NDEBUG) || defined(_NDEBUG)
#define CUPRINTF_DEBUG_PRINT 0
#else
#define CUPRINTF_DEBUG_PRINT 1
#endif
#endif

// Abuse of templates to simulate varargs
inline __device__ int cuPrintf(const char *fmt);
template <typename T1> __device__ int cuPrintf(const char *fmt, T1 arg1);
template <typename T1, typename T2> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12);

#if CUPRINTF_DEBUG_PRINT==1
#define cuPrintf_D(...) cuPrintf(__VA_ARGS__)
#else
#define cuPrintf_D(...) ((void)0)
#endif

//
//      cuPrintfRestrict
//
//      Called to restrict output to a given thread/block. Pass
//      the constant CUPRINTF_UNRESTRICTED to unrestrict output
//      for thread/block IDs. Note you can therefore allow
//      "all printfs from block 3" or "printfs from thread 2
//      on all blocks", or "printfs only from block 1, thread 5".
//
//      Arguments:
//              threadid - Thread ID to allow printfs from
//              blockid - Block ID to allow printfs from
//
//      NOTE: Restrictions last between invocations of
//      kernels unless cudaPrintfInit() is called again.
//
#define CUPRINTF_UNRESTRICTED   -1
__device__ void cuPrintfRestrict(int threadid, int blockid);



///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

//
//      cudaPrintfInit
//
//      Call this once to initialise the printf system. If the output
//      file or buffer size needs to be changed, call cudaPrintfEnd()
//      before re-calling cudaPrintfInit().
//
//      The default size for the buffer is 1 megabyte. For CUDA
//      architecture 1.1 and above, the buffer is filled linearly and
//      is completely used;     however for architecture 1.0, the buffer
//      is divided into as many segments are there are threads, even
//      if some threads do not call cuPrintf().
//
//      Arguments:
//              bufferLen - Length, in bytes, of total space to reserve
//                          (in device global memory) for output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern cudaError_t cudaPrintfInit(size_t bufferLen = 1048576);   // 1-meg - that's enough for 4096 printfs by all threads put together

																	 //
																	 //      cudaPrintfEnd
																	 //
																	 //      Cleans up all memories allocated by cudaPrintfInit().
																	 //      Call this at exit, or before calling cudaPrintfInit() again.
																	 //
extern void cudaPrintfEnd();

//
//      cudaPrintfDisplay
//
//      Dumps the contents of the output buffer to the specified
//      file pointer. If the output pointer is not specified,
//      the default "stdout" is used.
//
//      Arguments:
//              os           - The output stream
//              showThreadID - If "true", output strings are prefixed
//                             by "[blockid, threadid] " at output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern void cudaPrintfDisplay(std::ostream& os, bool showThreadID = false);

#if CUPRINTF_DEBUG_PRINT==1
//
//      cudaPrintfDisplay - but only in debug mode
//
//      Dumps the contents of the output buffer to the specified
//      file pointer. If the output pointer is not specified,
//      the default "stdout" is used.
//
//      Arguments:
//              os           - The output stream
//              showThreadID - If "true", output strings are prefixed
//                             by "[blockid, threadid] " at output.
//
//      Returns:
//              cudaSuccess if all is well.
//
#define cudaPrintfDisplay_D(...) cudaPrintfDisplay(__VA_ARGS__)
#else
#define cudaPrintfDisplay_D(...) ((void)0)
#endif


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE - Implementation
// External function definitions for device-side code

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single printf() can take up
#define CUPRINTF_MAX_LEN 2048

// Because we can't write an element which is not aligned to its bit-size,
// we have to align all sizes and variables on maximum-size boundaries.
// That means sizeof(double) in this case, but we'll use (long long) for
// better arch<1.3 support
#define CUPRINTF_ALIGN_SIZE      sizeof(long long)

// This is the maximal number of parameters that are possible to pass to cuPrintf
// This defines how many functions are created
#define CUPRINTF_MAX_ARGS 25

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct __align__(8) {
	unsigned short magic;                   // Magic number says we're valid
	unsigned short fmtoffset;               // Offset of fmt string into buffer
	unsigned short blockid;                 // Block ID of author
	unsigned short threadid;                // Thread ID of author
} cuPrintfHeader;

extern __device__ char *cuPrintfGetNextPrintfBufPtr();
extern __device__ void cuPrintfWritePrintfHeader(char *ptr, char *fmtptr);
extern __device__ char *cuPrintfStrncpy(char *dest, const char *src, int n, char *end);

template <typename T>
__device__ static char *cuPrintfCopyArg(char *ptr, T &arg, char *end)
{
	// Initisalisation and overflow check. Alignment rules mean that
	// we're at least CUPRINTF_ALIGN_SIZE away from "end", so we only need
	// to check that one offset.
	if (!ptr || ((ptr + CUPRINTF_ALIGN_SIZE) >= end))
		return NULL;

	// Write the length and argument
	*(int *)(void *)ptr = sizeof(arg);
	ptr += CUPRINTF_ALIGN_SIZE;
	*(T *)(void *)ptr = arg;
	ptr += CUPRINTF_ALIGN_SIZE;
	*ptr = 0;

	return ptr;
}

__device__ static char *cuPrintfCopyArg(char *ptr, const char *arg, char *end)
{
	// Initialisation check
	if (!ptr || !arg)
		return NULL;

	// strncpy does all our work. We just terminate.
	if ((ptr = cuPrintfStrncpy(ptr, arg, CUPRINTF_MAX_LEN, end)) != NULL)
		*ptr = 0;

	return ptr;
}

//
//  cuPrintf
//
//  Templated printf functions to handle multiple arguments.
//  Note we return the total amount of data copied, not the number
//  of characters output. But then again, who ever looks at the
//  return from printf() anyway?
//
//  The format is to grab a block of circular buffer space, the
//  start of which will hold a header and a pointer to the format
//  string. We then write in all the arguments, and finally the
//  format string itself. This is to make it easy to prevent
//  overflow of our buffer (we support up to 10 arguments, each of
//  which can be 12 bytes in length - that means that only the
//  format string (or a %s) can actually overflow; so the overflow
//  check need only be in the strcpy function.
//
//  The header is written at the very last because that's what
//  makes it look like we're done.
//
//  Errors, which are basically lack-of-initialisation, are ignored
//  in the called functions because NULL pointers are passed around
//

// All printf variants basically do the same thing, setting up the
// buffer, writing all arguments, then finalising the header. For
// clarity, we'll pack the code into some big macros.
#define CUPRINTF_PREAMBLE \
    char *start, *end, *bufptr, *fmtstart; \
    if((start = cuPrintfGetNextPrintfBufPtr()) == NULL) return 0; \
    end = start + CUPRINTF_MAX_LEN; \
    bufptr = start + sizeof(cuPrintfHeader);

// Posting an argument is easy
#define CUPRINTF_ARG(argname) \
        bufptr = cuPrintfCopyArg(bufptr, argname, end);

// After args are done, record start-of-fmt and write the fmt and header
#define CUPRINTF_POSTAMBLE \
    fmtstart = bufptr; \
    end = cuPrintfStrncpy(bufptr, fmt, CUPRINTF_MAX_LEN, end); \
    cuPrintfWritePrintfHeader(start, end ? fmtstart : NULL); \
    return end ? (int)(end - start) : 0;

inline __device__ int cuPrintf(const char *fmt)
{
	CUPRINTF_PREAMBLE;

	CUPRINTF_POSTAMBLE;
}

#define CUPRINTF_PP_ARG(z, n, unused) CUPRINTF_ARG(BOOST_PP_CAT(p, n));

#define CUPRINTF_PP_FUNC(z, n, unused)                                                                  \
  template <BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), class T)>                                        \
  __device__ int cuPrintf(const char *fmt, BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), T, p) ) {  \
      CUPRINTF_PREAMBLE;                                                                                \
      BOOST_PP_REPEAT(BOOST_PP_INC(n), CUPRINTF_PP_ARG, ~)                                              \
      CUPRINTF_POSTAMBLE;                                                                               \
  }

BOOST_PP_REPEAT(CUPRINTF_MAX_ARGS, CUPRINTF_PP_FUNC, ~)

#undef CUPRINTF_PP_ARG
#undef CUPRINTF_PP_FUNC

#undef CUPRINTF_PREAMBLE
#undef CUPRINTF_ARG
#undef CUPRINTF_POSTAMBLE

#endif  // CUPRINTF_H