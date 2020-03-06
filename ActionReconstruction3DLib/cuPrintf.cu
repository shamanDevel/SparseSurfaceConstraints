#include "cuPrintf.cuh"
#include <device_launch_parameters.h>
#include "cuMat/src/Errors.h"

// This structure is used internally to track block/thread output restrictions.
typedef struct __align__(8) {
	int threadid;                           // CUPRINTF_UNRESTRICTED for unrestricted
	int blockid;                            // CUPRINTF_UNRESTRICTED for unrestricted
} cuPrintfRestriction;

// The main storage is in a global print buffer, which has a known
// start/end/length. These are atomically updated so it works as a
// circular buffer.
// Since the only control primitive that can be used is atomicAdd(),
// we cannot wrap the pointer as such. The actual address must be
// calculated from printfBufferPtr by mod-ing with printfBufferLength.
// For sm_10 architecture, we must subdivide the buffer per-thread
// since we do not even have an atomic primitive.
__constant__ char *globalPrintfBuffer = nullptr;         // Start of circular buffer (set up by host)
__constant__ int printfBufferLength = 0;              // Size of circular buffer (set up by host)
__device__ cuPrintfRestriction restrictRules;         // Output restrictions
__device__ volatile char *printfBufferPtr = nullptr;     // Current atomically-incremented non-wrapped offset


// All our headers are prefixed with a magic number so we know they're ready
#define CUPRINTF_SM11_MAGIC  (unsigned short)0xC811        // Not a valid ascii character


//
//  getNextPrintfBufPtr
//
//  Grabs a block of space in the general circular buffer, using an
//  atomic function to ensure that it's ours. We handle wrapping
//  around the circular buffer and return a pointer to a place which
//  can be written to.
//
//  Important notes:
//      1. We always grab CUPRINTF_MAX_LEN bytes
//      2. Because of 1, we never worry about wrapping around the end
//      3. Because of 1, printfBufferLength *must* be a factor of CUPRINTF_MAX_LEN
//
//  This returns a pointer to the place where we own.
//
__device__ char *cuPrintfGetNextPrintfBufPtr()
{
	// Initialisation check
	if (!printfBufferPtr)
		return NULL;

	// Thread/block restriction check
	if ((restrictRules.blockid != CUPRINTF_UNRESTRICTED) && (restrictRules.blockid != (blockIdx.x + gridDim.x*blockIdx.y)))
		return NULL;
	if ((restrictRules.threadid != CUPRINTF_UNRESTRICTED) && (restrictRules.threadid != (threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z)))
		return NULL;

	size_t offset = atomicAdd((unsigned int *)&printfBufferPtr, CUPRINTF_MAX_LEN) - (size_t)globalPrintfBuffer;
	offset %= printfBufferLength;
	return globalPrintfBuffer + offset;
}

//
//  writePrintfHeader
//
//  Inserts the header for containing our UID, fmt position and
//  block/thread number. We generate it dynamically to avoid
//      issues arising from requiring pre-initialisation.
//
__device__ void cuPrintfWritePrintfHeader(char *ptr, char *fmtptr)
{
	if (ptr)
	{
		cuPrintfHeader header;
		header.magic = CUPRINTF_SM11_MAGIC;
		header.fmtoffset = (unsigned short)(fmtptr - ptr);
		header.blockid = blockIdx.x + gridDim.x*blockIdx.y;
		header.threadid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
		*(cuPrintfHeader *)(void *)ptr = header;
	}
}


//
//  cuPrintfStrncpy
//
//  This special strncpy outputs an aligned length value, followed by the
//  string. It then zero-pads the rest of the string until a 64-aligned
//  boundary. The length *includes* the padding. A pointer to the byte
//  just after the \0 is returned.
//
//  This function could overflow CUPRINTF_MAX_LEN characters in our buffer.
//  To avoid it, we must count as we output and truncate where necessary.
//
__device__ char *cuPrintfStrncpy(char *dest, const char *src, int n, char *end)
{
	// Initialisation and overflow check
	if (!dest || !src || (dest >= end))
		return NULL;

	// Prepare to write the length specifier. We're guaranteed to have
	// at least "CUPRINTF_ALIGN_SIZE" bytes left because we only write out in
	// chunks that size, and CUPRINTF_MAX_LEN is aligned with CUPRINTF_ALIGN_SIZE.
	int *lenptr = (int *)(void *)dest;
	int len = 0;
	dest += CUPRINTF_ALIGN_SIZE;

	// Now copy the string
	while (n--)
	{
		if (dest >= end)     // Overflow check
			break;

		len++;
		*dest++ = *src;
		if (*src++ == '\0')
			break;
	}

	// Now write out the padding bytes, and we have our length.
	while ((dest < end) && (((size_t)dest & (CUPRINTF_ALIGN_SIZE - 1)) != 0))
	{
		len++;
		*dest++ = 0;
	}
	*lenptr = len;
	return (dest < end) ? dest : NULL;        // Overflow means return NULL
}

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

#include <stdio.h>
#include <ostream>

static char *printfbuf_start = NULL;
static char *printfbuf_device = NULL;
static int printfbuf_len = 0;

static constexpr int BUFFER_SIZE = 1024;
static char buffer[BUFFER_SIZE];

//
//  outputPrintfData
//
//  Our own internal function, which takes a pointer to a data buffer
//  and passes it through libc's printf for output.
//
//  We receive the formate string and a pointer to where the data is
//  held. We then run through and print it out.
//
//  Returns 0 on failure, 1 on success
//
static int outputPrintfData(char *fmt, char *data, std::ostream& os)
{
	// Format string is prefixed by a length that we don't need
	fmt += CUPRINTF_ALIGN_SIZE;

	// Now run through it, printing everything we can. We must
	// run to every % character, extract only that, and use printf
	// to format it.
	char *p = strchr(fmt, '%');
	while (p != NULL)
	{
		// Print up to the % character
		*p = '\0';
		os << fmt;
		*p = '%';           // Put back the %

							// Now handle the format specifier
		char *format = p++;         // Points to the '%'
		p += strcspn(p, "%cdiouxXeEfgGaAnps");
		if (*p == '\0')              // If no format specifier, print the whole thing
		{
			fmt = format;
			break;
		}

		// Cut out the format bit and use printf to print it. It's prefixed
		// by its length.
		int arglen = *(int *)data;
		if (arglen > CUPRINTF_MAX_LEN)
		{
			os << "Corrupt printf buffer data - aborting\n";
			return 0;
		}

		data += CUPRINTF_ALIGN_SIZE;

		char specifier = *p++;
		char c = *p;        // Store for later
		*p = '\0';
		switch (specifier)
		{
			// These all take integer arguments
		case 'c':
		case 'd':
		case 'i':
		case 'o':
		case 'u':
		case 'x':
		case 'X':
		case 'p':
			snprintf(buffer, BUFFER_SIZE, format, *((int *)data));
			os << buffer;
			break;

			// These all take double arguments
		case 'e':
		case 'E':
		case 'f':
		case 'g':
		case 'G':
		case 'a':
		case 'A':
			if (arglen == 4)     // Float vs. Double thing
				snprintf(buffer, BUFFER_SIZE, format, *((float *)data));
			else
				snprintf(buffer, BUFFER_SIZE, format, *((double *)data));
			os << buffer;
			break;

			// Strings are handled in a special way
		case 's':
			snprintf(buffer, BUFFER_SIZE, format, *((char *)data));
			os << buffer;
			break;

			// % is special
		case '%':
			os << "%";
			break;

			// Everything else is just printed out as-is
		default:
			os << format;
			break;
		}
		data += CUPRINTF_ALIGN_SIZE;         // Move on to next argument
		*p = c;                     // Restore what we removed
		fmt = p;                    // Adjust fmt string to be past the specifier
		p = strchr(fmt, '%');       // and get the next specifier
	}

	// Print out the last of the string
	os << fmt;
	return 1;
}


//
//  doPrintfDisplay
//
//  This runs through the blocks of CUPRINTF_MAX_LEN-sized data, calling the
//  print function above to display them. We've got this separate from
//  cudaPrintfDisplay() below so we can handle the SM_10 architecture
//  partitioning.
//
static int doPrintfDisplay(int headings, int clear, char *bufstart, char *bufend, char *bufptr, char *endptr, std::ostream& os)
{
	// Grab, piece-by-piece, each output element until we catch
	// up with the circular buffer end pointer
	int printf_count = 0;
	char printfbuf_local[CUPRINTF_MAX_LEN + 1];
	printfbuf_local[CUPRINTF_MAX_LEN] = '\0';

	while (bufptr != endptr)
	{
		// Wrap ourselves at the end-of-buffer
		if (bufptr == bufend)
			bufptr = bufstart;

		// Adjust our start pointer to within the circular buffer and copy a block.
		cudaMemcpy(printfbuf_local, bufptr, CUPRINTF_MAX_LEN, cudaMemcpyDeviceToHost);

		// If the magic number isn't valid, then this write hasn't gone through
		// yet and we'll wait until it does (or we're past the end for non-async printfs).
		cuPrintfHeader *hdr = (cuPrintfHeader *)printfbuf_local;
		if ((hdr->magic != CUPRINTF_SM11_MAGIC) || (hdr->fmtoffset >= CUPRINTF_MAX_LEN))
		{
			//fprintf(printf_fp, "Bad magic number in printf header\n");
			break;
		}

		// Extract all the info and get this printf done
		if (headings) {
			snprintf(buffer, BUFFER_SIZE, "[%d, %d]: ", hdr->blockid, hdr->threadid);
			os << buffer;
		}
		if (hdr->fmtoffset == 0)
			os << "printf buffer overflow\n";
		else if (!outputPrintfData(printfbuf_local + hdr->fmtoffset, printfbuf_local + sizeof(cuPrintfHeader), os))
			break;
		printf_count++;

		// Clear if asked
		if (clear)
			cudaMemset(bufptr, 0, CUPRINTF_MAX_LEN);

		// Now advance our start location, because we're done, and keep copying
		bufptr += CUPRINTF_MAX_LEN;
	}

	return printf_count;
}


//
//  cudaPrintfInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
cudaError_t cudaPrintfInit(size_t bufferLen)
{
	// Fix up bufferlen to be a multiple of CUPRINTF_MAX_LEN
	bufferLen = (bufferLen < (size_t)CUPRINTF_MAX_LEN) ? CUPRINTF_MAX_LEN : bufferLen;
	if ((bufferLen % CUPRINTF_MAX_LEN) > 0)
		bufferLen += (CUPRINTF_MAX_LEN - (bufferLen % CUPRINTF_MAX_LEN));
	printfbuf_len = (int)bufferLen;

	// Allocate a print buffer on the device and zero it
	if (cudaMalloc((void **)&printfbuf_device, printfbuf_len) != cudaSuccess)
		return cudaErrorInitializationError;
	CUMAT_SAFE_CALL(cudaMemset(printfbuf_device, 0, printfbuf_len));
	printfbuf_start = printfbuf_device;         // Where we start reading from

												// No restrictions to begin with
	cuPrintfRestriction restrict;
	restrict.threadid = restrict.blockid = CUPRINTF_UNRESTRICTED;
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(restrictRules, &restrict, sizeof(restrict)));

	// Initialise the buffer and the respective lengths/pointers.
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(globalPrintfBuffer, &printfbuf_device, sizeof(char *)));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(printfBufferPtr, &printfbuf_device, sizeof(char *)));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(printfBufferLength, &printfbuf_len, sizeof(printfbuf_len)));

	return cudaSuccess;
}


//
//  cudaPrintfEnd
//
//  Frees up the memory which we allocated
//
void cudaPrintfEnd()
{
	if (!printfbuf_start || !printfbuf_device)
		return;

	CUMAT_SAFE_CALL(cudaFree(printfbuf_device));
	printfbuf_start = printfbuf_device = NULL;
}


//
//  cudaPrintfDisplay
//
//  Each call to this function dumps the entire current contents
//      of the printf buffer to the pre-specified FILE pointer. The
//      circular "start" pointer is advanced so that subsequent calls
//      dumps only new stuff.
//
//  In the case of async memory access (via streams), call this
//  repeatedly to keep trying to empty the buffer. If it's a sync
//  access, then the whole buffer should empty in one go.
//
//      Arguments:
//              outputFP     - File descriptor to output to (NULL => stdout)
//              showThreadID - If true, prints [block,thread] before each line
//
void cudaPrintfDisplay(std::ostream& os, bool showThreadID)
{
	// For now, we force "synchronous" mode which means we're not concurrent
	// with kernel execution. This also means we don't need clearOnPrint.
	// If you're patching it for async operation, here's where you want it.
	bool sync_printfs = true;
	bool clearOnPrint = false;

	// Initialisation check
	if (!printfbuf_start || !printfbuf_device)
		throw std::exception("Missing initialization");

	// To determine which architecture we're using, we read the
	// first short from the buffer - it'll be the magic number
	// relating to the version.
	unsigned short magic;
	CUMAT_SAFE_CALL(cudaMemcpy(&magic, printfbuf_device, sizeof(unsigned short), cudaMemcpyDeviceToHost));

	// Grab the current "end of circular buffer" pointer.
	char *printfbuf_end = NULL;
	CUMAT_SAFE_CALL(cudaMemcpyFromSymbol(&printfbuf_end, printfBufferPtr, sizeof(char *)));

	// Adjust our starting and ending pointers to within the block
	char *bufptr = ((printfbuf_start - printfbuf_device) % printfbuf_len) + printfbuf_device;
	char *endptr = ((printfbuf_end - printfbuf_device) % printfbuf_len) + printfbuf_device;

	// For synchronous (i.e. after-kernel-exit) printf display, we have to handle circular
	// buffer wrap carefully because we could miss those past "end".
	if (sync_printfs)
		doPrintfDisplay(showThreadID, clearOnPrint, printfbuf_device, printfbuf_device + printfbuf_len, endptr, printfbuf_device + printfbuf_len, os);
	doPrintfDisplay(showThreadID, clearOnPrint, printfbuf_device, printfbuf_device + printfbuf_len, bufptr, endptr, os);

	printfbuf_start = printfbuf_end;

	// If we were synchronous, then we must ensure that the memory is cleared on exit
	// otherwise another kernel launch with a different grid size could conflict.
	if (sync_printfs)
		CUMAT_SAFE_CALL(cudaMemset(printfbuf_device, 0, printfbuf_len));

}
