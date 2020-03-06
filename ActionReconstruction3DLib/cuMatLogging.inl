//This file should never be included directly,
//rather, a preprocessing macro
//   CUMAT_LOGGING_PLUGIN="#include "cuMatLogging.inl""
//should be defined globally

#include <cinder/Log.h>

#define CUMAT_LOG_DEBUG(...) CI_LOG_V(__VA_ARGS__)
#define CUMAT_LOG_INFO(...) CI_LOG_I(__VA_ARGS__)
#define CUMAT_LOG_WARNING(...) CI_LOG_W(__VA_ARGS__)
#define CUMAT_LOG_SEVERE(...) CI_LOG_E(__VA_ARGS__)
