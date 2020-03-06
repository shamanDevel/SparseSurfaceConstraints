//This file should never be included directly,
//rather, a preprocessing macro
//   CUMAT_LOGGING_PLUGIN="#include "cuMatLogging.inl""
//should be defined globally

#include <cinder/app/AppBase.h>

#define CUMAT_LOG_DEBUG(...) CINDER_LOG_I(__VA_ARGS__)
#define CUMAT_LOG_INFO(...) CINDER_LOG_I(__VA_ARGS__)
#define CUMAT_LOG_WARNING(...) CINDER_LOG_W(__VA_ARGS__)
#define CUMAT_LOG_SEVERE(...) CINDER_LOG_E(__VA_ARGS__)
