#ifndef PFMINPUT_H_INCLUDED
#define PFMINPUT_H_INCLUDED

#include "codecs.h"

#if USE_PFM_FILES
int ReadPFM(const char * filename,struct Image * pic,char read_only_header);
#endif // USE_PFM_FILES


#endif // PFMINPUT_H_INCLUDED
