#ifndef _ASCIIINPUT_H_INCLUDED
#define _ASCIIINPUT_H_INCLUDED

#include "codecs.h"


#if USE_ASCII_FILES
int ReadASCII(const char * filename,struct Image * pic,char read_only_header);
int WriteASCII(const char * filename,struct Image * pic,int packed);
#endif // USE_ASCII_FILES

#endif
