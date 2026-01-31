#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED

#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>


#if INTEL_OPTIMIZATIONS
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#endif // INTEL_OPTIMIZATIONS

static unsigned long tickBaseMN = 0;

/*
  This is an optimized/aligned memory allocation
  Only use this for a continuous block of data like a 1 channel image etc.
*/
static void * db_malloc( size_t size )
{
  #if INTEL_OPTIMIZATIONS
   return (void*) _mm_malloc(size, 32);
  #else
   return malloc(size);
  #endif
}



static unsigned long GetTickCountMicrosecondsMN()
{
    struct timespec ts;
    if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
        {
            return 0;
        }

    if (tickBaseMN==0)
        {
            tickBaseMN = ts.tv_sec*1000000 + ts.tv_nsec/1000;
            return 0;
        }

    return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseMN;
}

static char directoryExists(const char * folder)
{
 struct stat sb;
 if (stat(folder, &sb) == 0 && S_ISDIR(sb.st_mode)) { return 1; }
 return 0;
}

static char fileExists(const char * filename)
{
    FILE *fp = fopen(filename,"r");
    if( fp )
        {
            /* exists */
            fclose(fp);
            return 1;
        }
    return 0;
}

// Function to swap two integers
static void swapULong(unsigned long *a, unsigned long *b)
{
    unsigned long temp = *a;
    *a = *b;
    *b = temp;
}
#endif // TOOLS_H_INCLUDED
