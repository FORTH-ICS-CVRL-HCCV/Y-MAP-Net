/**
 * @file PrepareBatch.h
 * @brief Header file for creating and populating batch data
 */
#ifndef PREPAREBATCH_H_INCLUDED
#define PREPAREBATCH_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif


struct workerThreadContext
{
  struct ImageDatabase * db;
  unsigned long sampleStart;
  unsigned long sampleEnd;
  unsigned int thisThreadNumber;
  unsigned int workerThreads;
  unsigned int gradientSize;
  unsigned int PAFSize;
  //------------------------
  float *gradientX;
  float *gradientY;
  //------------------------
  unsigned long lagToStartWorkMicroseconds;
  unsigned long lagToFinishWorkMicroseconds;
  //------------------------


  unsigned int fullfilledWork;
  //------------------------
  int computationOutput;
};

void preloadAllFiles(struct ImageDatabase * db);

//Thread work loading mechanism
void logThreadProgress(unsigned int threadID,char start,const char * part);
void logMainThreadProgress(char start,const char * part);
void logAllThreadsProgress(unsigned int numberOfThreads,char start,const char * part);


void *workerThread(void * arg);


#ifdef __cplusplus
}
#endif

#endif
