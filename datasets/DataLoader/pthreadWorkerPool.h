/** @file pthreadWorkerPool.h
 *  @brief  A header-only thread automization library to make your multithreaded-lives easier.
 *  To add to your project just copy this header to your code and don't forget to link with
 *  pthreads, for example : gcc -O3 -pthread yourProject.c -o threadsExample
 *  Repository : https://github.com/AmmarkoV/PThreadWorkerPool
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef PTHREADWORKERPOOL_H_INCLUDED
#define PTHREADWORKERPOOL_H_INCLUDED

//The star of the show
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
#include <unistd.h>
#include <time.h>
#include <pwd.h>
#include <sched.h>
#include <sys/resource.h>

#ifdef __cplusplus
extern "C"
{
#endif

static const char pthreadWorkerPoolVersion[]="0.42"; //0.42: fix threadpoolCreate hang/UB on partial pthread_create failure, use atomics for activeWorkers/workersWaiting, drop the mainThreadWaiting busy-wait spin in threadpoolWorkerLoopEnd

#define SPIN_SLEEP_TIME_MICROSECONDS 120

#define DEBUG_LOG 0

/**
 * @brief Structure representing a thread context.
 */
struct threadContext
{
    void * argumentToPass;
    struct workerPool * pool;
    unsigned int threadID;
    char threadInitialized;
    char alive; //1 if pthread_create succeeded for this slot, 0 if this slot was never started
};

/**
 * @brief Structure representing a worker pool.
 */
struct workerPool
{
    //---------------------
    volatile char initialized;      //Only ever touched by the controlling (main) thread, never by workers
    volatile char work;             //Always read/written while holding startWorkMutex - protected by that lock
    //---------------------
    // activeWorkers and workersWaiting are each written under one mutex (startWorkMutex) and read/decremented
    // under a *different* mutex (completeWorkMutex), or read with no lock held at all (the busy-wait in
    // threadpoolMainThreadPrepareWorkForWorkers). A single pthread mutex only guarantees memory visibility for
    // accesses made while holding that same mutex, so relying purely on "volatile" here is not formally safe -
    // access these two fields only through the __atomic_* builtins below.
    volatile int activeWorkers;
    volatile int workersWaiting;  // workers currently inside pthread_cond_wait on startWorkCondition
    //---------------------
    pthread_attr_t initializationAttribute;

    //Start conditions..
    pthread_mutex_t startWorkMutex;
    pthread_cond_t  startWorkCondition;
    //---------------------

    //End conditions..
    pthread_mutex_t completeWorkMutex;
    pthread_cond_t  completeWorkCondition;
    //---------------------

    unsigned int numberOfThreads;      //Number of threads that were successfully created (i.e. alive)
    unsigned int numberOfThreadSlots;  //Number of slots allocated in workerPoolContext/workerPoolIDs (the requested spawn count)
    //---------------------
    struct threadContext *workerPoolContext;
    pthread_t * workerPoolIDs;
};

static void logmsg(const char *format, ...)
{
 #if DEBUG_LOG
    va_list args;

    // Start processing the variable arguments
    va_start(args, format);

    // Log to stderr (you can change this to write to a file if needed)
    vfprintf(stderr, format, args);

    // End processing the variable arguments
    va_end(args);
 #else
    (void) format;
 #endif // DEBUG_LOG
}


static unsigned long tickBaseTPMN = 0;
static pthread_once_t tickBaseOnce = PTHREAD_ONCE_INIT;
static void initTickBase(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        tickBaseTPMN = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/**
 * @brief Return number of clock ticks for our system in microseconds
 * @return Returns number of ticks in microseconds.
 */
static unsigned long GetTickCountMicrosecondsT()
{
    pthread_once(&tickBaseOnce, initTickBase);
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000) - tickBaseTPMN;
}

/**
 * @brief Function for sleeping for a specified amount of time.
 * @param nanoseconds The number of nanoseconds to sleep.
 * @return Returns 0 on success, -1 on failure.
 */
static int nanoSleepT(long nanoseconds)
{
    struct timespec req, rem;

    req.tv_sec = 0;
    req.tv_nsec = nanoseconds;

    return nanosleep(&req, &rem);
}

/**
 * @brief Function to stick the thread that calls the function to a specific CPU core, if the core does not exist it will wrap-around depending on the number of cores.
 * @param Core we want to associate this thread with.
 * @return Returns 0 on success, see pthread_setaffinity_np return values for possible failure codes.
 */
static int stick_this_thread_to_core(int core_id)
{
   #if _GNU_SOURCE
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   int core_id_mod_cores = core_id % num_cores;

   //if (core_id < 0 || core_id >= num_cores)
   //   return EINVAL;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id_mod_cores, &cpuset);

   pthread_t current_thread = pthread_self();
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
   #else
    (void) core_id;
    fprintf(stderr,"Cannot stick thread to core without GNU source extensions during compilation\n");
    return EFAULT;
   #endif
}

/**
 * @brief Function to change the priority of the process using `nice()`
 * @param priority The niceness value to set (-20 for highest priority, 19 for lowest)
 * @return Returns 1 on success, 0 on failure.
 */
static int set_process_nice(int priority)
{
    errno = 0; // nice() can legitimately return -1 on success, so errno must be cleared first
    int ret = nice(priority);
    if (ret == -1 && errno != 0)  // Check for errors
    {
        perror("Failed to change process priority");
        return 0;
    }

    fprintf(stderr, "Process priority (nice value) set to %d\n", priority);
    return 1;
}



/**
 * @brief Function to increase the process priority using setpriority().
 * @param priority The nice value to set (-20 for highest priority, 19 for lowest)
 * @return Returns 1 on success, 0 on failure.
 */
static int elevate_nice_priority(int priority)
{
    if (setpriority(PRIO_PROCESS, 0, priority) == -1)
    {
        perror("Failed to elevate process priority");
        return 0;
    }

    fprintf(stderr, "Process priority successfully elevated to %d\n", priority);
    return 1;
}



/**
 * @brief Function for setting the real-time priority of a thread.
 * @return Returns 1 on success, 0 on failure.
 */
static int set_realtime_thread_priority()
{
    int ret;

    // We'll operate on the currently running thread.
    pthread_t this_thread = pthread_self();
    // struct sched_param is used to store the scheduling priority
    struct sched_param params;

    // We'll set the priority to the maximum.
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);

    fprintf(stderr,"Trying to set thread realtime prio = %u \n",params.sched_priority);

    // Attempt to set thread real-time priority to the SCHED_FIFO policy
    ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    if (ret != 0)
    {
        // Print the error
        fprintf(stderr,"Failed setting thread realtime priority\n");
        return 0;
    }

    // Now verify the change in thread priority
    int policy = 0;
    ret = pthread_getschedparam(this_thread, &policy, &params);
    if (ret != 0)
    {
        fprintf(stderr,"Couldn't retrieve real-time scheduling paramers\n");
        return 0;
    }

    // Check the correct policy was applied
    if(policy != SCHED_FIFO)
    {
        fprintf(stderr,"Scheduling is NOT SCHED_FIFO!\n");
    }
    else
    {
        fprintf(stderr,"SCHED_FIFO OK\n");
    }

    // Print thread scheduling priority
    fprintf(stderr,"Thread priority is now %u\n",params.sched_priority);
    return 1;
}




/**
 * @brief Function for retrieving the OMP_NUM_THREADS
 * @return Returns the number of threads or 0 if OMP_NUM_THREADS is not declared.
 */
static int get_omp_num_threads_from_env(void)
{
    const char *env = getenv("OMP_NUM_THREADS");
    if (env != NULL) {
        return atoi(env);  // convert string to int
    }
    return 0;  // not set
}


/**
 * @brief Function for initializing a worker thread and waiting for start signal.
 * @param ctx Pointer to the thread context.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolWorkerInitialWait(struct threadContext * ctx)
{
    if (ctx!=0)
    {
     pthread_mutex_lock(&ctx->pool->completeWorkMutex);
     ctx->threadInitialized = 1;
     pthread_cond_signal(&ctx->pool->completeWorkCondition);
     pthread_mutex_unlock(&ctx->pool->completeWorkMutex);
     pthread_mutex_lock(&ctx->pool->startWorkMutex);
     __atomic_fetch_add(&ctx->pool->workersWaiting, 1, __ATOMIC_SEQ_CST);
     pthread_cond_wait(&ctx->pool->startWorkCondition,&ctx->pool->startWorkMutex);
     __atomic_fetch_sub(&ctx->pool->workersWaiting, 1, __ATOMIC_SEQ_CST);
     return 1;
    }
    return 0;
}

/**
 * @brief Function for checking the loop condition of a worker thread.
 * @param ctx Pointer to the thread context.
 * @return Returns 1 if the condition is met, 0 otherwise.
 */
static int threadpoolWorkerLoopCondition(struct threadContext * ctx)
{
    if (ctx==0) { return 0; }

    if (ctx->pool->work)
    {
        pthread_mutex_unlock(&ctx->pool->startWorkMutex);
        return 1;
    } else
    {
        pthread_mutex_unlock(&ctx->pool->startWorkMutex);
        pthread_exit(NULL);
        return 0;
    }
}

/**
 * @brief Function for handling the end of a worker thread's loop.
 * @param ctx Pointer to the thread context.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolWorkerLoopEnd(struct threadContext * ctx)
{
    if (ctx==0) { return 0; }

    // Report completion: decrement activeWorkers and signal completeWorkCondition while holding
    // completeWorkMutex. The main thread's wait loop (threadpoolMainThreadWaitFor...) always
    // re-checks "activeWorkers>0" under this same mutex immediately before (and after) sleeping
    // on completeWorkCondition, so it is safe to report completion even if the main thread has not
    // reached its wait call yet - it will simply see activeWorkers already decremented once it gets
    // there and will not block at all. This used to be gated behind a "mainThreadWaiting" flag that
    // workers spun on (usleep(SPIN_SLEEP_TIME_MICROSECONDS) loop) until the main thread announced it
    // was ready to collect - that spin was unnecessary given the predicate re-check above, and was
    // wasting a full CPU core whenever a worker finished before the main thread got around to calling
    // the Wait function (e.g. while pipelining with threadpoolMainThreadKickWorkers).
    pthread_mutex_lock(&ctx->pool->completeWorkMutex);

    //We are officially no longer an active worker!
    __atomic_fetch_sub(&ctx->pool->activeWorkers, 1, __ATOMIC_SEQ_CST);
    pthread_cond_signal(&ctx->pool->completeWorkCondition);

    //We are done communicating with Coordinator on our completed work
    pthread_mutex_unlock(&ctx->pool->completeWorkMutex);


    // Lock the "StartWorkMutex" so we can enter a waiting state for the next round before the main
    // thread broadcasts the "StartWorkCondition".
    pthread_mutex_lock(&ctx->pool->startWorkMutex); //<- This will get released by threadpoolWorkerLoopCondition
    __atomic_fetch_add(&ctx->pool->workersWaiting, 1, __ATOMIC_SEQ_CST);

    // Wait for the Main thread to send us the next "StartWorkCondition" broadcast.
    // Be sure to unlock the corresponding mutex immediately so that the other worker threads can exit their waiting state as well.
    unsigned long workerLoopBlockStartTime = GetTickCountMicrosecondsT();
    pthread_cond_wait(&ctx->pool->startWorkCondition, &ctx->pool->startWorkMutex);
    __atomic_fetch_sub(&ctx->pool->workersWaiting, 1, __ATOMIC_SEQ_CST);
    unsigned long workerLoopBlockFinishTime = GetTickCountMicrosecondsT();

    //report Resyncing lag with the rest of the thread pool
    logmsg("Thread %u | Re-Sync Lag: %lu μsec \n",ctx->threadID , workerLoopBlockFinishTime-workerLoopBlockStartTime);

    return 1;
}

/**
 * @brief Function for preparing work for worker threads by the main thread.
 * @param pool Pointer to the worker pool.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolMainThreadPrepareWorkForWorkers(struct workerPool * pool)
{
    if (pool==0) { return 0; }

    if (pool->initialized)
    {
        // Spin until every worker has entered pthread_cond_wait on startWorkCondition.
        // Without this, a worker that finished the previous batch but hasn't yet reached
        // pthread_cond_wait can miss the upcoming broadcast and deadlock indefinitely.
        while (__atomic_load_n(&pool->workersWaiting, __ATOMIC_SEQ_CST) < (int)pool->numberOfThreads)
        {
            usleep(SPIN_SLEEP_TIME_MICROSECONDS);
        }

        pthread_mutex_lock(&pool->startWorkMutex); //This will be released by threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds
        return 1;
    }
    return 0;
}

/**
 * @brief Actually RELEASE the workers that threadpoolMainThreadPrepareWorkForWorkers
 *        armed, WITHOUT waiting for them — the broadcast half of
 *        threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds. This is what makes a true
 *        start/collect pipeline possible: historically "StartUpdate" only called Prepare (which
 *        merely takes the start lock), so the workers did not begin until the combined wait
 *        function ran — i.e. nothing ever overlapped. Call sequence for async use:
 *        Prepare -> Kick -> (main thread does other things) -> WaitForKickedWorkers.
 *        A worker that finishes while the main thread is away does not block or spin at all: it
 *        reports completion immediately (see threadpoolWorkerLoopEnd) and goes straight back to
 *        sleep waiting for the next round, so the main thread's later WaitForKickedWorkers call
 *        just finds activeWorkers already decremented.
 * @param pool Pointer to the worker pool.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolMainThreadKickWorkers(struct workerPool * pool)
{
    if (pool==0) { return 0; }
    if (pool->initialized)
    {
        pool->work=1;
        //We consider all workers as active from now on!
        //Written under startWorkMutex but read/decremented under completeWorkMutex by workers -
        //use an atomic store so that cross-mutex visibility does not depend on incidental barriers.
        __atomic_store_n(&pool->activeWorkers, (int)pool->numberOfThreads, __ATOMIC_SEQ_CST);
        pthread_cond_broadcast(&pool->startWorkCondition); //Broadcast starting condition
        //Release lock for starting work that main thread held since threadpoolMainThreadPrepareWorkForWorkers was called
        pthread_mutex_unlock(&pool->startWorkMutex);       //Now start worker threads
        return 1;
    }
    return 0;
}

/**
 * @brief Wait for workers previously released by threadpoolMainThreadKickWorkers.
 *        ONLY the waiting half — it does NOT re-broadcast / reset activeWorkers (using the
 *        combined WaitForWorkersToFinish function after a Kick would reset activeWorkers to N
 *        after some workers already completed, waiting for completions that will never come).
 * @param pool Pointer to the worker pool.
 * @param timeout Number of seconds to wait, otherwise abort process as stuck (0 = wait forever).
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(struct workerPool * pool, int timeoutSeconds)
{
    if (pool==0) { return 0; }
    if (pool->initialized)
    {
        struct timespec ts;
        if (timeoutSeconds!=0)
        {
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeoutSeconds;
        }

        pthread_mutex_lock(&pool->completeWorkMutex);

        while (__atomic_load_n(&pool->activeWorkers, __ATOMIC_SEQ_CST)>0)
        {
            if (timeoutSeconds!=0)
            {
                int ret = pthread_cond_timedwait(&pool->completeWorkCondition, &pool->completeWorkMutex, &ts);
                if (ret == ETIMEDOUT)
                {
                    fprintf(stderr, "\n\npthreadWorkerPool: Timeout (%u sec) occurred @ %u of %u, threads are stuck.\n", timeoutSeconds, __atomic_load_n(&pool->activeWorkers, __ATOMIC_SEQ_CST), pool->numberOfThreads);
                    abort();
                }
            } else
            {
                pthread_cond_wait(&pool->completeWorkCondition, &pool->completeWorkMutex);
            }
        }

        pthread_mutex_unlock(&pool->completeWorkMutex);
        return 1;
    }
    return 0;
}

/**
 * @brief Function for waiting worker for threads to finish their task by the main thread.
 * @param pool Pointer to the worker pool.
 * @param timeout Number of seconds to wait for threads to complete their work, otherwise abort process as stuck (0 = wait forever).
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds(struct workerPool * pool, int timeoutSeconds)
{
    if (pool==0) { return 0; }

    if (pool->initialized)
    {
        pool->work=1;

        //We consider all workers as active from now on!
        //Written under startWorkMutex but read/decremented under completeWorkMutex by workers -
        //use an atomic store so that cross-mutex visibility does not depend on incidental barriers.
        __atomic_store_n(&pool->activeWorkers, (int)pool->numberOfThreads, __ATOMIC_SEQ_CST);

        pthread_cond_broadcast(&pool->startWorkCondition); //Broadcast starting condition

        //Release lock for starting work that main thread held since threadpoolMainThreadPrepareWorkForWorkers was called
        pthread_mutex_unlock(&pool->startWorkMutex);       //Now start worker threads

        //At this point of the code for the particular iteration all single threaded chains have been executed
        //All parallel threads are running and now we must wait until they are done and gather their output

         struct timespec ts;
         //If we are using a timeout check the time and
         //set the deadline for the correct amount of seconds
         //or else don't bother getting the time
         if (timeoutSeconds!=0)
         {
          clock_gettime(CLOCK_REALTIME, &ts);
          ts.tv_sec += timeoutSeconds;
         }


        //We now wait for "numberOfWorkerThreads" worker threads to finish
        //for (int numberOfWorkerThreadsToWaitFor=0;  numberOfWorkerThreadsToWaitFor<pool->numberOfThreads; numberOfWorkerThreadsToWaitFor++)
        //Signal that we can start and wait for finish...
        pthread_mutex_lock(&pool->completeWorkMutex);      //Make sure worker threads wont fall through after completion
        //From now on we are waiting for the workers to hand-over their finished jobs..

        while (__atomic_load_n(&pool->activeWorkers, __ATOMIC_SEQ_CST)>0)
        {
            // pthread_cond_wait atomically releases completeWorkMutex and sleeps,
            // so no completion signal can be lost between the activeWorkers check and the wait.
            if (timeoutSeconds!=0)
            {
             int ret = pthread_cond_timedwait(&pool->completeWorkCondition, &pool->completeWorkMutex, &ts);
             if (ret == ETIMEDOUT)
                     {
                        fprintf(stderr, "\n\npthreadWorkerPool: Timeout (%u sec) occurred @ %u of %u, threads are stuck.\n", timeoutSeconds, __atomic_load_n(&pool->activeWorkers, __ATOMIC_SEQ_CST), pool->numberOfThreads);
                        abort();
                     }
            } else
            {
             pthread_cond_wait(&pool->completeWorkCondition, &pool->completeWorkMutex);
            }
        }

        //We finished waiting..
        pthread_mutex_unlock(&pool->completeWorkMutex);

        //--------------------------------------------------
        return 1;
    }
    return 0;
}

/**
 * @brief Function for waiting worker for threads to finish their task by the main thread. This function will wait forever if something goes wrong with workers
 * @param pool Pointer to the worker pool.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolMainThreadWaitForWorkersToFinish(struct workerPool * pool)
{
  return threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds(pool,0);
}

/**
 * @brief Function for creating a worker pool.
 * @param pool Pointer to the worker pool to be created.
 * @param numberOfThreadsToSpawn Number of threads to spawn in the pool.
 * @param workerFunction Pointer to the worker function.
 * @param argument Argument to pass to the worker function.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolCreate(struct workerPool * pool,unsigned int numberOfThreadsToSpawn,void *  workerFunction, void * argument)
{
    if (pool==0)
    {
        return 0;
    }
    if (pool->workerPoolIDs!=0)
    {
        return 0;
    }
    if (pool->workerPoolContext!=0)
    {
        return 0;
    }

    //--------------------------------------------------
    pool->work = 0;
    pool->workersWaiting = 0;
    pool->numberOfThreads = 0;
    pool->numberOfThreadSlots = numberOfThreadsToSpawn;
    pool->workerPoolIDs     = (pthread_t*) malloc(sizeof(pthread_t) * numberOfThreadsToSpawn);
    pool->workerPoolContext = (struct threadContext*) malloc(sizeof(struct threadContext) * numberOfThreadsToSpawn);
    //--------------------------------------------------

    if ( (pool->workerPoolIDs==0) ||  (pool->workerPoolContext==0) )
    {
        fprintf(stderr,"Failed allocating worker pool resources\n");
        if (pool->workerPoolContext!=0)
        {
          free(pool->workerPoolContext);
          pool->workerPoolContext=0;
        }
        if (pool->workerPoolIDs!=0)
        {
          free(pool->workerPoolIDs);
          pool->workerPoolIDs=0;
        }
        return 0;
    }

    //--------------------------------------------------
    pthread_cond_init(&pool->startWorkCondition,0);
    pthread_mutex_init(&pool->startWorkMutex,0);
    pthread_cond_init(&pool->completeWorkCondition,0);
    pthread_mutex_init(&pool->completeWorkMutex,0);
    //--------------------------------------------------
    pthread_attr_init(&pool->initializationAttribute);
    pthread_attr_setdetachstate(&pool->initializationAttribute,PTHREAD_CREATE_JOINABLE);

    int threadsCreated = 0;

    for (unsigned int i=0; i<numberOfThreadsToSpawn; i++)
    {
        pool->workerPoolContext[i].threadID=i;
        pool->workerPoolContext[i].threadInitialized = 0;
        pool->workerPoolContext[i].alive = 0;
        pool->workerPoolContext[i].argumentToPass=argument;
        pool->workerPoolContext[i].pool=pool;

        //Wrap into a call..
        //void ( *callWrapped) (void *) =0;
        //callWrapped = (void(*) (void *) ) workerFunction;

        int result = pthread_create(
                                     &pool->workerPoolIDs[i],
                                     &pool->initializationAttribute,
                                     (void * (*)(void*)) workerFunction,
                                     (void*) &pool->workerPoolContext[i]
                                   );

        pool->workerPoolContext[i].alive = (result == 0);
        threadsCreated += (result == 0);
    }

    if (threadsCreated!=(int)numberOfThreadsToSpawn)
    {
        fprintf(stderr,"Failed spawning %u of %u worker threads, continuing with the ones that started\n",numberOfThreadsToSpawn-threadsCreated,numberOfThreadsToSpawn);
    }

    //Wait for all threads that actually started to signal that they are initialized.
    //We must scan every requested slot (not just the first threadsCreated of them) since a
    //failed pthread_create can leave a "dead" slot anywhere in the array - skip those via "alive".
    fprintf(stderr,"Waiting for threads to start : ");
    pthread_mutex_lock(&pool->completeWorkMutex);
    while (1)
    {
      unsigned int threadsThatAreReady=0;
      for (unsigned int i=0; i<numberOfThreadsToSpawn; i++)
      {
          if (pool->workerPoolContext[i].alive)
          {
              threadsThatAreReady+=pool->workerPoolContext[i].threadInitialized;
          }
      }

      if (threadsThatAreReady==(unsigned int)threadsCreated)
      {
          break;
      }
      pthread_cond_wait(&pool->completeWorkCondition, &pool->completeWorkMutex);
    }
    pthread_mutex_unlock(&pool->completeWorkMutex);
    fprintf(stderr," done \n");

    pool->numberOfThreads = threadsCreated;
    pool->initialized     = ((unsigned int)threadsCreated==numberOfThreadsToSpawn);
    return ((unsigned int)threadsCreated==numberOfThreadsToSpawn);
}



/**
 * @brief Function for destroying a worker pool.
 * @param pool Pointer to the worker pool to be destroyed.
 * @return Returns 1 on success, 0 on failure.
 */
static int threadpoolDestroy(struct workerPool *pool)
{
   if ( (pool!=0) && (pool->workerPoolIDs!=0) && (pool->workerPoolContext!=0) )
    {

    pthread_mutex_lock(&pool->startWorkMutex);
    // Set the conditions to stop all threads.
    pool->work = 0;
    pthread_cond_broadcast(&pool->startWorkCondition);
    pthread_mutex_unlock(&pool->startWorkMutex);


    // Join only the slots that actually got a thread started (skip any "dead" slot
    // left behind by a failed pthread_create in threadpoolCreate).
    for (unsigned int i=0; i<pool->numberOfThreadSlots; i++)
    {
        if (pool->workerPoolContext[i].alive)
        {
            pthread_join(pool->workerPoolIDs[i],0);
        }
    }

    // Clean up and exit.
    pthread_attr_destroy(&pool->initializationAttribute);
    pthread_mutex_destroy(&pool->completeWorkMutex);
    pthread_cond_destroy(&pool->completeWorkCondition);
    pthread_mutex_destroy(&pool->startWorkMutex);
    pthread_cond_destroy(&pool->startWorkCondition);

    free(pool->workerPoolContext);
    pool->workerPoolContext=0;
    free(pool->workerPoolIDs);
    pool->workerPoolIDs=0;
    pool->initialized=0;
    return 1;
    }

  return 0;
}



#ifdef __cplusplus
}
#endif

#endif // PTHREADWORKERPOOL_H_INCLUDED



