#include "cache.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

static int compare_records(const void *a, const void *b)
{
    const struct cache_record *ra = (const struct cache_record *)a;
    const struct cache_record *rb = (const struct cache_record *)b;
    if (ra->hash < rb->hash) return -1;
    if (ra->hash > rb->hash) return 1;
    return strncmp(ra->filename, rb->filename, CACHE_FILENAME_LEN);
}

struct cache * cache_init(unsigned long max_records)
{
    struct cache *c = (struct cache *) malloc(sizeof(struct cache));
    if (c!=0)
    {
      c->start_time = GetTickCountMicrosecondsMN();
      c->CACHE_MAX_RECORDS = max_records;
      c->count = 0;
      c->arranged = 0;

      for (int i=0; i<MAX_CACHE_THREADS; i++)
      {
        c->commonCache[i] = NULL;
        c->commonCache_MaxSize[i] = 0;
        c->commonCache_CurrentSize[i] = 0;
        c->commonCache_ReadSizeBytes[i]             = 0;
        c->commonCache_TotalReadTimeMicroseconds[i] = 0;
      }

      if (max_records>0)
      {
       c->total_size = sizeof(struct cache_record) * max_records; //Also count the size occupied by the strings/pointers etc.
       c->records = (struct cache_record *) malloc(sizeof(struct cache_record) * max_records);
      } else
      {
        c->total_size = 0;
        c->records = NULL;
      }

    }
    return c;
}

void cache_destroy(struct cache *c)
{
    if (c == NULL)
        return;

    // Free per-thread commonCache allocations
    for (int i = 0; i < MAX_CACHE_THREADS; i++)
    {
        if (c->commonCache[i] != NULL)
        {
            free(c->commonCache[i]);
            c->commonCache[i] = NULL;
        }
        c->commonCache_MaxSize[i] = 0;
        c->commonCache_CurrentSize[i] = 0;
    }

    // Free individual cache_record->data if needed
    if (c->records != NULL)
    {
        for (unsigned long i = 0; i < c->count; i++)
        {
            if (c->records[i].data != NULL)
            {
                free(c->records[i].data);
                c->records[i].data = NULL;
            }
        }

        // Free the array of records itself
        free(c->records);
        c->records = NULL;
    }

    // Finally free the cache struct itself
    free(c);
}


unsigned long hash_filename(const char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + (unsigned char) c; // hash * 33 + c

    return hash;
}



#define NORMAL  "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */


void progress_bar(unsigned long current,unsigned long finish)
{
  #define PROGRESS_BAR_LENGTH 10  // Length of the progress bar

  double progress = (double)current / finish;
  int filledLength = (int)(progress * PROGRESS_BAR_LENGTH);
  if (filledLength > PROGRESS_BAR_LENGTH) filledLength = PROGRESS_BAR_LENGTH;

  // Draw the progress bar
  printf(" [");
  for (int i = 0; i < PROGRESS_BAR_LENGTH; i++)
         {
            if (i < filledLength)  { printf(GREEN "█" NORMAL); } else
                                   { printf("-"); }
        }
  printf("] ");
}


void *cache_add(struct cache *c, const char *filename, unsigned long hash, unsigned long *size,char printStatus)
{
    if (c->count >= c->CACHE_MAX_RECORDS) return NULL;

    if (printStatus)
    {
        progress_bar(c->count,c->CACHE_MAX_RECORDS);
        printf("%0.2f%% |",(float) (100.0 * c->count) / c->CACHE_MAX_RECORDS);
        printf(" %06lu/%06lu |",c->count,c->CACHE_MAX_RECORDS);

        unsigned long usedMemoryInMegabytes = (unsigned long) c->total_size/1048576;
        printf(" %09lu MB |",usedMemoryInMegabytes);


        unsigned long elapsedTime = GetTickCountMicrosecondsMN() - c->start_time;
        float elapsedAcquisitionTimeSeconds  = (float) elapsedTime / 1000000.0;
        printf(" %0.2f GB/sec\n",(float) usedMemoryInMegabytes / (1024 * elapsedAcquisitionTimeSeconds ) );

        printf(" %s     \r",filename);
    }

    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *data = malloc(fsize);
    if (!data)
    {
        fclose(f);
        return NULL;
    }

    if (fread(data, 1, fsize, f) != fsize)
    {
        fclose(f);
        free(data);
        return NULL;
    }

    fclose(f);

    struct cache_record *rec = &c->records[c->count++];
    strncpy(rec->filename, filename, CACHE_FILENAME_LEN - 1);
    rec->filename[CACHE_FILENAME_LEN - 1] = '\0';
    rec->data = data;
    rec->size = fsize;
    rec->hash = hash;
    *size = fsize;

    c->total_size += fsize;
    c->arranged = 0;

    return data;
}

void *cache_open(struct cache *c, const char *filename,  unsigned long *size)
{
    unsigned long hash = hash_filename(filename);
    if (c->arranged)
    {
        // Binary search
        size_t left = 0, right = c->count;
        while (left < right)
         {
            size_t mid = (left + right) / 2;
            int cmp = 0;
            if (hash < c->records[mid].hash)
            {
                cmp = -1;
            } else if (hash > c->records[mid].hash)
            {
                cmp = 1;
            } else
            {
                cmp = strncmp(filename, c->records[mid].filename, CACHE_FILENAME_LEN);
            }

            if (cmp == 0)
            {
                *size = c->records[mid].size;
                return c->records[mid].data;
            } else
            if (cmp < 0) { right = mid;    } else
                         { left = mid + 1; }
        }
    } else
    {
        // Linear search
        for (size_t i = 0; i < c->count; ++i)
        {
            if (c->records[i].hash == hash && strncmp(c->records[i].filename, filename, CACHE_FILENAME_LEN) == 0)
            {
                *size = c->records[i].size;
                return c->records[i].data;
            }
        }
    }

    // Not found, add new
    return cache_add(c, filename, hash, size, 0);
}


void cache_arrange(struct cache *c)
{
    qsort(c->records, c->count, sizeof(struct cache_record), compare_records);
    c->arranged = 1;
}


void cache_print_stats(const struct cache *c)
{
    printf("Cache contains %zu file(s), total memory used: %zu bytes (%.2f MB)\n",
           c->count, c->total_size, c->total_size / (1024.0 * 1024.0));
}



float cache_readSpeedMBPerSecond(struct cache *c, unsigned int threadID)
{
    if (c == NULL)
        return 0.0f;

    if (threadID >= MAX_CACHE_THREADS)
        return 0.0f;

    unsigned long timeMicroseconds = c->commonCache_TotalReadTimeMicroseconds[threadID];
    if (timeMicroseconds == 0)
        return 0.0f;

    if (timeMicroseconds < 100) // Less than 0.1 ms
        return 0.0f;

    double timeInSeconds     = (double) timeMicroseconds / 1000000.0f;
    double avgBytesPerSecond = (double) c->commonCache_ReadSizeBytes[threadID] / timeInSeconds;
    double avgMBPerSecond    = (double) avgBytesPerSecond / ((double) 1024.0f * 1024.0f ); // Convert to megabytes

    return (float) avgMBPerSecond;
}

float cache_readSpeedGBPerSecond(struct cache *c, unsigned int threadID)
{
    return (float) cache_readSpeedMBPerSecond(c,threadID) / 1024.0f;
}

size_t read_all(FILE *fp, void *buf, size_t total_bytes)
{
    size_t total_read = 0;
    char *ptr = (char*)buf;

    while (total_read < total_bytes)
    {
        size_t n = fread(ptr + total_read, 1, total_bytes - total_read, fp);
        if (n == 0)
        {
            if (ferror(fp))
            {
                perror("fread");
                return total_read; // will signal failure
            }
            // EOF reached before expected
            break;
        }
        total_read += n;
    }

    return total_read;
}



void *read_file_to_common_memory_of_cache(struct cache *c,unsigned int threadID, const char *filename, size_t *size)
{
    if (c == NULL)
    {
        fprintf(stderr, "read_file_to_common_memory_of_cache called without cache\n");
        return NULL;
    }

    if (threadID >= MAX_CACHE_THREADS)
    {
        fprintf(stderr, "Invalid threadID %d (allowed 0..%d)\n", threadID, MAX_CACHE_THREADS - 1);
        return NULL;
    }

    unsigned long ioStart = GetTickCountMicrosecondsMN();

    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr,"read_file_to_common_memory_of_cache: Failed to open file");
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fprintf(stderr,"read_file_to_common_memory_of_cache: Failed to seek file");
        fclose(fp);
        return NULL;
    }

    long file_size = ftell(fp);
    if (file_size < 0)
    {
        fprintf(stderr,"read_file_to_common_memory_of_cache: Failed to tell file size");
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    if ((c->commonCache[threadID] == NULL) || (c->commonCache_MaxSize[threadID] < (size_t)file_size))
    {
        //Give console output showing reallocations
        //fprintf(stderr, "Thread %d:    Allocating %lu -> %lu                      \n",threadID,(unsigned long)c->commonCache_MaxSize[threadID],(unsigned long)file_size);

        // If there was a smaller buffer previously then release it
        if (c->commonCache[threadID] != NULL)
           {
            c->commonCache_MaxSize[threadID] = 0;
            c->commonCache_CurrentSize[threadID] = 0;
            free(c->commonCache[threadID]);
            c->commonCache[threadID] = 0;
           }
        c->commonCache[threadID] = malloc(file_size);
        c->commonCache_MaxSize[threadID] = file_size;
        c->commonCache_CurrentSize[threadID] = file_size;
    }

    if (!c->commonCache[threadID])
    {
        fprintf(stderr,"read_file_to_common_memory_of_cache: Failed to allocate memory\n");
        fclose(fp);
        return NULL;
    }

    //size_t read_size = fread(c->commonCache[threadID], 1, file_size, fp);
    size_t read_size = read_all(fp,c->commonCache[threadID],file_size);

    #if STRICT_CHECK_WHEN_OPENING_FILES
    if (read_size != (size_t)file_size)
    #else
    if (read_size<=0) //Just fail on completely failed files..
    #endif
    {
        fprintf(stderr,"read_file_to_common_memory_of_cache: Failed to read file completely (read %lu instead of %lu bytes)\n",read_size,file_size);
        if (feof(fp)) fprintf(stderr, "EOF reached early\n");
        if (ferror(fp)) fprintf(stderr, "File error occurred\n");

        free(c->commonCache[threadID]);
        c->commonCache[threadID] = NULL;
        c->commonCache_MaxSize[threadID] = 0;
        c->commonCache_CurrentSize[threadID] = 0;
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    c->commonCache_CurrentSize[threadID] = file_size;


    c->commonCache_ReadSizeBytes[threadID]             += (unsigned long) file_size;
    c->commonCache_TotalReadTimeMicroseconds[threadID] += (unsigned long) GetTickCountMicrosecondsMN()-ioStart;
    //fprintf(stderr,"T:%02u  Bytes :%lu  /  Elapsed Time %lu \n",threadID,c->commonCache_ReadSizeBytes[threadID],c->commonCache_TotalReadTimeMicroseconds[threadID]);

    //Keep Magnitude of values manageable
    if ( (c->commonCache_ReadSizeBytes[threadID]>10000000) &&
           (c->commonCache_TotalReadTimeMicroseconds[threadID]>10000000) )
    {
        c->commonCache_ReadSizeBytes[threadID] = (unsigned long) c->commonCache_ReadSizeBytes[threadID] / 2;
        c->commonCache_TotalReadTimeMicroseconds[threadID] = (unsigned long) c->commonCache_TotalReadTimeMicroseconds[threadID] / 2;
    }

    if (size) *size = read_size;
    return c->commonCache[threadID];
}


