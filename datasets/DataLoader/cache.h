// cache.h
#ifndef CACHE_H
#define CACHE_H

#include <stddef.h>

#define CACHE_FILENAME_LEN 128

struct cache_record
{
    char filename[CACHE_FILENAME_LEN];
    void *data;
    unsigned long hash;
    size_t size;
};

#define MAX_CACHE_THREADS 128  // Adjust this as needed

struct cache
{
    unsigned long start_time;
    unsigned long CACHE_MAX_RECORDS;
    struct cache_record * records;
    size_t count;
    int arranged;
    size_t total_size;  // total bytes cached


    void *commonCache[MAX_CACHE_THREADS];
    size_t commonCache_MaxSize[MAX_CACHE_THREADS];
    size_t commonCache_CurrentSize[MAX_CACHE_THREADS];
    unsigned long commonCache_ReadSizeBytes[MAX_CACHE_THREADS];
    unsigned long commonCache_TotalReadTimeMicroseconds[MAX_CACHE_THREADS];
};

// Initializes the cache
struct cache * cache_init(unsigned long max_records);

void cache_destroy(struct cache *c);

unsigned long hash_filename(const char *str);

void *cache_add(struct cache *c, const char *filename, unsigned long hash, unsigned long *size,char printStatus);

// Opens a file and caches it if not already present
void *cache_open(struct cache *c, const char *filename, unsigned long *size);

// Arranges the cache for fast lookup
void cache_arrange(struct cache *c);

void cache_print_stats(const struct cache *c);


float cache_readSpeedMBPerSecond(struct cache *c, unsigned int threadID);
float cache_readSpeedGBPerSecond(struct cache *c, unsigned int threadID);

void *read_file_to_common_memory_of_cache(struct cache *c,unsigned int threadID, const char *filename, size_t *size);
#endif // CACHE_H
