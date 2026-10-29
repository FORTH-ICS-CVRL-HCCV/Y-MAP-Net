/** @file PZPDLoader.h
 *  @brief .pzpd archive sources for the DataLoader (written by datasets/convertToPZPD.py).
 *
 *  A dataset source whose "DB file" is a .pzpd archive replaces the .db file, the image / depth /
 *  segmentation / combined directories and the .dinov3 descriptor file:
 *  - the PoseDatabase is filled from the archive's tables (joints, image, persons, descriptions,
 *    descriptor_dinov3), see readPoseDatabaseFromArchive();
 *  - the workers take each sample's files as bytes from the archive and decode them from memory,
 *    see pzpdSampleBegin() / pzpdSampleReadImage() / pzpdSampleEnd();
 *  - a pzpdir prefetcher reads the epoch order ahead of the workers, see pzpdPrefetchBatch().
 *  All archive sources of a DatabaseList are opened as one pzpdir handle, member i = sourceID i.
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef PZPDLOADER_H_INCLUDED
#define PZPDLOADER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include "DataLoader.h"

/** @brief The streams of an archive, as written by convertToPZPD.py (index into DatabaseList.archiveStream). */
enum PZPDStream
{
  PZPD_STREAM_RGB = 0,
  PZPD_STREAM_ALL,     // combined depth + segmentation
  PZPD_STREAM_DEPTH,
  PZPD_STREAM_SEG,
  PZPD_STREAM_COUNT
};

/** @brief A sample's blobs while a worker decodes it (from the prefetcher, or mmap views without one). */
struct ArchiveSample
{
  pzpd_blob_ref refs[PZPD_MAX_STREAMS];
  pzpd_ticket   ticket;
  char          haveTicket;
};

// 1 if the file starts with a pzpdir magic (manifest, shard or collection), else 0
int pzpdIsArchive(const char * path);

// Sample count and joint count of an archive source (for db_set_source_entry), 1 on success
int pzpdProbeSource(const char * path,unsigned long * numberOfSamples,unsigned short * numberOfKeypoints);

// Open every source of the list as one handle (member = sourceID), 1 on success
int pzpdOpenSources(struct DatabaseList * dbl);
void pzpdCloseSources(struct DatabaseList * dbl);

// The archive counterpart of readPoseDatabase(): same slots, same ignoreNoSkeletonSamples handling
struct PoseDatabase* readPoseDatabaseFromArchive(struct PoseDatabase* pdb,struct DatabaseList * dbl,unsigned int sourceID,unsigned long startOffset,unsigned long * loadedSamples,int ignoreNoSkeletonSamples);

// Worker side: get a sample's blobs, decode one stream, give the blobs back
void pzpdSampleBegin(struct ImageDatabase * db,unsigned long sampleNumber,struct ArchiveSample * s);
struct Image * pzpdSampleReadImage(struct ImageDatabase * db,unsigned long sampleNumber,struct ArchiveSample * s,enum PZPDStream stream);
void pzpdSampleEnd(struct ImageDatabase * db,struct ArchiveSample * s);

// Prefetcher: create after db_create, (re)submit the epoch order before a batch, destroy before closing
int  pzpdPrefetchCreate(struct ImageDatabase * db);
void pzpdPrefetchBatch(struct ImageDatabase * db,unsigned long sampleStart,unsigned long sampleEnd);
void pzpdPrefetchDestroy(struct ImageDatabase * db);

// Python: prefetch mode (-1 off, 0 auto, 1 map, 2 pagecache, 3 buffers), I/O threads, budget (MB) and window
// (records), 0 = pzpdir default. Call BEFORE db_create.
void db_set_prefetch_config(int mode,int ioThreads,int budgetMB,int window);

// Python: the caption of the sample at a position (copied, NUL-terminated), its length, or -1
int db_get_sample_description(struct ImageDatabase * db,unsigned long sampleID,char * out,unsigned int outSize);

#ifdef __cplusplus
}
#endif

#endif
