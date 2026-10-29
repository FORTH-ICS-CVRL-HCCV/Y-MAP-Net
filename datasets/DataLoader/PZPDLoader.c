/** @file PZPDLoader.c
 *  @brief .pzpd archive sources for the DataLoader, see PZPDLoader.h.
 *
 *  Like the .db path, every data problem found at load is reported and makes the load fail (the
 *  caller aborts "to protect training consistency"): the checks that the workers would otherwise
 *  hit mid-epoch (missing RGB, depth that is not 1 channel / 16 bit, unknown formats) are done
 *  here once, from the archive index, without reading any image data.
 *  @author Ammar Qammaz (AmmarkoV)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "PZPDLoader.h"
#include "DBLoader.h"
#include "codecs/codecs.h"

static const char * streamNames[PZPD_STREAM_COUNT] = { "rgb", "all", "depth", "seg" };

// Prefetcher settings (db_set_prefetch_config), module globals like db_set_geolocation_config
static int g_prefetchMode      = PZPD_PF_AUTO;
static int g_prefetchIOThreads = 0;
static int g_prefetchBudgetMB  = 0;
static int g_prefetchWindow    = 0;

void db_set_prefetch_config(int mode,int ioThreads,int budgetMB,int window)
{
  g_prefetchMode      = mode;
  g_prefetchIOThreads = (ioThreads>0) ? ioThreads : 0;
  g_prefetchBudgetMB  = (budgetMB>0)  ? budgetMB  : 0;
  g_prefetchWindow    = (window>0)    ? window    : 0;
}
//----------------------------------------------------------------------------------------------------
int pzpdIsArchive(const char * path)
{
  FILE * fp = fopen(path,"rb");
  if (fp==0) { return 0; }
  char magic[8]={0};
  size_t n = fread(magic,1,8,fp);
  fclose(fp);
  if (n!=8) { return 0; }
  return (memcmp(magic,"PZPDMANI",8)==0) || (memcmp(magic,"PZPDSHRD",8)==0) || (memcmp(magic,"PZPDCOLL",8)==0);
}
//----------------------------------------------------------------------------------------------------
// Byte offset of a column with the expected type and count, or -1 (reported).
static int columnOffset(const pzpd_schema * sc,const char * name,int type,unsigned int count)
{
  if (sc==0) { return -1; }
  for (unsigned int c=0; c<sc->ncols; c++)
  {
    if ( (strcmp(sc->cols[c].name,name)==0) && (sc->cols[c].type==type) && ( (count==0) || (sc->cols[c].count==count) ) )
       { return (int) sc->cols[c].offset; }
  }
  fprintf(stderr,RED "Archive table %s has no column %s of the expected type\n" NORMAL,sc->name,name);
  return -1;
}
//----------------------------------------------------------------------------------------------------
int pzpdProbeSource(const char * path,unsigned long * numberOfSamples,unsigned short * numberOfKeypoints)
{
  pzpd * a = pzpd_open(path,0);
  if (a==0)
  {
    fprintf(stderr,RED "Cannot open archive %s: %s\n" NORMAL,path,pzpd_last_error());
    return 0;
  }
  int ok = 0;
  int tJoints = pzpd_table_id(a,"joints");
  if (tJoints>=0)
  {
    const void * rows = 0;
    uint32_t joints = pzpd_global_rows(a,0,(unsigned) tJoints,&rows);
    *numberOfSamples   = (unsigned long) pzpd_count(a);
    *numberOfKeypoints = (unsigned short) joints;
    ok = (joints>0);
  }
  if (!ok) { fprintf(stderr,RED "Archive %s has no joints table\n" NORMAL,path); }
  pzpd_close(a);
  return ok;
}
//----------------------------------------------------------------------------------------------------
int pzpdOpenSources(struct DatabaseList * dbl)
{
  if ( (dbl==0) || (dbl->numberOfSources==0) ) { return 0; }
  const char ** paths = (const char **) malloc(sizeof(char*) * dbl->numberOfSources);
  if (paths==0) { return 0; }
  for (unsigned int i=0; i<dbl->numberOfSources; i++) { paths[i] = dbl->source[i].pathToDBFile; }
  dbl->archive = pzpd_open_many(paths,0,dbl->numberOfSources,0);
  free(paths);
  if (dbl->archive==0)
  {
    fprintf(stderr,RED "Cannot open the archive sources as one: %s\n" NORMAL,pzpd_last_error());
    return 0;
  }
  for (int s=0; s<PZPD_STREAM_COUNT; s++) { dbl->archiveStream[s] = pzpd_stream_id(dbl->archive,streamNames[s]); }
  if (dbl->archiveStream[PZPD_STREAM_RGB]<0)
  {
    fprintf(stderr,RED "The archive sources have no rgb stream\n" NORMAL);
    pzpdCloseSources(dbl);
    return 0;
  }
  fprintf(stderr,GREEN "Opened %u .pzpd sources (%lu records) with pzpdir v%s\n" NORMAL,dbl->numberOfSources,(unsigned long) pzpd_count(dbl->archive),pzpdirVersion);
  return 1;
}
//----------------------------------------------------------------------------------------------------
void pzpdCloseSources(struct DatabaseList * dbl)
{
  if ( (dbl!=0) && (dbl->archive!=0) )
  {
    pzpd_close(dbl->archive);
    dbl->archive = 0;
  }
}
//----------------------------------------------------------------------------------------------------
// Presence + codec of one optional stream of a record, checked from the index (no data I/O).
// Returns 1 and sets *kind (DL_FILE_NONE when absent), 0 on a blob the workers could not use.
static int streamKind(struct DatabaseList * dbl,uint64_t ordinal,enum PZPDStream stream,unsigned char * kind)
{
  *kind = DL_FILE_NONE;
  if (dbl->archiveStream[stream]<0) { return 1; }
  pzpd_blob_info bi;
  if (!pzpd_blob_info_get(dbl->archive,ordinal,(unsigned) dbl->archiveStream[stream],&bi))
  {
    fprintf(stderr,RED "Archive record %lu: %s\n" NORMAL,(unsigned long) ordinal,pzpd_last_error());
    return 0;
  }
  if (!bi.present) { return 1; }
  //PZPC = a PZP container (e.g. all_*PZPF), decoded by the same PZP codec as a single-frame PZP
  if      ( (bi.meta.format==PZPD_FORMAT_PZP) || (bi.meta.format==PZPD_FORMAT_PZPC) ) { *kind = DL_FILE_PZP; }
  else if (bi.meta.format==PZPD_FORMAT_PNG) { *kind = DL_FILE_PNG; }
  else
  {
    char f[5];
    fprintf(stderr,RED "Archive record %lu: %s blob %.*s has format %s, expected PZP, PZPC or PNG\n" NORMAL,(unsigned long) ordinal,streamNames[stream],(int) bi.name_len,bi.name,pzpd_format_name(bi.meta.format,f));
    return 0;
  }
  //The workers abort on depth that is not 1 channel / 16 bit, so refuse it here, before training starts.
  //(The "depth" blob is the source depth FILE byte for byte, PNG or PZP: it decodes big-endian and gets
  // DEPTH STEP 2's byte swap. A record that also has an "all" blob takes its depth from that one instead.)
  if ( (stream==PZPD_STREAM_DEPTH) && (bi.meta.meta_flags & PZPD_META_VALID) && ( (bi.meta.channels!=1) || (bi.meta.bits!=16) ) )
  {
    fprintf(stderr,RED "Archive record %lu: depth %.*s is %u channels / %u bit, expected 1 / 16\n" NORMAL,(unsigned long) ordinal,(int) bi.name_len,bi.name,bi.meta.channels,bi.meta.bits);
    return 0;
  }
  return 1;
}
//----------------------------------------------------------------------------------------------------
static int rgbUsable(struct DatabaseList * dbl,uint64_t ordinal)
{
  pzpd_blob_info bi;
  if (!pzpd_blob_info_get(dbl->archive,ordinal,(unsigned) dbl->archiveStream[PZPD_STREAM_RGB],&bi))
  {
    fprintf(stderr,RED "Archive record %lu: %s\n" NORMAL,(unsigned long) ordinal,pzpd_last_error());
    return 0;
  }
  if (!bi.present)
  {
    fprintf(stderr,RED "Archive record %lu has no rgb blob\n" NORMAL,(unsigned long) ordinal);
    return 0;
  }
  uint32_t f = bi.meta.format;
  if ( (f!=PZPD_FORMAT_JPEG) && (f!=PZPD_FORMAT_PNG) && (f!=PZPD_FORMAT_PZP) && (f!=PZPD_FORMAT_PZPC) && (f!=PZPD_FORMAT_PNM) )
  {
    char fs[5];
    fprintf(stderr,RED "Archive record %lu: rgb %.*s has format %s\n" NORMAL,(unsigned long) ordinal,(int) bi.name_len,bi.name,pzpd_format_name(f,fs));
    return 0;
  }
  return 1;
}
//----------------------------------------------------------------------------------------------------
struct PoseDatabase* readPoseDatabaseFromArchive(struct PoseDatabase* pdb,struct DatabaseList * dbl,unsigned int sourceID,unsigned long startOffset,unsigned long * loadedSamples,int ignoreNoSkeletonSamples)
{
  if ( (pdb==0) || (dbl==0) || (dbl->archive==0) || (sourceID>=dbl->numberOfSources) ) { return 0; }
  pzpd * a = dbl->archive;
  const char * path = dbl->source[sourceID].pathToDBFile;
  unsigned short J = pdb->keypointsForEachSample;

  uint64_t first = 0, count = 0;
  if (!pzpd_member_range(a,sourceID,&first,&count)) { fprintf(stderr,RED "%s: %s\n" NORMAL,path,pzpd_last_error()); return 0; }

  int tJoints  = pzpd_table_id(a,"joints");
  int tImage   = pzpd_table_id(a,"image");
  int tPersons = pzpd_table_id(a,"persons");
  int tDesc    = pzpd_table_id(a,"descriptions");
  int tVec     = pzpd_table_id(a,"descriptor_dinov3");
  int tVecInfo = pzpd_table_id(a,"descriptor_info");
  if ( (tJoints<0) || (tImage<0) || (tPersons<0) || (tDesc<0) )
  {
    fprintf(stderr,RED "%s: needs the joints, image, persons and descriptions tables (convertToPZPD.py)\n" NORMAL,path);
    return 0;
  }
  if (3*J > (1+MAX_KEYPOINT_NUMBER)*3)
  {
    fprintf(stderr,RED "%s: %u joints, the DataLoader holds at most %u\n" NORMAL,path,J,MAX_KEYPOINT_NUMBER);
    return 0;
  }

  const pzpd_schema * scJ = pzpd_table_schema(a,(unsigned) tJoints);
  const pzpd_schema * scI = pzpd_table_schema(a,(unsigned) tImage);
  const pzpd_schema * scP = pzpd_table_schema(a,(unsigned) tPersons);
  const pzpd_schema * scD = pzpd_table_schema(a,(unsigned) tDesc);
  int oJName = columnOffset(scJ,"name",PZPD_TYPE_STR,1),   oJParent = columnOffset(scJ,"parent",PZPD_TYPE_U16,1);
  int oW     = columnOffset(scI,"width",PZPD_TYPE_U16,1),  oH       = columnOffset(scI,"height",PZPD_TYPE_U16,1);
  int oPId   = columnOffset(scP,"id",PZPD_TYPE_U16,1),     oBBox    = columnOffset(scP,"bbox",PZPD_TYPE_U16,4);
  int oKp    = columnOffset(scP,"kp",PZPD_TYPE_U16,3*J),   oText    = columnOffset(scD,"text",PZPD_TYPE_STR,1);
  if ( (oJName<0) || (oJParent<0) || (oW<0) || (oH<0) || (oPId<0) || (oBBox<0) || (oKp<0) || (oText<0) ) { return 0; }

  //----------------------------------------------------------------------------------------
  // Joints: the first source defines them, every other source must have exactly the same set
  //----------------------------------------------------------------------------------------
  const unsigned char * jointRows = 0;
  uint32_t joints = pzpd_global_rows(a,sourceID,(unsigned) tJoints,(const void **) &jointRows);
  if (joints!=J)
  {
    fprintf(stderr,RED "Inconsistency, we where expecting %u joints but %s has %u\n" NORMAL,J,path,joints);
    return 0;
  }
  for (unsigned int j=0; j<J; j++)
  {
    const unsigned char * row = jointRows + (size_t) j * scJ->row_stride;
    size_t len = 0;
    const char * name = pzpd_global_str(a,sourceID,(unsigned) tJoints,row+oJName,&len);
    unsigned short parent;
    memcpy(&parent,row+oJParent,sizeof(parent));
    if ( (name==0) || (len>MAX_JOINT_NAME) )
    {
      fprintf(stderr,RED "%s: joint %u has no usable name\n" NORMAL,path,j);
      return 0;
    }
    if (sourceID==0)
    {
      memcpy(pdb->joint[j].name,name,len);
      pdb->joint[j].name[len] = 0;
      pdb->joint[j].parent    = parent;
    } else
    if ( (strlen(pdb->joint[j].name)!=len) || (memcmp(pdb->joint[j].name,name,len)!=0) || (pdb->joint[j].parent!=parent) )
    {
      fprintf(stderr,RED "%s: joint %u is %.*s (parent %u), the first source has %s (parent %u)\n" NORMAL,path,j,(int) len,name,parent,pdb->joint[j].name,pdb->joint[j].parent);
      return 0;
    }
  }

  //----------------------------------------------------------------------------------------
  // Descriptors: zero-copy rows of descriptor_dinov3, normalised as this build expects
  //----------------------------------------------------------------------------------------
  unsigned int D = 0;
  if (tVec>=0)
  {
    const pzpd_schema * scV = pzpd_table_schema(a,(unsigned) tVec);
    const unsigned char * info = 0;
    uint32_t infoRows = (tVecInfo>=0) ? pzpd_global_rows(a,sourceID,(unsigned) tVecInfo,(const void **) &info) : 0;
    if (infoRows>0)
    {
      const pzpd_schema * scN = pzpd_table_schema(a,(unsigned) tVecInfo);
      int oDim = columnOffset(scN,"dim",PZPD_TYPE_U32,1), oNorm = columnOffset(scN,"l2norm",PZPD_TYPE_U8,1);
      if ( (oDim<0) || (oNorm<0) || (scV->ncols!=1) || (scV->cols[0].type!=PZPD_TYPE_F32) ) { return 0; }
      uint32_t dim; memcpy(&dim,info+oDim,sizeof(dim));
      unsigned char l2norm = info[oNorm];
      D = scV->cols[0].count;
      if ( (dim!=D) || (D==0) || (D>MAX_DESCRIPTOR_LENGTH) )
      {
        fprintf(stderr,RED "%s: descriptor length %u (info says %u), the DataLoader holds at most %u\n" NORMAL,path,D,dim,MAX_DESCRIPTOR_LENGTH);
        return 0;
      }
      if (l2norm != (L2_NORMALIZE_DESCRIPTORS ? 1 : 0))
      {
        fprintf(stderr,RED "%s: descriptors are %sL2-normalised but this build has L2_NORMALIZE_DESCRIPTORS %u\n" NORMAL,path,l2norm ? "" : "not ",L2_NORMALIZE_DESCRIPTORS);
        return 0;
      }
    }
  }
  dbl->source[sourceID].descriptorLength = D;
  if (D) { fprintf(stderr,GREEN "%s: DinoV3 descriptors from the archive (D = %u)\n" NORMAL,path,D); }
    else { fprintf(stderr,RED "%s: no DinoV3 descriptors in the archive\n" NORMAL,path); }

  //----------------------------------------------------------------------------------------
  // Samples, in record order, with the same slot / skip logic as readPoseDatabase()
  //----------------------------------------------------------------------------------------
  unsigned long poseIndex = startOffset;
  for (uint64_t i=0; i<count; i++)
  {
    uint64_t ordinal = first + i;
    struct PoseEntry * e = &pdb->sample[poseIndex];
    e->width = 0; e->height = 0; e->imagePath[0] = 0; e->numberOfSkeletons = 0;

    size_t keyLen = 0;
    const char * key = pzpd_record_key(a,ordinal,&keyLen);
    if ( (key==0) || (keyLen>MAX_IMAGE_PATH) )
    {
      fprintf(stderr,RED "%s: record %lu has no usable key\n" NORMAL,path,(unsigned long) i);
      return 0;
    }
    memcpy(e->imagePath,key,keyLen);
    e->imagePath[keyLen] = 0;

    const unsigned char * row = 0;
    if (pzpd_table_rows(a,ordinal,(unsigned) tImage,(const void **) &row)!=1)
    {
      fprintf(stderr,RED "%s: %s has no image row\n" NORMAL,path,e->imagePath);
      return 0;
    }
    memcpy(&e->width, row+oW,sizeof(e->width));
    memcpy(&e->height,row+oH,sizeof(e->height));

    const unsigned char * persons = 0;
    uint32_t P = pzpd_table_rows(a,ordinal,(unsigned) tPersons,(const void **) &persons);
    if (P > MAX_SKELETONS_PER_IMAGE)
    {
      fprintf(stderr,RED "Implausible skeleton count %u for %s in %s — aborting load\n" NORMAL,P,e->imagePath,path);
      return 0;
    }
    e->numberOfSkeletons = (unsigned short) P;

    e->descriptor = NULL;
    if (D)
    {
      const void * v = 0;
      if (pzpd_table_rows(a,ordinal,(unsigned) tVec,&v)!=1)
      {
        fprintf(stderr,RED "Descriptor Mismatch @ %lu of %s | Image (%s) has no descriptor row\n" NORMAL,poseIndex,path,e->imagePath);
        return 0;
      }
      e->descriptor = (float *) v;
    }
    e->superpoint = NULL;

    const unsigned char * caption = 0;
    if (pzpd_table_rows(a,ordinal,(unsigned) tDesc,(const void **) &caption)!=1)
    {
      fprintf(stderr,RED "%s: %s has no caption\n" NORMAL,path,e->imagePath);
      return 0;
    }
    size_t captionLen = 0;
    e->description       = pzpd_table_str(a,ordinal,(unsigned) tDesc,caption+oText,&captionLen);
    e->descriptionLength = (unsigned int) captionLen;
    e->numberOfTokens    = 0; //Tokens are not stored in archives, only the caption text

    //Presence and codecs from the index: the same cache resolvePathToRequestedFiles() fills on first touch
    if (!rgbUsable(dbl,ordinal)) { return 0; }
    if ( (!streamKind(dbl,ordinal,PZPD_STREAM_ALL,&e->combinedKind)) ||
         (!streamKind(dbl,ordinal,PZPD_STREAM_DEPTH,&e->depthKind)) ||
         (!streamKind(dbl,ordinal,PZPD_STREAM_SEG,&e->segKind)) )
       { return 0; }
    e->geoKind        = DL_FILE_NONE;
    e->pathResolved   = 1;
    e->archiveOrdinal = (unsigned long) ordinal;

    int skipAdvance = 0;
    if (P==0)
    {
      e->sk = 0;
      if (ignoreNoSkeletonSamples) { skipAdvance = 1; }
    } else
    {
      e->sk = (struct Skeleton*) malloc(P * sizeof(struct Skeleton));
      if (e->sk==0) { fprintf(stderr,RED "Out of memory reading %s\n" NORMAL,path); return 0; }
      memset(e->sk,0,P * sizeof(struct Skeleton));
      for (uint32_t k=0; k<P; k++)
      {
        const unsigned char * p = persons + (size_t) k * scP->row_stride;
        struct Skeleton * sk = &e->sk[k];
        unsigned short id, bbox[4];
        memcpy(&id,p+oPId,sizeof(id));
        memcpy(bbox,p+oBBox,sizeof(bbox));
        sk->bboxX = bbox[0]; sk->bboxY = bbox[1]; sk->bboxX2 = bbox[2]; sk->bboxY2 = bbox[3];
        memcpy(sk->coords,p+oKp,3 * J * sizeof(unsigned short));
        if ((int) k != (int)(unsigned short)(id-1))
           { fprintf(stderr,"Inconsistency in sample %lu / expected skeleton %u / found %u \n",poseIndex,k,(unsigned short)(id-1)); }
      }
    }

    if (!skipAdvance) { poseIndex += 1; }
  }

  *loadedSamples = poseIndex; //as readPoseDatabase(): the next free slot, i.e. including startOffset
  return pdb;
}
//----------------------------------------------------------------------------------------------------
// Worker side
//----------------------------------------------------------------------------------------------------
// Streams the worker will decode for a sample (PrepareBatch.c): rgb, then either the combined file
// or depth / segmentation, as the flags and the sample's presence cache say.
static uint32_t sampleMask(struct ImageDatabase * db,unsigned long sampleNumber)
{
  struct DatabaseList * dbl = db->dbSources;
  struct PoseEntry * e = &db->pdb->sample[sampleNumber];
  uint32_t mask = 1u << dbl->archiveStream[PZPD_STREAM_RGB];
  int doOutputs = db->doHeatmapOutput;
  if (e->combinedKind!=DL_FILE_NONE)
  {
    if (doOutputs) { mask |= 1u << dbl->archiveStream[PZPD_STREAM_ALL]; }
  } else
  {
    if ( (doOutputs) && (db->addDepthHeatmap) && (e->depthKind!=DL_FILE_NONE) )        { mask |= 1u << dbl->archiveStream[PZPD_STREAM_DEPTH]; }
    if ( (doOutputs) && (db->addSegmentationHeatmaps) && (e->segKind!=DL_FILE_NONE) )  { mask |= 1u << dbl->archiveStream[PZPD_STREAM_SEG]; }
  }
  return mask;
}
//----------------------------------------------------------------------------------------------------
void pzpdSampleBegin(struct ImageDatabase * db,unsigned long sampleNumber,struct ArchiveSample * s)
{
  memset(s,0,sizeof(*s));
  if (db->prefetcher==0) { return; } //mmap views are taken per stream in pzpdSampleReadImage()
  int n = pzpd_prefetch_get(db->prefetcher,db->pdb->sample[sampleNumber].archiveOrdinal,sampleMask(db,sampleNumber),s->refs,&s->ticket);
  if (n<0)
  {
    fprintf(stderr,RED "Reading %s from its archive failed: %s\n" NORMAL,db->pdb->sample[sampleNumber].imagePath,pzpd_last_error());
    fprintf(stderr,RED "Terminating to avoid training inconsistency\n" NORMAL);
    abort();
  }
  s->haveTicket = 1;
}
//----------------------------------------------------------------------------------------------------
struct Image * pzpdSampleReadImage(struct ImageDatabase * db,unsigned long sampleNumber,struct ArchiveSample * s,enum PZPDStream stream)
{
  struct DatabaseList * dbl = db->dbSources;
  int id = dbl->archiveStream[stream];
  if (id<0) { return 0; }
  uint64_t ordinal = db->pdb->sample[sampleNumber].archiveOrdinal;

  const void * data = s->refs[id].data;
  size_t size       = s->refs[id].size;
  uint32_t format   = s->refs[id].format;
  if (data==0) //not prefetched (no prefetcher, or not in the submitted mask): a view of the mapped shard
  {
    pzpd_blob_info bi;
    data = pzpd_view(dbl->archive,ordinal,(unsigned) id,&size);
    if ( (data==0) || (!pzpd_blob_info_get(dbl->archive,ordinal,(unsigned) id,&bi)) )
    {
      fprintf(stderr,RED "Reading %s of %s from its archive failed: %s\n" NORMAL,streamNames[stream],db->pdb->sample[sampleNumber].imagePath,pzpd_last_error());
      return 0;
    }
    format = bi.meta.format;
  }

  unsigned int codec = NO_CODEC;
  const char * extension = "";
  if      (format==PZPD_FORMAT_JPEG) { codec = JPG_CODEC; extension = ".jpg"; }
  else if (format==PZPD_FORMAT_PNG)  { codec = PNG_CODEC; extension = ".png"; }
  else if ( (format==PZPD_FORMAT_PZP) || (format==PZPD_FORMAT_PZPC) ) { codec = PZP_CODEC; extension = ".pzp"; }
  else if (format==PZPD_FORMAT_PNM)  { codec = PNM_CODEC; extension = ".pnm"; }
  //The decoders only read the buffer (it may be a read-only mapping)
  return readImageFromMemory(extension,(void *) data,(unsigned long) size,codec);
}
//----------------------------------------------------------------------------------------------------
void pzpdSampleEnd(struct ImageDatabase * db,struct ArchiveSample * s)
{
  if ( (db->prefetcher!=0) && (s->haveTicket) )
  {
    pzpd_prefetch_release(db->prefetcher,&s->ticket);
    s->haveTicket = 0;
  }
}
//----------------------------------------------------------------------------------------------------
// Prefetcher
//----------------------------------------------------------------------------------------------------
int pzpdPrefetchCreate(struct ImageDatabase * db)
{
  if ( (db==0) || (db->dbSources==0) || (db->dbSources->archive==0) ) { return 1; }
  if (g_prefetchMode<0)
  {
    fprintf(stderr,YELLOW "Archive prefetching is off, workers read mmap views on demand\n" NORMAL);
    return 1;
  }
  pzpd_prefetch_opts o;
  memset(&o,0,sizeof(o));
  o.mode         = (unsigned) g_prefetchMode;
  o.io_threads   = (unsigned) g_prefetchIOThreads;
  o.budget_bytes = (uint64_t) g_prefetchBudgetMB * 1024 * 1024;
  o.window       = (unsigned) g_prefetchWindow;
  db->prefetcher = pzpd_prefetcher_create(db->dbSources->archive,&o);
  if (db->prefetcher==0)
  {
    fprintf(stderr,RED "Cannot create the archive prefetcher: %s\n" NORMAL,pzpd_last_error());
    return 0;
  }
  db->prefetchedVersion = db->indicesVersion - 1; //nothing submitted yet
  db->prefetchNext      = (unsigned long) -1;
  const char * modes[] = { "auto", "map", "pagecache", "buffers" };
  fprintf(stderr,GREEN "Archive prefetcher: mode %s, shard 0 %s\n" NORMAL,
          (g_prefetchMode<=3) ? modes[g_prefetchMode] : "?",
          modes[ (pzpd_prefetch_auto_mode(db->dbSources->archive,0)>=0) ? pzpd_prefetch_auto_mode(db->dbSources->archive,0) : 0 ]);
  return 1;
}
//----------------------------------------------------------------------------------------------------
// Called before every batch (idle pipeline): the whole rest of the epoch order is submitted when the
// order changed (a shuffle) or the batch is not the one the schedule expects next.
void pzpdPrefetchBatch(struct ImageDatabase * db,unsigned long sampleStart,unsigned long sampleEnd)
{
  if ( (db==0) || (db->prefetcher==0) ) { return; }
  if ( (db->prefetchedVersion!=db->indicesVersion) || (db->prefetchNext!=sampleStart) )
  {
    pzpd_prefetch_clear(db->prefetcher);
    unsigned long n = (sampleStart<db->numberOfSamples) ? db->numberOfSamples - sampleStart : 0;
    uint64_t * ordinals = (uint64_t *) malloc(sizeof(uint64_t) * (n+1));
    uint32_t * masks    = (uint32_t *) malloc(sizeof(uint32_t) * (n+1));
    if ( (ordinals!=0) && (masks!=0) )
    {
      unsigned long k = 0;
      for (unsigned long pos=sampleStart; pos<db->numberOfSamples; pos++)
      {
        SampleNumber sampleNumber = 0;
        if (db_resolve_sample(db,pos,&sampleNumber))
        {
          ordinals[k] = db->pdb->sample[sampleNumber].archiveOrdinal;
          masks[k]    = sampleMask(db,sampleNumber);
          k++;
        }
      }
      if (!pzpd_prefetch_submit(db->prefetcher,ordinals,masks,k))
        { fprintf(stderr,RED "Archive prefetch submit failed: %s (samples are read on demand)\n" NORMAL,pzpd_last_error()); }
    }
    free(ordinals);
    free(masks);
    db->prefetchedVersion = db->indicesVersion;
  }
  db->prefetchNext = sampleEnd;
}
//----------------------------------------------------------------------------------------------------
void pzpdPrefetchDestroy(struct ImageDatabase * db)
{
  if ( (db==0) || (db->prefetcher==0) ) { return; }
  pzpd_prefetch_stats st;
  pzpd_prefetch_stats_get(db->prefetcher,&st);
  fprintf(stderr,"Archive prefetcher: %lu gets, %lu prefetched hits, %lu waits, %lu read on demand, %.1f MB read ahead\n",
          (unsigned long) st.released,(unsigned long) st.hits,(unsigned long) st.waits,(unsigned long) (st.sync_misses + st.unscheduled),
          (double) st.bytes_prefetched / (1024.0*1024.0));
  pzpd_prefetcher_destroy(db->prefetcher);
  db->prefetcher = 0;
}
//----------------------------------------------------------------------------------------------------
int db_get_sample_description(struct ImageDatabase * db,unsigned long sampleID,char * out,unsigned int outSize)
{
  SampleNumber sampleNumber = 0;
  if ( (db==0) || (out==0) || (outSize==0) || (!db_resolve_sample(db,sampleID,&sampleNumber)) ) { return -1; }
  struct PoseEntry * e = &db->pdb->sample[sampleNumber];
  unsigned int n = e->descriptionLength;
  if (n >= outSize) { n = outSize - 1; }
  if ( (n>0) && (e->description!=0) ) { memcpy(out,e->description,n); }
  out[n] = 0;
  return (int) e->descriptionLength;
}
