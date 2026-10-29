#ifndef SUPERPOINTLOADER_H_INCLUDED
#define SUPERPOINTLOADER_H_INCLUDED

/**
 * @file superpointLoader.h
 * @brief Loader for the ragged ".superpoint" keypoint files and the frozen PCA
 *        basis, plus the load-time projection used to render SuperPoint keypoints
 *        as heatmap channels.
 *
 * Unlike the DINOv2/v3 descriptor path (descriptorConverter.h — one fixed-length
 * vector per image), SuperPoint emits a VARIABLE number of keypoints per image,
 * each with (x, y, score, descDim-d descriptor). We keep the raw descriptors and
 * project them to K PCA components at LOAD time, so K (= the channel count) is a
 * runtime knob with no re-extraction.
 *
 * This path is NOT gated by USE_DINOV2_FEATURES — it is runtime-gated by
 * db->addSuperpointHeatmaps and the presence of a ".superpoint" file, exactly
 * like addNormalHeatmaps / addSegmentationHeatmaps.
 *
 * File formats are written by datasets/superpoint/extract_superpoint.py and
 * datasets/superpoint/fit_superpoint_pca.py (kept in sync there).
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ── .superpoint ragged keypoint file ──────────────────────────────────────────
//   char  magic[4] = 'SPT1'
//   int32 numImages,  int32 descDim
//   per image: uint8 nameLen, char filename[nameLen], int32 numKeypoints,
//              numKeypoints * (float32 x, y, score, desc[descDim])
typedef struct
{
    uint8_t  name_len;
    char    *filename;       // null-terminated, basename
    int32_t  numKeypoints;
    float   *xyscore;        // numKeypoints * 3  (x, y, score) interleaved
    float   *descriptors;    // numKeypoints * descDim
} SuperpointImageEntry;

typedef struct
{
    int32_t numImages;
    int32_t descDim;
    SuperpointImageEntry *entries;
} SuperpointDataset;

// ── superpoint_pca.bin frozen basis ──────────────────────────────────────────
//   int32 descDim, int32 maxComponents
//   float32 mean[descDim]
//   float32 components[maxComponents * descDim]   (row j = component j)
//   float32 scale[maxComponents]
typedef struct
{
    int32_t descDim;
    int32_t maxComponents;
    float  *mean;            // descDim
    float  *components;      // maxComponents * descDim (row-major)
    float  *scale;           // maxComponents
} SuperpointPCA;


static void free_superpoint_dataset(SuperpointDataset *ds)
{
    if (ds == NULL) { return; }
    if (ds->entries != NULL)
    {
        for (int i = 0; i < ds->numImages; ++i)
        {
            free(ds->entries[i].filename);
            free(ds->entries[i].xyscore);
            free(ds->entries[i].descriptors);
        }
        free(ds->entries);
    }
    free(ds);
}

static SuperpointDataset * load_superpoint_bin(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { return NULL; }   // absent file is not an error — superpoint is optional

    char magic[4];
    int32_t numImages = 0, descDim = 0;
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "SPT1", 4) != 0 ||
        fread(&numImages, sizeof(int32_t), 1, fp) != 1 ||
        fread(&descDim,   sizeof(int32_t), 1, fp) != 1 ||
        numImages < 0 || descDim <= 0)
    {
        fprintf(stderr, "SuperPoint file '%s': bad header.\n", filename);
        fclose(fp);
        return NULL;
    }

    SuperpointDataset *ds = (SuperpointDataset*) malloc(sizeof(SuperpointDataset));
    if (!ds) { fclose(fp); return NULL; }
    memset(ds, 0, sizeof(SuperpointDataset));
    ds->numImages = numImages;
    ds->descDim   = descDim;
    ds->entries   = (SuperpointImageEntry*) calloc(numImages, sizeof(SuperpointImageEntry));
    if (!ds->entries && numImages > 0) { free(ds); fclose(fp); return NULL; }

    for (int i = 0; i < numImages; ++i)
    {
        SuperpointImageEntry *e = &ds->entries[i];

        if (fread(&e->name_len, 1, 1, fp) != 1) { goto fail; }
        e->filename = (char*) malloc((size_t)e->name_len + 1);
        if (!e->filename) { goto fail; }
        if (fread(e->filename, 1, e->name_len, fp) != e->name_len) { goto fail; }
        e->filename[e->name_len] = '\0';

        if (fread(&e->numKeypoints, sizeof(int32_t), 1, fp) != 1 || e->numKeypoints < 0) { goto fail; }

        if (e->numKeypoints > 0)
        {
            size_t nk = (size_t) e->numKeypoints;
            e->xyscore     = (float*) malloc(nk * 3 * sizeof(float));
            e->descriptors = (float*) malloc(nk * (size_t)descDim * sizeof(float));
            if (!e->xyscore || !e->descriptors) { goto fail; }
            // Stream is interleaved per keypoint: x,y,score, desc[descDim]
            for (size_t k = 0; k < nk; ++k)
            {
                if (fread(e->xyscore + k * 3, sizeof(float), 3, fp) != 3) { goto fail; }
                if (fread(e->descriptors + k * (size_t)descDim, sizeof(float), (size_t)descDim, fp) != (size_t)descDim) { goto fail; }
            }
        }
    }

    fclose(fp);
    return ds;

fail:
    fprintf(stderr, "SuperPoint file '%s': truncated/unreadable.\n", filename);
    free_superpoint_dataset(ds);
    fclose(fp);
    return NULL;
}


static void free_superpoint_pca(SuperpointPCA *p)
{
    if (p == NULL) { return; }
    free(p->mean);
    free(p->components);
    free(p->scale);
    free(p);
}

static SuperpointPCA * load_superpoint_pca(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "SuperPoint PCA '%s' could not be opened.\n", filename);
        return NULL;
    }

    SuperpointPCA *p = (SuperpointPCA*) malloc(sizeof(SuperpointPCA));
    if (!p) { fclose(fp); return NULL; }
    memset(p, 0, sizeof(SuperpointPCA));

    if (fread(&p->descDim,       sizeof(int32_t), 1, fp) != 1 ||
        fread(&p->maxComponents, sizeof(int32_t), 1, fp) != 1 ||
        p->descDim <= 0 || p->maxComponents <= 0 || p->maxComponents > p->descDim)
    {
        fprintf(stderr, "SuperPoint PCA '%s': bad header.\n", filename);
        free(p); fclose(fp); return NULL;
    }

    size_t dd = (size_t) p->descDim;
    size_t mc = (size_t) p->maxComponents;
    p->mean       = (float*) malloc(dd * sizeof(float));
    p->components = (float*) malloc(mc * dd * sizeof(float));
    p->scale      = (float*) malloc(mc * sizeof(float));
    if (!p->mean || !p->components || !p->scale ||
        fread(p->mean,       sizeof(float), dd,      fp) != dd ||
        fread(p->components, sizeof(float), mc * dd, fp) != mc * dd ||
        fread(p->scale,      sizeof(float), mc,      fp) != mc)
    {
        fprintf(stderr, "SuperPoint PCA '%s': truncated.\n", filename);
        free_superpoint_pca(p); fclose(fp); return NULL;
    }

    fclose(fp);
    fprintf(stderr, "SuperPoint PCA: descDim=%d, maxComponents=%d\n", p->descDim, p->maxComponents);
    return p;
}

/**
 * Project one keypoint descriptor onto PCA component j and scale it into the
 * heatmap value range. Caller clips to [MINV, MAXV].
 *   v_j = scale[j] * dot( components[j], desc - mean )
 */
static inline float superpoint_project_component(const SuperpointPCA *pca, const float *desc, int j)
{
    const float *comp = pca->components + (size_t)j * (size_t)pca->descDim;
    float acc = 0.0f;
    for (int d = 0; d < pca->descDim; ++d)
    {
        acc += comp[d] * (desc[d] - pca->mean[d]);
    }
    return acc * pca->scale[j];
}

#ifdef __cplusplus
}
#endif

#endif // SUPERPOINTLOADER_H_INCLUDED
