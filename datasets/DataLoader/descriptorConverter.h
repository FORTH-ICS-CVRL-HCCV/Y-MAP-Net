#ifndef DESCRIPTORCONVERTER_H_INCLUDED
#define DESCRIPTORCONVERTER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// configuration.h is normally pulled in AFTER this header (DataLoader.h:19 vs :34), so it
// is included explicitly here to make its toggles visible; it has its own include guard.
#include "configuration.h"

#ifndef L2_NORMALIZE_DESCRIPTORS
#define L2_NORMALIZE_DESCRIPTORS 0
#endif

#define MAX_LINE_LEN 32000

typedef struct
{
    uint8_t name_len;
    char *filename;     // null-terminated
    float *values;      // dynamically allocated based on descriptor dimension
    int value_count;    // store how many floats are allocated
} DescriptorEntry;

typedef struct
{
    int32_t count;
    int32_t max_filename_len;
    int value_count;    // global descriptor dimensionality
    DescriptorEntry *entries;
} DescriptorDataset;


static int count_lines_in_csv(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "CSV open failed for line count\n");
        return 0;
    }

    int count = 0;
    char line[MAX_LINE_LEN] = {0};

    // Skip header
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp))
    {
        count++;
    }

    fclose(fp);
    return count;
}

static void free_descriptor_dataset(DescriptorDataset *ds)
{
  if (ds!=NULL)
  {
  if (ds->entries!=NULL)
  {
    for (int i = 0; i < ds->count; ++i)
    {
        if (ds->entries[i].filename!=NULL)
        {
          free(ds->entries[i].filename);
          ds->entries[i].filename = NULL;
        }

        if (ds->entries[i].values!=NULL)
        {
           free(ds->entries[i].values);
           ds->entries[i].values = NULL;
        }
    }
    free(ds->entries);
    ds->entries = NULL;
    }
  }
}

// Scale every descriptor to unit L2 norm, in place, once per file load.
// DescriptorLoss (NNLosses.py) documents its targets as unit-norm, but the shipped DINOv3
// dumps arrive with ||v|| ~ 14.2, which makes its norm-reg term dominate the objective.
// A vector that is exactly zero means "this sample has no descriptor" (its source has no
// .dinov3 sidecar); those are left as zeros so the Python-side guard can still spot them
// and so we never divide by zero.
static void l2_normalize_descriptor_dataset(DescriptorDataset *ds, const char *filename)
{
#if L2_NORMALIZE_DESCRIPTORS
    if (ds == NULL || ds->entries == NULL) { return; }

    unsigned long normalized = 0, zeros = 0;
    for (int i = 0; i < ds->count; ++i)
    {
        DescriptorEntry *e = &ds->entries[i];
        if (e->values == NULL || e->value_count <= 0) { continue; }

        double sum = 0.0;
        for (int d = 0; d < e->value_count; ++d) { sum += (double) e->values[d] * (double) e->values[d]; }

        if (sum <= 0.0) { zeros++; continue; }   // missing descriptor -> deliberately left all-zero

        const float inverse = (float) (1.0 / sqrt(sum));
        for (int d = 0; d < e->value_count; ++d) { e->values[d] *= inverse; }
        normalized++;
    }
    fprintf(stderr, "Descriptor file '%s': L2-normalised %lu vectors (%lu all-zero left untouched)\n",
            filename, normalized, zeros);
#endif
}

static DescriptorDataset * load_descriptor_csv(const char *filename)
{
    int line_count = count_lines_in_csv(filename);
    if (line_count == 0) return NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "CSV open failed\n");
        return NULL;
    }

    char line[MAX_LINE_LEN] = {0};

    DescriptorDataset *ds = malloc(sizeof(DescriptorDataset));
    if (!ds)
    {
        fclose(fp);
        return NULL;
    }
    memset(ds, 0, sizeof(DescriptorDataset));

    ds->count = line_count;
    ds->entries = malloc(ds->count * sizeof(DescriptorEntry));
    if (!ds->entries)
    {
        free(ds);
        fclose(fp);
        return NULL;
    }
    memset(ds->entries, 0, ds->count * sizeof(DescriptorEntry));

    // Read header to determine dimensionality
    fgets(line, sizeof(line), fp);
    char *token = strtok(line, ",");
    int value_count = 0;
    while ((token = strtok(NULL, ",")) != NULL)
       {
        value_count++;
       }
    ds->value_count = value_count;

    // Read each entry
    int index = 0;
    while (fgets(line, sizeof(line), fp))
    {
        token = strtok(line, ",");
        int len = strlen(token);
        while (len > 0 && (token[len-1] == ' ' || token[len-1] == '\n')) token[--len] = 0;

        ds->entries[index].name_len = (uint8_t)len;
        ds->entries[index].filename = malloc(len + 1);
        memcpy(ds->entries[index].filename, token, len);
        ds->entries[index].filename[len] = '\0';
        if (len > ds->max_filename_len) ds->max_filename_len = len;

        ds->entries[index].value_count = value_count;
        ds->entries[index].values = malloc(value_count * sizeof(float));
        if (!ds->entries[index].values)
        {
            fprintf(stderr, "Memory allocation failed for descriptor values\n");
            exit(1);
        }

        for (int i = 0; i < value_count; ++i)
        {
            token = strtok(NULL, ",");
            if (!token)
            {
                fprintf(stderr, "Invalid line (not enough floats) at line %u float %u\n", index, i);
                exit(1);
            }
            ds->entries[index].values[i] = strtof(token, NULL);
        }

        index++;
        if (index % 100 == 0)
            fprintf(stderr, "Loaded %u/%u\n", index, ds->count);
    }

    fclose(fp);
    l2_normalize_descriptor_dataset(ds, filename);
    return ds;
}

static int save_descriptor_bin(const char *filename, const DescriptorDataset *ds)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Binary write failed\n");
        return 0;
    }

    fwrite(&ds->count, sizeof(int32_t), 1, fp);
    fwrite(&ds->max_filename_len, sizeof(int32_t), 1, fp);
    fwrite(&ds->value_count, sizeof(int32_t), 1, fp);

    for (int i = 0; i < ds->count; ++i)
    {
        DescriptorEntry *e = &ds->entries[i];
        fwrite(&e->name_len, sizeof(uint8_t), 1, fp);
        fwrite(e->filename, 1, e->name_len, fp);
        fwrite(e->values, sizeof(float), e->value_count, fp);
    }

    fclose(fp);
    return 1;
}

// Maximum plausible descriptor dimensionality. Real embeddings are at most a few
// thousand (DINOv2/v3: 384/768/1024/1280/1536). A "value_count" far above this means
// we are reading a pre-2026-01-03 file whose 8-byte header has NO value_count field,
// so the int32 we read as value_count is actually entry-0's name_len+filename bytes.
#define DESCRIPTOR_MAX_PLAUSIBLE_DIM 8192

// Recover the per-entry float count of a legacy (8-byte header) descriptor file by
// trying the known embedding dimensions and keeping whichever makes the entry walk
// land EXACTLY on EOF after 'count' entries. 'dataStart' is the byte offset of the
// first entry (8 for legacy files). Returns the dim, or -1 if none fit.
// The file position is left undefined; the caller must fseek before reading entries.
static int infer_legacy_descriptor_value_count(FILE *fp, int count, long dataStart)
{
    static const int candidates[] = { 768, 1024, 1280, 384, 512, 1536, 2048, 300 };
    const int nCand = (int)(sizeof(candidates) / sizeof(candidates[0]));

    if (fseek(fp, 0, SEEK_END) != 0) { return -1; }
    long fileSize = ftell(fp);
    if (fileSize < dataStart) { return -1; }

    for (int c = 0; c < nCand; ++c)
    {
        const long V = candidates[c];
        long pos = dataStart;
        int ok = 1;
        for (int i = 0; i < count; ++i)
        {
            uint8_t nl;
            if (fseek(fp, pos, SEEK_SET) != 0 || fread(&nl, 1, 1, fp) != 1) { ok = 0; break; }
            pos += 1 + (long)nl + V * (long)sizeof(float);
            if (pos > fileSize) { ok = 0; break; }   // overshoot -> wrong dim
        }
        if (ok && pos == fileSize) { return (int)V; }  // exact fit -> this is the dim
    }
    return -1;
}

static DescriptorDataset * load_descriptor_bin(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Binary Descriptors could not be read from path: %s\n", filename);
        return NULL;
    }

    DescriptorDataset *ds = malloc(sizeof(DescriptorDataset));
    if (!ds)
    {
        fclose(fp);
        return NULL;
    }
    memset(ds, 0, sizeof(DescriptorDataset));

    if (fread(&ds->count, sizeof(int32_t), 1, fp) != 1 ||
        fread(&ds->max_filename_len, sizeof(int32_t), 1, fp) != 1 ||
        fread(&ds->value_count, sizeof(int32_t), 1, fp) != 1)
    {
        fprintf(stderr, "Failed to read dataset header\n");
        free(ds);
        fclose(fp);
        return NULL;
    }

    // Backward compatibility with legacy (pre-2026-01-03) descriptor files: those
    // have an 8-byte header (count, max_filename_len) with NO value_count field, so
    // the third int32 we just read really belongs to entry 0 and comes out garbage.
    // Detect that, rewind the data to offset 8, and recover the dim by counting.
    long dataStart = 12;  // current format: 12-byte header incl. value_count
    if (ds->value_count <= 0 || ds->value_count > DESCRIPTOR_MAX_PLAUSIBLE_DIM)
    {
        dataStart = 8;    // legacy format: entries begin right after count+max_filename_len
        int inferred = infer_legacy_descriptor_value_count(fp, ds->count, dataStart);
        if (inferred <= 0)
        {
            fprintf(stderr, "Descriptor file '%s': legacy 8-byte header but descriptor "
                            "dimension could not be inferred; ignoring file.\n", filename);
            free(ds);
            fclose(fp);
            return NULL;
        }
        ds->value_count = inferred;
        fprintf(stderr, "Descriptor file '%s': legacy 8-byte header, inferred descriptor dim = %d\n",
                filename, ds->value_count);
    }
    // infer_legacy_descriptor_value_count moved the file pointer; (re)position it at
    // the first entry for both formats before the per-entry read loop below.
    if (fseek(fp, dataStart, SEEK_SET) != 0)
    {
        fprintf(stderr, "Failed to seek to descriptor data in '%s'\n", filename);
        free(ds);
        fclose(fp);
        return NULL;
    }

    ds->entries = malloc(ds->count * sizeof(DescriptorEntry));
    if (!ds->entries)
    {
        free(ds);
        fclose(fp);
        return NULL;
    }
    memset(ds->entries, 0, ds->count * sizeof(DescriptorEntry));

    /*
    for (int i = 0; i < ds->count; ++i)
    {
        DescriptorEntry *e = &ds->entries[i];

        fread(&e->name_len, sizeof(uint8_t), 1, fp);
        e->filename = malloc(e->name_len + 1);
        fread(e->filename, 1, e->name_len, fp);
        e->filename[e->name_len] = '\0';

        e->value_count = ds->value_count;
        e->values = malloc(e->value_count * sizeof(float));
        fread(e->values, sizeof(float), e->value_count, fp);
    }*/
    for (int i = 0; i < ds->count; ++i)
    {
    DescriptorEntry *e = &ds->entries[i];

    // Read name_len
    if (fread(&e->name_len, sizeof(uint8_t), 1, fp) != 1)
    {
        fprintf(stderr, "Failed to read name_len for entry %d\n", i);
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }

    // Read filename
    e->filename = malloc(e->name_len + 1);
    if (!e->filename)
    {
        fprintf(stderr, "Memory allocation failed for filename of entry %d\n", i);
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }

    if (fread(e->filename, 1, e->name_len, fp) != e->name_len)
    {
        fprintf(stderr, "Failed to read filename for entry %d\n", i);
        // e->filename is owned by ds->entries[i]; free_descriptor_dataset frees it.
        // Do NOT free it here too (that double-frees -> "free(): double free in tcache").
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }
    e->filename[e->name_len] = '\0'; // Null terminator

    // Allocate values
    e->value_count = ds->value_count;
    e->values = malloc(e->value_count * sizeof(float));
    if (!e->values)
    {
        fprintf(stderr, "Memory allocation failed for values of entry %d\n", i);
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }

    // Read values
    size_t actuallyRead = fread(e->values, sizeof(float), e->value_count, fp);
    if ( actuallyRead != (size_t)e->value_count)
    {
        fprintf(stderr, "Failed to read descriptor values for entry %d (actually read %lu)\n", i,actuallyRead);
        // e->filename and e->values are owned by ds->entries[i]; free_descriptor_dataset
        // frees both. Do NOT free them here too (double-free -> tcache abort, the exact
        // crash seen when libDataLoader.so reads an old 8-byte-header descriptor file).
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }
    }


    fclose(fp);
    l2_normalize_descriptor_dataset(ds, filename);
    return ds;
}

static void print_descriptor_dataset_summary(const DescriptorDataset *ds)
{
    printf("Entries: %d\n", ds->count);
    printf("Max filename length: %d\n", ds->max_filename_len);
    printf("Descriptor dimension: %d\n", ds->value_count);
    for (int i = 0; i < ds->count && i < 3; ++i)
    {
        printf("File[%d]: %.*s, First float: %.4f\n",
               i, ds->entries[i].name_len, ds->entries[i].filename,
               ds->entries[i].values[0]);
    }
}

static float* get_descriptor_vector_by_id(const DescriptorDataset *ds, int id)
{
    if (id < 0 || id >= ds->count) return NULL;
    return ds->entries[id].values;
}

static float* get_descriptor_vector_by_filename(const DescriptorDataset *ds, const char *filename)
{
    for (int i = 0; i < ds->count; ++i)
    {
        DescriptorEntry *e = &ds->entries[i];
        if (strlen(filename) == e->name_len && strncmp(e->filename, filename, e->name_len) == 0)
            return e->values;
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif

#endif // DESCRIPTORCONVERTER_H_INCLUDED
