#ifndef DESCRIPTORCONVERTER_H_INCLUDED
#define DESCRIPTORCONVERTER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
        free(e->filename);
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
        free(e->filename);
        free(e->values);
        free_descriptor_dataset(ds);
        fclose(fp);
        return NULL;
    }
    }


    fclose(fp);
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
