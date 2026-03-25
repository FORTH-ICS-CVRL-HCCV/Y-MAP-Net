#ifndef EMBEDDINGS_H_INCLUDED
#define EMBEDDINGS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_KEY_LENGTH 50  // Maximum length of each string

// Struct to hold the array of strings
struct EmbeddingVector
{
    float *vector;
};

struct Embeddings
{
    int D;
    int Classes;
    float offset;
    float scaling;
    float * emptyVector;
    struct EmbeddingVector *embeddings;
};

// Magic bytes stored at the start of the binary cache to detect corrupt/stale files.
#define EMBEDDINGS_BIN_MAGIC 0x454D4200u

static int _load_embeddings_alloc(struct Embeddings *emb)
{
    emb->embeddings = (struct EmbeddingVector *) malloc(emb->Classes * sizeof(struct EmbeddingVector));
    if (!emb->embeddings) return 0;
    memset(emb->embeddings, 0, emb->Classes * sizeof(struct EmbeddingVector));

    emb->emptyVector = (float*) malloc(emb->D * sizeof(float));
    if (!emb->emptyVector) { free(emb->embeddings); emb->embeddings = 0; return 0; }
    memset(emb->emptyVector, 0, emb->D * sizeof(float));

    for (int keyID = 0; keyID < emb->Classes; keyID++)
    {
        emb->embeddings[keyID].vector = (float*) malloc(emb->D * sizeof(float));
        if (!emb->embeddings[keyID].vector) return 0;
    }
    return 1;
}

// Binary cache format:
//   uint32  magic (EMBEDDINGS_BIN_MAGIC)
//   int32   D
//   int32   Classes
//   float   offset
//   float   scaling
//   float[Classes * D]  pre-scaled embedding vectors (offset+scaling already applied)
static int _load_embeddings_from_binary(const char *bin_path, struct Embeddings *emb)
{
    FILE *bf = fopen(bin_path, "rb");
    if (!bf) return 0;

    unsigned int magic = 0;
    if (fread(&magic, sizeof(magic), 1, bf) != 1 || magic != EMBEDDINGS_BIN_MAGIC)
        { fclose(bf); return 0; }

    if (fread(&emb->D,       sizeof(int),   1, bf) != 1 ||
        fread(&emb->Classes, sizeof(int),   1, bf) != 1 ||
        fread(&emb->offset,  sizeof(float), 1, bf) != 1 ||
        fread(&emb->scaling, sizeof(float), 1, bf) != 1)
        { fclose(bf); return 0; }

    fprintf(stderr, "D = %u / Classes = %u / Offset %f / Scaling %f (binary cache)\n",
            emb->D, emb->Classes, emb->offset, emb->scaling);

    if (!_load_embeddings_alloc(emb)) { fclose(bf); return 0; }

    for (int keyID = 0; keyID < emb->Classes; keyID++)
    {
        // Vectors are stored already-scaled; load directly with no transform.
        if (fread(emb->embeddings[keyID].vector, sizeof(float), emb->D, bf) != (size_t)emb->D)
            { fclose(bf); return 0; }
    }

    fclose(bf);
    printf("%u embeddings loaded from binary cache\n", emb->Classes);
    return 1;
}

static void _save_embeddings_binary(const char *bin_path, const struct Embeddings *emb)
{
    FILE *bf = fopen(bin_path, "wb");
    if (!bf) { fprintf(stderr, "Warning: could not write embeddings binary cache %s\n", bin_path); return; }

    unsigned int magic = EMBEDDINGS_BIN_MAGIC;
    fwrite(&magic,       sizeof(magic),    1, bf);
    fwrite(&emb->D,      sizeof(int),      1, bf);
    fwrite(&emb->Classes,sizeof(int),      1, bf);
    fwrite(&emb->offset, sizeof(float),    1, bf);
    fwrite(&emb->scaling,sizeof(float),    1, bf);

    for (int keyID = 0; keyID < emb->Classes; keyID++)
        fwrite(emb->embeddings[keyID].vector, sizeof(float), emb->D, bf);  // already scaled

    fclose(bf);
    fprintf(stderr, "Embeddings binary cache written to %s\n", bin_path);
}

static int load_embeddings(const char *filename, struct Embeddings *emb)
{
    // Build sidecar path:  "foo.embeddings" -> "foo.embeddings.bin"
    char bin_path[512];
    snprintf(bin_path, sizeof(bin_path), "%s.bin", filename);

    if (_load_embeddings_from_binary(bin_path, emb))
        return 1;

    // Binary cache missing or stale — fall back to text parsing.
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Fatal Error: Could not open embeddings file %s\n", filename);
        return 0;
    }

    int res = fscanf(file, "%d\n", &emb->D);
    res = fscanf(file, "%d\n", &emb->Classes);
    res = fscanf(file, "%f\n", &emb->offset);
    res = fscanf(file, "%f\n", &emb->scaling);

    fprintf(stderr, "D = %u / Classes = %u / Offset %f / Scaling %f \n",
            emb->D, emb->Classes, emb->offset, emb->scaling);

    char boundary[MAX_KEY_LENGTH] = {0};
    res = fscanf(file, "%s\n", boundary);
    if (strcmp("start", boundary) != 0)
    {
        fprintf(stderr, "Corrupted embeddings file (%s), could not start reading it\n", filename);
        fclose(file);
        return 0;
    }

    if (!_load_embeddings_alloc(emb)) { fclose(file); return 0; }

    for (int keyID = 0; keyID < emb->Classes; keyID++)
    {
        char key[MAX_KEY_LENGTH] = {0};
        res = fscanf(file, "%s\n", key);

        float *vec = emb->embeddings[keyID].vector;
        for (int embeddingID = 0; embeddingID < emb->D; embeddingID++)
        {
            res = fscanf(file, "%f\n", &vec[embeddingID]);
            vec[embeddingID] = (vec[embeddingID] + emb->offset) * emb->scaling;
        }

        boundary[0] = 0;
        boundary[1] = 0;
        res = fscanf(file, "%s\n", boundary);
        if (strcmp("next_token", boundary) != 0)
        {
            fprintf(stderr, "Corrupted embeddings file (%s) @ key %s / ID %u", filename, key, keyID);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    printf("%u embeddings loaded successfully\n", emb->Classes);

    // Write binary cache for future runs.
    _save_embeddings_binary(bin_path, emb);

    return 1;
}

static void free_embeddings(struct Embeddings *emb)
{
    if (emb->emptyVector!=0)
    {
        free(emb->emptyVector);
        emb->emptyVector = 0;
    }

    if (emb->embeddings!=0)
    {
      for (int i = 0; i < emb->Classes; i++)
      {
        if (emb->embeddings[i].vector!=0)
        {
         free(emb->embeddings[i].vector);
         emb->embeddings[i].vector = 0;
        }
      }
      free(emb->embeddings);
      emb->embeddings = 0;
    }
}



static int testEmbeddings()
{
    struct Embeddings emb;
    load_embeddings("GloVe_D300.embeddings", &emb);

    // Example usage: access the first embedding vector
    printf("First vector value: %f\n", emb.embeddings[0].vector[0]);

    free_embeddings(&emb);
    return 0;
}



#ifdef __cplusplus
}
#endif

#endif // EMBEDDINGS_H_INCLUDED
