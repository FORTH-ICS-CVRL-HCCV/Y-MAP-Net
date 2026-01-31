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

static int load_embeddings(const char *filename,struct  Embeddings *emb)
{
    //float offset  = -0.1662235361903037;
    //float scaling =  0.27269877187819924;

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr,"Fatal Error: Could not open embeddings file %s\n", filename);
        return 0;
    }

    int res = fscanf(file, "%d\n", &emb->D);   // Read number of dimensions (D)
    res = fscanf(file, "%d\n", &emb->Classes);  // Read number of classes (keys)
    res = fscanf(file, "%f\n", &emb->offset);  // Read number of classes (keys)
    res = fscanf(file, "%f\n", &emb->scaling);  // Read number of classes (keys)

    fprintf(stderr,"D = %u / Classes = %u / Offset %f / Scaling %f \n",emb->D,emb->Classes,emb->offset,emb->scaling);

    char boundary[MAX_KEY_LENGTH]={0};
    res = fscanf(file, "%s\n", boundary);  // Read and skip 'next_token'
    if (strcmp("start",boundary)!=0)
        {
            fprintf(stderr,"Corrupted embeddings file (%s), could not start reading it\n",filename);
            return 0;
        }


    // Allocate memory for the embeddings
    emb->embeddings = (struct EmbeddingVector *) malloc(emb->Classes * sizeof(struct EmbeddingVector));
    if (emb->embeddings!=0)
    {
     memset(emb->embeddings,0,emb->Classes * sizeof(struct EmbeddingVector));

     emb->emptyVector = (float*) malloc(emb->D * sizeof(float));
     memset(emb->emptyVector,0,emb->D * sizeof(float));

     // Read the embeddings for each key
     for (int keyID = 0; keyID < emb->Classes; keyID++)
     {
        char key[MAX_KEY_LENGTH]={0};
        res = fscanf(file, "%s\n", key);  // Read the key (ignoring it because we use ID everywhere to simplify code)

        // Allocate memory for each embedding vector
        emb->embeddings[keyID].vector = (float *)malloc(emb->D * sizeof(float));

        if (emb->embeddings[keyID].vector!=0)
        {
         memset(emb->embeddings[keyID].vector, 0, emb->D * sizeof(float)); // Explicitly empty memory!

         for (int embeddingID=0; embeddingID<emb->D; embeddingID++)
          {
            res = fscanf(file, "%f\n", &emb->embeddings[keyID].vector[embeddingID]);  // Read embedding values

            //Perform scaling
            emb->embeddings[keyID].vector[embeddingID] += emb->offset;
            emb->embeddings[keyID].vector[embeddingID] *= emb->scaling;
          }
        } else
        {
            fprintf(stderr,"Failed allocating memory for embedding key %s / ID %u",key,keyID);
            return 0;
        }

        boundary[0]=0; //Taint previous boundary
        boundary[1]=0;
        res = fscanf(file, "%s\n", boundary);  // Read and skip 'next_token'
        if (strcmp("next_token",boundary)!=0)
        {
            fprintf(stderr,"Corrupted embeddings file (%s) @ key %s / ID %u",filename,key,keyID);
            return 0;
        }
     }
    }

    fclose(file);
    printf("%u embeddings loaded successfully\n",emb->Classes);
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
