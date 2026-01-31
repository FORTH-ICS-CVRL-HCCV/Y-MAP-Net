#ifndef LABELS_H_INCLUDED
#define LABELS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define MAX_STRINGS 17862      // Assuming there are at most 2048 strings
#define MAX_STRING_LENGTH 256  // Maximum length of each string

// Struct to hold the array of strings
struct StringArray
{
    char **strings;  // Dynamically allocated array of string pointers
    int count;  // Number of strings
    int max_count;  // Number of strings
};

// Function to count the number of strings in the file
static int countStringsInFile(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return -1;  // Return -1 to indicate an error
    }

    char buffer[MAX_STRING_LENGTH * 2];  // Buffer to read lines
    int count = 0;

    while (fgets(buffer, sizeof(buffer), file))
    {
        // Look for the numeric key part and then the quoted string
        char *start = strchr(buffer, ':');  // Find the colon
        if (start)
        {
            start = strchr(start, '"');  // Find the first quote after the colon
            if (start)
            {
                start++;  // Move past the first quote
                char *end = strchr(start, '"');  // Find the second quote
                if (end)
                {
                    count++;  // Increment count for each valid string
                }
            }
        }
    }

    fclose(file);
    return count;  // Return the count of strings
}

// Function to read file and populate the struct
static void populateStructFromFile(const char *filename, struct StringArray *array)
{
  if (array==0) { fprintf(stderr,"populateStructFromFile(%s) without array\n",filename); return; }

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr,"Error opening vocabulary file: %s\n",filename);
        exit(EXIT_FAILURE);
    }

    char buffer[MAX_STRING_LENGTH * 2];  // Buffer to read lines
    int index = 0;

    while (fgets(buffer, sizeof(buffer), file) && index < array->max_count)
    {
        // Skip the numeric key part, look for the second quote after the colon
        char *start = strchr(buffer, ':');  // Find the colon
        if (start)
        {
            start = strchr(start, '"');  // Find the first quote after the colon
            if (start)
            {
                start++;  // Move past the first quote
                char *end = strchr(start, '"');  // Find the second quote
                if (end)
                {
                    *end = '\0';  // Null-terminate the string
                    array->strings[index] = malloc(strlen(start) + 1);
                    strcpy(array->strings[index], start);
                    index++;
                }
            }
        }
    }

    array->count = index;  // Set the count of strings
    fclose(file);
}

static struct StringArray * loadStringArrayFromFile(const char *filename)
{
   struct StringArray * sa = 0;
   int countOfStrings = countStringsInFile(filename);
   if ( countOfStrings!= -1 )
   {
      sa = (struct StringArray *) malloc(sizeof(struct StringArray));
      if (sa!=0)
      {
        sa->count     = 0;
        sa->max_count = countOfStrings+1;
        sa->strings   = (char **) malloc(sa->max_count * sizeof(char *));  // Dynamically allocate strings array

        if (sa->strings != NULL)
            {
                for (int i = 0; i < countOfStrings; i++)  // Initialize to NULL
                {
                    sa->strings[i] = NULL;
                }

                populateStructFromFile(filename, sa);
            }
            else
            {
                free(sa);
                sa = NULL;  // If allocation fails, return NULL
            }

      }
   }

   return sa;
}

static void freeStringArray(struct StringArray *array)
{
    if (array == NULL)
    {
        return;  // Nothing to free if the pointer is NULL
    }

    // Free each string in the array
    for (int i = 0; i < array->count; i++)
    {
        if (array->strings[i] != NULL)
        {
            free(array->strings[i]);
            array->strings[i] = NULL;  // Avoid dangling pointer
        }
    }

    // Free the strings array
    free(array->strings);
    array->strings = NULL;  // Avoid dangling pointer

    // Free the struct itself
    free(array);
}

// Function to print the array of strings
static void printStringArray(struct StringArray *array)
{
  if (array!=0)
  {
    for (int i = 0; i < array->count; i++)
    {
        printf("%d: %s\n", i, array->strings[i]);
    }
  }
}

static int testLabels()
{
    struct StringArray array = { .count = 0 };

    // Populate the struct from the file
    populateStructFromFile("../descriptions/vocabulary.json", &array);

    // Print the populated strings
    printStringArray(&array);

    // Free the allocated memory
    freeStringArray(&array);

    return 0;
}

#ifdef __cplusplus
}
#endif

#endif // DRAW_H_INCLUDED
