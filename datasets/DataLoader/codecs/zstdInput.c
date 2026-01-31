#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zstd.h>
//sudo apt install libzstd-dev

#define CHUNK_SIZE 16384

#define PRINT_COMMENTS 1
#define PPMREADBUFLEN 256

const int headerSize = sizeof(unsigned int) * 4; //width,height,bitsperpixel,channels


unsigned int simplePowPPM(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}

int swapDepthEndianness(unsigned char * pixels,unsigned int width,unsigned int height,unsigned int bitsperpixel, unsigned int channels)
{
    if (pixels==0)
    {
        return 0;
    }
    if (bitsperpixel!=16)
    {
        fprintf(stderr,"Only 16bit PNM files need swapping ( we have a %u bit x %u channels file )..\n",bitsperpixel, channels);
        return 0;
    }

    unsigned char * traverser=(unsigned char * ) pixels;
    unsigned char * traverserSwap1;//=(unsigned char * ) pixels;
    unsigned char * traverserSwap2;//=(unsigned char * ) pixels;

    unsigned int bytesperpixel = (bitsperpixel/8);
    unsigned char * endOfMem = traverser + width * height * channels * bytesperpixel;

    while ( ( traverser < endOfMem)  )
    {
        traverserSwap1 = traverser;
        traverserSwap2 = traverser+1;

        unsigned char tmp = *traverserSwap1;
        *traverserSwap1 = *traverserSwap2;
        *traverserSwap2 = tmp;

        traverser += bytesperpixel;
    }

    return 1;
}

unsigned char * ReadPNM(unsigned char * buffer,const char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp, unsigned int * bytesPerPixel, unsigned int * channels)
{
    * bytesPerPixel = 0;
    * channels = 0;

    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0;
        *height=0;
        *timestamp=0;

        char buf[PPMREADBUFLEN]= {0};
        char *t;
        unsigned int w=0, h=0, d=0;
        int r=0, z=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0)
        {
            fclose(pf);
            return buffer;
        }

        if ( strncmp(buf,"P6\n", 3) == 0 )
        {
            *channels=3;
        }
        else if ( strncmp(buf,"P5\n", 3) == 0 )
        {
            *channels=1;
        }
        else
        {
            fprintf(stderr,"Could not understand/Not supported file format\n");
            fclose(pf);
            return buffer;
        }
        do
        {
            /* Px formats can have # comments after first line */
#if PRINT_COMMENTS
            memset(buf,0,PPMREADBUFLEN);
#endif
            t = fgets(buf, PPMREADBUFLEN, pf);
            if (strstr(buf,"TIMESTAMP")!=0)
            {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
            }

            if ( t == 0 )
            {
                fclose(pf);
                return buffer;
            }
        }
        while ( strncmp(buf, "#", 1) == 0 );
        z = sscanf(buf, "%u %u", &w, &h);
        if ( z < 2 )
        {
            fclose(pf);
            fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h);
            return buffer;
        }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if (r < 1)
        {
            fprintf(stderr,"Could not understand how many bytesPerPixel there are on this image\n");
            fclose(pf);
            return buffer;
        }
        if (d==255)
        {
            *bytesPerPixel=1;
        }
        else if (d==65535)
        {
            *bytesPerPixel=2;
        }
        else
        {
            fprintf(stderr,"Incoherent payload received %u bits per pixel \n",d);
            fclose(pf);
            return buffer;
        }


        //This is a super ninja hackish patch that fixes the case where fscanf eats one character more on the stream
        //It could be done better  ( no need to fseek ) but this will have to do for now
        //Scan for border case
        unsigned long startOfBinaryPart = ftell(pf);
        if ( fseek (pf, 0, SEEK_END)!=0 )
        {
            fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n");
            fclose(pf);
            return 0;
        }
        unsigned long totalFileSize = ftell (pf); //lSize now holds the size of the file..

        //fprintf(stderr,"totalFileSize-startOfBinaryPart = %u \n",totalFileSize-startOfBinaryPart);
        //fprintf(stderr,"bytesPerPixel*channels*w*h = %u \n",bytesPerPixel*channels*w*h);
        if (totalFileSize-startOfBinaryPart < *bytesPerPixel*(*channels)*w*h )
        {
            fprintf(stderr," Detected Border Case\n\n\n");
            startOfBinaryPart-=1;
        }
        if ( fseek (pf, startOfBinaryPart, SEEK_SET)!=0 )
        {
            fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n");
            fclose(pf);
            return 0;
        }
        //-----------------------
        //----------------------

        *width=w;
        *height=h;
        if (pixels==0)
        {
            pixels= (unsigned char*) malloc(w*h*(*bytesPerPixel)*(*channels)*sizeof(char));
        }

        if ( pixels != 0 )
        {
            size_t rd = fread(pixels,*bytesPerPixel*(*channels), w*h, pf);
            if (rd < w*h )
            {
                fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd, w*h);
                fprintf(stderr,"Dimensions ( %u x %u ) , Depth %u bytes , Channels %u \n",w,h,*bytesPerPixel,*channels);
            }

            fclose(pf);

#if PRINT_COMMENTS
            if ( (*channels==1) && (*bytesPerPixel==2) && (timestamp!=0) )
            {
                printf("DEPTH %lu\n",*timestamp);
            }
            else if ( (*channels==3) && (*bytesPerPixel==1) && (timestamp!=0) )
            {
                printf("COLOR %lu\n",*timestamp);
            }
#endif

            return pixels;
        }
        else
        {
            fprintf(stderr,"Could not Allocate enough memory for file %s \n",filename);
        }
        fclose(pf);
    }
    else
    {
        fprintf(stderr,"File %s does not exist \n",filename);
    }
    return buffer;
}


int WritePNM(const char * filename, unsigned char * pixels, unsigned int width, unsigned int height, unsigned int bitsperpixel, unsigned int channels)
{
    if ((width == 0) || (height == 0) || (channels == 0) || (bitsperpixel == 0))
    {
        fprintf(stderr, "saveRawImageToFile(%s) called with zero dimensions ( %ux%u %u channels %u bpp\n", filename, width, height, channels, bitsperpixel);
        return 0;
    }
    if (pixels == 0)
    {
        fprintf(stderr, "saveRawImageToFile(%s) called for an unallocated (empty) frame, will not write any file output\n", filename);
        return 0;
    }
    if (bitsperpixel / channels > 16)
    {
        fprintf(stderr, "PNM does not support more than 2 bytes per pixel..!\n");
        return 0;
    }

    FILE *fd = fopen(filename, "wb");
    if (fd != 0)
    {
        if (channels == 3)
        {
            fprintf(fd, "P6\n");
        }
        else if (channels == 1)
        {
            fprintf(fd, "P5\n");
        }
        else
        {
            fprintf(stderr, "Invalid channels arg (%u) for SaveRawImageToFile\n", channels);
            fclose(fd);
            return 1;
        }

        unsigned int bitsperchannelpixel = bitsperpixel / channels;
        fprintf(fd, "%u %u\n%u\n", width, height, simplePowPPM(2,bitsperchannelpixel) - 1);

        unsigned int n = width * height * channels * (bitsperchannelpixel / 8);

        fwrite(pixels, 1, n, fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr, "SaveRawImageToFile could not open output file %s\n", filename);
        return 0;
    }
    return 0;
}

int swap_endianness(int value)
{
    return ((value >> 24) & 0xFF) | ((value >> 8) & 0xFF00) | ((value << 8) & 0xFF0000) | ((value << 24) & 0xFF000000);
}

void split_channels_and_filter(const unsigned char *image, unsigned char **buffers, int num_buffers, int WIDTH, int HEIGHT)
{
    int total_size = WIDTH * HEIGHT;

    // Split channels
    for (int i = 0; i < total_size; i++)
    {
        for (int ch = 0; ch < num_buffers; ch++)
        {
            buffers[ch][i] = image[i * num_buffers + ch];
        }
    }

    // Apply left-pixel delta filtering
    for (int i = total_size - 1; i > 0; i--)
    {
        for (int ch = 0; ch < num_buffers; ch++)
        {
            buffers[ch][i] -= buffers[ch][i - 1];
        }
    }
}

void restore_channels(unsigned char **buffers, int num_buffers, int WIDTH, int HEIGHT)
{
    int total_size = WIDTH * HEIGHT;
    for (int i = 1; i < total_size; i++)
    {
        for (int ch = 0; ch < num_buffers; ch++)
        {
            buffers[ch][i] += buffers[ch][i - 1];
        }
    }
}
void compress_combined(unsigned char **buffers, unsigned int width,unsigned int height, unsigned int bitsperpixel, unsigned int channels, const char *output_filename)
{
    FILE *output = fopen(output_filename, "wb");
    if (!output)
    {
        perror("File error");
        exit(EXIT_FAILURE);
    }


    unsigned int combined_buffer_size = (width * height * (bitsperpixel/8)* channels) + headerSize;

    unsigned int size = combined_buffer_size; //width * height;
    fwrite(&size, sizeof(unsigned int), 1, output); // Store size for decompression


    size_t max_compressed_size = ZSTD_compressBound(combined_buffer_size);
    void *compressed_buffer = malloc(max_compressed_size);
    if (!compressed_buffer)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    unsigned char *combined_buffer_raw = (unsigned char *) malloc(combined_buffer_size);
    if (!combined_buffer_raw)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Store header information
    unsigned int *bitsperpixelTarget = (unsigned int*) combined_buffer_raw;
    unsigned int *channelsTarget     = bitsperpixelTarget + 1; // Move by 1, not sizeof(unsigned int)
    unsigned int *widthTarget        = bitsperpixelTarget + 2; // Move by 1, not sizeof(unsigned int)
    unsigned int *heightTarget       = bitsperpixelTarget + 3; // Move by 1, not sizeof(unsigned int)

    *bitsperpixelTarget = bitsperpixel;
    *channelsTarget     = channels;
    *widthTarget        = width;
    *heightTarget       = height;

    fprintf(stderr, "Storing %ux%u / %u bitsperpixel / %u channels \n",width,height, bitsperpixel, channels);

    // Store separate image planes
    unsigned char *combined_buffer = combined_buffer_raw + headerSize;
    for (int i = 0; i < width*height; i++)
    {
        for (unsigned int ch = 0; ch < channels; ch++)
        {
            combined_buffer[i * channels + ch] = buffers[ch][i];
        }
    }

    size_t compressed_size = ZSTD_compress(compressed_buffer, max_compressed_size, combined_buffer_raw, combined_buffer_size, 1);
    if (ZSTD_isError(compressed_size))
    {
        fprintf(stderr, "Zstd compression error: %s\n", ZSTD_getErrorName(compressed_size));
        exit(EXIT_FAILURE);
    }

    fwrite(compressed_buffer, 1, compressed_size, output);

    free(compressed_buffer);
    free(combined_buffer_raw);
    fclose(output);
}

void decompress_combined(const char *input_filename, unsigned char ***buffers,unsigned int *size,unsigned int *width,unsigned int *height,unsigned int *bitsperpixel, unsigned int *channels)
{
    FILE *input = fopen(input_filename, "rb");
    if (!input)
    {
        perror("File error");
        exit(EXIT_FAILURE);
    }

    // Read stored size
    fread(size, sizeof(unsigned int), 1, input);

    printf("Debug: Read size (raw bytes) = %02X %02X %02X %02X\n",
           ((unsigned char*)size)[0], ((unsigned char*)size)[1],
           ((unsigned char*)size)[2], ((unsigned char*)size)[3]);
    if (*size <= 0 || *size > 100000000)   // Sanity check
    {
        fprintf(stderr, "Error: Invalid size read from file (%d)\n", *size);
        exit(EXIT_FAILURE);
    }
    printf("Read size: %d\n", *size);

    // Read compressed data
    fseek(input, 0, SEEK_END);
    size_t compressed_size = ftell(input) - sizeof(int);
    fseek(input, sizeof(int), SEEK_SET);

    void *compressed_buffer = malloc(compressed_size);
    if (!compressed_buffer)
    {
        perror("Memory allocation #1 failed");
        exit(EXIT_FAILURE);
    }

    fread(compressed_buffer, 1, compressed_size, input);
    fclose(input);

    size_t decompressed_size = (*size);// * (*channels) + headerSize;
    void *decompressed_buffer = malloc(decompressed_size);
    if (!decompressed_buffer)
    {
        perror("Memory allocation #2 failed");
        exit(EXIT_FAILURE);
    }

    size_t actual_decompressed_size = ZSTD_decompress(decompressed_buffer, decompressed_size, compressed_buffer, compressed_size);
    if (ZSTD_isError(actual_decompressed_size))
    {
        fprintf(stderr, "Zstd decompression error: %s\n", ZSTD_getErrorName(actual_decompressed_size));
        exit(EXIT_FAILURE);
    }

    free(compressed_buffer);

    // Read header information
    unsigned int *bitsperpixelSource = (unsigned int *) decompressed_buffer;
    unsigned int *channelsSource     = bitsperpixelSource + 1; // Move by 1, not sizeof(unsigned int)
    unsigned int *widthSource        = bitsperpixelSource + 2; // Move by 1, not sizeof(unsigned int)
    unsigned int *heightSource       = bitsperpixelSource + 3; // Move by 1, not sizeof(unsigned int)

    *bitsperpixel = *bitsperpixelSource;
    *channels     = *channelsSource;
    *width        = *widthSource;
    *height       = *heightSource;

    fprintf(stderr, "Detected %ux%u / %u bitsperpixel / %u channels \n", *width, *height, *bitsperpixel, *channels);

    *bitsperpixel *= *channels; //This is needed because of what writePNM expects..

    // Allocate memory for all channels
    *buffers = (unsigned char **)malloc(*channels * sizeof(unsigned char *));
    if (!*buffers)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (unsigned int ch = 0; ch < *channels; ch++)
    {
        (*buffers)[ch] = (unsigned char *)malloc(*size);
        if (!(*buffers)[ch])
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
    }

    // Copy decompressed data into the channel buffers
    unsigned char *decompressed_bytes = (unsigned char *) decompressed_buffer + headerSize;
    for (int i = 0; i < *size; i++)
    {
        for (unsigned int ch = 0; ch < *channels; ch++)
        {
            (*buffers)[ch][i] = decompressed_bytes[i * (*channels) + ch];
        }
    }

    free(decompressed_buffer);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <compress|decompress> <input_file> <output_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char * operation                    = argv[1];
    const char * input_commandline_parameter  = argv[2];
    const char * output_commandline_parameter = argv[3];

    if (strcmp(operation, "compress") == 0)
    {
        fprintf(stderr, "Compress %s \n", input_commandline_parameter);

        unsigned char *image = NULL;
        unsigned int width = 0, height = 0;
        unsigned long timestamp = 0;
        unsigned int bytesPerPixel = 0, channels = 0;

        image = ReadPNM(0, input_commandline_parameter, &width, &height, &timestamp, &bytesPerPixel, &channels);
        unsigned int bitsperpixel = bytesPerPixel * 8;

        fprintf(stderr, "Width %u / Height %u \n", width, height);
        fprintf(stderr, "bitsperpixel %u / channels %u \n", bitsperpixel, channels);

        unsigned char **buffers = malloc(channels * sizeof(unsigned char *));
        for (unsigned int ch = 0; ch < channels; ch++)
        {
            buffers[ch] = malloc(width * height * sizeof(unsigned char));
        }

        split_channels_and_filter(image, buffers, channels, width, height);
        compress_combined(buffers, width,height, bitsperpixel, channels, output_commandline_parameter);

        free(image);
        for (unsigned int ch = 0; ch < channels; ch++)
        {
            free(buffers[ch]);
        }
        free(buffers);
    }
    else if (strcmp(operation, "decompress") == 0)
    {
        fprintf(stderr, "Decompress %s \n", input_commandline_parameter);

        unsigned char **buffers = NULL;
        unsigned int size = 0;
        unsigned int width = 0, height = 0;
        unsigned int bitsperpixel = 24, channels = 3;

        decompress_combined(input_commandline_parameter, &buffers, &size, &width, &height, &bitsperpixel, &channels);


        restore_channels(buffers, channels, width, height);

        unsigned char *reconstructed = malloc( width * height * (bitsperpixel/8)* channels );
        for (size_t i = 0; i < width * height * (bitsperpixel/8); i++)
        {
            for (unsigned int ch = 0; ch < channels; ch++)
            {
                reconstructed[i * channels + ch] = buffers[ch][i];
            }
        }

        WritePNM(output_commandline_parameter, reconstructed, width, height, bitsperpixel, channels);

        for (unsigned int ch = 0; ch < channels; ch++)
        {
            free(buffers[ch]);
        }
        free(buffers);
        free(reconstructed);
    }
    else
    {
        fprintf(stderr, "Invalid mode: %s. Use 'compress' or 'decompress'.\n", operation);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
