#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include <time.h>
#include <math.h>

//To create an input image use :
//   convert -channel 0 -separate ../../sample00002_C8bit_hm29.png depth.pgm
//   convert ../../sample00002_C8bit_hm29.png depth.ppm

//To compile use :
//   gcc test_sobel_avx2.c -O3 -march=native -mtune=native -lm -o test_sobel_avx2  &&  ./test_sobel_avx2

//To debug use:
//   valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes  ./test_sobel_avx2

//To profile use:
//   valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./test_sobel_avx2 && kcachegrind && rm callgrind.out.*

#define PPMREADBUFLEN 256

float getRandomFloat(float minVal, float maxVal)
{
    // Generate a random float in the range [0, 1]
    float random_float = (float) rand() / RAND_MAX;

    // Scale and shift the random float to fit within the specified range
    float result = random_float * (maxVal - minVal) + minVal;

    return result;
}



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


unsigned char * ReadPNM(unsigned char * buffer ,const char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp , unsigned int * bytesPerPixel , unsigned int * channels)
{
   * bytesPerPixel = 0;
   * channels = 0;

    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0; *height=0; *timestamp=0;

        char buf[PPMREADBUFLEN]={0};
        char *t;
        unsigned int w=0, h=0, d=0;
        int r=0 , z=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0) { fclose(pf); return buffer; }

        if ( strncmp(buf,"P6\n", 3) == 0 ) { *channels=3; } else
        if ( strncmp(buf,"P5\n", 3) == 0 ) { *channels=1; } else
                                           { fprintf(stderr,"Could not understand/Not supported file format\n"); fclose(pf); return buffer; }
        do
        { /* Px formats can have # comments after first line */
           #if PRINT_COMMENTS
             memset(buf,0,PPMREADBUFLEN);
           #endif
           t = fgets(buf, PPMREADBUFLEN, pf);
           if (strstr(buf,"TIMESTAMP")!=0)
              {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
              }

           if ( t == 0 ) { fclose(pf); return buffer; }
        } while ( strncmp(buf, "#", 1) == 0 );
        z = sscanf(buf, "%u %u", &w, &h);
        if ( z < 2 ) { fclose(pf); fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h); return buffer; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if (r < 1) { fprintf(stderr,"Could not understand how many bytesPerPixel there are on this image\n"); fclose(pf); return buffer; }
        if (d==255) { *bytesPerPixel=1; }  else
        if (d==65535) { *bytesPerPixel=2; } else
                       { fprintf(stderr,"Incoherent payload received %u bits per pixel \n",d); fclose(pf); return buffer; }


        //This is a super ninja hackish patch that fixes the case where fscanf eats one character more on the stream
        //It could be done better  ( no need to fseek ) but this will have to do for now
        //Scan for border case
           unsigned long startOfBinaryPart = ftell(pf);
           if ( fseek (pf , 0 , SEEK_END)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
           unsigned long totalFileSize = ftell (pf); //lSize now holds the size of the file..

           //fprintf(stderr,"totalFileSize-startOfBinaryPart = %u \n",totalFileSize-startOfBinaryPart);
           //fprintf(stderr,"bytesPerPixel*channels*w*h = %u \n",bytesPerPixel*channels*w*h);
           if (totalFileSize-startOfBinaryPart < *bytesPerPixel*(*channels)*w*h )
           {
              fprintf(stderr," Detected Border Case\n\n\n");
              startOfBinaryPart-=1;
           }
           if ( fseek (pf , startOfBinaryPart , SEEK_SET)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
         //-----------------------
         //----------------------

        *width=w; *height=h;
        if (pixels==0) {  pixels= (unsigned char*) malloc(w*h*(*bytesPerPixel)*(*channels)*sizeof(char)); }

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
             if ( (*channels==1) && (*bytesPerPixel==2) && (timestamp!=0) ) { printf("DEPTH %lu\n",*timestamp); } else
             if ( (*channels==3) && (*bytesPerPixel==1) && (timestamp!=0) ) { printf("COLOR %lu\n",*timestamp); }
           #endif

          return pixels;
        } else
        {
            fprintf(stderr,"Could not Allocate enough memory for file %s \n",filename);
        }
        fclose(pf);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }
  return buffer;
}


// Function to write PPM (Portable PixMap) files
void write_ppm(const char *filename, uint8_t *image, int width, int height, int channels) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Write image data
    fwrite(image, 1, width * height * channels, fp);

    fclose(fp);
}



static void sobel8Bit2Way_ignore_center_AVX2(uint8_t *pixelsInput, int width, int height, int channels,
                          float *gradientX, int sobelKernelX[3][3],
                          float *gradientY, int sobelKernelY[3][3], int channel)
{
  if ( (sobelKernelX[1][1]!=0.0)  || (sobelKernelY[1][1]!=0.0) )
   {
       fprintf(stderr,"sobel8Bit2Way_ignore_center_AVX2 will not work with non null center kernel 3x3 element\n");
       fprintf(stderr,"AVX2 has 8 registers not 9\n");
       abort();
   }


   __m256 kX = _mm256_set_ps( sobelKernelX[0][0], sobelKernelX[0][1], sobelKernelX[0][2],
                              sobelKernelX[1][0],      /*empty*/      sobelKernelX[1][2],
                              sobelKernelX[2][0], sobelKernelX[2][1], sobelKernelX[2][2]);

   __m256 kY = _mm256_set_ps( sobelKernelY[0][0], sobelKernelY[0][1], sobelKernelY[0][2],
                              sobelKernelY[1][0],      /*empty*/      sobelKernelY[1][2],
                              sobelKernelY[2][0], sobelKernelY[2][1], sobelKernelY[2][2]);


   int gradientTarget;
   int indexStart,nextLineIndexStart,nextnextLineIndexStart;
   int start_x = 0;
   float gradX = 0.0f, gradY = 0.0f;

   for (int y = 1; y < height - 1; ++y)
   {
    indexStart = (y * width + start_x) * channels + channel;
    for (int x = start_x; x < width - 2; x += 1)
    { // Ensure valid blocks
        nextLineIndexStart     = indexStart + (width*channels);
        nextnextLineIndexStart = nextLineIndexStart + (width*channels);

        // Load pixel values into AVX2 register
        __m256 pixels = _mm256_set_ps((float) pixelsInput[indexStart + 0],              (float) pixelsInput[indexStart + channels],        (float) pixelsInput[indexStart + (2*channels)],
                                      (float) pixelsInput[nextLineIndexStart + 0],      /* empty */                                      (float) pixelsInput[nextLineIndexStart + (2*channels)],
                                      (float) pixelsInput[nextnextLineIndexStart + 0], (float)  pixelsInput[nextnextLineIndexStart + channels],  (float) pixelsInput[nextnextLineIndexStart + (2*channels)]);

        // Multiply pixels with kernels
        __m256 productX = _mm256_mul_ps(kX, pixels);
        __m256 productY = _mm256_mul_ps(kY, pixels);

        // Sum all elements in the AVX2 registers
        //float gradX = 0.0f, gradY = 0.0f;
        gradX = productX[0] + productX[1] + productX[2] + productX[3] + productX[4] + productX[5] + productX[6] + productX[7];
        gradY = productY[0] + productY[1] + productY[2] + productY[3] + productY[4] + productY[5] + productY[6] + productY[7];

        //Maybe also do sum using AVX ?
        //__m256 sumX =  _mm256_setzero_ps();
        //__m256 sumY =  _mm256_setzero_ps();

        // Store the results
        gradientTarget = (y+1) * width + (x+1) + channel;
        gradientX[gradientTarget] = gradX;
        gradientY[gradientTarget] = gradY;

        indexStart+=channels;
     }
   }

}




static void sobel8Bit2Way_AVX2(uint8_t *pixelsInput, int width, int height, int channels,
                          float *gradientX, int sobelKernelX[3][3],
                          float *gradientY, int sobelKernelY[3][3], int channel)
{

for (int y = 1; y < height - 1; ++y)
{
    for (int x = 1; x < width - 1; x += 1)
    { // Ensure valid blocks
        __m256i sumX = _mm256_setzero_si256();
        __m256i sumY = _mm256_setzero_si256();

        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int pixelOffset = ((y + ky) * width + (x + kx)) * channels + channel;

                // Load and convert pixels
                __m128i pixels8 = _mm_loadu_si128((__m128i *)&pixelsInput[pixelOffset]);
                __m256i pixels = _mm256_cvtepu8_epi32(pixels8);

                // Apply Sobel kernels
                int kX = sobelKernelX[ky + 1][kx + 1];
                int kY = sobelKernelY[ky + 1][kx + 1];
                __m256i kernelX = _mm256_set1_epi32(kX);
                __m256i kernelY = _mm256_set1_epi32(kY);

                // Accumulate results
                sumX = _mm256_add_epi32(sumX, _mm256_mullo_epi32(pixels, kernelX));
                sumY = _mm256_add_epi32(sumY, _mm256_mullo_epi32(pixels, kernelY));
            }
        }

        // Store results
        __m256 gradXF = _mm256_cvtepi32_ps(sumX);
        __m256 gradYF = _mm256_cvtepi32_ps(sumY);
        _mm256_storeu_ps(&gradientX[y * width + x], gradXF);
        _mm256_storeu_ps(&gradientY[y * width + x], gradYF);
    }
}


}




static void sobel8Bit2Way_Simple(uint8_t *pixelsInput, int width, int height,int channels,
                          float *gradientX, int sobelKernelX[3][3],
                          float *gradientY, int sobelKernelY[3][3], int channel)
{
    unsigned char * pixels = (unsigned char* ) pixelsInput;

    int sobX,sobY,pixelValue,yIndex;

    //#pragma omp parallel for
    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            int sumX = 0.0, sumY = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                yIndex = (y + ky) * width;

                for (int kx = -1; kx <= 1; ++kx)
                {
                    sobX = sobelKernelX[ky + 1][kx + 1];
                    sobY = sobelKernelY[ky + 1][kx + 1];

                    pixelValue = pixels[(yIndex + (x+kx)) * channels + channel];

                    sumX += (pixelValue * sobX);
                    sumY += (pixelValue * sobY);
                }
            }
            gradientX[y * width + x] = (float) sumX;
            gradientY[y * width + x] = (float) sumY;
        }
    }
}

static void sobelXY8BitLoop(uint8_t *pixelsInput, int width, int height,int channels, float *gradientX, float *gradientY,int channel)
{
    int sobelXKernel[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobelYKernel[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };


     sobel8Bit2Way_ignore_center_AVX2(pixelsInput, width, height, channels, gradientX, sobelXKernel, gradientY, sobelYKernel, channel);
     sobel8Bit2Way_AVX2(pixelsInput, width, height, channels, gradientX, sobelXKernel, gradientY, sobelYKernel, channel);
     sobel8Bit2Way_Simple(pixelsInput, width, height, channels, gradientX, sobelXKernel, gradientY, sobelYKernel, channel);
}



static void computeNormals8Bit(uint8_t *pixelsInput, int width, int height,int channels)
{
    unsigned char * pixels = (unsigned char*) pixelsInput;

    float *gradientX  = (float *) malloc(width * height * sizeof(float));
    if (gradientX!=0)
    {
     float *gradientY = (float *) malloc(width * height * sizeof(float));
     if (gradientY!=0)
     {

      //Perform it many times to profile
      for (int loop=0; loop<10; loop++)
      {
        sobelXY8BitLoop(pixelsInput,width,height,channels, gradientX, gradientY, 0);
      }

      float epsilon = 1e-8;

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
            int idx  = y * width + x;
            float dx = gradientX[idx];
            float dy = gradientY[idx];
            float dz = 1.0;

            float norm = sqrt(dx * dx + dy * dy + dz * dz);

            float nx = (float) dx / (norm + epsilon);
            float ny = (float) dy / (norm + epsilon);
            float nz = (float) dz / (norm + epsilon);

            pixels[(idx * 3) + 0] = (unsigned char) (((nx + 1.0) / 2.0) * 255);
            pixels[(idx * 3) + 1] = (unsigned char) (((ny + 1.0) / 2.0) * 255);
            pixels[(idx * 3) + 2] = (unsigned char) (((nz + 1.0) / 2.0) * 255);
        }
     }

     free(gradientY);
     }
    free(gradientX);
    }
}




int main(int argc, char *argv[])
{
    // Load a PPM image
    const char *input_file = "depth.ppm";
    unsigned int width, height, channels;

    unsigned long timestamp;
    unsigned int bytesPerPixel;

    uint8_t *image = (uint8_t *) ReadPNM(0,input_file, &width, &height, &timestamp , &bytesPerPixel , &channels);
    printf("Loaded image: %dx%dx%d\n", width, height, channels);

    computeNormals8Bit(image,width,height,channels);


    // Save the resized image as PPM
    const char *output_file = "normals.ppm";
    write_ppm(output_file, image,width,height,channels);

    printf("Resized image saved as %s.\n", output_file);

    // Free allocated memory
    free(image);

    return 0;
}

