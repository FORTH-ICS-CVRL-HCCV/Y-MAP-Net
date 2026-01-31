#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include <time.h>

//To create an input use convert path/to/an/animage.jpg input.ppm
//To compile use :
//   gcc test_brightness_contrast_avx2.c -O3 -march=native -mtune=native -o test_brightness_contrast_avx2  &&  ./test_brightness_contrast_avx2

#define CHANNELS 3

float getRandomFloat(float minVal, float maxVal)
{
    // Generate a random float in the range [0, 1]
    float random_float = (float) rand() / RAND_MAX;

    // Scale and shift the random float to fit within the specified range
    float result = random_float * (maxVal - minVal) + minVal;

    return result;
}


// Helper to load PPM image
uint8_t *load_ppm(const char *filename, int *width, int *height, int *channels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        fprintf(stderr,"File : %s\n",filename);
        exit(EXIT_FAILURE);
    }

    char format[3];
    fscanf(fp, "%2s", format);
    if (strcmp(format, "P6") != 0) {
        fprintf(stderr, "Unsupported PPM format: %s\n", format);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Read image dimensions and max color value
    fscanf(fp, "%d %d", width, height);
    int max_val;
    fscanf(fp, "%d", &max_val);
    if (max_val != 255) {
        fprintf(stderr, "Unsupported max value: %d\n", max_val);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Consume the newline character after max_val
    fgetc(fp);

    *channels = CHANNELS;

    // Allocate aligned memory and read the image data
    int size = (*width) * (*height) * (*channels);
    uint8_t *image = (uint8_t *)_mm_malloc(size, 32);
    if (!image) {
        perror("Error allocating memory");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    fread(image, 1, size, fp);

    fclose(fp);
    return image;
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


// Resize function using AVX2 (nearest-neighbor)
void resize_image_avx2(uint8_t *src, int src_width, int src_height,
                       uint8_t *dst, int dst_width, int dst_height, int channels)
{
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++)
    {
        int src_y = (int)(y * y_ratio);

        for (int x = 0; x < dst_width; x += 8) { // Process 8 pixels at a time
            int max_process = (x + 8 > dst_width) ? (dst_width - x) : 8;

            uint8_t temp[32] = {0};  // Temporary buffer for unaligned AVX2 storage

            for (int i = 0; i < max_process; i++) {
                int src_x = (int)((x + i) * x_ratio);
                for (int c = 0; c < channels; c++) {
                    temp[i * channels + c] = src[(src_y * src_width + src_x) * channels + c];
                }
            }

            // Load into AVX2 register
            __m256i pixels = _mm256_loadu_si256((__m256i *)temp);

            // Store resized pixels
            _mm256_storeu_si256((__m256i *)&dst[(y * dst_width + x) * channels], pixels);
        }
    }
}


void adjustBrightnessContrast_Simple(uint8_t *pixelsInput, int width, int height,int channels,
                                     float brightnessR, float contrastR,
                                     float brightnessG, float contrastG,
                                     float brightnessB, float contrastB,
                                     float borderX, float borderY
                                     )
{
    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = width  - start_x;
    signed int end_y = height - start_y;

    if (start_x<0)    { start_x = 0;   }
    if (start_y<0)    { start_y = 0;   }

    if (end_x>width)  { end_x = width; }
    if (end_y>height) { end_y = height;}


    float pixel_value;

    // Adjust brightness
    for (unsigned int y = start_y; y < end_y; ++y)
    {
        unsigned int index = (y * width * channels) + (start_x * channels);
        for (unsigned int x = start_x; x < end_x; ++x)
        {
            //R
            pixel_value = (float) pixelsInput[index];                                        // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastR;                                                        // Adjust contrast
            pixel_value += brightnessR;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            pixelsInput[index] = (unsigned char)pixel_value;                                 // Update pixel value
            index+=1;

            //G
            pixel_value = (float) pixelsInput[index];                                        // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastG;                                                        // Adjust contrast
            pixel_value += brightnessG;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            pixelsInput[index] = (unsigned char)pixel_value;                                 // Update pixel value
            index+=1;

            //B
            pixel_value = (float) pixelsInput[index];                                        // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastB;                                                        // Adjust contrast
            pixel_value += brightnessB;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            pixelsInput[index] = (unsigned char) pixel_value;                                // Update pixel value
            index+=1;
        }
    }
   return;
}


//This is a function that should alter the brightness and contrast of the pixelsInput
//each of the channels has its own brightness and contrast change that should be changed
//See adjustBrightnessContrast_Simple for a non AVX template for this function
void adjustBrightnessContrast_AVX2(uint8_t *pixelsInput, int width, int height,int channels,
                                   float brightnessR, float contrastR,
                                   float brightnessG, float contrastG,
                                   float brightnessB, float contrastB,
                                   float borderX, float borderY)
{
    if (pixelsInput==0) { return; }

    // Boundary handling
    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = width  - start_x;
    signed int end_y = height - start_y;

    if (start_x<0)    { start_x = 0;    }
    if (start_y<0)    { start_y = 0;    }
    if (end_x>width)  { end_x = width;  }
    if (end_y>height) { end_y = height; }


    __m256 zero     = _mm256_setzero_ps();
    __m256 max_val  = _mm256_set1_ps(255.0f);

    __m256 channel_brightness[3];
    // Depending on offset we will encounter one of these three situations for our 8 values
    channel_brightness[2] = _mm256_set_ps( brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG ); //Offset 0
    channel_brightness[1] = _mm256_set_ps( brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB ); //Offset 1
    channel_brightness[0] = _mm256_set_ps( brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR ); //Offset 2

    __m256 channel_contrast[3];
    // Depending on offset we will encounter one of these three situations for our 8 values
    channel_contrast[2] = _mm256_set_ps(contrastR, contrastG, contrastB, contrastR, contrastG, contrastB, contrastR, contrastG); //Offset 0
    channel_contrast[1] = _mm256_set_ps(contrastG, contrastB, contrastR, contrastG, contrastB, contrastR, contrastG, contrastB); //Offset 1
    channel_contrast[0] = _mm256_set_ps(contrastB, contrastR, contrastG, contrastB, contrastR, contrastG, contrastB, contrastR); //Offset 2


    for (int y = start_y; y < end_y; ++y)
    {
        int offset = 0;
        int indexStart = (y * width + start_x) * channels;
        //for (int x = start_x; x < end_x; x+=3)
        for (int index = indexStart;  index<indexStart+(end_x-start_x)*channels; index+=8)
        {
            // Load 3 pixels (RGB)              //   Offset
            __m256 pixels_f = _mm256_set_ps(    //  0   1    2
                (float)pixelsInput[index + 0],  // R0   G0   B0
                (float)pixelsInput[index + 1],  // G0   B0   R1
                (float)pixelsInput[index + 2],  // B0   R1   G1
                (float)pixelsInput[index + 3],  // R1   G1   B1
                (float)pixelsInput[index + 4],  // G1   B1   R2
                (float)pixelsInput[index + 5],  // B1   R2   G2
                (float)pixelsInput[index + 6],  // R2   G2   B2
                (float)pixelsInput[index + 7]   // G2   B2   R3
            );

            // Apply contrast first, then brightness
            __m256 adjusted_pixels = _mm256_add_ps( _mm256_mul_ps(pixels_f, channel_contrast[offset%3]), channel_brightness[offset%3] );

            // Clamp to [0, 255]
            adjusted_pixels = _mm256_min_ps( _mm256_max_ps(adjusted_pixels, zero), max_val );

            //                                                          Offset
            // Store back adjusted pixel values                    // 0    1    2
            pixelsInput[index]     = (uint8_t)adjusted_pixels[7];  // R    G    B
            pixelsInput[index + 1] = (uint8_t)adjusted_pixels[6];  // G    B    R
            pixelsInput[index + 2] = (uint8_t)adjusted_pixels[5];  // B    R    G
            pixelsInput[index + 3] = (uint8_t)adjusted_pixels[4];  // R    G    B
            pixelsInput[index + 4] = (uint8_t)adjusted_pixels[3];  // G    B    R
            pixelsInput[index + 5] = (uint8_t)adjusted_pixels[2];  // B    R    G
            pixelsInput[index + 6] = (uint8_t)adjusted_pixels[1];  // R    G    B
            pixelsInput[index + 7] = (uint8_t)adjusted_pixels[0];  // G    B    R
            //                                                        Next Group
            //                                     Starts with        B    R    G

            //index+=8;
            offset+=1;
        }
    }
}


int main(int argc, char *argv[])
{
    // Load a PPM image
    const char *input_file = "input.ppm";
    int width, height, channels;
    uint8_t *image = load_ppm(input_file, &width, &height, &channels);
    printf("Loaded image: %dx%dx%d\n", width, height, channels);

    // Set new dimensions for resizing
    int new_width = 128;
    int new_height = 128;

    // Allocate aligned memory for resized image
    uint8_t *resized_image = (uint8_t *)_mm_malloc(new_width * new_height * channels, 32);



    // Perform resizing using AVX2
    resize_image_avx2(image, width, height, resized_image, new_width, new_height, channels);

    srand(time(NULL)); // randomize seed
    float borderX       = getRandomFloat(0.0, 10.0);
    float borderY       = getRandomFloat(0.0, 10.0);
    float brightnessR   = getRandomFloat(-55.0,55.0);
    float contrastR     = getRandomFloat(0.8, 1.2);
    float brightnessG   = getRandomFloat(-55.0,55.0);
    float contrastG     = getRandomFloat(0.8, 1.2);
    float brightnessB   = getRandomFloat(-55.0,55.0);
    float contrastB     = getRandomFloat(0.8, 1.2);

    printf("Brightness RGB(%0.1f,%0.1f,%0.1f)\n",brightnessR,brightnessG,brightnessB);
    printf("Contrast +- (%0.1f,%0.1f,%0.1f) \n",contrastR,contrastG,contrastB);
    printf("Border X/Y (%0.1f,%0.1f) \n",borderX,borderY);
    //This works
    /*
    adjustBrightnessContrast_Simple(resized_image,new_width,new_height,channels,
                                  brightnessR,contrastR,
                                  brightnessG,contrastG,
                                  brightnessB,contrastB,
                                  borderX,borderY);*/
    //This doesn't
    adjustBrightnessContrast_AVX2(resized_image,new_width,new_height,channels,
                                  brightnessR,contrastR,
                                  brightnessG,contrastG,
                                  brightnessB,contrastB,
                                  borderX,borderY);

    // Save the resized image as PPM
    const char *output_file = "resized.ppm";
    write_ppm(output_file, resized_image, new_width, new_height, channels);

    printf("Resized image saved as %s.\n", output_file);

    // Free allocated memory
    _mm_free(image);
    _mm_free(resized_image);

    return 0;
}

