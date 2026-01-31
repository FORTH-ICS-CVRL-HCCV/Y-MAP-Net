#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHANNELS 3
//gcc resize_avx2.c -march=native -mtune=native -o resize_avx2


// Helper to load PPM image
uint8_t *load_ppm(const char *filename, int *width, int *height, int *channels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
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

// Resize function using AVX2 (nearest-neighbor)
void resize_image_avx2(uint8_t *src, int src_width, int src_height,
                       uint8_t *dst, int dst_width, int dst_height, int channels) {
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
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

int main() {
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

    // Save the resized image as PPM
    const char *output_file = "resized.ppm";
    write_ppm(output_file, resized_image, new_width, new_height, channels);

    printf("Resized image saved as %s.\n", output_file);

    // Free allocated memory
    _mm_free(image);
    _mm_free(resized_image);

    return 0;
}

