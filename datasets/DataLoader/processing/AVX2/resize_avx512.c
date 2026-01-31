#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//gcc -o resize_avx512 resize_avx512.c -mavx512f

#define WIDTH 256
#define HEIGHT 256
#define CHANNELS 3

// Helper to generate random RGB image
void generate_image(uint8_t *image, int width, int height, int channels) {
    for (int i = 0; i < width * height * channels; i++) {
        image[i] = rand() % 256; // Random values for RGB channels
    }
}

// Resize function using AVX512 (nearest-neighbor)
void resize_image_avx512(uint8_t *src, int src_width, int src_height,
                         uint8_t *dst, int dst_width, int dst_height, int channels) {
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
        int src_y = (int)(y * y_ratio);

        for (int x = 0; x < dst_width; x += 16) { // Process 16 pixels at a time
            int src_x[16];
            for (int i = 0; i < 16; i++) {
                src_x[i] = (int)((x + i) * x_ratio) * channels;
            }

            // Load source pixels
            __m512i r = _mm512_set_epi32(
                src[src_y * src_width * channels + src_x[15]],
                src[src_y * src_width * channels + src_x[14]],
                src[src_y * src_width * channels + src_x[13]],
                src[src_y * src_width * channels + src_x[12]],
                src[src_y * src_width * channels + src_x[11]],
                src[src_y * src_width * channels + src_x[10]],
                src[src_y * src_width * channels + src_x[9]],
                src[src_y * src_width * channels + src_x[8]],
                src[src_y * src_width * channels + src_x[7]],
                src[src_y * src_width * channels + src_x[6]],
                src[src_y * src_width * channels + src_x[5]],
                src[src_y * src_width * channels + src_x[4]],
                src[src_y * src_width * channels + src_x[3]],
                src[src_y * src_width * channels + src_x[2]],
                src[src_y * src_width * channels + src_x[1]],
                src[src_y * src_width * channels + src_x[0]]
            );

            // Store destination pixels
            _mm512_storeu_si512((__m512i*)&dst[y * dst_width * channels + x * channels], r);
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
    // Generate original 256x256 RGB image
    uint8_t *image = (uint8_t *)malloc(WIDTH * HEIGHT * CHANNELS);
    generate_image(image, WIDTH, HEIGHT, CHANNELS);

    // Set new dimensions for resizing
    int new_width = 128;
    int new_height = 128;

    // Allocate memory for resized image
    uint8_t *resized_image = (uint8_t *)malloc(new_width * new_height * CHANNELS);

    // Perform resizing using AVX512
    resize_image_avx512(image, WIDTH, HEIGHT, resized_image, new_width, new_height, CHANNELS);

    // Save the images as PPM files
    write_ppm("source.ppm", image, WIDTH, HEIGHT, CHANNELS);
    write_ppm("resized.ppm", resized_image, new_width, new_height, CHANNELS);

    printf("Images saved as source.ppm and resized.ppm.\n");

    // Free allocated memory
    free(image);
    free(resized_image);

    return 0;
}

