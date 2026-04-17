#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

// Build a tiled grid image of all heatmap channels.
//
// hm         : CHW planar float32, channel c starts at hm + c*W*H
// values in [-hm_scale, hm_scale]
// tile_size  : each channel rendered as tile_size × tile_size pixels
// cols       : number of tile columns in the grid
//
// Returns a BGR Mat suitable for cv::imshow.
inline cv::Mat make_heatmap_grid(
        const float* hm, int W, int H, int C,
        float hm_scale = 120.f,
        int tile_size = 48,
        int cols = 12)
{
    const int rows = (C + cols - 1) / cols;
    const int lbl_h = 10;  // pixels for channel-label strip above each tile
    const int cell_h = tile_size + lbl_h;

    cv::Mat grid(rows * cell_h, cols * tile_size, CV_8UC3, cv::Scalar(30, 30, 30));

    for (int c = 0; c < C; ++c) {
        const float* src = hm + c * W * H;
        cv::Mat ch(H, W, CV_32FC1, const_cast<float*>(src));

        // Map [-hm_scale, hm_scale] → [0, 255]
        cv::Mat u8;
        ch.convertTo(u8, CV_8UC1, 127.5f / hm_scale, 127.5f);

        cv::Mat resized;
        cv::resize(u8, resized, {tile_size, tile_size}, 0, 0, cv::INTER_LINEAR);

        cv::Mat colored;
        cv::applyColorMap(resized, colored, cv::COLORMAP_JET);

        const int row = c / cols;
        const int col = c % cols;
        const int x = col * tile_size;
        const int y = row * cell_h + lbl_h;

        colored.copyTo(grid(cv::Rect(x, y, tile_size, tile_size)));

        // Channel label
        char lbl[8];
        std::snprintf(lbl, sizeof(lbl), "%d", c);
        cv::putText(grid, lbl,
                    {x + 2, row * cell_h + lbl_h - 1},
                    cv::FONT_HERSHEY_PLAIN, 0.6,
                    cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    return grid;
}
