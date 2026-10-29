/** @file DataAugmentation.h
 *  @brief  Runtime-configurable data-augmentation parameters (header-only).
 *
 *  Previously these were compile-time #defines in configuration.h.  They now
 *  live in struct DataAugmentation inside struct ImageDatabase so they can be
 *  read and overridden at runtime (e.g. from Python / configuration.json)
 *  without recompilation.  augmentationParamsSetDefaults() seeds the struct
 *  with the same values the old defines carried.
 */
#ifndef DATAAUGMENTATION_H_INCLUDED
#define DATAAUGMENTATION_H_INCLUDED

#include <string.h>

struct DataAugmentation {
    /* --- Chance percentages (0.0 – 100.0) -------------------------------- */
    float chanceDestroy;
    float chancePerturbed;
    float chancePanAndZoom;
    float chanceBurnedPixels;
    float chanceBrightnessContrast;
    float chanceBrightnessContrastUniform;
    float chanceHorizontalFlip;
    float chanceRotate90;
    float chanceCoarseDropout;
    float chanceGaussianBlur;
    float chanceDefocusBlur;
    float chanceMotionBlur;

    /* --- Zoom / pan ------------------------------------------------------- */
    float maxZoomFactor;

    /* --- Perturbation / burned pixels ------------------------------------- */
    int   perturbationMagnitude;
    int   maxBurnedPixels;

    /* --- Brightness / contrast -------------------------------------------- */
    float minBrightnessChange;
    float maxBrightnessChange;
    float minRelContrastChange;
    float maxRelContrastChange;
    float minUniformBrightnessChange;
    float maxUniformBrightnessChange;

    /* --- Coarse dropout --------------------------------------------------- */
    int   coarseDropoutHolesMin;
    int   coarseDropoutHolesMax;
    int   coarseDropoutMinSize;
    int   coarseDropoutMaxSize;

    /* --- Gaussian blur ---------------------------------------------------- */
    float gaussianBlurSigmaMin;
    float gaussianBlurSigmaMax;

    /* --- Defocus blur ----------------------------------------------------- */
    int   defocusBlurRadiusMin;
    int   defocusBlurRadiusMax;

    /* --- Motion blur ------------------------------------------------------ */
    int   motionBlurLengthMin;
    int   motionBlurLengthMax;
};

static inline void augmentationParamsSetDefaults(struct DataAugmentation *a)
{
    memset(a, 0, sizeof(*a));

    a->chanceDestroy                   = 0.0f;
    a->chancePerturbed                 = 35.0f;
    a->chancePanAndZoom                = 45.0f;
    a->chanceBurnedPixels              = 50.0f;
    a->chanceBrightnessContrast        = 50.0f;
    a->chanceBrightnessContrastUniform = 50.0f;
    a->chanceHorizontalFlip            = 0.0f;
    a->chanceRotate90                  = 10.0f;
    a->chanceCoarseDropout             = 30.0f;
    a->chanceGaussianBlur              = 20.0f;
    a->chanceDefocusBlur               = 15.0f;
    a->chanceMotionBlur                = 15.0f;

    a->maxZoomFactor                   = 1.1f;

    a->perturbationMagnitude           = 100;
    a->maxBurnedPixels                 = 10;

    a->minBrightnessChange             = -55.0f;
    a->maxBrightnessChange             =  55.0f;
    a->minRelContrastChange            =   0.8f;
    a->maxRelContrastChange            =   1.2f;
    a->minUniformBrightnessChange      = -100.0f;
    a->maxUniformBrightnessChange      =  120.0f;

    a->coarseDropoutHolesMin           = 1;
    a->coarseDropoutHolesMax           = 8;
    a->coarseDropoutMinSize            = 8;
    a->coarseDropoutMaxSize            = 24;

    a->gaussianBlurSigmaMin            = 0.5f;
    a->gaussianBlurSigmaMax            = 3.0f;

    a->defocusBlurRadiusMin            = 1;
    a->defocusBlurRadiusMax            = 5;

    a->motionBlurLengthMin             = 5;
    a->motionBlurLengthMax             = 15;
}

#endif // DATAAUGMENTATION_H_INCLUDED
