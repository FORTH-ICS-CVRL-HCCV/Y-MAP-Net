#ifndef RECALIBRATION_H_INCLUDED
#define RECALIBRATION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif
 

//DEPTH STEP 8 of PrepareBatch.c (after all augmentation): per-image re-normalisation of the depth targets over
//the content (the borders bx/by are skipped; after a rotate90 there are none). Both map the content's min..max
//linearly to the full range (rounded half away from zero), larger = nearer is kept.
//  recalibrate16BitDepth: ALWAYS stretches to [MINV_16BIT, MAXV_16BIT] = [-32767, 32767] (returns only for a
//                         constant depth) -- the 16-bit target's per-image normalisation happens here.
//  recalibrate8BitDepth : to [MINV, MAXV] = [-120, 120]; a no-op when STEP 4 already produced the full range
//                         (the usual case).
//The normals and depth-level channels are NOT recomputed afterwards.
static void recalibrate16BitDepth(struct Heatmaps *hm16,
                                   Heatmap16BitIndex ch,
                                   SampleNumber sampleID,
                                   unsigned int bx,
                                   unsigned int by)
{
    if (!hm16 || !hm16->pixels || hm16->channels == 0) { return; }
    int W = hm16->width, H = hm16->height, C = hm16->channels;
    signed short *base = (signed short *)hm16->pixels + ((size_t)W * H * C * sampleID) + ch;

    signed short mn = MAXV_16BIT, mx = MINV_16BIT;
    for (int y = (int)by; y < H - (int)by; y++) {
        const signed short *row = base + y * W * C;
        for (int x = (int)bx; x < W - (int)bx; x++) {
            signed short v = row[x * C];
            if (v < mn) { mn = v; }
            if (v > mx) { mx = v; }
        }
    }
    if (mn >= mx) { return; }

    float scale = (float)(MAXV_16BIT - MINV_16BIT) / (float)(mx - mn);
    float bias  = (float)MINV_16BIT - (float)mn * scale;
    for (int y = (int)by; y < H - (int)by; y++) {
        signed short *row = base + y * W * C;
        for (int x = (int)bx; x < W - (int)bx; x++) {
            signed short v = row[x * C];
            float r = (float)v * scale + bias;
            if (r < MINV_16BIT) { r = MINV_16BIT; }
            if (r > MAXV_16BIT) { r = MAXV_16BIT; }
            row[x * C] = (signed short)(r + (r >= 0.0f ? 0.5f : -0.5f));
        }
    }
}

static void recalibrate8BitDepth(struct Heatmaps *hm8,
                                  Heatmap8BitIndex ch,
                                  SampleNumber sampleID,
                                  unsigned int bx,
                                  unsigned int by)
{
    if (!hm8 || !hm8->pixels || hm8->channels == 0) { return; }
    int W = hm8->width, H = hm8->height, C = hm8->channels;
    signed char *base = (signed char *)hm8->pixels + ((size_t)W * H * C * sampleID) + ch;

    signed char mn = MAXV, mx = MINV;
    for (int y = (int)by; y < H - (int)by; y++) {
        const signed char *row = base + y * W * C;
        for (int x = (int)bx; x < W - (int)bx; x++) {
            signed char v = row[x * C];
            if (v < mn) { mn = v; }
            if (v > mx) { mx = v; }
        }
    }
    if (mn >= mx) { return; }
    if (mn == MINV && mx == MAXV) { return; } // already full range

    float scale = (float)MAXV_MINUS_MMINV / (float)(mx - mn);
    float bias  = (float)MINV - (float)mn * scale;
    for (int y = (int)by; y < H - (int)by; y++) {
        signed char *row = base + y * W * C;
        for (int x = (int)bx; x < W - (int)bx; x++) {
            float v = (float)row[x * C];
            float r = v * scale + bias;
            if (r < MINV) { r = MINV; }
            if (r > MAXV) { r = MAXV; }
            row[x * C] = (signed char)(r + (r >= 0.0f ? 0.5f : -0.5f));
        }
    }
}


#ifdef __cplusplus
}
#endif

#endif // RECALIBRATION_H_INCLUDED
