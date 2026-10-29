#include "HeatmapGenerator.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "processing/augmentations.h"

#include "DBLoader.h"

/*
horizontalJointFlip Assumes:
0: head
1: endsite_eye.l
2: endsite_eye.r
3: lear
4: rear
5: lshoulder
6: rshoulder
7: lelbow
8: relbow
9: lhand
10: rhand
11: lhip
12: rhip
13: lknee
14: rknee
15: lfoot
16: rfoot
*/

int isRightJoint(int jID)
{
    switch (jID)
    {
        case 2:
        case 4:
        case 6:
        case 8:
        case 10:
        case 12:
        case 14:
        case 16:
          return 1;
        default:
          return 0;
    }
   return 0;
}


 int horizontalJointFlip(int jID, int doFlip)
{
    if (!doFlip) { return jID; }

    switch (jID)
    {
        // eyes
        case 1: return 2;
        case 2: return 1;

        // ears
        case 3: return 4;
        case 4: return 3;

        // shoulders
        case 5: return 6;
        case 6: return 5;

        // elbows
        case 7: return 8;
        case 8: return 7;

        // hands
        case 9:  return 10;
        case 10: return 9;

        // hips
        case 11: return 12;
        case 12: return 11;

        // knees
        case 13: return 14;
        case 14: return 13;

        // feet
        case 15: return 16;
        case 16: return 15;

        // center joints remain the same
        case 0:  // head
        default:
            return jID;
    }
}


static inline void writeSignedPAF(
                                  signed char *heatmapPTR,
                                  int width, int height, int channels,
                                  int xx, int yy,
                                  int paf_ch,
                                  float value   // float in [-127..127]
                                 )
{
    if (xx >= 0 && xx < width && yy >= 0 && yy < height)
    {
        int idx = (yy * width * channels) + (xx * channels) + paf_ch;

        // Write signed char safely
        int iv = (int)value;
        if (iv > 127)  { iv = 127;  }
        if (iv < -127) { iv = -127; }

        heatmapPTR[idx] = (signed char)iv;
    }
}


void drawSignedPAFLine(
                       signed char *heatmapPTR,
                       int paf_ch,
                       int width, int height, int channels,
                       int x1, int y1,       // parent
                       int x2, int y2,       // child
                       int lineWidth,
                       int flipGradient
                      )
{
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2 ? 1 : -1);
    int sy = (y1 < y2 ? 1 : -1);
    int err = dx - dy;

    // Limb vector and length
    float fx = (float)(x2 - x1);
    float fy = (float)(y2 - y1);
    float L = sqrtf(fx*fx + fy*fy);
    if (L < 1e-6f) return;  // no meaningful limb

    float invL = 1.0f / L;

    // For projection: dot( (px,py) , (fx,fy) ) / L^2
    float invL2 = 1.0f / (L*L);

    int half = lineWidth / 2;
    int x = x1;
    int y = y1;

    // === SAFE TERMINATION BASED ON LIMB LENGTH ===
    int maxSteps = (int)(L * 1.5f);
    if (maxSteps < 100) maxSteps = 100;
    int stepCount = 0;
    // =============================================

    for (;;)
    {
        // Compute normalized projection t in [0..1]
        float dxp = (float)(x - x1);
        float dyp = (float)(y - y1);

        float t = (dxp*fx + dyp*fy) * invL2;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;

        // Signed PAF: -127 at parent → +127 at child
        float value = -127.0f * (1.0f - 2.0f * t);

        // Flipped signed PAF: +127 at parent → -127 at child
        if (flipGradient)
            { value = -1.0 * value; }

        // Draw thickness box
        for (int ox = -half; ox <= half; ox++)
        for (int oy = -half; oy <= half; oy++)
        {
            int xx = x + ox;
            int yy = y + oy;
            writeSignedPAF(
                heatmapPTR, width, height, channels,
                xx, yy,
                paf_ch,
                value
            );
        }

        // End if reached child
        if (x == x2 && y == y2)
            break;

        // Bresenham step
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }

        // === SAFETY EXIT ===
        stepCount++;
        if (stepCount >= maxSteps)
            break;
        // ====================
    }
}


/*
 * Limb PAF with PAF_ENCODING_CHILD_LIT (PLAN.md Observation 29 / Experiment W):
 * value goes linearly from PAF_CHILD_VALUE at the child joint to PAF_PARENT_VALUE at the parent,
 * for every limb regardless of body side. The whole limb stays well above the -120 background and
 * above the loss foreground threshold (NNLosses _fn_penalty: y_true > 0.33 ≈ +40), and the bright
 * end is always at the child, where resolution must pick the right wrist/ankle/elbow/knee.
 * (drawSignedPAFLine is kept unchanged: the person_lr_bridge instance channel still uses it.)
 */
void drawChildLitPAFLine(
                         signed char *heatmapPTR,
                         int paf_ch,
                         int width, int height, int channels,
                         int cx, int cy,       // child joint
                         int px, int py,       // parent joint
                         int lineWidth
                        )
{
    int dx = abs(px - cx);
    int dy = abs(py - cy);
    int sx = (cx < px ? 1 : -1);
    int sy = (cy < py ? 1 : -1);
    int err = dx - dy;

    float fx = (float)(px - cx);
    float fy = (float)(py - cy);
    float L2 = fx*fx + fy*fy;
    if (L2 < 1e-12f) return;  // no meaningful limb
    float invL2 = 1.0f / L2;

    int half = lineWidth / 2;
    int x = cx;
    int y = cy;

    int maxSteps = (int)(sqrtf(L2) * 1.5f);   // safe termination, as in drawSignedPAFLine
    if (maxSteps < 100) maxSteps = 100;

    for (int stepCount = 0; stepCount < maxSteps; stepCount++)
    {
        // normalized projection t in [0..1]: 0 at the child, 1 at the parent
        float t = ((float)(x - cx)*fx + (float)(y - cy)*fy) * invL2;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float value = PAF_CHILD_VALUE + (PAF_PARENT_VALUE - PAF_CHILD_VALUE) * t;

        for (int ox = -half; ox <= half; ox++)
        for (int oy = -half; oy <= half; oy++)
            { writeSignedPAF(heatmapPTR, width, height, channels, x + ox, y + oy, paf_ch, value); }

        if (x == px && y == py) break;   // reached the parent

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
}



void drawThickLineOnHeatmapSteep(signed char *heatmapPTR, int heatmapTargetChannel, unsigned int widthR, unsigned int heightR, unsigned int channels, int x0, int y0, int x1, int y1, int lineWidth, signed char foregroundValue, signed char backgroundValue)
{
    // Determine if the line is steep
    int temp;
    // The line is steep so we must swap x0 with y0 and x1 with y1
    temp = x0;
    x0 = y0;
    y0 = temp;

    temp = x1;
    x1 = y1;
    y1 = temp;

    // Ensure x0 <= x1
    if (x0 > x1)
    {
        temp = x0;
        x0 = x1;
        x1 = temp;

        temp = y0;
        y0 = y1;
        y1 = temp;
    }

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int x = x0;
    int y = y0;

    int width  = widthR;
    int height = heightR;
    int lineWidthSquared = lineWidth * lineWidth;
    int halfLineWidth    = lineWidth / 2;

    //Make sure this function will always terminate
    int stepCount = 0;          // Step counter
    // Calculate the Euclidean distance
    float distance = sqrtf((float)((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)));
    int maxSteps = (int)(distance * 1.5); // Add a safety margin, e.g., 1.5 times the distance
    if (maxSteps<100) {maxSteps=100;} //minimum value for degenerate inputs
    //const int maxSteps = 10000; // Set a reasonable maximum step count

    // Loop through the original line
    do
    {
      for (int i = 0; i <= lineWidthSquared; i++)
       {
        int x_offset = i % lineWidth - halfLineWidth;
        int y_offset = i / lineWidth - halfLineWidth;

        int xx = y + y_offset;
        int yy = x + x_offset;
        //Basically if the xx,yy coordinates are out of bounds then a foreground pixel will be stored on 0 without affecting image
        //Apparently skipping the jump in the next lines 300M times is preferable to accessing heatmapPTR so this jmp is best preserved!
        if ( (xx >= 0) && (xx < width) && (yy >= 0) && (yy < height) )
          { heatmapPTR[(yy * width * channels) + (xx * channels) + heatmapTargetChannel] = foregroundValue; }
       } //add pixels

       int e2 = 2 * err;
       int e2GreaterThanMinusDy = (e2 > -dy);
       int e2SmallerThanDx      = (e2 < dx);
       //-----------------------------------
       err -= e2GreaterThanMinusDy * dy;
       x   += e2GreaterThanMinusDy * sx;
       //-----------------------------------
       err +=  e2SmallerThanDx * dx;
       y   +=  e2SmallerThanDx * sy;
       //-----------------------------------
       ++stepCount;
       //-----------------------------------
    } while ( (stepCount<maxSteps) && ( (x != x1) || (y != y1) ) ); //line ended

}


void drawThickLineOnHeatmapNotSteep(signed char *heatmapPTR, int heatmapTargetChannel, unsigned int widthR, unsigned int heightR, unsigned int channels, int x0, int y0, int x1, int y1, int lineWidth, signed char foregroundValue, signed char backgroundValue)
{
    // Determine if the line is steep
    //int isSteep = 0;
    int temp;

    // Ensure x0 <= x1
    if (x0 > x1)
    {
        temp = x0;
        x0 = x1;
        x1 = temp;

        temp = y0;
        y0 = y1;
        y1 = temp;
    }

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int x = x0;
    int y = y0;

    int width  = widthR;
    int height = heightR;
    int lineWidthSquared = lineWidth * lineWidth;
    int halfLineWidth    = lineWidth / 2;

    //Make sure this function will always terminate
    int stepCount = 0;          // Step counter
    // Calculate the Euclidean distance
    float distance = sqrtf((float)((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)));
    int maxSteps = (int)(distance * 1.5); // Add a safety margin, e.g., 1.5 times the distance
    if (maxSteps<100) {maxSteps=100;} //minimum value for degenerate inputs
    //const int maxSteps = 10000; // Set a reasonable maximum step count

    //Remember the first element
    signed char remember = heatmapPTR[0];

    // Loop through the original line
    do
    {
      for (int i = 0; i <= lineWidthSquared; i++)
       {
        int x_offset = i % lineWidth - halfLineWidth;
        int y_offset = i / lineWidth - halfLineWidth;

        int xx = x + x_offset;
        int yy = y + y_offset;

        if ( (xx >= 0) && (xx < width) && (yy >= 0) && (yy < height) )
          { heatmapPTR[(yy * width * channels) + (xx * channels) + heatmapTargetChannel] = foregroundValue; }
       } //add pixels

       int e2 = 2 * err;
       int e2GreaterThanMinusDy = (e2 > -dy);
       int e2SmallerThanDx      = (e2 < dx);
       //-----------------------------------
       err -= e2GreaterThanMinusDy * dy;
       x   += e2GreaterThanMinusDy * sx;
       //-----------------------------------
       err +=  e2SmallerThanDx * dx;
       y   +=  e2SmallerThanDx * sy;
       //-----------------------------------
       ++stepCount;
       //-----------------------------------
    } while ( (stepCount<maxSteps) && ( (x != x1) || (y != y1) ) );//line ended

    //Restore first element
    heatmapPTR[0] = remember;
}

//TODO: There is a corner case where the first pixel (of the first channel of the heatmap) will not be drawn over, this never happens for heatmap with target channel #17 though
void drawThickLineOnHeatmap(signed char *heatmapPTR, int heatmapTargetChannel, unsigned int width, unsigned int height, unsigned int channels, int x0, int y0, int x1, int y1, int lineWidth, signed char foregroundValue, signed char backgroundValue)
{
    if ((x0==x1) && (y0==y1)) { return; } //Fast path out

    int isSteep = abs(y1 - y0) > abs(x1 - x0);
    if (isSteep)
    {
        drawThickLineOnHeatmapSteep(heatmapPTR,heatmapTargetChannel,width,height,channels,x0,y0,x1,y1,lineWidth,foregroundValue,backgroundValue);
    } else
    {
        drawThickLineOnHeatmapNotSteep(heatmapPTR,heatmapTargetChannel,width,height,channels,x0,y0,x1,y1,lineWidth,foregroundValue,backgroundValue);
    }
}


// Left joint are simply "the ones who are not right"
int isLeftJoint(int jID)
{
    if (!isRightJoint(jID)) { return 1; }
    return 0;
}



void drawSingleLRHeatmapGaussian(
                                   struct ImageDatabase *db,
                                   signed char *heatmapPTR,
                                   int lrChannel,                 // which channel to draw into
                                   char right,                    // 1 = right-only, 0 = left-only
                                   unsigned long sourceSampleID,
                                   unsigned int doLRFlip,
                                   int gradientSizeDefault,
                                   int originalInputWidth,
                                   int originalInputHeight,
                                   float zoom_factor,
                                   int pan_x,
                                   int pan_y,
                                   float offsetX,
                                   float offsetY,
                                   float scaleX,
                                   float scaleY
                                 )
{
    if (sourceSampleID >= db->numberOfSamples) { return; }
    if (db->pdb->sample[sourceSampleID].numberOfSkeletons == 0) { return; }

    float IORatioX = (float) db->out8bit.width  / db->in.width;
    float IORatioY = (float) db->out8bit.height / db->in.height;

    // Loop through joints
    for (int fjID = 0; fjID < db->pdb->keypointsForEachSample; fjID++)
    {

        // Apply left-right augmentation flips: the side is the OUTPUT label fjID, the position is read
        // from its mirror jID (on a flipped sample the right channel shows the original left joints)
        int jID = horizontalJointFlip(fjID, doLRFlip);

        // =============================
        // Filter LEFT or RIGHT joints
        // =============================
        if (right)
        {
            if (!isRightJoint(fjID)) { continue; }
        }
        else
        {
            if (!isLeftJoint(fjID)) { continue; }
        }


        for (unsigned short skID = 0; skID < db->pdb->sample[sourceSampleID].numberOfSkeletons; skID++)
        {
            unsigned short x = db->pdb->sample[sourceSampleID].sk[skID].coords[jID * 3 + 0];
            unsigned short y = db->pdb->sample[sourceSampleID].sk[skID].coords[jID * 3 + 1];

            if (x == 0 || y == 0) { continue; }

            float xF = (float)x;
            float yF = (float)y;

            transformCoordinatesPanAndZoom(
                                            &xF, &yF,
                                            (float)pan_x, (float)pan_y,
                                            zoom_factor,
                                            originalInputWidth,
                                            originalInputHeight
                                          );

            xF = (offsetX + (xF * scaleX)) * IORatioX;
            yF = (offsetY + (yF * scaleY)) * IORatioY;

            //signed + floorf so joints just off the left/top edge keep their visible part (as the joint heatmaps)
            int px = (int) floorf(xF);
            int py = (int) floorf(yF);
            if (doLRFlip) { px = db->out8bit.width - 1 - px; } //pixel i -> W-1-i, as flipImageHoriz_8bit

            // Gradient size selection
            int gradientSize                = gradientSizeDefault + db->pdb->joint[jID].jointDifficulty;
            const signed char *gradient     = find_heatmap_by_gradient_size(db->gradients, gradientSize);

            int g2 = gradientSize * 2;

            if (gradient!=0)
            {
            // Draw Gaussian blob
            for (int gx = 0; gx < g2; gx++)
            {
             for (int gy = 0; gy < g2; gy++)
             {
                int xx = px + gx - gradientSize;
                int yy = py + gy - gradientSize;

                if (xx < 0 || xx >= db->out8bit.width || yy < 0 || yy >= db->out8bit.height) {  continue; }

                signed char g = gradient[gy * g2 + gx];   // 0…127

                signed char *dst = heatmapPTR + (yy * db->out8bit.width * db->out8bit.channels) + (xx * db->out8bit.channels) + lrChannel;

                // Use max magnitude blending (same as joint heatmaps)
                if (g > *dst)
                    { *dst = g; }
             }
            }
           } //Have a valid gradient to draw
        }
    }
}



int countSkeletonsJointsInHeatmaps(
                            struct ImageDatabase *db,
                            unsigned long sampleID,
                            int originalInputWidth,
                            int originalInputHeight,
                            int padX,
                            int padY,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY,
                            int *totalJointsOut   // may be NULL
                           )
{
   int totalJoints = 0;
   int jointCount  = 0;

   float IORatioX = (float) db->out8bit.width  / db->in.width;
   float IORatioY = (float) db->out8bit.height / db->in.height;

   float startX = 0.0 + (float) padX;
   float startY = 0.0 + (float) padY;
   float endX   = db->out8bit.width  - (float) padX;
   float endY   = db->out8bit.height - (float) padY;

   unsigned short skID, jID;
   unsigned short x1, y1;
   float xF,yF;
   for (skID = 0; skID < db->pdb->sample[sampleID].numberOfSkeletons; skID++)
        {
          for (jID = 0; jID < db->pdb->keypointsForEachSample; jID++)
            {
                x1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 0];
                y1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 1];
                xF = (float) x1;
                yF = (float) y1;

                totalJoints = totalJoints + ((x1!=0) && (y1!=0));

                transformCoordinatesPanAndZoom(&xF, &yF, (float) pan_x,(float) pan_y, zoom_factor, originalInputWidth, originalInputHeight);

                xF =  (offsetX + (xF * scaleX)) * IORatioX;
                yF =  (offsetY + (yF * scaleY)) * IORatioY;

                if ( (startX<xF) && (xF<endX) && (startY<yF) && (yF<endY) )
                {
                   jointCount = jointCount + 1;
                }
            }
        }

   if (totalJointsOut != NULL) { *totalJointsOut = totalJoints; }
   return jointCount;
}



int ensurePercentageOfJointsInHeatmap(
                                       float percentage,
                                       struct ImageDatabase *db,
                                       unsigned long sampleID,
                                       int originalInputWidth,
                                       int originalInputHeight,
                                       int padX,
                                       int padY,
                                       float zoom_factor,
                                       int pan_x,
                                       int pan_y,
                                       float offsetX,
                                       float offsetY,
                                       float scaleX,
                                       float scaleY
                                      )
{
    int totalJoints = 0;
    int jointCount  = countSkeletonsJointsInHeatmaps(
                            db, sampleID,
                            originalInputWidth, originalInputHeight,
                            padX, padY,
                            zoom_factor, pan_x, pan_y,
                            offsetX, offsetY, scaleX, scaleY,
                            &totalJoints);

    if (totalJoints == 0)  { return 1; }

    if ( ((float) jointCount / totalJoints) > percentage )  { return 1; }

    return 0;
}




//Parent joint of jID's limb PAF (PAF_HIPS_TO_SHOULDERS: hips -> same-side shoulder instead of the nose).
static int pafParentJoint(struct ImageDatabase * db, int jID)
{
    if (PAF_HIPS_TO_SHOULDERS)
    {   //joint ids: see horizontalJointFlip
        if (jID==11) { return 5; } //lhip -> lshoulder
        if (jID==12) { return 6; } //rhip -> rshoulder
    }
    return db->pdb->joint[jID].parent;
}

void drawPAFsOnHeatmaps(
                            struct ImageDatabase * db,
                            signed char * heatmapPTR,
                            int heatmapTargetChannel,
                            unsigned short numberOfPAFJoints,
                            const int * PAFJoints,
                            unsigned long sampleID,
                            int thickness,
                            int doLRFlip,
                            signed char foregroundValue,
                            signed char backgroundValue,
                            int originalInputWidth,
                            int originalInputHeight,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY
                           )
{
   float IORatioX = (float) db->out8bit.width  / db->in.width;
   float IORatioY = (float) db->out8bit.height / db->in.height;

   unsigned int pafID, skID, jID, parentjID;
   int x1, y1, x2, y2;
   float xF,yF;

   for (pafID = 0; pafID<numberOfPAFJoints; pafID ++)
   {
     for (skID = 0; skID < db->pdb->sample[sampleID].numberOfSkeletons; skID++)
        {
                jID       = PAFJoints[pafID];
                parentjID = pafParentJoint(db,jID);
                if (PAF_SKIP_MISSING_PARENT)
                {   //parent not annotated (raw coords 0, as instanceTransformJoint): draw to the grandparent instead
                    const int fp = horizontalJointFlip(parentjID,doLRFlip);
                    if ( (db->pdb->sample[sampleID].sk[skID].coords[fp * 3 + 0]==0) || (db->pdb->sample[sampleID].sk[skID].coords[fp * 3 + 1]==0) )
                       { parentjID = pafParentJoint(db,parentjID); }
                }

                const int labelJ = jID, labelP = parentjID; //what this channel encodes (pre-flip ids)
                jID       = horizontalJointFlip(jID,doLRFlip);
                parentjID = horizontalJointFlip(parentjID,doLRFlip);
                //fprintf(stderr,"Parent of %s is %s \n",db->pdb->joint[jID].name,db->pdb->joint[parentjID].name);
                x1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 0];
                y1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 1];
                xF = (float) x1;
                yF = (float) y1;
                transformCoordinatesPanAndZoom(&xF,&yF,(float) pan_x,(float) pan_y,zoom_factor, originalInputWidth, originalInputHeight);
                xF =  (offsetX + (xF * scaleX)) * IORatioX;
                yF =  (offsetY + (yF * scaleY)) * IORatioY;
                x1 = (int) xF;
                y1 = (int) yF;

                //X coordinates need to be also horizontally flipped! (pixel i -> W-1-i, as flipImageHoriz_8bit)
                if (doLRFlip) { x1 = db->out8bit.width - 1 - x1;  }
                //--------------------------------------------

                if ((parentjID != jID) && (parentjID<db->pdb->keypointsForEachSample) && (x1!=0) && (y1!=0) && (x1<db->out8bit.width) && (y1<db->out8bit.height) )
                {
                    x2 = db->pdb->sample[sampleID].sk[skID].coords[parentjID * 3 + 0];
                    y2 = db->pdb->sample[sampleID].sk[skID].coords[parentjID * 3 + 1];
                    xF = (float) x2;
                    yF = (float) y2;
                    transformCoordinatesPanAndZoom(&xF,&yF,(float) pan_x,(float) pan_y,zoom_factor, originalInputWidth, originalInputHeight);
                    xF =  (offsetX + (xF * scaleX)) * IORatioX;
                    yF =  (offsetY + (yF * scaleY)) * IORatioY;
                    x2 = (int) xF;
                    y2 = (int) yF;

                    //X coordinates need to be also horizontally flipped! (pixel i -> W-1-i)
                    if (doLRFlip) { x2 = db->out8bit.width - 1 - x2;  }
                    //--------------------------------------------

                    if ((x2!=0) && (y2!=0) && ( x2!=x1 || y2!=y1 ) && (x2<db->out8bit.width) && (y2<db->out8bit.height)  )
                      {
                        //NOT USED drawLineOnHeatmap(heatmapPTR, heatmapTargetChannel+pafID, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, foregroundValue, backgroundValue);
                        //STABLE drawThickLineOnHeatmap(heatmapPTR, heatmapTargetChannel+pafID, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, thickness, foregroundValue, backgroundValue);
                        //(x1,y1) = child joint jID, (x2,y2) = its parent parentjID
                        if (PAF_ENCODING_CHILD_LIT)
                        {
                          drawChildLitPAFLine(heatmapPTR,heatmapTargetChannel+pafID, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, thickness);
                        } else
                        { //Legacy (serials <= 288): signed -127..+127 ramp, lit at the child only for left/central limbs
                          int flipGradient = 0;
                          //side of the CHANNEL (labelJ/labelP), not of the mirrored source joints, so a flip keeps each channel's convention
                          if ( (FLIP_PAF_GRADIENTS_FOR_LEFT_JOINTS) && ( (!isRightJoint(labelJ) || !isRightJoint(labelP))  ) ) { flipGradient=1; }
                          drawSignedPAFLine(heatmapPTR,heatmapTargetChannel+pafID, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, thickness, flipGradient ); //TEST
                        }
                      }
                }
        } // for each skeleton ID loop
   } // for each pafID loop
}





void drawSkeletonsOnHeatmap(
                            struct ImageDatabase *db,
                            signed char *heatmapPTR,
                            int heatmapTargetChannel,
                            unsigned long sampleID,
                            int thickness,
                            int doLRFlip,
                            signed char foregroundValue,
                            signed char backgroundValue,
                            int originalInputWidth,
                            int originalInputHeight,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY
                           )
{
   float IORatioX = (float) db->out8bit.width  / db->in.width;
   float IORatioY = (float) db->out8bit.height / db->in.height;

   unsigned short skID, jID, fjID, parentjID;
   unsigned short x1, y1, x2, y2;
   float xF,yF;
   for (skID = 0; skID < db->pdb->sample[sampleID].numberOfSkeletons; skID++)
        {
          for (fjID = 0; fjID < db->pdb->keypointsForEachSample; fjID++)
            {
                parentjID = db->pdb->joint[fjID].parent;
                //---------------------------------------------------
                jID       = horizontalJointFlip(fjID,doLRFlip);
                parentjID = horizontalJointFlip(parentjID,doLRFlip);

                //fprintf(stderr,"Parent of %s is %s \n",db->pdb->joint[jID].name,db->pdb->joint[parentjID].name);
                x1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 0];
                y1 = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 1];
                xF = (float) x1;
                yF = (float) y1;
                transformCoordinatesPanAndZoom(&xF,&yF,(float) pan_x,(float) pan_y,zoom_factor, originalInputWidth, originalInputHeight);
                xF =  (offsetX + (xF * scaleX)) * IORatioX;
                yF =  (offsetY + (yF * scaleY)) * IORatioY;
                x1 = (unsigned short) xF;
                y1 = (unsigned short) yF;

                if ((parentjID != jID) && (parentjID<db->pdb->keypointsForEachSample) && (x1!=0) && (y1!=0))
                {
                    x2 = db->pdb->sample[sampleID].sk[skID].coords[parentjID * 3 + 0];
                    y2 = db->pdb->sample[sampleID].sk[skID].coords[parentjID * 3 + 1];
                    xF = (float) x2;
                    yF = (float) y2;
                    transformCoordinatesPanAndZoom(&xF,&yF,(float) pan_x,(float) pan_y,zoom_factor, originalInputWidth, originalInputHeight);
                    xF =  (offsetX + (xF * scaleX)) * IORatioX;
                    yF =  (offsetY + (yF * scaleY)) * IORatioY;
                    x2 = (unsigned short) xF;
                    y2 = (unsigned short) yF;

                    if ((x2!=0) && (y2!=0) && ( x2!=x1 || y2!=y1 ) )
                      {
                        //drawLineOnHeatmap(heatmapPTR, heatmapTargetChannel, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, foregroundValue, backgroundValue);
                        drawThickLineOnHeatmap(heatmapPTR, heatmapTargetChannel, db->out8bit.width, db->out8bit.height, db->out8bit.channels, x1, y1, x2, y2, thickness, foregroundValue, backgroundValue);
                      }
                }
            }
        }
}

//Per-instance person SEGMENTATION targets (SEGMENTATION_PLAN §E redesign, 2026-06-20).
//Relative to instanceHeatmapChannel:
//  +0 = person_blob       large size-adaptive Gaussian anchor at the box centre (sigma ~ box size)
//  +1 = person_lr_bridge  SIGNED cross-body line, -127 (anat. left) -> +127 (anat. right): shoulders + hips
//  +2 = person_head       Gaussian anchor at the neck (shoulder midpoint; fallback nose)
//All three are local "name the part you sit on" targets so the decoder can learn them — the old
//center+size (PLAN.md Obs 14) and box-edge-ridge (Obs 17) encodings both collapsed / mislocated.
//Box geometry and pixel ownership now live in the Python decode (SEGMENTATION_PLAN §E), not pixels.

//Transform one skeleton joint through the SAME pipeline as the joints/PAFs (pan/zoom -> offset/
//scale -> IORatio -> LR flip). Returns 1 and fills *outX/*outY if the joint is present (raw!=0).
static int instanceTransformJoint(
                            struct ImageDatabase *db, unsigned long sampleID, int skID, int jID,
                            int doLRFlip, int originalInputWidth, int originalInputHeight,
                            float zoom_factor, int pan_x, int pan_y,
                            float offsetX, float offsetY, float scaleX, float scaleY,
                            int *outX, int *outY)
{
   const float IORatioX = (float) db->out8bit.width  / db->in.width;
   const float IORatioY = (float) db->out8bit.height / db->in.height;
   const int rx = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 0];
   const int ry = db->pdb->sample[sampleID].sk[skID].coords[jID * 3 + 1];
   if ((rx==0) || (ry==0)) { return 0; }   //absent joint — mirror the joint/PAF (x,y)!=0 skip
   float xF = (float) rx, yF = (float) ry;
   transformCoordinatesPanAndZoom(&xF,&yF,(float)pan_x,(float)pan_y,zoom_factor,originalInputWidth,originalInputHeight);
   xF = (offsetX + (xF * scaleX)) * IORatioX;
   yF = (offsetY + (yF * scaleY)) * IORatioY;
   int xo = (int) floorf(xF), yo = (int) floorf(yF);
   if (doLRFlip) { xo = (int) db->out8bit.width - 1 - xo; } //pixel i -> W-1-i, as flipImageHoriz_8bit
   *outX = xo; *outY = yo;
   return 1;
}

//Splat a Gaussian ramping from MINV (patch edge) up to MAXV (centre), sigma = radius/3, centred at
//(px,py) into channel ch, max-blended like the joint / LR heatmaps. Ramping to MINV (not 0) is what
//lets the tail blend into the MINV background — so the blob reads as a Gaussian, not a 0-floored
//rectangle (matches how the precomputed joint/head gradient patches decay to MINV).
static void instanceSplatGaussian(
                            signed char *heatmapPTR, int ch, int width, int height, int channels,
                            int px, int py, int radius)
{
   if (radius < 1) { radius = 1; }
   const float inv2s2 = 9.0f / (2.0f * (float)radius * (float)radius);   //1/(2 sigma^2), sigma=radius/3
   int dx,dy;
   for (dy=-radius; dy<=radius; dy++)
   {
      const int yy = py + dy;
      if (yy<0 || yy>=height) { continue; }
      for (dx=-radius; dx<=radius; dx++)
      {
         const int xx = px + dx;
         if (xx<0 || xx>=width) { continue; }
         const signed char gv = (signed char)((float)MINV + (float)(MAXV - MINV) * expf(-((float)(dx*dx+dy*dy)) * inv2s2));
         signed char * p = heatmapPTR + ((yy*width*channels)+(xx*channels)+ch);
         if (gv > *p) { *p = gv; }   //max-blend (tail == MINV leaves background untouched)
      }
   }
}

void drawInstanceCentersOnHeatmaps(
                            struct ImageDatabase *db,
                            signed char *heatmapPTR,
                            int instanceHeatmapChannel,
                            unsigned long sampleID,
                            int doLRFlip,
                            const signed char * gradient,
                            int gradientSize,
                            int sizeDiskRadius,
                            int originalInputWidth,
                            int originalInputHeight,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY
                           )
{
   const int width    = (int) db->out8bit.width;
   const int height   = (int) db->out8bit.height;
   const int channels = (int) db->out8bit.channels;
   const int blobCh   = instanceHeatmapChannel + 0;
   const int bridgeCh = instanceHeatmapChannel + 1;
   const int headCh   = instanceHeatmapChannel + 2;
   const float IORatioX = (float) db->out8bit.width  / db->in.width;
   const float IORatioY = (float) db->out8bit.height / db->in.height;
   const int bridgeThickness = (sizeDiskRadius>1) ? sizeDiskRadius : 1;   //reuse the disk-radius param as line thickness

   //COCO keypoint indices (channels 0..16): 0=nose 5=left_shoulder 6=right_shoulder 11=left_hip 12=right_hip
   const int NOSE=0, LSHOULDER=5, RSHOULDER=6, LHIP=11, RHIP=12;

   unsigned short skID;
   for (skID = 0; skID < db->pdb->sample[sampleID].numberOfSkeletons; skID++)
   {
       struct Skeleton * sk = &db->pdb->sample[sampleID].sk[skID];

       //--- +0 person_blob: large size-adaptive Gaussian at the (transformed) box centre. ---
       //The box is stored as corners (x1,y1)-(x2,y2), not x,y,w,h (DATALOADER.md A9)
       if ((sk->bboxX2>sk->bboxX) && (sk->bboxY2>sk->bboxY))
       {
          float x1 = (float) sk->bboxX;
          float y1 = (float) sk->bboxY;
          float x2 = (float) sk->bboxX2;
          float y2 = (float) sk->bboxY2;
          transformCoordinatesPanAndZoom(&x1,&y1,(float)pan_x,(float)pan_y,zoom_factor,originalInputWidth,originalInputHeight);
          transformCoordinatesPanAndZoom(&x2,&y2,(float)pan_x,(float)pan_y,zoom_factor,originalInputWidth,originalInputHeight);
          x1 = (offsetX + (x1 * scaleX)) * IORatioX;
          y1 = (offsetY + (y1 * scaleY)) * IORatioY;
          x2 = (offsetX + (x2 * scaleX)) * IORatioX;
          y2 = (offsetY + (y2 * scaleY)) * IORatioY;
          if (doLRFlip) { x1 = (float)width - x1; x2 = (float)width - x2; }

          const float cx   = (x1 + x2) * 0.5f;
          const float cy   = (y1 + y2) * 0.5f;
          const float boxW = (x2>x1) ? (x2-x1) : (x1-x2);
          const float boxH = (y2>y1) ? (y2-y1) : (y1-y2);
          const int   cX   = (int) cx;
          const int   cY   = (int) cy;

          //Drop boxes whose centre fell outside the (possibly cropped) frame — mirror the
          //joint (x,y)!=(0,0) skip so the head never learns phantom anchors.
          if ((cX>0) && (cY>0) && (cX<width) && (cY<height))
          {
             int ri = (int)(INSTANCE_BLOB_RADIUS_FRAC * ((boxW<boxH)?boxW:boxH));
             if (ri < MINIMUM_GRADIENT_SIZE)    { ri = MINIMUM_GRADIENT_SIZE; }
             if (ri > INSTANCE_BLOB_RADIUS_MAX) { ri = INSTANCE_BLOB_RADIUS_MAX; }
             instanceSplatGaussian(heatmapPTR, blobCh, width, height, channels, cX, cY, ri);
          }
       }

       //--- +1 person_lr_bridge: signed cross-body lines, -127 (anat. left) -> +127 (anat. right).
       //--- Sign is tied to joint identity (left=parent, right=child), matching the leftjoints/
       //--- rightjoints (37/38) convention; positions mirror under flip via instanceTransformJoint,
       //--- and on a flip the left end is the mirrored RIGHT joint (as the joint/side channels).
       {
          const int LS = horizontalJointFlip(LSHOULDER,doLRFlip), RS = horizontalJointFlip(RSHOULDER,doLRFlip);
          const int LH = horizontalJointFlip(LHIP,doLRFlip),      RH = horizontalJointFlip(RHIP,doLRFlip);
          int lx,ly,rx,ry;
          if ( instanceTransformJoint(db,sampleID,skID,LS,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&lx,&ly)
            && instanceTransformJoint(db,sampleID,skID,RS,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&rx,&ry) )
          {
             if (BRIDGE_ENCODING_LIT) { drawChildLitPAFLine(heatmapPTR, bridgeCh, width, height, channels, lx,ly, rx,ry, bridgeThickness); } else
                                      { drawSignedPAFLine(heatmapPTR, bridgeCh, width, height, channels, lx,ly, rx,ry, bridgeThickness, 0); }
          }
          if ( instanceTransformJoint(db,sampleID,skID,LH,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&lx,&ly)
            && instanceTransformJoint(db,sampleID,skID,RH,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&rx,&ry) )
          {
             if (BRIDGE_ENCODING_LIT) { drawChildLitPAFLine(heatmapPTR, bridgeCh, width, height, channels, lx,ly, rx,ry, bridgeThickness); } else
                                      { drawSignedPAFLine(heatmapPTR, bridgeCh, width, height, channels, lx,ly, rx,ry, bridgeThickness, 0); }
          }
       }

       //--- +2 person_head: Gaussian anchor at the neck = midpoint(shoulders); fallback nose. ---
       //--- Joint-sized splat from the precomputed gradient patch, max-blended like a joint.
       if (gradientSize>MINIMUM_GRADIENT_SIZE)
       {
          int hx=0, hy=0, haveHead=0;
          int lsx,lsy,rsx,rsy;
          const int haveLS = instanceTransformJoint(db,sampleID,skID,LSHOULDER,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&lsx,&lsy);
          const int haveRS = instanceTransformJoint(db,sampleID,skID,RSHOULDER,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&rsx,&rsy);
          if (haveLS && haveRS) { hx=(lsx+rsx)/2; hy=(lsy+rsy)/2; haveHead=1; }
          else { int nx,ny; if (instanceTransformJoint(db,sampleID,skID,NOSE,doLRFlip,originalInputWidth,originalInputHeight,zoom_factor,pan_x,pan_y,offsetX,offsetY,scaleX,scaleY,&nx,&ny)) { hx=nx; hy=ny; haveHead=1; } }

          if (haveHead && (hx>0) && (hy>0) && (hx<width) && (hy<height))
          {
             signed int xG,yG,newX,newY,gradientSizeX2 = gradientSize*2;
             for (xG=0; xG<gradientSizeX2; xG++)
             {
                for (yG=0; yG<gradientSizeX2; yG++)
                {
                   newX = hx + xG - gradientSize;
                   newY = hy + yG - gradientSize;
                   if (newY>=0 && newY<height && newX>=0 && newX<width)
                   {
                      signed char * p = heatmapPTR + ((newY*width*channels) + (newX*channels) + headCh);
                      const signed char gv = gradient[(yG*gradientSizeX2)+xG];
                      if (gv > *p) { *p = gv; }
                   }
                }
             }
          }
       }
   }
}

// Render the sample's SuperPoint keypoints as K heatmap channels. Each keypoint
// is transformed through the SAME pan/zoom -> offset/scale -> IORatio -> LR-flip
// pipeline as the skeleton joints (so it tracks augmentation), then for each of
// the K PCA components a Gaussian blob is splatted whose PEAK equals the scaled,
// clipped projection value v_j and whose tail decays to MINV (the background /
// "no keypoint" value). Because v_j >= MINV (clipped), max-blending is correct
// and consistent with the joint/instance splats.
//
// Independent of skeletons — called outside the numberOfSkeletons>0 block.
static void drawSuperpointOnHeatmaps(
                            struct ImageDatabase *db,
                            signed char *heatmapPTR,
                            int superpointChannelStart,
                            unsigned long sourceSampleID,
                            unsigned int doLRFlip,
                            int gradientSizeDefault,
                            int originalInputWidth,
                            int originalInputHeight,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY
                           )
{
    const SuperpointImageEntry *sp = db->pdb->sample[sourceSampleID].superpoint;
    const SuperpointPCA *pca = db->superpointPCA;
    if (sp == NULL || pca == NULL || sp->numKeypoints <= 0) { return; }

    int K = db->superpointChannels;
    if (K > pca->maxComponents) { K = pca->maxComponents; }
    if (K <= 0) { return; }

    const int width    = (int) db->out8bit.width;
    const int height   = (int) db->out8bit.height;
    const int channels = (int) db->out8bit.channels;
    const float IORatioX = (float) width  / (float) db->in.width;
    const float IORatioY = (float) height / (float) db->in.height;

    int radius = gradientSizeDefault;
    if (radius < 1) { radius = 1; }
    const float inv2s2 = 9.0f / (2.0f * (float)radius * (float)radius);  // sigma = radius/3

    for (int k = 0; k < sp->numKeypoints; ++k)
    {
        float xF = sp->xyscore[k * 3 + 0];
        float yF = sp->xyscore[k * 3 + 1];
        transformCoordinatesPanAndZoom(&xF, &yF, (float)pan_x, (float)pan_y, zoom_factor, originalInputWidth, originalInputHeight);
        xF = (offsetX + (xF * scaleX)) * IORatioX;
        yF = (offsetY + (yF * scaleY)) * IORatioY;
        int px = (int) xF;
        int py = (int) yF;
        if (doLRFlip) { px = width - 1 - px; } //pixel i -> W-1-i, as flipImageHoriz_8bit
        if (px < 0 || px >= width || py < 0 || py >= height) { continue; }

        const float *desc = sp->descriptors + (size_t)k * (size_t)pca->descDim;
        for (int j = 0; j < K; ++j)
        {
            float v = superpoint_project_component(pca, desc, j);
            if (v > (float)MAXV) { v = (float)MAXV; }
            if (v < (float)MINV) { v = (float)MINV; }
            const int ch = superpointChannelStart + j;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                const int yy = py + dy;
                if (yy < 0 || yy >= height) { continue; }
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int xx = px + dx;
                    if (xx < 0 || xx >= width) { continue; }
                    // Gaussian ramp from MINV (tail) up to v (peak).
                    const float t  = expf(-((float)(dx*dx + dy*dy)) * inv2s2);
                    const signed char gv = (signed char)((float)MINV + (v - (float)MINV) * t);
                    signed char *p = heatmapPTR + ((yy*width*channels) + (xx*channels) + ch);
                    if (gv > *p) { *p = gv; }   // max-blend (tail == MINV leaves background untouched)
                }
            }
        }
    }
}

int populateHeatmaps(
                     struct ImageDatabase *db,
                     unsigned long sourceSampleID,
                     unsigned long targetImageID,
                     unsigned int gradientSizeDefaultRAW,
                     unsigned int PAFSize,
                     unsigned int doLRFlip,
                     int originalInputWidth,
                     int originalInputHeight,
                     float zoom_factor,
                     int pan_x,
                     int pan_y,
                     float offsetX,
                     float offsetY,
                     float scaleX,
                     float scaleY
                     )
{
    //fprintf(stderr,"Asked to populate heatmaps for sample %lu/%lu\n",sampleID,db->numberOfSamples);
    //fprintf(stderr,"Gradient Size = %u\n",gradientSize);
    //fprintf(stderr," db->out8bit.pixels = %p \n", db->out8bit.pixels);
    //fprintf(stderr,"Outputs are : %hux%hu:%hu\n",db->out8bit.width,db->out8bit.height,db->out8bit.channels);

    if (sourceSampleID<db->numberOfSamples)
    {
     //Select correct gradient based on gradient Size
     int gradientSizeDefault = gradientSizeDefaultRAW;
     //TODO: Did i break this ?
     const signed char * defaultGradient         = find_heatmap_by_gradient_size(db->gradients,gradientSizeDefault); // HEATMAP
     const signed char * defaultPositiveGradient = find_positive_heatmap_by_gradient_size(db->gradients,gradientSizeDefault); // HEATMAP
     int gradientSize = (int) gradientSizeDefault;


     //Calculate pointer addresses
     unsigned long allHeatmapsOfASampleSize = (unsigned long) db->out8bit.width * (unsigned long) db->out8bit.height * (unsigned long) db->out8bit.channels;
     signed char *heatmapPTR = (signed char*) db->out8bit.pixels + (allHeatmapsOfASampleSize * targetImageID);


     if ((char *) db->out8bit.pixels + allHeatmapsOfASampleSize >= (char *) db->out8bit.pixelsLimit)
     {
       fprintf(stderr," Stopping before reaching out of memory limit %p!\n", db->out8bit.pixelsLimit);
       exit(1);
     }

     // Flush all particular heatmaps for sourceSampleID
     //This is now done in cleanHeatmapsOfTargetSample
     //memset(heatmapPTR, (signed char) MINV, allHeatmapsOfASampleSize); // signed char range is MINV to 127

     if (db->pdb->sample[sourceSampleID].numberOfSkeletons > 0)
        {
         //unsigned short originalSampleWidth  = db->pdb->sample[sourceSampleID].width;
         //unsigned short originalSampleHeight = db->pdb->sample[sourceSampleID].height;

         float IORatioX = (float) db->out8bit.width  / db->in.width;
         float IORatioY = (float) db->out8bit.height / db->in.height;
         int fjID, jID, skID;

         unsigned short x,y;
         //fprintf(stderr,"Sample %lu %s (%hux%hu)",sourceSampleID,db->pdb->sample[sourceSampleID].imagePath,originalSampleWidth,originalSampleHeight);
         //fprintf(stderr," %hu keypoints",db->pdb->keypointsForEachSample);
         //fprintf(stderr," %hu skeletons\n",db->pdb->sample[sourceSampleID].numberOfSkeletons);
         for (fjID = 0; fjID < db->pdb->keypointsForEachSample; fjID++)
         {
            jID = horizontalJointFlip(fjID,doLRFlip);

            //Take care of the appropriate gradient size for the specific joint
            gradientSize = (int) gradientSizeDefault;
            const signed char * gradient = defaultGradient;
            if (db->pdb->joint[jID].jointDifficulty != 0) //Update gradient for more/less difficult case
                 {
                     //Update gradient size accordingly to the difficulty
                     gradientSize = (int) gradientSizeDefault + (int) db->pdb->joint[jID].jointDifficulty;
                     gradient     = find_heatmap_by_gradient_size(db->gradients,gradientSize); // HEATMAP
                 }


            for (skID = 0; skID < db->pdb->sample[sourceSampleID].numberOfSkeletons; skID++)
            {
                x = db->pdb->sample[sourceSampleID].sk[skID].coords[jID * 3 + 0];
                y = db->pdb->sample[sourceSampleID].sk[skID].coords[jID * 3 + 1];

                if ((x!=0) && (y!=0))
                {
                 //fprintf(stderr,"KP %u/%hu / SK %u/%hu ",jID,db->pdb->keypointsForEachSample,skID,db->pdb->sample[sourceSampleID].numberOfSkeletons);
                 //fprintf(stderr,"OFFSET %0.2f/%0.2f / SCALE %0.2f/%0.2f ",db->pdb->sample[sourceSampleID].offsetX,db->pdb->sample[sourceSampleID].offsetY,db->pdb->sample[sourceSampleID].scaleX,db->pdb->sample[sampleID].scaleY);
                 //fprintf(stderr," @  %hu,%u ",x,y);
                 float xF = (float) x;
                 float yF = (float) y;
                 transformCoordinatesPanAndZoom(&xF,&yF,(float) pan_x,(float) pan_y,zoom_factor, originalInputWidth, originalInputHeight);
                 xF =  (offsetX + (xF * scaleX)) * IORatioX;
                 yF =  (offsetY + (yF * scaleY)) * IORatioY;
                 //signed + floorf: a joint just off the left/top edge still splats its visible part
                 //(an unsigned cast wrapped negatives far off-canvas, only right/bottom ones kept it)
                 int px = (int) floorf(xF);
                 int py = (int) floorf(yF);
                 //fprintf(stderr," ->  %d,%d \n",px,py);

                 //X coordinates need to be also horizontally flipped! (pixel i -> W-1-i, as flipImageHoriz_8bit)
                 if (doLRFlip) { px = db->out8bit.width - 1 - px; }
                 //--------------------------------------------

                 if (gradientSize>MINIMUM_GRADIENT_SIZE)
                 {
                 // Iterate over the gradient matrix and blend with existing values in heatmap
                 signed int xG, yG, newX, newY, blendValue, existingValueIsHigher, gradientSizeX2 = gradientSize*2;
                 for (xG=0; xG<gradientSizeX2; xG++)
                  {
                    for (yG=0; yG<gradientSizeX2; yG++)
                    {
                        newX = px + xG - gradientSize;
                        newY = py + yG - gradientSize;
                        if (newY >= 0 && newY < db->out8bit.height && newX >= 0 && newX < db->out8bit.width)
                        {
                            //-------------------------------------------------------------------------------------------------------------------------
                            //Get Back pointers / values
                            //channel fjID: on a flip it receives the mirrored joint jID (left <-> right), like the PAFs
                            signed char * heatmapJIDPTR     = heatmapPTR + ((newY * db->out8bit.width * db->out8bit.channels) + (newX* db->out8bit.channels) + fjID);
                            const signed char gradientValue = gradient[ (yG * gradientSizeX2) + xG];
                            //Jumpless overflow check
                            existingValueIsHigher = (gradientValue < *heatmapJIDPTR);
                            blendValue  = ((existingValueIsHigher) * (*heatmapJIDPTR)) + (!(existingValueIsHigher) * gradientValue);
                            //fprintf(stderr,"xG,yG [%d,%d]=%d |  x,y [%d,%d]=%d -> %d\n",xG,yG,gradientValue,newX,newY,*heatmapJIDPTR,blendValue);
                            * heatmapJIDPTR = (signed char) blendValue;

                            if (db->addBackgroundHeatmap)
                            {
                             signed char * heatmapBKGPTR     = heatmapPTR + ((newY * db->out8bit.width * db->out8bit.channels) + (newX* db->out8bit.channels) + db->backgroundHeatmapIndex); //<- TODO: 17 is fixed
                             existingValueIsHigher = (gradientValue < *heatmapBKGPTR);
                             blendValue  = ((existingValueIsHigher) * (*heatmapBKGPTR)) + (!(existingValueIsHigher) * gradientValue);
                             *heatmapBKGPTR = (signed char)  blendValue;
                            }
                           //-------------------------------------------------------------------------------------------------------------------------
                        } // Valid point
                    } // yG loop
                 } // xG loop

                 } else//Gradient Size is big enough!
                 {
                   fprintf(stderr,"Gradient size %u requested, this should never happen! \n",gradientSize);
                   abort();
                 }

               }// x,y != 0,0

            } //skID loop
        }//jID loop

        //Draw skelington with lines :)
        /* Disabled
        drawSkeletonsOnHeatmap(
                               db,
                               heatmapPTR,
                               17,
                               sourceSampleID,
                               5,
                               doLRFlip,
                               MAXV, MINV,
                               originalInputWidth,
                               originalInputHeight,
                               zoom_factor,
                               pan_x, pan_y,
                               offsetX, offsetY,
                               scaleX, scaleY
                              );*/


        if (db->addPAFHeatmap)
        {
            drawPAFsOnHeatmaps(
                               db,
                               heatmapPTR,
                               db->PAFHeatmapIndex,
                               numberOfPAFJoints,
                               PAFJoints,
                               sourceSampleID,
                               PAFSize,
                               doLRFlip,
                               MAXV, MINV,
                               originalInputWidth,
                               originalInputHeight,
                               zoom_factor,
                               pan_x, pan_y,
                               offsetX, offsetY,
                               scaleX, scaleY
                              );
        } //Done adding heatmaps


        if (db->addInstanceHeatmaps)
        {
            drawInstanceCentersOnHeatmaps(
                               db,
                               heatmapPTR,
                               db->instanceHeatmapIndex,
                               sourceSampleID,
                               doLRFlip,
                               defaultGradient,
                               gradientSizeDefault,
                               INSTANCE_SIZE_DISK_RADIUS,
                               originalInputWidth,
                               originalInputHeight,
                               zoom_factor,
                               pan_x, pan_y,
                               offsetX, offsetY,
                               scaleX, scaleY
                              );
        } //Done adding instance center+size targets


        #if ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT
              drawSingleLRHeatmapGaussian(
                                           db,
                                           heatmapPTR,
                                           db->leftRightHeatmapIndex,
                                           1,
                                           sourceSampleID,
                                           doLRFlip,
                                           gradientSizeDefault,
                                           originalInputWidth,
                                           originalInputHeight,
                                           zoom_factor,
                                           pan_x,
                                           pan_y,
                                           offsetX,
                                           offsetY,
                                           scaleX,
                                           scaleY
                                         );

              drawSingleLRHeatmapGaussian(
                                           db,
                                           heatmapPTR,
                                           db->leftRightHeatmapIndex+1,
                                           0,
                                           sourceSampleID,
                                           doLRFlip,
                                           gradientSizeDefault,
                                           originalInputWidth,
                                           originalInputHeight,
                                           zoom_factor,
                                           pan_x,
                                           pan_y,
                                           offsetX,
                                           offsetY,
                                           scaleX,
                                           scaleY
                                         );
       #endif

    } //We have skeletons in the image


    // SuperPoint keypoints are independent of human skeletons — render them for
    // every sample (even background-only ones) when the modality is enabled.
    if (db->addSuperpointHeatmaps && db->superpointChannels > 0)
    {
        drawSuperpointOnHeatmaps(
                               db,
                               heatmapPTR,
                               db->superpointHeatmapIndex,
                               sourceSampleID,
                               doLRFlip,
                               (int) gradientSizeDefaultRAW,
                               originalInputWidth,
                               originalInputHeight,
                               zoom_factor,
                               pan_x, pan_y,
                               offsetX, offsetY,
                               scaleX, scaleY
                              );
    } //Done adding SuperPoint heatmaps


    if (db->addBackgroundHeatmap)
    {
    //WE do flips even if there are no humans (i.e. only background)
      #if FLIP_ODD_HEATMAPS
        //Flip BKG heatmap — yG outer for row-major cache locality
        signed int xG, yG,hm;
        for (yG=0; yG<db->out8bit.height; yG++)
                  {
                    for (xG=0; xG<db->out8bit.width; xG++)
                    {
                        for (hm=0; hm < db->backgroundHeatmapIndex; hm+=2)
                        {
                         signed char * heatmapBKGPTR = heatmapPTR + ((yG * db->out8bit.width * db->out8bit.channels) + (xG* db->out8bit.channels) + hm);
                         signed int avoidOverflow    = (signed int) MAXV - (signed int) *heatmapBKGPTR;
                         *heatmapBKGPTR              = (signed char)  avoidOverflow;
                        }
                    }
                  }
       #else
        //Flip only BKG heatmap — yG outer for row-major cache locality
        signed int xG, yG;
        for (yG=0; yG<db->out8bit.height; yG++)
                  {
                    for (xG=0; xG<db->out8bit.width; xG++)
                    {
                        signed char * heatmapBKGPTR = heatmapPTR + ((yG * db->out8bit.width * db->out8bit.channels) + (xG* db->out8bit.channels) + db->backgroundHeatmapIndex);
                        signed int avoidOverflow    = (signed int) MAXV - (signed int) *heatmapBKGPTR;
                        *heatmapBKGPTR              = (signed char)  avoidOverflow;
                    }
                  }
       #endif
     } //Add background

     return 1;

    }
    return 0;
}
