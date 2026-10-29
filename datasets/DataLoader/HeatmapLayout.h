/** @file HeatmapLayout.h
 *  @brief  Single source of truth for the 8-bit heatmap output channel layout.
 *  @author Ammar Qammaz (AmmarkoV)
 *
 *  The list of output "modalities" (keypoints, PAFs, depth, normals, segmentation,
 *  …) and the exact channel range each occupies used to be restated by hand in
 *  several disconnected places (the index arithmetic in db_create, the
 *  configuration.h START_* macros, the Python loss/decode slices, …).  Keeping
 *  them in sync was error-prone.
 *
 *  This header makes the layout a *declared-then-resolved* table:
 *
 *      struct HeatmapLayout L;
 *      heatmapLayoutClear(&L);
 *      heatmapLayoutAddGroup(&L,"keypoints", 17, 1);            // always on
 *      heatmapLayoutAddGroup(&L,"paf",       12, addPAFs);      // conditional
 *      ...  one line per modality, in channel order  ...
 *      heatmapLayoutResolve(&L);                                // prefix-sum start indices
 *
 *      unsigned int start,count;
 *      if (heatmapLayoutGetRange(&L,"segmentation",&start,&count)) { ... }
 *
 *  Each *enabled* modality occupies the half-open channel range
 *  [start, start+count).  A *disabled* modality is still recorded (so the full
 *  catalogue can be enumerated / printed) but consumes no channels and is
 *  reported as absent by heatmapLayoutGetRange().  Adding a new modality is a
 *  single extra heatmapLayoutAddGroup() call in the right position — every
 *  derived index follows automatically.
 *
 *  Header-only (static inline): include it wherever the layout is needed; no
 *  extra translation unit / Makefile change required.
 */
#ifndef HEATMAP_LAYOUT_H_INCLUDED
#define HEATMAP_LAYOUT_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdio.h>

/** @brief Hard cap on the number of declared modalities. Bump if you ever exceed it. */
#define MAX_HEATMAP_GROUPS 24

/** @brief One output modality and the channel range it occupies. */
struct HeatmapGroup
{
   const char * name;     ///< Stable identifier (expects a string literal — static lifetime).
   unsigned int start;    ///< First channel index. Valid after heatmapLayoutResolve().
   unsigned int count;    ///< Number of channels (0 when disabled).
   int          enabled;  ///< Whether this modality is part of the output.
};

/** @brief Ordered table of modalities describing the full 8-bit heatmap output. */
struct HeatmapLayout
{
   struct HeatmapGroup group[MAX_HEATMAP_GROUPS];
   unsigned int numberOfGroups;
   unsigned int totalChannels;   ///< Sum of counts of enabled groups. Valid after resolve().
   int          resolved;        ///< 1 after heatmapLayoutResolve(), 0 if a group was added since.
};

/** @brief Reset a layout to empty. */
static inline void heatmapLayoutClear(struct HeatmapLayout * L)
{
   if (L==0) { return; }
   memset(L,0,sizeof(struct HeatmapLayout));
}

/** @brief Declare a modality (in channel order) and whether it is included.
 *  @param count    Channel width of this modality.
 *  @param enabled  Non-zero to include it in the output (a 0-width group is forced off).
 *  @return 1 on success, 0 on overflow / bad args. Marks the layout as needing resolve(). */
static inline int heatmapLayoutAddGroup(struct HeatmapLayout * L, const char * name, unsigned int count, int enabled)
{
   if ( (L==0) || (name==0) ) { return 0; }
   if (L->numberOfGroups >= MAX_HEATMAP_GROUPS)
   {
       fprintf(stderr,"heatmapLayoutAddGroup: too many groups for `%s`, raise MAX_HEATMAP_GROUPS\n",name);
       return 0;
   }
   struct HeatmapGroup * g = &L->group[L->numberOfGroups++];
   g->name    = name;
   g->count   = count;
   g->enabled = ( enabled && (count>0) ) ? 1 : 0;
   g->start   = 0;
   L->resolved = 0;
   return 1;
}

/** @brief Compute every group's start index (prefix-sum over enabled groups) and the total.
 *  @return Total number of channels occupied. */
static inline unsigned int heatmapLayoutResolve(struct HeatmapLayout * L)
{
   if (L==0) { return 0; }
   unsigned int cursor = 0;
   unsigned int i=0;
   for (i=0; i<L->numberOfGroups; i++)
   {
      // A disabled group keeps the cursor as its (empty) start so the field is
      // still meaningful ("where it would have begun") for legacy derivations.
      L->group[i].start = cursor;
      if (L->group[i].enabled) { cursor += L->group[i].count; }
      else                     { L->group[i].count = 0;        }
   }
   L->totalChannels = cursor;
   L->resolved      = 1;
   return cursor;
}

/** @brief Index of a group by name, or -1 if not declared. */
static inline int heatmapLayoutFindGroup(const struct HeatmapLayout * L, const char * name)
{
   if ( (L==0) || (name==0) ) { return -1; }
   unsigned int i=0;
   for (i=0; i<L->numberOfGroups; i++)
   {
      if ( (L->group[i].name!=0) && (strcmp(L->group[i].name,name)==0) ) { return (int)i; }
   }
   return -1;
}

/** @brief Channel range of an ENABLED modality by name.
 *  @return 1 and fills start/count (either may be NULL) if present and enabled; 0 otherwise. */
static inline int heatmapLayoutGetRange(const struct HeatmapLayout * L, const char * name, unsigned int * outStart, unsigned int * outCount)
{
   int idx = heatmapLayoutFindGroup(L,name);
   if (idx<0)                  { return 0; }
   if (!L->group[idx].enabled) { return 0; }
   if (outStart!=0) { *outStart = L->group[idx].start; }
   if (outCount!=0) { *outCount = L->group[idx].count; }
   return 1;
}

/** @brief Start index of a group whether or not it is enabled (-1 if not declared).
 *         Useful for legacy index fields that must be set even when the modality is off. */
static inline int heatmapLayoutGetStart(const struct HeatmapLayout * L, const char * name)
{
   int idx = heatmapLayoutFindGroup(L,name);
   return (idx<0) ? -1 : (int)L->group[idx].start;
}

/** @brief Pretty-print the resolved layout to stderr (for configuration sanity checks). */
static inline void heatmapLayoutPrint(const struct HeatmapLayout * L)
{
   if (L==0) { return; }
   fprintf(stderr,"\n8-bit heatmap channel layout (%u declared groups, %u channels total):\n",L->numberOfGroups,L->totalChannels);
   unsigned int i=0;
   for (i=0; i<L->numberOfGroups; i++)
   {
      if (L->group[i].enabled)
        { fprintf(stderr,"   [%2u..%2u] %-16s (%u ch)\n",L->group[i].start,L->group[i].start+L->group[i].count-1,L->group[i].name,L->group[i].count); }
      else
        { fprintf(stderr,"   [  off  ] %-16s\n",L->group[i].name); }
   }
   fprintf(stderr,"\n");
}

#ifdef __cplusplus
}
#endif

#endif // HEATMAP_LAYOUT_H_INCLUDED
