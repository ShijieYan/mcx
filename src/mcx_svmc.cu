/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2024
**
**  \section sref Reference
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section sformat Formatting
**          Please always run "make pretty" inside the \c src folder before each commit.
**          The above command requires \c astyle to perform automatic formatting.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_svmc.cu

@brief    GPU kernel for volume preprocessing for SVMC simulations(Yan2020).
*******************************************************************************/

#include "mcx_svmc.h"

#include <stdint.h>

#include "mcx_tictoc.h"
#include "mcx_const.h"
#include "mcx_vector_math.cu"

// host function signatures
void pad_replicate_volume(unsigned int* vol, unsigned int** new_vol, unsigned int pad_size, unsigned int dimx,
                          unsigned int dimy, unsigned int dimz);
void gaussian_filter(float** filter, unsigned int sizex, unsigned int sizey, unsigned int sizez, float std);
void mcx_cu_assess(cudaError_t cuerr, const char* file, const int linenum);

// device function signatures
__global__ void init_lower_label(unsigned char* vol_new, unsigned int* vol);
__global__ void create_binary_mask(unsigned int* vol, float* binary_mask, unsigned int label);
__global__ void gaussian_blur(float* binary_mask, float* mask);
__global__ void split_voxel(float* scalar_field, unsigned char* new_vol, unsigned int label);
__device__ float3 interpolate(float3 a, float3 b, float a_val, float b_val, float isovalue);
__device__ unsigned int flatten_3d_to_1d(uint3 idx3d, uint3 dim);

/**
 * @brief      macro to report CUDA errors
 */
#define CUDA_ASSERT(a)      mcx_cu_assess((a),__FILE__,__LINE__)

/**
 * gaussian filter parameters
 */
#define MCX_SVMC_GKERNEL_SIZE 3U    // must be an odd positive integer
#define MCX_SVMC_GKERNEL_STD  1.0f
#define MCX_SVMC_ISOVALUE     0.5f

/**
 * Indices of the vertices in the local coordinate system
 */
__constant__ uint3 cube_vertices_local[8] = {
    {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
    {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}
};

/**
 * gaussian filter for smoothing binary volume
 */
__constant__ float gfilter[MCX_SVMC_GKERNEL_SIZE * MCX_SVMC_GKERNEL_SIZE * MCX_SVMC_GKERNEL_SIZE];


/**
 * edge[i] joins edge_vertices[i][0] and edge_vertices[i][1]
 * adapted from https://paulbourke.net/geometry/polygonise/
 */
__constant__ uint8_t edge_vertices[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {0, 3}, {4, 5}, {5, 6}, {6, 7}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

/**
 * (edge_intersections[i] >> j) & 1 is 1 if isosurface intersects edge[j], where 0 <= j <= 11
 * adapted from https://paulbourke.net/geometry/polygonise/
 */
__constant__ uint16_t edge_intersections[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

/**
 * In ith (0-255) configuration, triangle_vertices[i][j*3], triangle_vertices[i][j*3+1]
 * and triangle_vertices[i][j*3+2] are vertices of the jth triangle
 * adapted from https://paulbourke.net/geometry/polygonise/
 */
__constant__ int8_t triangle_vertices[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

/**
 * @brief      Preprocess media volume for SVMC simulation
 *
 * @param      cfg   Simulation Configuration
 * @param      gpu   GPU info
 */
void mcx_svmc_preprocess(Config* cfg, GPUInfo* gpu) {
    // if volume is not an 3D integer array, do nothing
    if (cfg->mediabyte > 4) {
        return;
    }

    // start timer
    MCX_FPRINTF(cfg->flog, "Preprocessing volume for SVMC simulation... \t");
    unsigned int tic = StartTimer();

    // activate a GPU
    int gpuid = cfg->deviceid[0] - 1;
    CUDA_ASSERT(cudaSetDevice(gpuid));

    // dimension of the volume and the padded volume
    uint3 vol_dim = cfg->dim;
    unsigned int pad_size = MCX_SVMC_GKERNEL_SIZE / 2;
    uint3 vol_padded_dim = make_uint3(vol_dim.x + 2 * pad_size, vol_dim.y + 2 * pad_size, vol_dim.z + 2 * pad_size);
    unsigned long long vol_length = vol_dim.x * vol_dim.y * vol_dim.z;
    unsigned long long vol_padded_length = vol_padded_dim.x * vol_padded_dim.y * vol_padded_dim.z;

    // pad 3D volume for filtering
    unsigned int* vol_padded = NULL;
    pad_replicate_volume(cfg->vol, &vol_padded, pad_size, vol_dim.x, vol_dim.y, vol_dim.z);

    // upload padded 3D volume to GPU
    unsigned int* gvol_padded = NULL;
    CUDA_ASSERT(cudaMalloc((void**)&gvol_padded, sizeof(unsigned int) * vol_padded_length));
    CUDA_ASSERT(cudaMemcpy(gvol_padded, vol_padded, sizeof(unsigned int) * vol_padded_length, cudaMemcpyHostToDevice));

    // generate gaussian filter and upload it to constant memory
    float* filter = NULL;
    gaussian_filter(&filter, MCX_SVMC_GKERNEL_SIZE, MCX_SVMC_GKERNEL_SIZE, MCX_SVMC_GKERNEL_SIZE, MCX_SVMC_GKERNEL_STD);
    CUDA_ASSERT(cudaMemcpyToSymbol(gfilter, filter,
                                   sizeof(float) * MCX_SVMC_GKERNEL_SIZE * MCX_SVMC_GKERNEL_SIZE * MCX_SVMC_GKERNEL_SIZE,
                                   0, cudaMemcpyHostToDevice));

    // allocate global memory buffers
    float* gbinary_mask = NULL;
    float* gmask = NULL;
    unsigned int* gvol = NULL;
    unsigned char* gvol_new = NULL;
    CUDA_ASSERT(cudaMalloc((void**)&gbinary_mask, sizeof(float) * vol_padded_length));
    CUDA_ASSERT(cudaMalloc((void**)&gmask, sizeof(float) * vol_length));
    CUDA_ASSERT(cudaMalloc((void**)&gvol, sizeof(unsigned int) * vol_length));
    CUDA_ASSERT(cudaMalloc((void**)&gvol_new, sizeof(unsigned int) * vol_length * 2));

    // copy old vol
    CUDA_ASSERT(cudaMemcpy(gvol, cfg->vol, sizeof(unsigned int) * vol_length, cudaMemcpyHostToDevice));

    // init new vol to 0
    CUDA_ASSERT(cudaMemset((void*)gvol_new, 0, sizeof(unsigned int) * vol_length * 2));

    // iterate over all medium labels
    dim3 grid, block;

    // init lower label of the gvol_new
    grid = dim3(vol_dim.x, vol_dim.y, vol_dim.z);
    block = dim3(1, 1, 1);
    init_lower_label <<< grid, block>>>(gvol_new, gvol);
    cudaDeviceSynchronize();

    for (unsigned int label = 0; label < cfg->medianum; ++label) {
        // generate binary mask for each label
        grid = dim3(vol_padded_dim.x, vol_padded_dim.y, vol_padded_dim.z);
        block = dim3(1, 1, 1);
        create_binary_mask <<< grid, block>>>(gvol_padded, gbinary_mask, label);
        cudaDeviceSynchronize();

        // smooth the binary mask using gaussian blur
        grid = dim3(vol_dim.x, vol_dim.y, vol_dim.z);
        block = dim3(1, 1, 1);
        gaussian_blur <<< grid, block>>>(gbinary_mask, gmask);
        cudaDeviceSynchronize();

        // generate intra-voxel boundary surface using marching cube algorithm
        grid = dim3(vol_dim.x - 1, vol_dim.y - 1, vol_dim.z - 1);
        block = dim3(1, 1, 1);
        split_voxel <<< grid, block>>>(gmask, gvol_new, label);
        cudaDeviceSynchronize();
    }

    // report elapsed time
    MCX_FPRINTF(cfg->flog, "complete:  \t%d ms\n", GetTimeMillis() - tic);

    // download new volume and overwrite the old volume
    cfg->vol = (unsigned int*)malloc(sizeof(unsigned int) * vol_length * 2);
    CUDA_ASSERT(cudaMemcpy(cfg->vol, gvol_new, sizeof(unsigned int) * vol_length * 2, cudaMemcpyDeviceToHost));

    // enable svmc mode
    cfg->mediabyte = MEDIA_2LABEL_SPLIT;

    // adjust source position to compensate for the grid offset between mcx and svmc
    cfg->srcpos.x += 0.5f;
    cfg->srcpos.y += 0.5f;
    cfg->srcpos.z += 0.5f;

    // TODO: adjust detector position to compensate for the grid offset between mcx and svmc

    // add detector mask
    mcx_maskdet(cfg);

    // clear
    if (vol_padded) {
        free(vol_padded);
    }

    if (filter) {
        free(filter);
    }

    if (gvol_padded) {
        CUDA_ASSERT(cudaFree(gvol_padded));
    }

    if (gbinary_mask) {
        CUDA_ASSERT(cudaFree(gbinary_mask));
    }

    if (gmask) {
        CUDA_ASSERT(cudaFree(gmask));
    }

    if (gvol) {
        CUDA_ASSERT(cudaFree(gvol));
    }

    if (gvol_new) {
        CUDA_ASSERT(cudaFree(gvol_new));
    }
}

/**
 * @brief      Pad a 3D volume by replicating values
 *
 * @param      vol       The volume
 * @param      new_vol   The new volume
 * @param[in]  pad_size  The pad size
 * @param[in]  dimx      The dimx
 * @param[in]  dimy      The dimy
 * @param[in]  dimz      The dimz
 */
void pad_replicate_volume(unsigned int* vol, unsigned int** new_vol, unsigned int pad_size,
                          unsigned int dimx, unsigned int dimy, unsigned int dimz) {
    unsigned int new_dimx = dimx + pad_size * 2;
    unsigned int new_dimy = dimy + pad_size * 2;
    unsigned int new_dimz = dimz + pad_size * 2;
    *new_vol = (unsigned int*)calloc(new_dimx * new_dimy * new_dimz, sizeof(unsigned int));

    // copy vol values
    for (unsigned int i = 0; i < dimx; ++i) {
        for (unsigned int j = 0; j < dimy; ++j) {
            for (unsigned int k = 0; k < dimz; ++k) {
                (*new_vol)[pad_size + i + (pad_size + j) * new_dimx + (pad_size + k) * new_dimx * new_dimy] =
                    vol[i + j * dimx + k * dimx * dimy];
            }
        }
    }

    // pad along -x
    for (int i = static_cast<int>(pad_size) - 1; i >= 0; --i) {
        for (unsigned int j = 0; j < new_dimy; ++j) {
            for (unsigned int k = 0; k < new_dimz; ++k) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[(i + 1) + j * new_dimx + k * new_dimx * new_dimy];
            }
        }
    }

    // pad along +x
    for (unsigned int i = new_dimx - pad_size; i < new_dimx; ++i) {
        for (unsigned int j = 0; j < new_dimy; ++j) {
            for (unsigned int k = 0; k < new_dimz; ++k) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[(i - 1) + j * new_dimx + k * new_dimx * new_dimy];
            }
        }
    }

    // pad along -y
    for (int j = static_cast<int>(pad_size) - 1; j >= 0; --j) {
        for (unsigned int i = 0; i < new_dimx; ++i) {
            for (unsigned int k = 0; k < new_dimz; ++k) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[i + (j + 1) * new_dimx + k * new_dimx * new_dimy];
            }
        }
    }

    // pad along +y
    for (unsigned int j = new_dimy - pad_size; j < new_dimy; ++j) {
        for (unsigned int i = 0; i < new_dimx; ++i) {
            for (unsigned int k = 0; k < new_dimz; ++k) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[i + (j - 1) * new_dimx + k * new_dimx * new_dimy];
            }
        }
    }


    // pad along -z
    for (int k = static_cast<int>(pad_size) - 1; k >= 0; --k) {
        for (unsigned int i = 0; i < new_dimx; ++i) {
            for (unsigned int j = 0; j < new_dimy; ++j) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[i + j * new_dimx + (k + 1) * new_dimx * new_dimy];
            }
        }
    }

    // pad along +z
    for (unsigned int k = new_dimz - pad_size; k < new_dimz; ++k) {
        for (unsigned int i = 0; i < new_dimx; ++i) {
            for (unsigned int j = 0; j < new_dimy; ++j) {
                (*new_vol)[i + j * new_dimx + k * new_dimx * new_dimy] =
                    (*new_vol)[i + j * new_dimx + (k - 1) * new_dimx * new_dimy];
            }
        }
    }
}

/**
 * @brief      Create a 3-D gaussian filter kernel
 *
 * @param      filter  The gaussian fileter kernel
 * @param[in]  sizex   The x dimension
 * @param[in]  sizey   The y dimension
 * @param[in]  sizez   The z dimension
 * @param[in]  std     The standard deviation
 */
void gaussian_filter(float** filter, unsigned int sizex, unsigned int sizey,
                     unsigned int sizez, float std) {
    float sum = 0.0f;
    *filter = (float*)calloc(sizex * sizey * sizez, sizeof(float));

    for (unsigned int i = 0; i < sizex; ++i) {
        for (unsigned int j = 0; j < sizey; ++j) {
            for (unsigned int k = 0; k < sizez; ++k) {
                float x = (static_cast<float>(i) - (static_cast<float>(sizex) - 1.0f) / 2.0f);
                float y = (static_cast<float>(j) - (static_cast<float>(sizey) - 1.0f) / 2.0f);
                float z = (static_cast<float>(k) - (static_cast<float>(sizez) - 1.0f) / 2.0f);
                (*filter)[i + j * sizex + k * sizex * sizey] =
                    expf(-(x * x + y * y + z * z) / (2.0f * std * std));
                sum += (*filter)[i + j * sizex + k * sizex * sizey];
            }
        }
    }

    // normalization
    float factor = 1.0f / sum;

    for (unsigned int i = 0; i < sizex; ++i) {
        for (unsigned int j = 0; j < sizey; ++j) {
            for (unsigned int k = 0; k < sizez; ++k) {
                (*filter)[i + j * sizex + k * sizex * sizey] *= factor;
            }
        }
    }
}

/**
 * @brief      Initialize the lower label.
 *
 * @param      vol_new  The new volume
 * @param      vol      The old volume
 */
__global__ void init_lower_label(unsigned char* vol_new, unsigned int* vol) {
    unsigned int idx1d = flatten_3d_to_1d(blockIdx, gridDim);
    vol_new[idx1d * sizeof(unsigned int) + 0] = vol[idx1d] & MED_MASK; // bytes[7]
}

/**
 * @brief      Creates a binary mask for the specified label
 *
 * @param      vol          The volume
 * @param      binary_mask  The binary mask
 * @param[in]  label        The label
 */
__global__ void create_binary_mask(unsigned int* vol, float* binary_mask, unsigned int label) {
    unsigned int idx1d = flatten_3d_to_1d(blockIdx, gridDim);
    binary_mask[idx1d] = ((vol[idx1d] & MED_MASK) == label ? 1.0f : 0.0f);
}

/**
 * @brief      Perform gaussian smoothing
 *
 * @param      binary_mask  The binary mask
 * @param      mask         The mask after gaussian smoothing
 */
__global__ void gaussian_blur(float* binary_mask, float* mask) {
    unsigned int pad_size = MCX_SVMC_GKERNEL_SIZE / 2;
    float temp = 0.0f;

    for (unsigned int i = 0; i < MCX_SVMC_GKERNEL_SIZE; ++i) {
        for (unsigned int j = 0; j < MCX_SVMC_GKERNEL_SIZE; ++j) {
            for (unsigned int k = 0; k < MCX_SVMC_GKERNEL_SIZE; ++k) {
                temp += binary_mask[flatten_3d_to_1d(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k),
                                                                                            make_uint3(gridDim.x + pad_size * 2, gridDim.y + pad_size * 2, gridDim.z + pad_size * 2))] *
                        gfilter[flatten_3d_to_1d(make_uint3(MCX_SVMC_GKERNEL_SIZE - 1 - i, MCX_SVMC_GKERNEL_SIZE - 1 - j, MCX_SVMC_GKERNEL_SIZE - 1 - k),
                                                                                                   make_uint3(MCX_SVMC_GKERNEL_SIZE, MCX_SVMC_GKERNEL_SIZE, MCX_SVMC_GKERNEL_SIZE))];
            }
        }
    }

    mask[flatten_3d_to_1d(blockIdx, gridDim)] = temp;
}

// Extract isosurface and get the new volume for svmc simulation
__global__ void split_voxel(float* scalar_field, unsigned char* new_vol, unsigned int label) {
    // grid idx3d
    uint3 cube_idx3d = blockIdx + make_uint3(1, 1, 1);

    // vol dimension
    uint3 vol_dim = gridDim + make_uint3(1, 1, 1);

    // get index of the polygon configurations (0 - 255)
    float cube_values[8];
    unsigned char cube_index = 0;

    for (unsigned int i = 0; i < 8; ++i) {
        cube_values[i] = scalar_field[flatten_3d_to_1d(blockIdx + cube_vertices_local[i], vol_dim)];

        if (cube_values[i] < MCX_SVMC_ISOVALUE) {
            cube_index |= (1 << i);
        }
    }

    // if the voxel does not need to be split, terminate
    if (cube_index == 0 || cube_index == 0xFF) {
        return;
    }

    // get intersections (in the local coordinate of the grid) between isosurface and cube edges
    uint16_t edge_intersection_mask = edge_intersections[cube_index];
    float3 isosurface_vertices[12];

    for (unsigned int i = 0; i < 12; ++i) {
        if (edge_intersection_mask & 1) {
            // mcx volume lower corner is [0 0 0] while svmc volume lower corner is [1 1 1].
            // It is because the medium type is defined at the grid vertices.
            isosurface_vertices[i] = interpolate(make_float3(cube_vertices_local[edge_vertices[i][0]]),
                                                 make_float3(cube_vertices_local[edge_vertices[i][1]]),
                                                 cube_values[edge_vertices[i][0]],
                                                 cube_values[edge_vertices[i][1]],
                                                 MCX_SVMC_ISOVALUE);
        }

        edge_intersection_mask >>= 1;
    }

    // get isosurface triangles (in the local coordinate of the grid)
    float3 isosurface_centroid = make_float3(0.0f, 0.0f, 0.0f);
    float3 isosurface_normal = make_float3(0.0f, 0.0f, 0.0f);
    float isosurface_area = 0.0f;

    for (unsigned int i = 0; triangle_vertices[cube_index][i] != -1; i += 3) {
        // get a triangle
        float3& A = isosurface_vertices[triangle_vertices[cube_index][i]];
        float3& B = isosurface_vertices[triangle_vertices[cube_index][i + 1]];
        float3& C = isosurface_vertices[triangle_vertices[cube_index][i + 2]];
        float3 AB_x_AC = cross(B - A, C - A);
        float triangle_area = 0.5f * length(AB_x_AC);
        float3 triangle_normal = 0.5f * AB_x_AC / triangle_area; // AB_x_AC / length(AB_x_AC)
        float3 triangle_centroid = 1.0f / 3.0f * (isosurface_vertices[triangle_vertices[cube_index][i]] +
                                   isosurface_vertices[triangle_vertices[cube_index][i + 1]] +
                                   isosurface_vertices[triangle_vertices[cube_index][i + 2]]);

        // compress triangle information if mulitple triangles are present
        isosurface_area += triangle_area;
        isosurface_normal += triangle_area * triangle_normal;
        isosurface_centroid += triangle_area * triangle_centroid;
    }

    isosurface_normal = -isosurface_normal / isosurface_area;
    isosurface_centroid /= isosurface_area;

    // 1D index of the current grid
    unsigned int idx1d = flatten_3d_to_1d(cube_idx3d, vol_dim);
    unsigned int vol_length = vol_dim.x * vol_dim.y * vol_dim.z;

    // check if we are processing for lower label or upper label
    unsigned int* temp = (unsigned int*)new_vol;

    if (temp[idx1d + vol_length]) { // if we have already init isosurface normal, bytes[3-0] must not be zero
        // update upper volume and return
        new_vol[idx1d * sizeof(unsigned int) + 1] = label; // bytes[6]
        // printf("Grid #[%u %u %u], [%u %u %u %u %u %u %u %u]\n", cube_idx3d.x + 1, cube_idx3d.y + 1, cube_idx3d.z + 1,
        //     new_vol[idx1d * sizeof(unsigned int) + 0],
        //     new_vol[idx1d * sizeof(unsigned int) + 1],
        //     new_vol[idx1d * sizeof(unsigned int) + 2],
        //     new_vol[idx1d * sizeof(unsigned int) + 3],
        //     new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 0],
        //     new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 1],
        //     new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 2],
        //     new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 3]);
        return;
    }

    // update lower label
    new_vol[idx1d * sizeof(unsigned int) + 0] = label; // bytes[7]

    // convert float vectors to gray-scale vectors (0-255) and update the new volume
    new_vol[idx1d * sizeof(unsigned int) + 2] = (unsigned char)(isosurface_centroid.x * 255.0f); // bytes[5]
    new_vol[idx1d * sizeof(unsigned int) + 3] = (unsigned char)(isosurface_centroid.y * 255.0f); // bytes[4]
    new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 0] = (unsigned char)(isosurface_centroid.z * 255.0f); // bytes[3]
    new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 1] = min((unsigned char)floorf((isosurface_normal.x + 1.0f) * 255.0f * 0.5f), 254); // bytes[2]
    new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 2] = min((unsigned char)floorf((isosurface_normal.y + 1.0f) * 255.0f * 0.5f), 254); // bytes[1]
    new_vol[(idx1d + vol_length) * sizeof(unsigned int) + 3] = min((unsigned char)floorf((isosurface_normal.z + 1.0f) * 255.0f * 0.5f), 254); // bytes[0]
}

/**
 * @brief      In a 3-D cartesian coordinate,
 *             given the isovalue, compute the interpolated point between two points.
 *
 * @param[in]  a         first point
 * @param[in]  b         second point
 * @param[in]  a_val     value of first point
 * @param[in]  b_val     value of second point
 * @param[in]  isovalue  value of interpolated point
 *
 * @return     Interpolated position
 */
__device__ float3 interpolate(float3 a, float3 b, float a_val, float b_val, float isovalue) {
    return b_val == a_val ? a : (b + (a - b) * ((isovalue - b_val) / (a_val - b_val)));
}

/**
 * @brief      Convert 3D index to 1D index
 *
 * @param[in]  idx3d  3-D index
 * @param[in]  dim    The dimension of the 3-D matrix
 *
 * @return     1-D index
 */
__device__ unsigned int flatten_3d_to_1d(uint3 idx3d, uint3 dim) {
    return idx3d.x + idx3d.y * dim.x + idx3d.z * dim.x * dim.y;
}