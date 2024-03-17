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
\file    mcx_vector_math.cu

@brief    Common math operations on vector types.
*******************************************************************************/

/**
 * @brief Adding two float3 vectors c=a+b
 */

inline __device__ float3 operator +(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Increatment a float3 vector by another float3, a+=b
 */

inline __device__ void operator +=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

/**
 * @brief Subtracting two float3 vectors c=a+b
 */

inline __device__ float3 operator -(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


/**
 * @brief Negating a float3 vector c=-a
 */

inline __device__ float3 operator -(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

/**
 * @brief Front-multiplying a float3 with a scalar c=a*b
 */

inline __device__ float3 operator *(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

/**
 * @brief Post-multiplying a float3 with a scalar c=a*b
 */

inline __device__ float3 operator *(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Multiplying two float3 vectors c=a*b
 */

inline __device__ float3 operator *(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

/**
 * @brief Dot-product of two float3 vectors c=a*b
 */

inline __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Adding two uint3 vectors c=a+b
 */

inline __device__ uint3 operator +(const uint3& a, const uint3& b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Cast uint3 to float3
 */

inline __device__ float3 make_float3(const uint3& a) {
    return make_float3(float(a.x), float(a.y), float(a.z));
}


/**
 * @brief Cross-product of two float3 vectors c=axb
 */

inline __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/**
 * @brief Length of a float3 vector
 */

inline __device__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

/**
 * @brief Division between a float3 vector and a float
 */

inline __device__ float3 operator/(const float3& a, const float& b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

/**
 * @brief Divide a float3 vector by a float
 */

inline __device__ void operator/=(float3& a, const float& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}