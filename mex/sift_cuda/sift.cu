#include <iostream>
#include <future>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>

#include <cuda_runtime.h>
#include <cmath>

#include "math.h"
#include "sift.h"
#include "matrix_helper.h"
#include "cuda_timer.h"

#include "spdlog/spdlog.h"

namespace cudautils {

template<typename T>
struct greater_tol : public thrust::binary_function<T, T, bool>
{
    // tolerance to compare doubles (15 decimal places)
    __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
        if (fabs(lhs - rhs) <= .000000000000001)
            return false;
        return lhs > rhs;
    }
};

struct is_negative {
    __host__ __device__ bool operator() (const long long a) const {
        return a < 0;
    }
};

struct isnan_test {
    __host__ __device__ bool operator() (const float a) const {
        return isnan(a);
    }
};

// row major order index into the descriptor vector
// note the descriptor vector length is determined by 
// sift_params.IndexSize ^ 3 * sift_params.nFaces
// this is why i, j, and k are dimensions of stride sift_params.IndexSize
__forceinline__ __device__ 
int bin_sub2ind_row(int i, int j, int k, uint16_t l, const cudautils::SiftParams sift_params) {
    return (int) l + sift_params.nFaces * (k + j * sift_params.IndexSize + i
            * pow(sift_params.IndexSize, 2));
}

// column major order index into the descriptor vector
// note the descriptor vector length is determined by 
// sift_params.IndexSize ^ 3 * sift_params.nFaces
// this is why i, j, and k are dimensions of stride sift_params.IndexSize
__forceinline__ __device__ 
int bin_sub2ind(int i, int j, int k, uint16_t l, const cudautils::SiftParams sift_params) {
    return (int) i + j * sift_params.IndexSize + k * pow(sift_params.IndexSize, 2)
        + l * pow(sift_params.IndexSize, 3);
}

// column major order index into the descriptor vector
// note the descriptor vector length is determined by 
// sift_params.IndexSize ^ 3 * sift_params.nFaces
// this is why i, j, and k are dimensions of stride sift_params.IndexSize
__global__
void bin_sub2ind_wrap(int i, int j, int k, uint16_t l, const cudautils::SiftParams sift_params, int* ind) {
    *ind = bin_sub2ind(i, j, k, l, sift_params);
    return ;
}

__forceinline__ __device__ 
void place_in_index(double* index, double mag, int i, int j, int k, 
        double* yy, uint16_t* ix, long long idx, const cudautils::SiftParams sift_params) {

    double tmpsum = 0.0;
    int bin_index;
    if (sift_params.Smooth_Flag) {
        for (int tessel=0; tessel < sift_params.Tessel_thresh; tessel++) {
            tmpsum += pow(yy[tessel], sift_params.Smooth_Var);
        }

        // Add three nearest tesselation faces
        for (int ii=0; ii < sift_params.Tessel_thresh; ii++) {
            bin_index = bin_sub2ind(i, j, k, ix[ii], sift_params);

#ifdef DEBUG_NUMERICAL
            printf("i%d j%d k%d ix[ii]%d bin_index%d yy[ii]%.54f, index+=%.54f, idx%lld\n",
                    i, j, k, ix[ii], bin_index, yy[ii], mag * pow(yy[ii],
                        sift_params.Smooth_Var ) / tmpsum, idx);
#endif

            index[bin_index] +=  mag * pow(yy[ii], sift_params.Smooth_Var ) / tmpsum;
        }
    } else {
        bin_index = bin_sub2ind(i, j, k, ix[0], sift_params);
        index[bin_index] += mag;
    }
    return;
}

// matrix multiply in row memory order 
// first is a matrix in row order
// second is the array multiply
// assumes length of second = cols of first
__forceinline__ __device__ 
void dot_product(double* first, double* second, double* out, int rows,
        int cols) {
    for (int i=0; i < rows; i++) {
        double sum = 0.0;
        for (int j=0; j < cols; j++) {
            sum += first[j + i * cols] * second[j];
        }
        out[i] = sum;
    }
}

// matrix multiply in row memory order 
// first is a matrix in row order
// second is the array multiply
// assumes length of second = cols of first
__global__
void dot_product_wrap(double* first, double* second, double* out, int rows,
        int cols) {
    dot_product(first, second, out, rows, cols);
    return;
}

// matrix multiply in col memory order 
// first is a matrix in column order
// second is the array multiply
// assumes length of second = cols of first
__forceinline__ __device__ 
void dot_product_col_ord(double* first, double* second, double* out, int rows,
        int cols) {
    for (int i=0; i < rows; i++) {
        double sum = 0.0;
        for (int j=0; j < cols; j++) {
            sum += first[i + j * rows] * second[j];
        }
        out[i] = sum;
    }
}

// assumes r,c,s lie within accessible image boundaries
__forceinline__ __device__ 
double get_grad_ori_vector(double* image, long long idx, unsigned int
        x_stride, unsigned int y_stride, int r, int c, int t, double vect[3],
        double* yy, uint16_t* ix, const cudautils::SiftParams sift_params,
        double* device_centers) {


    int last_row = sift_params.image_size0 - 1;
    int last_col = sift_params.image_size1 - 1;
    int last_slice = sift_params.image_size2 - 1;

    /* this is literal translation from Scovanner et al. 3DSIFT, 
       even though it seems xgrad and ygrad are switched, and ygrad seems to be
       in wrong direction. Protect edge cases explicitly rather than 
       by padding
    */
    double xgrad, ygrad, zgrad;
    if (c == 0) {
        xgrad = 2.0 * (image[idx + x_stride] - image[idx]);
    } else if (c == last_col) {
        xgrad = 2.0 * (image[idx] - image[idx - x_stride]);
    } else {
        xgrad = image[idx + x_stride] - image[idx - x_stride];
    }

    if (r == 0) {
        ygrad = 2.0 * (image[idx] - image[idx + 1]);
    } else if (r == last_row) {
        ygrad = 2.0 * (image[idx - 1] - image[idx]);
    } else {
        ygrad = image[idx - 1] - image[idx + 1];
    }

    if (t == 0) {
        zgrad = 2.0 * (image[idx + x_stride * y_stride] - image[idx]);
    } else if (t == last_slice) {
        zgrad = 2.0 * (image[idx] - image[idx - x_stride * y_stride]);
    } else {
        zgrad = image[idx + x_stride * y_stride] - image[idx - x_stride * y_stride];
    }

    double mag = sqrt(xgrad * xgrad + ygrad * ygrad + zgrad * zgrad);

    xgrad /= mag;
    ygrad /= mag;
    zgrad /= mag;

    if (mag != 0.0) {
        vect[0] = xgrad;
        vect[1] = ygrad;
        vect[2] = zgrad;
    } else {
        vect[0] = 1.0;
        vect[1] = 0.0;
        vect[2] = 0.0;
    }

    //Find the nearest tesselation face indices
    // N = sift_params.nFaces 
    int N = sift_params.fv_centers_len / DIMS;
    dot_product(device_centers, vect, yy, N, DIMS);

    // overwrite idxs 1 : N, N can not exceed the length of ori_hist
    thrust::sequence(thrust::device, ix, ix + sift_params.nFaces);
    thrust::stable_sort_by_key(thrust::device, yy, yy + sift_params.nFaces, ix, thrust::greater<double>());

#ifdef DEBUG_NUMERICAL
    printf("ggov N%d fv_len%d DIMS%d idx%lld vect0 %.4f vect1 %.4f vect2 %.4f image[idx] %.4f r%d c%d t%d yy %.4f %.4f %.4f %.4f ix %d %d %d %d eq:%d diff:%.54f\n",
        N, sift_params.fv_centers_len, DIMS, idx, vect[0], vect[1], vect[2],
        image[idx], r, c, t, yy[0], yy[1], yy[2], yy[3], ix[0], ix[1], ix[2],
        ix[3], yy[2] == yy[3], yy[2] - yy[3]);
    printf("fv[%d] %.4f %.4f %.4f\n", ix[0], device_centers[3 * ix[0]], device_centers[3 * ix[0] + 1], device_centers[3 * ix[0] + 2]);
    printf("fv[%d] %.4f %.4f %.4f\n", ix[1], device_centers[3 * ix[1]], device_centers[3 * ix[1] + 1], device_centers[3 * ix[1] + 2]);
    printf("fv[%d] %.4f %.4f %.4f\n", ix[2], device_centers[3 * ix[2]], device_centers[3 * ix[2] + 1], device_centers[3 * ix[2] + 2]);
    printf("fv[%d] %.4f %.4f %.4f\n", ix[3], device_centers[3 * ix[3]], device_centers[3 * ix[3] + 1], device_centers[3 * ix[3] + 2]);
#endif

    return mag;
}

__global__
void get_grad_ori_vector_wrap(double* image, long long idx, unsigned int
        x_stride, unsigned int y_stride, int r, int c, int t, double vect[3], double* yy, uint16_t* ix,
        const cudautils::SiftParams sift_params, double* device_centers, double* mag) {

    *mag = cudautils::get_grad_ori_vector(image, 
        idx, x_stride, y_stride, r, c, t, vect,
        yy, ix, sift_params, device_centers);
    return;
}

/*r, c, s is the pixel index (x, y, z dimensions respect.) in the image within the radius of the */
/*keypoint before clamped*/
/*For each pixel, take a neighborhhod of xyradius and tiradius,*/
/*bin it down to the sift_params.IndexSize dimensions*/
/*thus, i_indx, j_indx, s_indx represent the binned index within the radius of the keypoint*/
__forceinline__ __device__
void add_sample(double* index, double* image, double distsq, long long
        idx, unsigned int x_stride, unsigned int y_stride, int i_bin, int j_bin, int k_bin, 
        int r, int c, int t, const cudautils::SiftParams sift_params, double*
        device_centers, uint16_t* ix, double* yy) {

    double sigma = sift_params.SigmaScaled;
    double weight = exp(-(distsq / (2.0 * sigma * sigma)));

    double vect[3] = {0.0, 0.0, 0.0};

    // gradient and orientation vectors calculated from 3D halo/neighboring
    // pixels
    double mag = get_grad_ori_vector(image, idx, x_stride, y_stride, r, c, t,
            vect, yy, ix, sift_params, device_centers);
    mag *= weight; // scale magnitude by gaussian 

    place_in_index(index, mag, i_bin, j_bin, k_bin, yy, ix, idx, sift_params);
    return;
}


// floor quotient, add 1
// clamp bin idx to IndexSize
__forceinline__ __device__ 
int get_bin_idx(int orig, int radius, int IndexSize) {
    int idx = (int) floor((orig + radius) / (2.0 * (double) radius / IndexSize));
    if (idx >= IndexSize) // clamp to IndexSize
        idx = IndexSize - 1;
    return idx;
}

// floor quotient, add 1
// clamp bin idx to IndexSize
__global__
void get_bin_idx_wrap(int orig, int radius, int IndexSize, int* idx) {
    *idx = get_bin_idx(orig, radius, IndexSize);
    return;
}


__forceinline__ __device__
double* key_sample(const cudautils::SiftParams sift_params, 
        cudautils::Keypoint key, double* image, long long idx,
        unsigned int x_stride, unsigned int y_stride, 
        double* device_centers, uint16_t* ix, double* yy,
        double* index) {

    double xySpacing = (double) sift_params.xyScale * sift_params.MagFactor;
    double tSpacing = (double) sift_params.tScale * sift_params.MagFactor;

    int xyiradius = rint(1.414 * xySpacing * (sift_params.IndexSize + 1) / 2.0);
    int tiradius = rint(1.414 * tSpacing * (sift_params.IndexSize + 1) / 2.0);

    // Surrounding radius of pixels are binned for computation 
    // according to sift_params.IndexSize
    int r, c, t, i_bin, j_bin, k_bin;
    double distsq;
    long long update_idx;
    for (int i = -xyiradius; i <= xyiradius; i++) {
        for (int j = -xyiradius; j <= xyiradius; j++) {
            for (int k = -tiradius; k <= tiradius; k++) {

                distsq = (double) pow(i,2) + pow(j,2) + pow(k,2);

                // Find bin idx
                i_bin = get_bin_idx(i, xyiradius, sift_params.IndexSize);
                j_bin = get_bin_idx(j, xyiradius, sift_params.IndexSize);
                k_bin = get_bin_idx(k, tiradius, sift_params.IndexSize);
                
                // Find original image pixel idx
                r = key.x + i;
                c = key.y + j;
                t = key.z + k;

                // only add if within image range
                if (!(r < 0  ||  r >= sift_params.image_size0 ||
                        c < 0  ||  c >= sift_params.image_size1
                        || t < 0 || t >= sift_params.image_size2)) {

                    // image is assumed as column order
                    // make sure it isn't cast to unsigned
                    update_idx = (long long) idx + i + (int) x_stride * j +
                        (int) x_stride * (int) y_stride * k;
                    add_sample(index, image, distsq, update_idx, x_stride, y_stride,
                            i_bin, j_bin, k_bin, r, c, t, sift_params,
                            device_centers, ix, yy);
                }
            }
        }
    }

    return index;
}

__forceinline__ __device__
double* build_ori_hists(int x, int y, int z, long long idx, unsigned int
        x_stride, unsigned int y_stride, int radius, double* image, 
        const cudautils::SiftParams sift_params, double* device_centers,
        uint16_t* ix, double* yy, double* ori_hist) {

    double mag;
    double vect[3] = {0.0, 0.0, 0.0};

    int r, c, t;
    long long update_idx;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            for (int k = -radius; k <= radius; k++) {
                // Find original image pixel idx
                r = x + i;
                c = y + j;
                t = z + k;

                // only add if within image range
                // NOTE from original source
                // Do not use last row or column, which are not valid.
                if (!(r < 0 || r >= sift_params.image_size0 - 2 ||
                      c < 0 || c >= sift_params.image_size1 - 2 ||
                      t < 0 || t >= sift_params.image_size2 - 2)) {
                    // image is assumed as column order
                    // make sure it isn't cast to unsigned
                    update_idx = (long long) idx + i + (int) x_stride * j +
                        (int) x_stride * (int) y_stride * k;
                    /*gradient and orientation vectors calculated from 3D halo/neighboring pixels*/
                    mag = get_grad_ori_vector(image, update_idx, x_stride, y_stride,
                            r, c, t, vect, yy, ix, sift_params, device_centers);
                    ori_hist[ix[0]] += mag;
                }
            }
        }
    }
    return ori_hist;
}

__forceinline__ __device__
void normalize_arr(double* arr, int len) {

    double sqlen = 0.0;
    for (int i=0; i < len; i++) {
        sqlen += arr[i] * arr[i];
    }

    double fac = 1.0 / sqrt(sqlen);
    for (int i=0; i < len; i++) {
        arr[i] = arr[i] * fac;
    }
    return;
}

__forceinline__ __device__
cudautils::Keypoint make_keypoint_sample(cudautils::Keypoint key, double*
        image, const cudautils::SiftParams sift_params, unsigned int thread_idx, long long idx,
        unsigned int x_stride, unsigned int y_stride, double * descriptors,
        double* device_centers, uint16_t* ix, double* yy) {

    bool changed = false;

    // default N=640; 5120 bytes
    int N = sift_params.descriptor_len;
    double* index = &(descriptors[thread_idx * sift_params.descriptor_len]);
    memset(index, 0.0, N * sizeof(double));

    key_sample(sift_params, key, image, idx, x_stride, y_stride,
            device_centers, ix, yy, index);

#ifdef DEBUG_NUMERICAL
    for (int i=0; i < sift_params.descriptor_len; i++) {
        if (index[i] != 0) 
            printf("index[%d]=%.4f\n",i, index[i]);
    }
    printf("\n");
#endif

    normalize_arr(index, N);

    for (int i=0; i < N; i++) {
        if (index[i] > sift_params.MaxIndexVal) {
            index[i] = sift_params.MaxIndexVal;
            changed = true;
        }
    }

    if (changed) {
        normalize_arr(index, N);
    }

    int intval;
    for (int i=0; i < N; i++) {
        intval = rint(512.0 * index[i]);
        index[i] =  (double) min(255, intval);
    }
    return key;
}

__forceinline__ __device__
cudautils::Keypoint make_keypoint(double* image, int x, int y, int z,
        unsigned int thread_idx, long long idx, unsigned int x_stride, unsigned int y_stride,
        const cudautils::SiftParams sift_params, double * descriptors, double*
        device_centers, uint16_t* ix, double* yy) {
    cudautils::Keypoint key;
    key.x = x;
    key.y = y;
    key.z = z;

    return make_keypoint_sample(key, image, sift_params, thread_idx, idx,
            x_stride, y_stride, descriptors, device_centers, ix, yy);
}

/* Main function of 3DSIFT Program from http://www.cs.ucf.edu/~pscovann/
Inputs:
image - a 3 dimensional matrix of double
xyScale and tScale - affects both the scale and the resolution, these are
usually set to 1 and scaling is done before calling this function
x, y, and z - the location of the center of the keypoint where a descriptor is requested

Outputs:
keypoint - the descriptor, varies in size depending on values in LoadParams.m
reRun - a flag (0 or 1) which is set if the data at (x,y,z) is not
descriptive enough for a good keypoint
*/
__global__
void create_descriptor(
        unsigned int x_stride,
        unsigned int y_stride,
        unsigned int x_sub_start,
        unsigned int y_sub_start,
        unsigned int dw,
        const unsigned int map_idx_size,
        long long *map_idx,
        int8_t *map,
        double *image,
        const cudautils::SiftParams sift_params, 
        double* device_centers,
        double *descriptors,
        uint16_t* idx_scratch,
        double* yy_scratch,
        uint16_t* ori_idx_scratch,
        double* ori_scratch) {

    // thread per keypoint in this substream
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= map_idx_size) return;
    // map_idx holds the relevant image idxs only for the substream
    // map_idx_size matchs total # of threads
    // idx describes the linear index for current GPUs section of the image and corresponding map
    long long idx = map_idx[thread_idx];

    // column-major order since image is from matlab
    int x, y, z;
    unsigned int padding_x;
    unsigned int padding_y;
    unsigned int padding_z;
    ind2sub(x_stride, y_stride, idx, padding_x, padding_y, padding_z);
    // correct for dw_ padding, 0-indexed for checking boundaries
    x = x_sub_start + padding_x - dw;
    y = y_sub_start + padding_y - dw;
    z = padding_z - dw;
    
    uint16_t* ix = (uint16_t*) &(idx_scratch[thread_idx * sift_params.nFaces]);
    cudaCheckPtrDevice(ix);
    thrust::sequence(thrust::device, ix, ix + sift_params.nFaces);

    double *yy = (double*) &(yy_scratch[thread_idx * sift_params.nFaces]);
    cudaCheckPtrDevice(yy);

    if (sift_params.TwoPeak_Flag) {
        int radius = rint(sift_params.xyScale * 3.0);

        // init ori hist indices
        int ori_hist_len = sift_params.nFaces; //default 80
        uint16_t* ori_hist_idx = &(ori_idx_scratch[ori_hist_len * thread_idx]);
        cudaCheckPtrDevice(ori_hist_idx);
        thrust::sequence(thrust::device, ori_hist_idx, ori_hist_idx + ori_hist_len);

        //init ori histogram
        double* ori_hist = &(ori_scratch[ori_hist_len * thread_idx]);
        cudaCheckPtrDevice(ori_hist);
        memset(ori_hist, 0.0, ori_hist_len * sizeof(double));

        build_ori_hists(x, y, z, idx, x_stride, y_stride, radius, image,
                sift_params, device_centers, ix, yy, ori_hist);
        // descending order according to ori_hist
        thrust::stable_sort_by_key(thrust::device, ori_hist, ori_hist +
                ori_hist_len, ori_hist_idx, thrust::greater<double>());
            
        double prod01, prod02;
        dot_product(&(device_centers[DIMS * ori_hist_idx[0]]),
            &(device_centers[DIMS * ori_hist_idx[1]]), &prod01, 1, DIMS);
        dot_product(&(device_centers[DIMS * ori_hist_idx[0]]),
            &(device_centers[DIMS * ori_hist_idx[2]]), &prod02, 1, DIMS);

#ifdef DEBUG_NUMERICAL
        printf("TPF x%d y%d z%d ori_hist %.25f %.25f %.25f ori_hist_idx %d %d %d %d prod01 %.25f prod02 %.25f eq:%d diff:%.54f\n",
                x, y, z, ori_hist[0], ori_hist[1], ori_hist[2], ori_hist_idx[0], ori_hist_idx[1], ori_hist_idx[2], ori_hist_idx[3],
                prod01, prod02, ori_hist[2] == ori_hist[3], ori_hist[2] - ori_hist[3]);
#endif

        if ( ( prod01 > sift_params.TwoPeak_Thresh) &&
             ( prod02 > sift_params.TwoPeak_Thresh) ) {
            // mark this keypoint as null in map
            map_idx[thread_idx] = -1;
#ifdef DEBUG_OUTPUT
            printf("Removed keypoint from thread: %u, desc index: %lld, x:%d y:%d z:%d\n",
                    thread_idx, idx, x, y, z);
#endif
            return ;
        }

    }

    cudautils::Keypoint key = make_keypoint(image, x, y, z, thread_idx, idx,
            x_stride, y_stride, sift_params, descriptors, device_centers, ix,
            yy);

    return;
}

/*Define the constructor for the SIFT class*/
/*See the class Sift definition in sift.h*/
Sift::Sift(
        const unsigned int x_size,
        const unsigned int y_size,
        const unsigned int z_size,
        const unsigned int x_sub_size,
        const unsigned int y_sub_size,
        const unsigned int dx,
        const unsigned int dy,
        const unsigned int dw,
        const unsigned int num_gpus,
        const unsigned int num_streams,
        const cudautils::SiftParams sift_params,
        const double* fv_centers)
    : x_size_(x_size), y_size_(y_size), z_size_(z_size),
        x_sub_size_(x_sub_size), y_sub_size_(y_sub_size),
        dx_(dx), dy_(dy), dw_(dw),
        num_gpus_(num_gpus), num_streams_(num_streams),
        sift_params_(sift_params),
        fv_centers_(fv_centers),
        subdom_data_(num_gpus) {

    logger_ = spdlog::get("console");
    if (! logger_) {
        logger_ = spdlog::stdout_logger_mt("console");
    }
#ifdef DEBUG_OUTPUT
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif

    size_t log_q_size = 4096;
    spdlog::set_async_mode(log_q_size);

    num_x_sub_ = get_num_blocks(x_size_, x_sub_size_);
    num_y_sub_ = get_num_blocks(y_size_, y_sub_size_);

    x_sub_stride_ = x_sub_size_ + 2 * dw_;
    y_sub_stride_ = y_sub_size_ + 2 * dw_;

    dx_stride_ = dx_ + 2 * dw_;
    dy_stride_ = dy_ + 2 * dw_;
    z_stride_ = z_size_ + 2 * dw_;
#ifdef DEBUG_OUTPUT
    logger_->info("x_size={}, x_sub_size={}, num_x_sub={}, x_sub_stride={}, dx={}, dx_stride={}",
            x_size_, x_sub_size_, num_x_sub_, x_sub_stride_, dx_, dx_stride_);
    logger_->info("y_size={}, y_sub_size={}, num_y_sub={}, y_sub_stride={}, dy={}, dy_stride={}",
            y_size_, y_sub_size_, num_y_sub_, y_sub_stride_, dy_, dy_stride_);
    logger_->info("z_size={}, dw={}, z_stride={}", z_size_, dw_, z_stride_);
#endif


    dom_data_ = std::make_shared<DomainDataOnHost>(x_size_, y_size_, z_size_);

    for (unsigned int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);

        subdom_data_[i] = std::make_shared<SubDomainDataOnGPU>(x_sub_stride_, y_sub_stride_, z_stride_, num_streams_);

        for (unsigned int j = 0; j < num_streams_; j++) {
            subdom_data_[i]->stream_data[j] = std::make_shared<SubDomainDataOnStream>(dx_stride_, dy_stride_, z_stride_);

            cudaStreamCreate(&subdom_data_[i]->stream_data[j]->stream);
        }
    }
    cudaSetDevice(0);


    unsigned int idx_gpu = 0;
    for (unsigned int y_sub_i = 0; y_sub_i < num_y_sub_; y_sub_i++) {
        for (unsigned int x_sub_i = 0; x_sub_i < num_x_sub_; x_sub_i++) {
            subdom_data_[idx_gpu]->x_sub_i_list.push_back(x_sub_i);
            subdom_data_[idx_gpu]->y_sub_i_list.push_back(y_sub_i);

            idx_gpu++;
            if (idx_gpu == num_gpus) {
                idx_gpu = 0;
            }
        }
    }

}

Sift::~Sift() {
    for (unsigned int i = 0; i < num_gpus_; i++) {
        for (unsigned int j = 0; j < num_streams_; j++) {
            cudaStreamDestroy(subdom_data_[i]->stream_data[j]->stream);
        }
    }

    //logger_->flush();
}

void Sift::setImage(const double *img)
{
    thrust::copy(img, img + (x_size_ * y_size_ * z_size_), dom_data_->h_image);
}

void Sift::setImage(const std::vector<double>& img)
{
    assert((x_size_ * y_size_ * z_size_) == img.size());

    thrust::copy(img.begin(), img.end(), dom_data_->h_image);
}

void Sift::setMap(const int8_t *map)
{
    thrust::copy(map, map + (x_size_ * y_size_ * z_size_), dom_data_->h_map);
}

void Sift::setMap(const std::vector<int8_t>& map)
{
    assert((x_size_ * y_size_ * z_size_) == map.size());

    thrust::copy(map.begin(), map.end(), dom_data_->h_map);
}

void Sift::getKeystore(cudautils::Keypoint_store *keystore)
{
    keystore->len = dom_data_->keystore->len;
    if (keystore->len) {
        keystore->buf = (cudautils::Keypoint*) malloc(keystore->len * sizeof(cudautils::Keypoint));
        thrust::copy(dom_data_->keystore->buf, dom_data_->keystore->buf + dom_data_->keystore->len, keystore->buf);
    }
}


void Sift::getImage(double *img)
{
    thrust::copy(dom_data_->h_image, dom_data_->h_image + x_size_ * y_size_ * z_size_, img);
}

void Sift::getImage(std::vector<double>& img)
{
    thrust::copy(dom_data_->h_image, dom_data_->h_image + x_size_ * y_size_ * z_size_, img.begin());
}


int Sift::getNumOfGPUTasks(const int gpu_id) {
    return subdom_data_[gpu_id]->x_sub_i_list.size();
}

int Sift::getNumOfStreamTasks(
        const int gpu_id,
        const int stream_id) {
    return 1;
}

void Sift::runOnGPU(
        const int gpu_id,
        const unsigned int gpu_task_id) {

    cudaSafeCall(cudaSetDevice(gpu_id));

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];
    std::shared_ptr<SubDomainDataOnStream> stream_data0 = subdom_data->stream_data[0];

    unsigned int x_sub_i = subdom_data->x_sub_i_list[gpu_task_id];
    unsigned int y_sub_i = subdom_data->y_sub_i_list[gpu_task_id];
#ifdef DEBUG_OUTPUT
    CudaTimer timer;
    logger_->info("===== gpu_id={} x_sub_i={} y_sub_i={}", gpu_id, x_sub_i, y_sub_i);
#endif

    unsigned int x_sub_start = x_sub_i * x_sub_size_;
    unsigned int y_sub_start = y_sub_i * y_sub_size_;
    // clamp delta to end value 
    unsigned int x_sub_delta = get_delta(x_size_, x_sub_i, x_sub_size_);
    unsigned int y_sub_delta = get_delta(y_size_, y_sub_i, y_sub_size_);
    // only add in pad factor at first
    unsigned int base_x_sub  = (x_sub_i > 0 ? 0 : dw_);
    unsigned int base_y_sub  = (y_sub_i > 0 ? 0 : dw_);

    // subtract pad factor after first
    unsigned int padding_x_sub_start = x_sub_start - (x_sub_i > 0 ? dw_ : 0);
    unsigned int padding_y_sub_start = y_sub_start - (y_sub_i > 0 ? dw_ : 0);
    unsigned int padding_x_sub_delta = x_sub_delta + (x_sub_i > 0 ? dw_ : 0) + (x_sub_i < num_x_sub_ - 1 ? dw_ : 0);
    unsigned int padding_y_sub_delta = y_sub_delta + (y_sub_i > 0 ? dw_ : 0) + (y_sub_i < num_y_sub_ - 1 ? dw_ : 0);

    // per GPU padded image size
    size_t padded_sub_volume_size = x_sub_stride_ * y_sub_stride_ * z_stride_;

#ifdef DEBUG_OUTPUT
    unsigned int x_sub_end = x_sub_start + x_sub_delta;
    unsigned int y_sub_end = y_sub_start + y_sub_delta;
    logger_->debug("x_sub=({},{},{}) y_sub=({},{},{})", x_sub_start, x_sub_delta, x_sub_end, y_sub_start, y_sub_delta, y_sub_end);
    logger_->debug("base_x_sub={},base_y_sub={}", base_x_sub, base_y_sub);

#ifdef DEBUG_OUTPUT_MATRIX
    // print the x, y, z image / map coordinates of the selected keypoints
    if (gpu_id == 0)  { // don't repeat this for every GPU
        for (long long idx=0; idx < x_size_ * y_size_ * z_size_; idx++) {
            if (! dom_data_->h_map[idx]) {
                unsigned int x;
                unsigned int y;
                unsigned int z;
                ind2sub(x_size_, y_size_, idx, x, y, z);

                logger_->info("h_map 0's: idx={}, x={}, y={}, z={}",
                        idx, x, y, z);
            }
        }
    }
#endif
#endif

    // allocate the per GPU padded map and image
    int8_t *padded_sub_map;
    double *padded_sub_image;
    cudaSafeCall(cudaHostAlloc(&padded_sub_map, padded_sub_volume_size *
                sizeof(int8_t), cudaHostAllocPortable));
        cudaCheckError();
    cudaSafeCall(cudaHostAlloc(&padded_sub_image, padded_sub_volume_size *
                sizeof(double), cudaHostAllocPortable));
        cudaCheckError();

    // First set all values to holder value -1
    thrust::fill(padded_sub_map, padded_sub_map + padded_sub_volume_size, -1);

    for (unsigned int k = 0; k < z_size_; k++) {
        for (unsigned int j = 0; j < padding_y_sub_delta; j++) {
            // get row-major / c-order linear index according orig. dim [x_size, y_size, z_size]
            size_t src_idx = dom_data_->sub2ind(padding_x_sub_start, padding_y_sub_start + j, k);
            size_t dst_idx = subdom_data->pad_sub2ind(base_x_sub, base_y_sub + j, dw_ + k);

            int8_t* src_map_begin = &(dom_data_->h_map[src_idx]);
            int8_t* dst_map_begin = &(padded_sub_map[dst_idx]);
            // note this assumes the rows to be contiguous in memory (row-order / c-order)
            thrust::copy(src_map_begin, src_map_begin + padding_x_sub_delta, dst_map_begin);

            double* src_image_begin = &(dom_data_->h_image[src_idx]);
            double* dst_image_begin = &(padded_sub_image[dst_idx]);
            thrust::copy(src_image_begin, src_image_begin + padding_x_sub_delta, dst_image_begin);
        }
    }
    
#ifdef DEBUG_OUTPUT_MATRIX

    // print the x, y, z in padded image / map coordinates of the selected keypoints
    for (long long i=0; i < padded_sub_volume_size; i++) {
        if (!padded_sub_map[i]) {
            unsigned int padding_x;
            unsigned int padding_y;
            unsigned int padding_z;
            ind2sub(x_sub_stride_, y_sub_stride_, i, padding_x, padding_y, padding_z);
            // correct for dw_ padding, matlab is 1-indexed
            unsigned int x = x_sub_start + padding_x - dw_ + 1;
            unsigned int y = y_sub_start + padding_y - dw_ + 1;
            unsigned int z = padding_z - dw_ + 1;

            logger_->info("padded_sub_map 0's (matlab 1-indexed): idx={}, x={}, y={}, z={}",
                    i, x, y, z);
        }
    }

#endif

    thrust::fill(thrust::device, subdom_data->padded_image, subdom_data->padded_image + padded_sub_volume_size, 0.0);

    cudaSafeCall(cudaMemcpyAsync(
            subdom_data->padded_image,
            padded_sub_image,
            padded_sub_volume_size * sizeof(double),
            cudaMemcpyHostToDevice, stream_data0->stream));

#ifdef DEBUG_OUTPUT
    cudaSafeCall(cudaStreamSynchronize(stream_data0->stream));
    logger_->info("transfer image data {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
    logger_->info("===== dev image");
    print_matrix3d(logger_, x_size_, y_size_, 0, 0, 0, x_size_, y_size_, z_size_, dom_data_->h_image);
    print_matrix3d_dev(logger_, x_sub_stride_, y_sub_stride_, z_stride_, 0, 0, 0, x_sub_stride_, y_sub_stride_, z_stride_, subdom_data->padded_image);
#endif

    timer.reset();
#endif

    cudaSafeCall(cudaMemcpyAsync(
            subdom_data->padded_map,
            padded_sub_map,
            padded_sub_volume_size * sizeof(int8_t),
            cudaMemcpyHostToDevice, stream_data0->stream));

#ifdef DEBUG_OUTPUT
    cudaSafeCall(cudaStreamSynchronize(stream_data0->stream));
    logger_->info("transfer map data {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
    logger_->debug("===== dev map");
    print_matrix3d(logger_, x_size_, y_size_, 0, 0, 0, x_size_, y_size_, z_size_, dom_data_->h_map);
    print_matrix3d_dev(logger_, x_sub_stride_, y_sub_stride_, z_stride_, 0, 0, 0, x_sub_stride_, y_sub_stride_, z_stride_, subdom_data->padded_map);
#endif

    timer.reset();
#endif

    // clear previous result to zero
    thrust::fill(thrust::device, subdom_data->padded_map_idx, subdom_data->padded_map_idx + padded_sub_volume_size, 0.0);

    /*Note: padded_sub_volume_size = x_sub_stride_ * y_sub_stride_ * z_stride_;*/
    auto end_itr = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator<unsigned int>(0), // count indexes from 0
            thrust::make_counting_iterator<unsigned int>(padded_sub_volume_size), // ...to padded_sub_volume_size
            subdom_data->padded_map, //beginning of stencil sequence
            subdom_data->padded_map_idx, // beginning of sequence to copy into
            thrust::logical_not<int8_t>());//predicate test on every value

    subdom_data->padded_map_idx_size = end_itr - subdom_data->padded_map_idx;

    // set all padded map boundaries (still -1) to 0 for correctness to
    // distinguish boundaries
    thrust::replace(thrust::device, subdom_data->padded_map, subdom_data->padded_map + padded_sub_volume_size, -1, 0);

#ifdef DEBUG_OUTPUT
    cudaSafeCall(cudaStreamSynchronize(stream_data0->stream));
    logger_->info("calculate map idx {}", timer.get_laptime());

    logger_->info("padded_map_idx_size={}", subdom_data->padded_map_idx_size);

    timer.reset();
#endif


    // Each GPU each subdom_data
    // this set the dx and dy start idx for each stream
    unsigned int num_dx = get_num_blocks(x_sub_delta, dx_);
    unsigned int num_dy = get_num_blocks(y_sub_delta, dy_);
    unsigned int stream_id = 0;
    for (unsigned int dy_i = 0; dy_i < num_dy; dy_i++) {
        for (unsigned int dx_i = 0; dx_i < num_dx; dx_i++) {
            subdom_data->stream_data[stream_id]->dx_i_list.push_back(dx_i);
            subdom_data->stream_data[stream_id]->dy_i_list.push_back(dy_i);

            stream_id++;
            if (stream_id == num_streams_) {
                stream_id = 0;
            }
        }
    }
    cudaSafeCall(cudaStreamSynchronize(stream_data0->stream));

    cudaSafeCall(cudaFreeHost(padded_sub_map));
    cudaSafeCall(cudaFreeHost(padded_sub_image));
}

const cudautils::SiftParams Sift::get_sift_params() {
    return sift_params_;
}

void Sift::postrun() {
    // count keypoints
    int total_keypoints = 0;
    for (int gpu_id = 0; gpu_id < num_gpus_; gpu_id++) {
        std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];

        for (int stream_id = 0; stream_id < num_streams_; stream_id++) {
            std::shared_ptr<SubDomainDataOnStream> stream_data =
                subdom_data->stream_data[stream_id];

            /*logger_->info("gpu_id {}, streamid {}, # of kypts {}", gpu_id, stream_id, stream_data->keystore.size());*/
            total_keypoints += stream_data->keystore.size();
        }
    }

    // allocate for number of keypoints
    dom_data_->keystore->len = total_keypoints;
    /*logger_->info("total_keypoints {}", total_keypoints);*/
    if (total_keypoints < 1)
        return;
    cudaHostAlloc(&(dom_data_->keystore->buf), dom_data_->keystore->len *
            sizeof(cudautils::Keypoint), cudaHostAllocPortable);

    // copy keypoints to host
    int counter = 0;
    for (int gpu_id = 0; gpu_id < num_gpus_; gpu_id++) {
        std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];

        for (int stream_id = 0; stream_id < num_streams_; stream_id++) {
            std::shared_ptr<SubDomainDataOnStream> stream_data =
                subdom_data->stream_data[stream_id];

            for (int i = 0; i < stream_data->keystore.size(); i++) {
                dom_data_->keystore->buf[counter] = stream_data->keystore[i];
                counter++;
            }
        }
    }
    assert(counter == total_keypoints);
    return;
}


void Sift::runOnStream(
        const int gpu_id,
        const int stream_id,
        const unsigned int gpu_task_id) {

    cudaSetDevice(gpu_id);

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];
    std::shared_ptr<SubDomainDataOnStream> stream_data = subdom_data->stream_data[stream_id];

    unsigned int x_sub_i = subdom_data->x_sub_i_list[gpu_task_id];
    unsigned int y_sub_i = subdom_data->y_sub_i_list[gpu_task_id];
    unsigned int x_sub_delta = get_delta(x_size_, x_sub_i, x_sub_size_);
    unsigned int y_sub_delta = get_delta(y_size_, y_sub_i, y_sub_size_);
    unsigned int x_sub_start = x_sub_i * x_sub_size_;
    unsigned int y_sub_start = y_sub_i * y_sub_size_;

#ifdef DEBUG_OUTPUT
    CudaTimer timer(stream_data->stream);
#endif

    // each stream has a individual subsections of data, that each kernel call will operate on
    // these subsections start/stop idx are determined by dx_i and dy_i lists
    for (auto dx_itr = stream_data->dx_i_list.begin(), dy_itr = stream_data->dy_i_list.begin();
            dx_itr != stream_data->dx_i_list.end() || dy_itr != stream_data->dy_i_list.end();
            dx_itr++, dy_itr++) {

        unsigned int dx_i = *dx_itr;
        unsigned int dy_i = *dy_itr;

        unsigned int dx_start = dx_i * dx_;
        unsigned int dx_delta = get_delta(x_sub_delta, dx_i, dx_);
        unsigned int dx_end   = dx_start + dx_delta;
        unsigned int dy_start = dy_i * dy_;
        unsigned int dy_delta = get_delta(y_sub_delta, dy_i, dy_);
        unsigned int dy_end   = dy_start + dy_delta;

#ifdef DEBUG_OUTPUT
        logger_->info("dx_i={}, dy_i={}", dx_i, dy_i);
        logger_->info("x=({},{},{}) y=({},{},{}), dw={}", dx_start, dx_delta, dx_end, dy_start, dy_delta, dy_end, dw_);
        logger_->info("subdom_data->padded_map_idx_size={}", subdom_data->padded_map_idx_size);
#endif

        // create each substream data on device
        long long int *substream_padded_map_idx;
        cudaSafeCall(cudaMalloc(&substream_padded_map_idx,
                    subdom_data->padded_map_idx_size * sizeof(long long int)));

        RangeCheck range_check { x_sub_stride_, y_sub_stride_,
            dx_start + dw_, dx_end + dw_, dy_start + dw_, dy_end + dw_, dw_, z_size_ + dw_ };

        // copy the relevant (in range) idx elements from the
        // global GPU padded_map_idx to the local substream_padded_map_idx 
        auto end_itr = thrust::copy_if(
                thrust::device,
                subdom_data->padded_map_idx,
                subdom_data->padded_map_idx + subdom_data->padded_map_idx_size,
                substream_padded_map_idx,
                range_check);

        const unsigned int substream_padded_map_idx_size = end_itr - substream_padded_map_idx;

#ifdef DEBUG_OUTPUT
        logger_->info("substream_padded_map_idx_size={}", substream_padded_map_idx_size);
        logger_->info("transfer map idx {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
        cudaSafeCall(cudaStreamSynchronize(stream_data->stream));
        thrust::device_vector<long long int> dbg_d_padded_map_idx(substream_padded_map_idx,
                substream_padded_map_idx + substream_padded_map_idx_size);
        thrust::host_vector<unsigned int> dbg_h_padded_map_idx(dbg_d_padded_map_idx);
        for (unsigned int i = 0; i < substream_padded_map_idx_size; i++) {
            logger_->debug("substream_padded_map_idx={}", dbg_h_padded_map_idx[i]);
        }
#endif
        timer.reset();
#endif

        if (substream_padded_map_idx_size == 0) {
#ifdef DEBUG_OUTPUT
            logger_->debug("no map to be padded");
#endif
            continue;
        }

        // only calculate location and save keypoints
        if (sift_params_.skipDescriptor) {
#ifdef DEBUG_OUTPUT
            logger_->debug("Skip calculatation of descriptors");
#endif
            // transfer index map to host for referencing correct index
            long long int *h_padded_map_idx;
            cudaSafeCall(cudaHostAlloc((void **) &h_padded_map_idx, 
                        substream_padded_map_idx_size * sizeof(long long int),
                        cudaHostAllocPortable));

            cudaSafeCall(cudaMemcpyAsync(
                    h_padded_map_idx,
                    substream_padded_map_idx,
                    substream_padded_map_idx_size * sizeof(long long int),
                    cudaMemcpyDeviceToHost, stream_data->stream));

            // make sure all async memcpys (above) are finished before access
            cudaSafeCall(cudaStreamSynchronize(stream_data->stream));

            // save data for all streams to global Sift object store
            for (int i = 0; i < substream_padded_map_idx_size; i++) {

                Keypoint temp;

                unsigned int padding_x;
                unsigned int padding_y;
                unsigned int padding_z;
                ind2sub(x_sub_stride_, y_sub_stride_, h_padded_map_idx[i], padding_x, padding_y, padding_z);
                // correct for dw_ padding, matlab is 1-indexed
                temp.x = x_sub_start + padding_x - dw_ + 1;
                temp.y = y_sub_start + padding_y - dw_ + 1;
                temp.z = padding_z - dw_ + 1;

                stream_data->keystore.push_back(temp);
            }
            cudaSafeCall(cudaFree(substream_padded_map_idx));
            continue; // do this for every substream forloop
        }

        /*
        Create an array to hold each descriptor ivec vector on VRAM 
        essentially a matrix of substream_padded_map_idx_size by descriptor length
        */
        double *descriptors, *yy_scratch, *ori_scratch;
        uint16_t * idx_scratch, *ori_idx_scratch;
        long desc_mem_size = sift_params_.descriptor_len * 
            substream_padded_map_idx_size * sizeof(double);
        cudaSafeCall(cudaMalloc(&descriptors, desc_mem_size));
        // default nFaces 80; 640 bytes per keypoint yy
        cudaSafeCall(cudaMalloc(&yy_scratch, sift_params_.nFaces *
                    substream_padded_map_idx_size * sizeof(double)));
        //  160 bytes per keypoint idx
        cudaSafeCall(cudaMalloc(&idx_scratch, sift_params_.nFaces *
                    substream_padded_map_idx_size * sizeof(uint16_t)));
        if (sift_params_.TwoPeak_Flag) {
            // default nFaces=80
            cudaSafeCall(cudaMalloc(&ori_idx_scratch, sift_params_.nFaces *
                        substream_padded_map_idx_size * sizeof(uint16_t)));
            cudaSafeCall(cudaMalloc(&ori_scratch, sift_params_.nFaces *
                        substream_padded_map_idx_size * sizeof(double)));
        }

        // One keypoint per thread, one thread per block
        unsigned int num_threads = 1;
        // round up by number of threads per block, to calc num of blocks
        unsigned int num_blocks = get_num_blocks(substream_padded_map_idx_size, num_threads);

#ifdef DEBUG_OUTPUT
        /*cudaSafeCall(cudaStreamSynchronize(stream_data->stream));*/
        logger_->debug("num_blocks={}", num_blocks);
        logger_->debug("num_threads={}", num_threads);
#endif

        if (num_blocks * num_threads < substream_padded_map_idx_size) {
            logger_->info("Error occured in numblocks and num_threads estimation... returning from stream"); 
            return;
        }

#ifdef DEBUG_OUTPUT
        logger_->debug("create_descriptor");
        timer.reset();
#endif

        // sift_params.fv_centers must be placed on device since array passed to cuda kernel
        double* device_centers;
        // default fv_centers_len 80 * 3 (3D) = 240;
        cudaSafeCall(cudaMalloc((void **) &device_centers,
                    sizeof(double) * sift_params_.fv_centers_len));
        cudaSafeCall(cudaMemcpy((void *) device_centers, (const void *) fv_centers_,
                (size_t) sizeof(double) * sift_params_.fv_centers_len,
                cudaMemcpyHostToDevice));
        
#ifdef DEBUG_OUTPUT_MATRIX

        /*printf("Print image\n");*/
        /*cudaStreamSynchronize(stream_data->stream);*/
        /*int sub_volume_size = x_sub_stride_ * y_sub_stride_ * z_stride_;*/
        /*double* dbg_h_image = (double*) malloc(sizeof(double) * sub_volume_size);*/
        /*cudaSafeCall(cudaMemcpy((void **) dbg_h_image, subdom_data->padded_image,*/
                /*sizeof(double) * sub_volume_size,*/
                /*cudaMemcpyDeviceToHost));*/
        /*// print*/
        /*for (int i=0; i < sub_volume_size; i++) {*/
            /*if (dbg_h_image[i] != 0.0) {*/
                /*printf("host image[%d]: %f\n", i, dbg_h_image[i]);*/
            /*}*/
        /*}*/

#endif

        create_descriptor<<<num_blocks, num_threads, 0, stream_data->stream>>>(
                x_sub_stride_, y_sub_stride_, x_sub_start, y_sub_start, 
                dw_, // pad width
                substream_padded_map_idx_size, // total number of keypoints to process
                substream_padded_map_idx, //substream map, filtered linear idx into per GPU padded_map and padded_image
                subdom_data->padded_map,//global map split per GPU
                subdom_data->padded_image,//image split per GPU
                sift_params_, 
                device_centers,
                descriptors,
                idx_scratch,
                yy_scratch,
                ori_idx_scratch,
                ori_scratch); 
        cudaCheckError();


#ifdef DEBUG_OUTPUT
        logger_->info("create descriptors elapsed: {}", timer.get_laptime());

        timer.reset();
#endif

        // transfer vector descriptors via host pinned memory for faster async cpy
        double *h_descriptors;
        cudaSafeCall(cudaHostAlloc((void **) &h_descriptors, desc_mem_size, cudaHostAllocPortable));
        
        cudaSafeCall(cudaMemcpyAsync(
                h_descriptors,
                descriptors,
                desc_mem_size,
                cudaMemcpyDeviceToHost, stream_data->stream));

        // transfer index map to host for referencing correct index
        long long int *h_padded_map_idx;
        cudaSafeCall(cudaHostAlloc((void **) &h_padded_map_idx, 
                    substream_padded_map_idx_size * sizeof(long long int),
                    cudaHostAllocPortable));

        cudaSafeCall(cudaMemcpyAsync(
                h_padded_map_idx,
                substream_padded_map_idx,
                substream_padded_map_idx_size * sizeof(long long int),
                cudaMemcpyDeviceToHost, stream_data->stream));

#ifdef DEBUG_OUTPUT_MATRIX
        for (int i=0; i < substream_padded_map_idx_size; i++) {
            printf("h_padded_map_idx:%lld\n", h_padded_map_idx[i]);
            if (i % sift_params_.descriptor_len == 0) {
                printf("\n\nDescriptor:%d\n", (int) i / sift_params_.descriptor_len);
            }
            printf("%d: %d\n", i, h_descriptors[i]);
        }
#endif

        // make sure all async memcpys (above) are finished before access
        cudaSafeCall(cudaStreamSynchronize(stream_data->stream));

        // save data for all streams to global Sift object store
        int skip_counter = 0;
        for (int i = 0; i < substream_padded_map_idx_size; i++) {
            Keypoint temp;

            if (sift_params_.TwoPeak_Flag) {
                if (h_padded_map_idx[i] == -1) {
                    skip_counter++;
                    continue;
                } 
            }

            unsigned int padding_x;
            unsigned int padding_y;
            unsigned int padding_z;
            ind2sub(x_sub_stride_, y_sub_stride_, h_padded_map_idx[i], padding_x, padding_y, padding_z);
            // correct for dw_ padding, matlab is 1-indexed
            temp.x = x_sub_start + padding_x - dw_ + 1;
            temp.y = y_sub_start + padding_y - dw_ + 1;
            temp.z = padding_z - dw_ + 1;

            temp.ivec = (double*) malloc(sift_params_.descriptor_len * sizeof(double));
            memcpy(temp.ivec, &(h_descriptors[i * sift_params_.descriptor_len]), 
                    sift_params_.descriptor_len * sizeof(double));
            temp.xyScale = sift_params_.xyScale;
            temp.tScale = sift_params_.tScale;

            // buffer the size of the whole image
            stream_data->keystore.push_back(temp);
        }

        cudaSafeCall(cudaFree(substream_padded_map_idx));
        cudaSafeCall(cudaFree(descriptors));
        cudaSafeCall(cudaFree(device_centers));
        cudaSafeCall(cudaFree(idx_scratch));
        cudaSafeCall(cudaFree(yy_scratch));
        if (sift_params_.TwoPeak_Flag) {
            cudaSafeCall(cudaFree(ori_idx_scratch));
            cudaSafeCall(cudaFree(ori_scratch));
        }

        cudaSafeCall(cudaFreeHost(h_descriptors));
        cudaSafeCall(cudaFreeHost(h_padded_map_idx));

#ifdef DEBUG_OUTPUT
        logger_->info("gpu:{}, stream:{}, substream_padded_map_idx_size={}, saved={}",
                gpu_id, stream_id, substream_padded_map_idx_size, 
                substream_padded_map_idx_size - skip_counter);

        logger_->info("transfer d2h and copy descriptor ivec values {}", timer.get_laptime());
#endif
    }
}


} // namespace cudautils

