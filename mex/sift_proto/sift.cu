#include <iostream>
#include <future>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>
#include <cmath>
/*#include <numeric> //std::inner_product*/

#include "sift.h"
#include "matrix_helper.h"
#include "cuda_timer.h"

#include "spdlog/spdlog.h"


namespace cudautils {

/*struct FV {*/
    /*double* vertices;*/
    /*double* faces;*/
    /*double* centers;*/
/*};*/

/*FV sphere_tri(int maxlevel, int r) {*/
    /*
     sphere_tri - generate a triangle mesh approximating a sphere
    
     Usage: FV = sphere_tri(Nrecurse,r)
    
       Nrecurse is int >= 0, setting the recursions (default 0)
    
       r is the radius of the sphere (default 1)
    
       FV has fields FV.vertices and FV.faces.  The vertices
       are listed in clockwise order in FV.faces, as viewed
       from the outside in a RHS coordinate system.
    
     The function uses recursive subdivision.  The first
     approximation is an icosahedron. Each level of refinement
     subdivides each triangle face by a factor of 4 (see also
     mesh_refine). At each refinement, the vertices are
     projected to the sphere surface (see sphere_project).
    
     A recursion level of 3 or 4 is a good sphere surface, if
     gouraud shading is used for rendering.
    
     The returned struct can be used in the patch command, eg:
    
      create and plot, vertices: [2562x3] and faces: [5120x3]
     FV = sphere_tri('ico',4,1);
     lighting phong; shading interp; figure;
     patch('vertices',FV.vertices,'faces',FV.faces,...
           'facecolor',[1 0 0],'edgecolor',[.2 .2 .6]);
     axis off; camlight infinite; camproj('perspective');
    
     See also: mesh_refine, sphere_project
    
     Cuda revision of:
     Licence:  GNU GPL, no implied or express warranties
     Jon Leech (leech @ cs.unc.edu) 3/24/89
     icosahedral code added by Jim Buddenhagen (jb1556@daditz.sbc.com) 5/93
     06/2002, adapted from c to matlab by Darren.Weber_at_radiology.ucsf.edu
     05/2004, reorder of the faces for the 'ico' surface so they are indeed
     clockwise!  Now the surface normals are directed outward.  Also reset the
     default recursions to zero, so we can get out just the platonic solids.
    
    */

    /*// default maximum subdivision level*/
    /*if (maxlevel < 0)*/
        /*maxlevel = 0;*/

    /*// default radius*/
    /*if (r < 0)*/
        /*r = 1;*/

    /*// define the icosehedron*/

    /*// Twelve vertices of icosahedron on unit sphere*/
    /*double tau = 0.8506508084; // t=(1+sqrt(5))/2, tau=t/sqrt(1+t^2)*/
    /*double one = 0.5257311121; // one=1/sqrt(1+t^2) , unit sphere*/

    /*FV fv;*/
    
    /*// store the vertices in column (Matlab) order */
    /*fv.vertices = new double[12][3];*/
    /*fv.vertices[ 0] = {  tau,  one,    0 }; // ZA*/
    /*fv.vertices[ 1] = { -tau,  one,    0 }; // ZB*/
    /*fv.vertices[ 2] = { -tau, -one,    0 }; // ZC*/
    /*fv.vertices[ 3] = {  tau, -one,    0 }; // ZD*/
    /*fv.vertices[ 4] = {  one,   0 ,  tau }; // YA*/
    /*fv.vertices[ 5] = {  one,   0 , -tau }; // YB*/
    /*fv.vertices[ 6] = { -one,   0 , -tau }; // YC*/
    /*fv.vertices[ 7] = { -one,   0 ,  tau }; // YD*/
    /*fv.vertices[ 8] = {   0 ,  tau,  one }; // XA*/
    /*fv.vertices[ 9] = {   0,  -tau,  one }; // XB*/
    /*fv.vertices[10] = {   0 , -tau, -one }; // XC*/
    /*fv.vertices[11] = {   0 ,  tau, -one }; // XD*/
    
    /*// Structure for unit icosahedron*/
    /*// Fixme check this is in correct col order*/
    /*// previous matlab code was in ; order*/
    /*fv.faces = {  5,  8,  9 ,*/
               /*5, 10,  8 ,*/
               /*6, 12,  7 ,*/
               /*6,  7, 11 ,*/
               /*1,  4,  5 ,*/
               /*1,  6,  4 ,*/
               /*3,  2,  8 ,*/
               /*3,  7,  2 ,*/
               /*9, 12,  1 ,*/
               /*9,  2, 12 ,*/
              /*10,  4, 11 ,*/
              /*10, 11,  3 ,*/
               /*9,  1,  5 ,*/
              /*12,  6,  1 ,*/
               /*5,  4, 10 ,*/
               /*6, 11,  4 ,*/
               /*8,  2,  9 ,*/
               /*7, 12,  2 ,*/
               /*8, 10,  3 ,*/
               /*7,  3, 11 };*/
    


    /*// -----------------*/
    /*// refine the starting shapes with subdivisions*/
    /*if maxlevel,*/
        
        /*// Subdivide each starting triangle (maxlevel) times*/
        /*for level = 1:maxlevel,*/
            
            /*// Subdivide each triangle and normalize the new points thus*/
            /*// generated to lie on the surface of a sphere radius r.*/
            /*fv = mesh_refine_tri4(fv);*/
            /*fv.vertices = sphere_project(fv.vertices,r);*/
            
            /*// An alternative might be to define a min distance*/
            /*// between vertices and recurse or use fminsearch*/
            
        /*end*/
    /*end*/

    /*for (int i=0; i < length(fv.faces); i++) {*/
        /*fv.centers(i,:) = mean(fv.vertices(fv.faces(i,:),:));*/
        /*// Unit Normalization*/
        /*fv.centers(i,:) = fv.centers(i,:) ./ sqrt(dot(fv.centers(i,:),fv.centers(i,:)));*/
    /*}*/
/*}*/

__device__
void place_in_index(double* index, double mag, int i, int j, int s, 
        double* yy, int* ix, cudautils::SiftParams sift_params) {

    double tmpsum = 0.0;
    /*FIXME*/
    int bin_index = 0;
    /*int bin_index = bin_sub2ind(i,j,s);*/
    if (sift_params.Smooth_Flag) {
        for (int tessel=0; tessel < sift_params.Tessel_thresh; tessel++) {
            tmpsum += pow(yy[tessel], sift_params.Smooth_Var);
        }

        // Add three nearest tesselation faces
        for (int ii=0; ii<sift_params.Tessel_thresh; ii++) {
            index[bin_index] +=  mag * pow(yy[ii], sift_params.Smooth_Var ) / tmpsum;
        }
    } else {
        index[bin_index] += mag;
    }
    return;
}

__device__
double dot_product(double* first, double* second, int N) {
    double sum = 0;
    for (int i=0; i < N; i++) {
        sum += first[i] * second[i];
    }
    return sum;
}

// assumes r,c,s lie within accessible image boundaries
__device__
double get_grad_ori_vector(double* image, int r, int c, int s, 
        double vect[3], double* yy, int* ix, cudautils::SiftParams sift_params) {

    /*//FIXME subscripts to linear ind*/
    /*double xgrad = image[r,c+1,s] - image[r,c-1,s];*/
    /*//FIXME is this correct direction?*/
    /*double ygrad = image[r-1,c,s] - image[r+1,c,s];*/
    /*double zgrad = image[r,c,s+1] - image[r,c,s-1];*/
    double xgrad = 0.0;
    double ygrad = 0.0;
    double zgrad = 0.0;

    double mag = sqrt(xgrad * xgrad + ygrad * ygrad + zgrad * zgrad);

    xgrad /= mag;
    ygrad /= mag;
    zgrad /= mag;

    if (mag !=0) {
        vect[0] = xgrad;
        vect[1] = ygrad;
        vect[2] = zgrad;
    } 

    //Find the nearest tesselation face indices
    //FIXME cublasSgemm() higher performance
    int N = sift_params.fv_centers_len;
    /*double* corr_array;*/
    cudaMalloc(&yy, N*sizeof(double));
    for (int i=0; i < N; i++) {
        yy[i] = dot_product(&sift_params.fv_centers[i],
                vect, 3);
    }
    thrust::sequence(thrust::device, ix, ix + N);
    // descending order by ori_hist
    thrust::sort_by_key(thrust::device, ix, ix + N, yy, thrust::greater<int>());
    /*yy = corr_array;*/
    return mag;
}

/*r, c, s is the pixel index (x, y, z dimensions respect.) in the image within the radius of the */
/*keypoint before clamped*/
/*For each pixel, take a neighborhhod of xyradius and tiradius,*/
/*bin it down to the sift_params.IndexSize dimensions*/
/*thus, i_indx, j_indx, s_indx represent the binned index within the radius of the keypoint*/
__device__
void add_sample(double* index, double* image, double distsq, int
        r, int c, int s, int i_bin, int j_bin, int k_bin, 
        cudautils::SiftParams sift_params) {

    double sigma = sift_params.SigmaScaled;
    double weight = exp(-(distsq / (2.0 * sigma * sigma)));

    double vect[3] = {1.0, 0.0, 0.0};
    int* ix;
    cudaMalloc(&ix, sift_params.fv_centers_len*sizeof(int));
    double *yy;
    /*gradient and orientation vectors calculated from 3D halo/neighboring pixels*/
    double mag = get_grad_ori_vector(image,r,c,s, vect, yy, ix, sift_params);
    mag *= weight; // scale magnitude by gaussian 

    place_in_index(index, mag, i_bin, j_bin, k_bin, yy, ix, sift_params);
    return;
}


// floor quotient, add 1
// clamp bin idx to IndexSize
__device__
inline int get_bin_idx(int orig, int radius, int IndexSize) {
    int idx = (int) 1 + ((orig + radius) / (2.0 * radius / IndexSize));
    if (idx > IndexSize)
        idx = IndexSize;
    return idx;
}

__device__
double* key_sample(cudautils::Keypoint key, double* image, cudautils::SiftParams sift_params) {

    /*FV fv = sphere_tri(sift_params.Tessellation_levels,1);*/

    double xySpacing = key.xyScale * sift_params.MagFactor;
    double tSpacing = key.tScale * sift_params.MagFactor;

    int xyiradius = round(1.414 * xySpacing * (sift_params.IndexSize + 1) / 2.0);
    int tiradius = round(1.414 * tSpacing * (sift_params.IndexSize + 1) / 2.0);

    int N = sift_params.IndexSize * sift_params.IndexSize * sift_params.IndexSize * sift_params.nFaces;
    /*double* index = (double*) calloc(N, sizeof(double));*/
    double* index;
    cudaMalloc(&index, N * sizeof(double));
    for (int i=0; i < N; i++) {
        index[i] = 0.0;
    }
    /*cudaMemset(index, 0.0, N * sizeof(double));*/

    int r, c, t, i_bin, j_bin, k_bin;
    double distsq;
    for (int i = -xyiradius; i <= xyiradius; i++) {
        for (int j = -xyiradius; j <= xyiradius; j++) {
            for (int k = -tiradius; j <= tiradius; k++) {

                distsq = (double) pow(i,2) + pow(j,2) + pow(k,2);

                // Find bin idx
                // FIXME check correct
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
                    add_sample(index, image, distsq, r, c, t,
                            i_bin, j_bin, k_bin, sift_params);
                }
            }
        }
    }

    return index;
}

__device__
double* build_ori_hists(int x, int y, int z, int radius, double* image, cudautils::SiftParams sift_params) {

    double* ori_hist;
    //FIXME thrust::fill
    cudaMalloc(&ori_hist, sift_params.nFaces * sizeof(double));
    for (int i=0; i < sift_params.nFaces; i++) {
        ori_hist[i] = 0.0;
    }
    /*cudaMemset(ori_hist, 0.0, sift_params.nFaces * sizeof(double));*/
    /*double* ori_hist = (double*) calloc(sift_params.nFaces,sizeof(double));*/

    double mag;
    double vect[3] = {1.0, 0.0, 0.0};
    int* ix;
    cudaMalloc(&ix, sift_params.fv_centers_len*sizeof(int));
    double* yy;
    int r, c, t;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            for (int k = -radius; j <= radius; k++) {
                // Find original image pixel idx
                r = x + i;
                c = y + j;
                t = z + k;

                // only add if within image range
                if (!(r < 0  ||  r >= sift_params.image_size0 ||
                        c < 0  ||  c >= sift_params.image_size1
                        || t < 0 || t >= sift_params.image_size2)) {
                    /*gradient and orientation vectors calculated from 3D halo/neighboring pixels*/
                    mag = get_grad_ori_vector(image,r,c,t, vect, yy, ix, sift_params);
                    ori_hist[ix[0]] += mag;
                }
            }
        }
    }
    return ori_hist;
}

__device__
void normalize_vec(double* vec, int len) {

    double sqlen = 0.0;
    for (int i=0; i < len; i++) {
        sqlen += vec[i] * vec[i];
    }

    double fac = 1.0 / sqrt(sqlen);
    for (int i=0; i < len; i++) {
        vec[i] = vec[i] * fac;
    }
    return;
}

__device__
cudautils::Keypoint make_keypoint_sample(cudautils::Keypoint key, double* image, cudautils::SiftParams sift_params) {

    //FIXME add to sift_params from Matlab side
    sift_params.MaxIndexVal = 0.2;
    bool changed = false;

    //FIXME make sure vec is in column order
    double* vec = key_sample(key, image, sift_params);
    int N = sift_params.IndexSize * sift_params.IndexSize * sift_params.IndexSize * sift_params.nFaces;

    normalize_vec(vec, N);

    for (int i=0; i < N; i++) {
        if (vec[i] > sift_params.MaxIndexVal) {
            vec[i] = sift_params.MaxIndexVal;
            changed = true;
        }
    }

    if (changed) {
        normalize_vec(vec, N);
    }

    int intval;
    for (int i=0; i < N; i++) {
        intval = round(512.0 * vec[i]);
        key.ivec[i] = (int) min(255, intval);
    }
    return key;
}


__device__
cudautils::Keypoint make_keypoint(double* image, int x, int y, int z, cudautils::SiftParams sift_params) {
    cudautils::Keypoint key;
    key.x = x;
    key.y = y;
    key.z = z;
    key.xyScale = sift_params.xyScale;
    key.tScale = sift_params.tScale;
    return make_keypoint_sample(key, image, sift_params);
}

/*__device__ __host__*/
/*void ind2sub(long idx*/

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
        unsigned int map_idx_size,
        unsigned int *map_idx,
        int8_t *map,
        double *image,
        cudautils::SiftParams sift_params,
        uint8_t *descriptors) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= map_idx_size) return;
    unsigned int idx = map_idx[i];

    //FIXME use reRun var
    // column-major order since image is from matlab
    int x, y, z;
    x = idx % sift_params.image_size0;
    y = (idx - x)/sift_params.image_size0 % sift_params.image_size1;
    z = ((idx - x)/sift_params.image_size0 - y)/sift_params.image_size1;
#ifdef DEBUG_OUTPUT
    printf("thread: %u, desc index: %u, x:%d y:%d z:%d\n", i, idx, x, y, z);
#endif
    return;
    
    cudautils::Keypoint key;
    bool reRun = false;
    int radius = round(sift_params.xyScale * 3.0);

    int ori_hist_len = sift_params.nFaces;
    int* ix;
    cudaMalloc(&ix, ori_hist_len*sizeof(int));

    thrust::sequence(thrust::device, ix, ix + ori_hist_len);
    double* ori_hist = build_ori_hists(x, y, z, radius, image, sift_params);
    // descending order by ori_hist
    thrust::sort_by_key(thrust::device, ix, ix + ori_hist_len, ori_hist, thrust::greater<int>());
        
    if (sift_params.TwoPeak_Flag &&
            //FIXME must be in row order
            (dot_product(&sift_params.fv_centers[ix[0]],
                &sift_params.fv_centers[ix[1]],
                3) > .9) &&
            (dot_product(&sift_params.fv_centers[ix[0]],
                &sift_params.fv_centers[ix[2]], 3) > .9)) {
        reRun = true;
        return ;
    }

    key = make_keypoint(image, x, y, z, sift_params);

    cudaMemcpy(&(descriptors[i * sift_params_.descriptor_len]), &(key.ivec), 
            sift_params.descriptor_len * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
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
        cudautils::SiftParams sift_params)
    : x_size_(x_size), y_size_(y_size), z_size_(z_size),
        x_sub_size_(x_sub_size), y_sub_size_(y_sub_size),
        dx_(dx), dy_(dy), dw_(dw),
        num_gpus_(num_gpus), num_streams_(num_streams),
        sift_params_(sift_params),
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

void Sift::setMapToBeInterpolated(const int8_t *map)
{
    thrust::copy(map, map + (x_size_ * y_size_ * z_size_), dom_data_->h_map);
}

void Sift::setMapToBeInterpolated(const std::vector<int8_t>& map)
{
    assert((x_size_ * y_size_ * z_size_) == map.size());

    thrust::copy(map.begin(), map.end(), dom_data_->h_map);
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

    cudaSetDevice(gpu_id);

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];
    std::shared_ptr<SubDomainDataOnStream> stream_data0 = subdom_data->stream_data[0];

    unsigned int x_sub_i = subdom_data->x_sub_i_list[gpu_task_id];
    unsigned int y_sub_i = subdom_data->y_sub_i_list[gpu_task_id];
#ifdef DEBUG_OUTPUT
    CudaTimer timer;
    logger_->info("===== gpu_id={} x_sub_i={} y_sub_i={}", gpu_id, x_sub_i, y_sub_i);
#endif

    unsigned int x_sub_start = x_sub_i * x_sub_size_;
    unsigned int x_sub_delta = get_delta(x_size_, x_sub_i, x_sub_size_);
    unsigned int y_sub_start = y_sub_i * y_sub_size_;
    unsigned int y_sub_delta = get_delta(y_size_, y_sub_i, y_sub_size_);
    unsigned int base_x_sub  = (x_sub_i > 0 ? 0 : dw_);
    unsigned int base_y_sub  = (y_sub_i > 0 ? 0 : dw_);

    unsigned int padding_x_sub_start = x_sub_start - (x_sub_i > 0 ? dw_ : 0);
    unsigned int padding_x_sub_delta = x_sub_delta + (x_sub_i > 0 ? dw_ : 0) + (x_sub_i < num_x_sub_ - 1 ? dw_ : 0);
    unsigned int padding_y_sub_start = y_sub_start - (y_sub_i > 0 ? dw_ : 0);
    unsigned int padding_y_sub_delta = y_sub_delta + (y_sub_i > 0 ? dw_ : 0) + (y_sub_i < num_y_sub_ - 1 ? dw_ : 0);
#ifdef DEBUG_OUTPUT
    unsigned int x_sub_end = x_sub_start + x_sub_delta;
    unsigned int y_sub_end = y_sub_start + y_sub_delta;
    logger_->debug("x_sub=({},{},{}) y_sub=({},{},{})", x_sub_start, x_sub_delta, x_sub_end, y_sub_start, y_sub_delta, y_sub_end);
    logger_->debug("base_x_sub={},base_y_sub={}", base_x_sub, base_y_sub);
#endif

    size_t padded_sub_volume_size = x_sub_stride_ * y_sub_stride_ * z_stride_;

    int8_t *padded_sub_map;
    double *padded_sub_image;
    cudaHostAlloc(&padded_sub_map,   padded_sub_volume_size * sizeof(int8_t), cudaHostAllocPortable);
    cudaHostAlloc(&padded_sub_image, padded_sub_volume_size * sizeof(double), cudaHostAllocPortable);

    // First set all values to -1
    thrust::fill(padded_sub_map, padded_sub_map + padded_sub_volume_size, -1);

    for (unsigned int k = 0; k < z_size_; k++) {
        for (unsigned int j = 0; j < padding_y_sub_delta; j++) {
            size_t src_idx = dom_data_->sub2ind(padding_x_sub_start, padding_y_sub_start + j, k);
            size_t dst_idx = subdom_data->pad_sub2ind(base_x_sub, base_y_sub + j, dw_ + k);

            int8_t* src_map_begin = &(dom_data_->h_map[src_idx]);
            int8_t* dst_map_begin = &(padded_sub_map[dst_idx]);
            thrust::copy(src_map_begin, src_map_begin + padding_x_sub_delta, dst_map_begin);

            double* src_image_begin = &(dom_data_->h_image[src_idx]);
            double* dst_image_begin = &(padded_sub_image[dst_idx]);
            thrust::copy(src_image_begin, src_image_begin + padding_x_sub_delta, dst_image_begin);
        }
    }

    thrust::fill(thrust::device, subdom_data->padded_image, subdom_data->padded_image + padded_sub_volume_size, 0.0);

    cudaMemcpyAsync(
            subdom_data->padded_image,
            padded_sub_image,
            padded_sub_volume_size * sizeof(double),
            cudaMemcpyHostToDevice, stream_data0->stream);

#ifdef DEBUG_OUTPUT
    cudaStreamSynchronize(stream_data0->stream);
    logger_->info("transfer image data {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
    logger_->info("===== dev image");
    print_matrix3d(logger_, x_size_, y_size_, 0, 0, 0, x_size_, y_size_, z_size_, dom_data_->h_image);
    print_matrix3d_dev(logger_, x_sub_stride_, y_sub_stride_, z_stride_, 0, 0, 0, x_sub_stride_, y_sub_stride_, z_stride_, subdom_data->padded_image);
#endif

    timer.reset();
#endif

    cudaMemcpyAsync(
            subdom_data->padded_map,
            padded_sub_map,
            padded_sub_volume_size * sizeof(int8_t),
            cudaMemcpyHostToDevice, stream_data0->stream);

#ifdef DEBUG_OUTPUT
    cudaStreamSynchronize(stream_data0->stream);
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

    auto end_itr = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(padded_sub_volume_size),
            subdom_data->padded_map,
            subdom_data->padded_map_idx,
            thrust::logical_not<int8_t>());

    subdom_data->padded_map_idx_size = end_itr - subdom_data->padded_map_idx;

    // set all padded map boundaries to 0 for correctness to
    // distinguish boundaries
    thrust::replace(thrust::device, subdom_data->padded_map, subdom_data->padded_map + padded_sub_volume_size, -1, 0);

#ifdef DEBUG_OUTPUT
    cudaStreamSynchronize(stream_data0->stream);
    logger_->info("calculate map idx {}", timer.get_laptime());

    logger_->info("padded_map_idx_size={}", subdom_data->padded_map_idx_size);
//    logger_->debug("===== padded_map idx");
//    std::copy(subdom_data->padded_map_idx.begin(), end_itr, std::ostream_iterator<unsigned int>(std::cout, ","));
//    std::cout << std::endl;

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
    cudaStreamSynchronize(stream_data0->stream);

    cudaFreeHost(padded_sub_map);
    cudaFreeHost(padded_sub_image);
}

cudautils::SiftParams Sift::get_sift_params() {
    return sift_params_;
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

#ifdef DEBUG_OUTPUT
    CudaTimer timer(stream_data->stream);
#endif

    for (auto dx_itr = stream_data->dx_i_list.begin(), dy_itr = stream_data->dy_i_list.begin();
            dx_itr != stream_data->dx_i_list.end() || dy_itr != stream_data->dy_i_list.end(); dx_itr++, dy_itr++) {

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
        logger_->info("padded_map_idx_size={}", subdom_data->padded_map_idx_size);
#endif


        // for each subdomain of stream of this for loop
        unsigned int *padded_map_idx;
        cudaMalloc(&padded_map_idx, subdom_data->padded_map_idx_size * sizeof(unsigned int));

        RangeCheck range_check { x_sub_stride_, y_sub_stride_,
            dx_start + dw_, dx_end + dw_, dy_start + dw_, dy_end + dw_, dw_, z_size_ + dw_ };

        // copy the relevant (in range) idx elements from the
        // global GPU padded_map_idx to the local padded_map_idx for each
        // sub stream (subdom_data->padded_map_idx[stream_id])
        auto end_itr = thrust::copy_if(
                thrust::device,
                subdom_data->padded_map_idx,
                subdom_data->padded_map_idx + subdom_data->padded_map_idx_size,
                padded_map_idx,
                range_check);

        unsigned int padded_map_idx_size = end_itr - padded_map_idx;

#ifdef DEBUG_OUTPUT
        logger_->info("padded_map_idx_size={}", padded_map_idx_size);
        logger_->info("transfer map idx {}", timer.get_laptime());

        cudaStreamSynchronize(stream_data->stream);

        thrust::device_vector<unsigned int> dbg_d_padded_map_idx(padded_map_idx, padded_map_idx + padded_map_idx_size);
        thrust::host_vector<unsigned int> dbg_h_padded_map_idx(dbg_d_padded_map_idx);
        for (unsigned int i = 0; i < padded_map_idx_size; i++) {
            logger_->debug("padded_map_idx={}", dbg_h_padded_map_idx[i]);
        }
        timer.reset();
#endif
        if (padded_map_idx_size == 0) {
#ifdef DEBUG_OUTPUT
            logger_->debug("no map to be padded");
#endif
            continue;
        }

        /*FIXME
        Get sift_params_.IndexSize, for full keypoint size
        each descriptor is placed in according 0 to padded_map_idx_size
        essentially a matrix of padded_map_idx_size by descriptor length
        */
        uint8_t *descriptors;
        long desc_mem_size = sift_params_.descriptor_len * padded_map_idx_size * sizeof(uint8_t);
        cudaMalloc(&descriptors, desc_mem_size);

        unsigned int num_threads = 1;
        unsigned int num_blocks = get_num_blocks(padded_map_idx_size, num_threads);
#ifdef DEBUG_OUTPUT
        logger_->info("num_blocks={}", num_blocks);
#endif

#ifdef DEBUG_OUTPUT
        logger_->info("create_descriptor");
#endif

        // create device keypoints
        /*Keypoints * h_keypoints = (Keypoints *) malloc(sizeof(Keypoint) * padded_map_idx_size);*/
        /*Keypoints * d_keypoints;*/
        /*cudaMalloc(&d_keypoints, sizeof(Keypoint) * padded_map_idx_size);*/

        // sift_params.fv_centers must be placed on device since array
        double* device_centers;
        cudaMalloc(&device_centers, sizeof(double) * sift_params_.fv_centers_len);
        cudaMemcpy(&device_centers, sift_params_.fv_centers,
                sizeof(*(sift_params_.fv_centers)) *
                sift_params_.fv_centers_len, cudaMemcpyHostToDevice);
        cudaMalloc(&(sift_params_.fv_centers), sizeof(double) * sift_params_.fv_centers_len);
        cudaMemcpy(&(sift_params_.fv_centers), &device_centers,
                sizeof(*(sift_params_.fv_centers)) *
                sift_params_.fv_centers_len, cudaMemcpyDeviceToDevice);

        create_descriptor<<<num_blocks, num_threads, 0, stream_data->stream>>>(
                x_sub_stride_, y_sub_stride_, padded_map_idx_size,
                padded_map_idx,//substream map
                subdom_data->padded_map,//global map for GPU
                subdom_data->padded_image,
                sift_params_,
                descriptors); 

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

        /*// memcpy the descriptors (do not contain ivec)*/
        /*cudaMemcpy(h_keypoints, d_keypoints, sizeof(Keypoint) * padded_map_idx_size);*/

#ifdef DEBUG_OUTPUT
        logger_->info("create descriptors {}", timer.get_laptime());

        //debug
//        cudaStreamSynchronize(stream_data->stream);
//        std::copy(interpolated_values.begin(),
//                  interpolated_values.begin() + padded_map_idx_size,
//                  std::ostream_iterator<double>(std::cout, ","));
//        std::cout << std::endl;

        timer.reset();
#endif

        // transfer vector descriptors via host pinned memory for faster async cpy
        double *h_descriptors;
        cudaHostAlloc(&h_descriptors, desc_mem_size, cudaHostAllocPortable);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

        cudaMemcpyAsync(
                h_descriptors,
                descriptors,
                desc_mem_size,
                cudaMemcpyDeviceToHost, stream_data->stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

        // transfer index map to host for referencing correct index
        unsigned int *h_padded_map_idx;
        cudaHostAlloc(&h_padded_map_idx, padded_map_idx_size * sizeof(unsigned
                    int), cudaHostAllocPortable);
        cudaMemcpyAsync(
                h_padded_map_idx,
                padded_map_idx,
                padded_map_idx_size * sizeof(unsigned int),
                cudaMemcpyDeviceToHost, stream_data->stream);

        /*//build list of keypoint structs*/
        /*for (unsigned int i = 0; i < padded_map_idx_size; i++) {*/
            /*Keypoint temp;*/
            /*temp.ivec = (double*) malloc(sift_params.descriptor_len * sizeof(double));*/
            /*memcpy(&(temp.ivec), h_descriptors[i * sift_params_.descriptor_len], */
                    /*sift_params_.descriptor_len, sizeof(double));*/
            /*temp.xyScale = sift_params_.xyScale;*/
            /*temp.tScale = sift_params_.tScale;*/
            /*h_keypoints[i] = temp;*/
        /*}*/

        // make sure all streams are done
        cudaStreamSynchronize(stream_data->stream);

        save data for all streams to global Sift object store
        for (unsigned int i = 0; i < padded_map_idx_size; i++) {
            Keypoint temp;
            temp.ivec = (double*) malloc(sift_params.descriptor_len * sizeof(double));
            memcpy(&(temp.ivec), h_descriptors[i * sift_params_.descriptor_len], 
                    sift_params_.descriptor_len, sizeof(double));
            temp.xyScale = sift_params_.xyScale;
            temp.tScale = sift_params_.tScale;

            unsigned int padding_x;
            unsigned int padding_y;
            unsigned int padding_z;
            ind2sub(x_sub_stride_, y_sub_stride_, h_padded_map_idx[i], padding_x, padding_y, padding_z);
            size_t idx = dom_data_->sub2ind(padding_x - dw_, padding_y - dw_, padding_z - dw_);

            //FIXME Need a unique id for each keypoint regardless of stream
            //dom_data will be switched to keypoint_store
            /*dom_data_->h_image[idx] = h_descriptors[i];*/
        }

        cudaFree(padded_map_idx);
        cudaFree(descriptors);

        cudaFreeHost(h_descriptors);
        cudaFreeHost(h_padded_map_idx);

#ifdef DEBUG_OUTPUT
        logger_->info("transfer d2h and copy descriptor ivec values {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
        logger_->debug("===== host interp image");
        print_matrix3d(logger_, x_size_, y_size_, 0, 0, 0, x_size_, y_size_, z_size_, dom_data_->h_image);
#endif
#endif
    }
}


} // namespace cudautils

