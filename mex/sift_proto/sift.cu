#include <iostream>
#include <future>

#include <thrust/copy.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>
#include <cmath>

#include "sift.h"
#include "matrix_helper.h"
#include "cuda_timer.h"

#include "spdlog/spdlog.h"


namespace cudautils {

struct FV {
    double* vertices;
    double* faces;
    double* centers;
}

struct Keypoint {
    int x;
    int y;
    int z;
    double xyScale;
    double tScale;
    double* ivec; //stores the flattened descriptor vector
};

struct SiftParams {
    double MagFactor;
    int IndexSize;
    int nFaces;
}

FV sphere_tri(int maxlevel, int r) {
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
     mesh_refine).  At each refinement, the vertices are
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

    // default maximum subdivision level
    if (maxlevel < 0)
        maxlevel = 0;

    // default radius
    if (r < 0)
        r = 1;

    // define the icosehedron

    // Twelve vertices of icosahedron on unit sphere
    double tau = 0.8506508084; // t=(1+sqrt(5))/2, tau=t/sqrt(1+t^2)
    double one = 0.5257311121; // one=1/sqrt(1+t^2) , unit sphere

    FV fv;
    
    // store the vertices in column (Matlab) order 
    fv.vertices( 1,:) = [  tau,  one,    0 ]; // ZA
    fv.vertices( 2,:) = [ -tau,  one,    0 ]; // ZB
    fv.vertices( 3,:) = [ -tau, -one,    0 ]; // ZC
    fv.vertices( 4,:) = [  tau, -one,    0 ]; // ZD
    fv.vertices( 5,:) = [  one,   0 ,  tau ]; // YA
    fv.vertices( 6,:) = [  one,   0 , -tau ]; // YB
    fv.vertices( 7,:) = [ -one,   0 , -tau ]; // YC
    fv.vertices( 8,:) = [ -one,   0 ,  tau ]; // YD
    fv.vertices( 9,:) = [   0 ,  tau,  one ]; // XA
    fv.vertices(10,:) = [   0 , -tau,  one ]; // XB
    fv.vertices(11,:) = [   0 , -tau, -one ]; // XC
    fv.vertices(12,:) = [   0 ,  tau, -one ]; // XD
    
    // Structure for unit icosahedron
    fv.faces = [  5,  8,  9 ;
               5, 10,  8 ;
               6, 12,  7 ;
               6,  7, 11 ;
               1,  4,  5 ;
               1,  6,  4 ;
               3,  2,  8 ;
               3,  7,  2 ;
               9, 12,  1 ;
               9,  2, 12 ;
              10,  4, 11 ;
              10, 11,  3 ;
               9,  1,  5 ;
              12,  6,  1 ;
               5,  4, 10 ;
               6, 11,  4 ;
               8,  2,  9 ;
               7, 12,  2 ;
               8, 10,  3 ;
               7,  3, 11 ];
    


    // -----------------
    // refine the starting shapes with subdivisions
    if maxlevel,
        
        // Subdivide each starting triangle (maxlevel) times
        for level = 1:maxlevel,
            
            // Subdivide each triangle and normalize the new points thus
            // generated to lie on the surface of a sphere radius r.
            fv = mesh_refine_tri4(fv);
            fv.vertices = sphere_project(fv.vertices,r);
            
            // An alternative might be to define a min distance
            // between vertices and recurse or use fminsearch
            
        end
    end

    for (int i=0; i < length(fv.faces); i++) {
        fv.centers(i,:) = mean(fv.vertices(fv.faces(i,:),:));
        // Unit Normalization
        fv.centers(i,:) = fv.centers(i,:) ./ sqrt(dot(fv.centers(i,:),fv.centers(i,:)));
    }
}

__device__
place_in_index(double* index, double mag, int i, int j, int s, 
        double* yy, double* ix, double* sift_params) {

    double tmpsum;
    /*FIXME*/
    /*int bin_index = bin_sub2ind(i,j,s);*/
    if (sift_params.Smooth_Flag) {
        tmpsum = sum(yy(1:sift_params.Tessel_thresh).^sift_params.Smooth_Var);
        // Add to the three nearest tesselation faces
        for (int ii=0; ii<sift_params.Tessel_thresh; ii++) {
            /*FIXME*/
            /*int bin_index = bin_sub2ind(i,j,s);*/
            index[bin_index] = index[bin_index] + ( mag * pow( yy(ii),
                        sift_params.Smooth_Var ) / tmpsum );
        }
    }
        index[bin_index] = index[bin_index] + mag;
    }

}

/*r, c, s is the pixel index (x, y, z dimensions respect.) in the image within the radius of the */
/*keypoint before clamped*/
/*For each pixel, take a neighborhhod of xyradius and tiradius,*/
/*bin it down to the sift_params.IndexSize dimensions*/
/*thus, i_indx, j_indx, s_indx represent the binned index within the radius of the keypoint*/
__device__
void add_sample(double* index, double* image, double distsq, int r, int c, int s, int i_indx, int j_indx, int s_indx, FV fv, SiftParams sift_params) {

    double sigma = sift_params.SigmaScaled;
    double weight = exp(-(distsq / (2.0 * sigma * sigma)));

    double mag;
    double* vect, yy, ix;
    /*gradient and orientation vectors calculated from 3D halo/neighboring pixels*/
    get_grad_ori_vector(image,r,c,s, fv, mag, vect, yy, ix, sift_params);
    double mag = weight * mag; // scale magnitude by gaussian 

    place_in_index(index, mag, i_indx, j_indx, s_indx, yy, ix, sift_params);
}

// assumes r,c,s lie within accessible image boundaries
__device__
void get_grad_ori_vector(double* image, int r, int c, int s, FV fv, 
        double mag, double* vect, double* yy, double* ix, SiftParams sift_params) {

    //FIXME subscripts to linear ind
    double xgrad = image[r,c+1,s] - image[r,c-1,s];
    double ygrad = image[r-1,c,s] - image[r+1,c,s];
    double zgrad = image[r,c,s+1] - image[r,c,s-1];

    double mag = sqrt(xgrad * xgrad + ygrad * ygrad + zgrad * zgrad);

    if (mag !=0)
        vect = {xgrad / mag, ygrad / mag, zgrad / mag};
    else
        vect = {1 0 0};
    end

    //Find the nearest tesselation face indices
    //FIXME this needs to be done in c++
    corr_array = fv.centers * vect';
    //FIXME this needs to be done in c++
    [yy ix] = sort(corr_array,'descend');
}

double* KeySample(key, image, sift_params) {

    FV fv = sphere_tri(sift_params.Tessellation_levels,1);

    irow = int16(key.x);
    icol = int16(key.y);
    islice = int16(key.z);

    xySpacing = key.xyScale * sift_params.MagFactor;
    tSpacing = key.tScale * sift_params.MagFactor;

    xyRadius = 1.414 * xySpacing * (sift_params.IndexSize + 1) / 2.0;
    tRadius = 1.414 * tSpacing * (sift_params.IndexSize + 1) / 2.0;
    xyiradius = int16(xyRadius);
    tiradius = int16(tRadius);

    index = zeros(sift_params.IndexSize,sift_params.IndexSize,sift_params.IndexSize,sift_params.nFaces);

    for i = -xyiradius:xyiradius
        for j = -xyiradius:xyiradius
            for s = -tiradius:tiradius

                % This is redundant and probably slows down the code, but at
                % some point this solved a major overflow headache, so leaving
                % as-is for now
                i2 = double(i);
                j2 = double(j);
                s2 = double(s);
                distsq = double(i2^2 + j2^2 + s2^2);

                v0 = [i2; j2; s2];

                
                i_indx = int16(floor(double((i + xyiradius)) / double((2*xyiradius/sift_params.IndexSize)))) + 1;
                j_indx = int16(floor(double((j + xyiradius)) / double((2*xyiradius/sift_params.IndexSize)))) + 1;
                s_indx = int16(floor(double((s + tiradius)) / double((2*tiradius/sift_params.IndexSize)))) + 1;
                
                if i_indx > sift_params.IndexSize
                    i_indx = sift_params.IndexSize;
                end
                if j_indx > sift_params.IndexSize
                    j_indx = sift_params.IndexSize;
                end
                if s_indx > sift_params.IndexSize
                    s_indx = sift_params.IndexSize;
                end

                %if (i_indx < 1 || j_indx < 1 || s_indx < 1)
                    %disp('Something wrong with the sub-histogram index');
                %end
                
                %For each pixel, take a neighborhhod of xyradius and tiradius,
                %bin it down to the sift_params.IndexSize dimensions
                r = irow + v0(1);
                c = icol + v0(2);
                t = islice + v0(3);

                // within image range
                if !(r < 1  ||  r > sift_params.image_size(1)  ||  c < 1  ||  c >
                        sift_params.image_size(2) || s < 1 || s > sift_params.image_size(3)) {
                    AddSample(index, image, distsq, r, c, t, i_indx, j_indx, s_indx, fv, sift_params, precomp_grads);
                }
            }
        }
    }

    return index;
}

// interpolate image data
//
__global__
void interpolate_volumes(
        unsigned int x_stride,
        unsigned int y_stride,
        unsigned int map_idx_size,
        unsigned int *map_idx,
        int8_t *map,
        double *image,
        double *interpolated_values) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= map_idx_size) return;

    unsigned int idx_zplane = map_idx[i] - 1 - x_stride - (x_stride * y_stride); // move current pos idx by (-1, -1, -1)
    unsigned int idx = idx_zplane;

    int sum_idx = 0;
    double sum = 0.0;

    // (-1, -1, -1)  ->  (1, -1, -1)
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 0, -1)  ->  (1, 0, -1)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 1, -1)  ->  (1, 1, -1)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    idx_zplane += x_stride * y_stride;
    idx = idx_zplane;

    // (-1, -1, 0)  ->  (1, -1, 0)
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 0, 0)  ->  (1, 0, 0)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 1, 0)  ->  (1, 1, 0)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    idx_zplane += x_stride * y_stride;
    idx = idx_zplane;

    // (-1, -1, 1)  ->  (1, -1, 1)
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 0, 1)  ->  (1, 0, 1)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    // (-1, 1, 1)  ->  (1, 1, 1)
    idx += x_stride;
    sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
    sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
    sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);

    if (sum_idx > 0) {
        interpolated_values[i] = sum / double(sum_idx);
    } else {
        idx_zplane = map_idx[i] - 2 * (1 + x_stride + (x_stride * y_stride)); // move current pos idx by (-2, -2, -2)

        // (u, v, w) <- (x, y, z)
        // u=0-4 v=0-4 w=0,4
        idx = idx_zplane;
        for (unsigned int v = 0; v < 5; v++) {
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
            sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);
            sum += image[idx + 3] * double(map[idx + 3]);   sum_idx += (map[idx + 3] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
            idx += x_stride;
        }
        idx = idx_zplane + 4 * x_stride * y_stride;
        for (unsigned int v = 0; v < 5; v++) {
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
            sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);
            sum += image[idx + 3] * double(map[idx + 3]);   sum_idx += (map[idx + 3] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
            idx += x_stride;
        }


        // u=0-4 v=0,4 w=1-3
        for (unsigned int w = 1; w < 4; w++) {
            idx = idx_zplane + w * x_stride * y_stride;
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
            sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);
            sum += image[idx + 3] * double(map[idx + 3]);   sum_idx += (map[idx + 3] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
            idx += 4 * x_stride;
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 1] * double(map[idx + 1]);   sum_idx += (map[idx + 1] > 0);
            sum += image[idx + 2] * double(map[idx + 2]);   sum_idx += (map[idx + 2] > 0);
            sum += image[idx + 3] * double(map[idx + 3]);   sum_idx += (map[idx + 3] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
        }

        // u=0,4 v=1-3 w=1-3
        for (unsigned int w = 1; w < 4; w++) {
            idx = idx_zplane + w * x_stride * y_stride;
            idx += x_stride;
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
            idx += x_stride;
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
            idx += x_stride;
            sum += image[idx + 0] * double(map[idx + 0]);   sum_idx += (map[idx + 0] > 0);
            sum += image[idx + 4] * double(map[idx + 4]);   sum_idx += (map[idx + 4] > 0);
        }

        if (sum_idx > 0) {
            interpolated_values[i] = sum / double(sum_idx);
        } else {
            interpolated_values[i] = 0.0;
        }
    }
    return;
}


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
        const unsigned int num_streams)
    : x_size_(x_size), y_size_(y_size), z_size_(z_size),
        x_sub_size_(x_sub_size), y_sub_size_(y_sub_size),
        dx_(dx), dy_(dy), dw_(dw),
        num_gpus_(num_gpus), num_streams_(num_streams),
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

    thrust::fill(thrust::device, subdom_data->padded_map_idx, subdom_data->padded_map_idx + padded_sub_volume_size, 0.0);

    auto end_itr = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(padded_sub_volume_size),
            subdom_data->padded_map,
            subdom_data->padded_map_idx,
            thrust::logical_not<int8_t>());

    subdom_data->padded_map_idx_size = end_itr - subdom_data->padded_map_idx;

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


        unsigned int *padded_map_idx;
        cudaMalloc(&padded_map_idx, subdom_data->padded_map_idx_size * sizeof(unsigned int));

        RangeCheck range_check { x_sub_stride_, y_sub_stride_,
            dx_start + dw_, dx_end + dw_, dy_start + dw_, dy_end + dw_, dw_, z_size_ + dw_ };

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

        double *interpolated_values;
        cudaMalloc(&interpolated_values, padded_map_idx_size * sizeof(double));


        unsigned int num_blocks = get_num_blocks(padded_map_idx_size, 1024);
#ifdef DEBUG_OUTPUT
        logger_->info("num_blocks={}", num_blocks);
#endif

        interpolate_volumes<<<num_blocks, 1024, 0, stream_data->stream>>>(
                x_sub_stride_, y_sub_stride_, padded_map_idx_size,
                padded_map_idx,
                subdom_data->padded_map,
                subdom_data->padded_image,
                interpolated_values);

#ifdef DEBUG_OUTPUT
        logger_->info("interpolate volumes {}", timer.get_laptime());

        //debug
//        cudaStreamSynchronize(stream_data->stream);
//        std::copy(interpolated_values.begin(),
//                  interpolated_values.begin() + padded_map_idx_size,
//                  std::ostream_iterator<double>(std::cout, ","));
//        std::cout << std::endl;

        timer.reset();
#endif

        double *h_interpolated_values;
        cudaHostAlloc(&h_interpolated_values, padded_map_idx_size * sizeof(double), cudaHostAllocPortable);

        cudaMemcpyAsync(
                h_interpolated_values,
                interpolated_values,
                padded_map_idx_size * sizeof(double),
                cudaMemcpyDeviceToHost, stream_data->stream);

        unsigned int *h_padded_map_idx;
        cudaHostAlloc(&h_padded_map_idx, padded_map_idx_size * sizeof(unsigned int), cudaHostAllocPortable);

        cudaMemcpyAsync(
                h_padded_map_idx,
                padded_map_idx,
                padded_map_idx_size * sizeof(unsigned int),
                cudaMemcpyDeviceToHost, stream_data->stream);

        cudaStreamSynchronize(stream_data->stream);
        for (unsigned int i = 0; i < padded_map_idx_size; i++) {
            unsigned int padding_x;
            unsigned int padding_y;
            unsigned int padding_z;
            ind2sub(x_sub_stride_, y_sub_stride_, h_padded_map_idx[i], padding_x, padding_y, padding_z);
            size_t idx = dom_data_->sub2ind(padding_x - dw_, padding_y - dw_, padding_z - dw_);

            dom_data_->h_image[idx] = h_interpolated_values[i];
        }

        cudaFree(padded_map_idx);
        cudaFree(interpolated_values);

        cudaFreeHost(h_interpolated_values);
        cudaFreeHost(h_padded_map_idx);

#ifdef DEBUG_OUTPUT
        logger_->info("transfer d2h and copy interpolated values {}", timer.get_laptime());

#ifdef DEBUG_OUTPUT_MATRIX
        logger_->debug("===== host interp image");
        print_matrix3d(logger_, x_size_, y_size_, 0, 0, 0, x_size_, y_size_, z_size_, dom_data_->h_image);
#endif
#endif
    }
}


} // namespace cudautils

