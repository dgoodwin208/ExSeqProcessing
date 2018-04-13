// Single GPU version of 3DSIFT via CUDA

// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

// includes, project
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cuda_runtime.h>

/*typedef thrust::host_vector<double, thrust::system::cuda::experimental::pinned_allocator<double>> pinnedDblHostVector;*/
/*typedef thrust::host_vector<unsigned int, thrust::system::cuda::experimental::pinned_allocator<int>> pinnedUIntHostVector;*/

namespace cudautils {

/*int calculate_3DSIFT(double* img, double* keypts, bool skipDescriptor, struct* keys) {*/

/*LoadParams;*/
/*sift_params.pix_size = size(img);*/
/*i = 0;*/
/*offset = 0;*/
/*precomp_grads = {};*/
/*precomp_grads.count = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));*/
/*precomp_grads.mag = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));*/
/*precomp_grads.ix = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...*/
    /*sift_params.Tessel_thresh, 1);*/
/*precomp_grads.yy = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...*/
    /*sift_params.Tessel_thresh, 1);*/
/*precomp_grads.vect = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), 1, 3);*/
/*while 1*/

    /*reRun = 1;*/
    /*i = i+1;*/
    
    /*while reRun == 1*/
        
        /*loc = keypts(i+offset,:);*/
        /*%fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);*/
        
        /*% Create a 3DSIFT descriptor at the given location*/
        /*if ~skipDescriptor*/
            /*[keys{i} reRun precomp_grads] = Create_Descriptor(img,1,1,loc(1),loc(2),loc(3),sift_params, precomp_grads);*/
        /*else         */
            /*clear k; reRun=0;*/
            /*k.x = loc(1); k.y = loc(2); k.z = loc(3);*/
            /*keys{i} = k;*/
        /*end*/

        /*if reRun == 1*/
            /*offset = offset + 1;*/
        /*end*/
        
        /*%are we out of data?*/
        /*if i+offset>=size(keypts,1)*/
            /*break;*/
        /*end*/
    /*end*/
    
    /*%are we out of data?*/
    /*if i+offset>=size(keypts,1)*/
            /*break;*/
    /*end*/
/*end*/

} // namespace cufftutils
