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

struct FV {
    double* vertices;
    double* faces;
    double* centers;
}

struct key {
    int x;
    int y;
    int z;
    double xyScale;
    double tScale;
};

struct sift_params {
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

    for i=1:length(fv.faces)
        fv.centers(i,:) = mean(fv.vertices(fv.faces(i,:),:));
        // Unit Normalization
        fv.centers(i,:) = fv.centers(i,:) ./ sqrt(dot(fv.centers(i,:),fv.centers(i,:)));
    end

}

int SIFT_handler() {
    return 0;
}

} // namespace cudautils

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
        /*//fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);*/
        
        /*// Create a 3DSIFT descriptor at the given location*/
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
        
        /*//are we out of data?*/
        /*if i+offset>=size(keypts,1)*/
            /*break;*/
        /*end*/
    /*end*/
    
    /*//are we out of data?*/
    /*if i+offset>=size(keypts,1)*/
            /*break;*/
    /*end*/
/*end*/

