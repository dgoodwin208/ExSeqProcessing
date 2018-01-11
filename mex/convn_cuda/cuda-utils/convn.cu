#include <cudnn.h>
#include <iostream>
#include <cstdlib>

#include "convn.h"

namespace cudautils {

#define checkCUDNN(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

float* convn(float *image, const int channels, const int height, const int width, float *kernel, const int kernel_channels, const int kernel_height, const int kernel_width)  {
    // process args
    // input, kernel, and output descriptors must all have the same dimensions : 3
    int batch_size = 1;
    int out_channels = 3; //feature maps
    int in_channels = 3;
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // define input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                /*format=*/CUDNN_TENSOR_NCHW,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/batch_size,
                /*channels=*/channels,
                /*image_height=*/height,
                /*image_width=*/width));

    /*Real code*/
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*format=*/CUDNN_TENSOR_NCHW, // for defining weights
                /*out_channels=*/out_channels,
                /*in_channels=*/in_channels,
                /*kernel_height=*/kernel_height,
                /*kernel_width=*/kernel_width));

    // describe the convolution
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    // define the zero padding size for each dimension
    const int padA[] = {1, 1, 1};
    // for each dim, define num elements to stride for eacr point
    const int filterStrideA[] = {1, 1, 1};
    const int dilationA[] = {1, 1, 1}; // dilation factor
    const int dimensions = 3; // number of dimensions of the convolution / input matrix
    checkCUDNN(cudnnSetConvolutionNdDescriptor(convolution_descriptor,
            /*array_length=*/dimensions,
            /*padA=*/padA,
            /*filterStrideA=*/filterStrideA,
            /*dilationA=*/dilationA,
            // /*mode=*/CUDNN_CONVOLUTION,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    /*// get output dimensions of convolutions for allocating correct space*/
    /*// holds size of output tensor*/
    /*// output dims of this function must be strictly respected for `cudnnConvolutionForward()`*/
    /*int tensorOutputDimA[] = {0, 0, 0};*/
    /*checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convolution_descriptor,*/
                /*input_descriptor,*/
                /*kernel_descriptor,*/
                /*[>nbDims=<]dimensions,*/
                /*tensorOutputDimA));*/

    /*std::cout << "Output dimensions: " << tensorOutputDimA[0] << ", " << tensorOutputDimA[1] << ", " << tensorOutputDimA[2] << std::endl;*/

    std::cout << "Output dimensions: " << channels << ", " << height << ", " << width << std::endl;

    //  output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                /*format=*/CUDNN_TENSOR_NCHW,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/batch_size,
                /*channels=*/channels,
                /*image_height=*/height,
                /*image_width=*/width));


    /*std::cout << "Starting convolution algorithm" << std::endl;*/
    /*// choose convolution algorithm*/
    /*cudnnConvolutionFwdAlgo_t convolution_algorithm;*/
    /*size_t memoryLimitInBytes = 0;*/
    /*// allow unlimited memory*/
    /*checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,*/
            /*input_descriptor,*/
            /*kernel_descriptor,*/
            /*convolution_descriptor,*/
            /*output_descriptor,*/
            /*CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,*/
            /*0, */
            /*&convolution_algorithm)); // save chosen algo*/
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

    // print convolution_algorithm to stdout
    std::cout << "Convolution algorithm: " << convolution_algorithm << std::endl;

    // choose workspace byte size
   std::cout << "Getting workspace size" << std::endl;
   size_t workspace_bytes{0};
   checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
               input_descriptor,
               kernel_descriptor,
               convolution_descriptor,
               output_descriptor,
               convolution_algorithm,
               &workspace_bytes));
   std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

   // allocation of mem buffers
   void* d_workspace{nullptr};
   cudaMalloc(&d_workspace, workspace_bytes);

   /*int image_bytes = batch_size * tensorOutputDimA[0] * tensorOutputDimA[1] * tensorOutputDimA[2] * sizeof(float);*/
   int image_bytes = batch_size * channels * height * width * sizeof(float);

   float* d_input{nullptr};
   cudaMalloc(&d_input, image_bytes);
   // copy to GPU device
   cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);

   float* d_output{nullptr};
   cudaMalloc(&d_output, image_bytes);
   // guarantee all vals 0
   cudaMemset(d_output, 0, image_bytes);

   float* d_kernel{nullptr};
   cudaMalloc(&d_kernel, sizeof(kernel));
   // copy to device
   cudaMemcpy(d_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);

   // Convolution
   std::cout << "Compute convolution" << std::endl;
   const float alpha = 1.0f;
   const float beta = 0.0f;
   checkCUDNN(cudnnConvolutionForward(cudnn,
               &alpha, 
               input_descriptor, 
               d_input,
               kernel_descriptor,
               d_kernel,
               convolution_descriptor,
               convolution_algorithm,
               d_workspace,
               workspace_bytes,
               &beta,
               output_descriptor,
               d_output));

   // Copy data back to host
   float* h_output = new float[image_bytes];
   cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

   // free
   // delete[] h_output
   cudaFree(d_kernel);
   cudaFree(d_input);
   cudaFree(d_output);
   cudaFree(d_workspace);

   cudnnDestroyTensorDescriptor(input_descriptor);
   cudnnDestroyTensorDescriptor(output_descriptor);
   cudnnDestroyFilterDescriptor(kernel_descriptor);
   cudnnDestroyConvolutionDescriptor(convolution_descriptor);

   cudnnDestroy(cudnn);
   std::cout << "Success, exiting" << std::endl;

   return h_output;
}

} //namespace
