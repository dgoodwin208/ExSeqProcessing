#include <cudnn.h>
#include <iostream>
#include <cstdlib>

#define checkCUDNN(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

float* convolution(int argc, char const *argv[])  {
    // process args
    int batch_size = 1;
    int channels = 1;
    int out_channels = 1; //feature maps
    int in_channels = 1;
    int height = 12;
    int width = 12;
    // printf("Creating image");
    std::cout << "Creating image" << std::endl;
    float image[batch_size][channels][height][width];
    /*for (int batch = 0; batch < batch_size; ++batch) {*/
        /*for (int channel = 0; channel < channels; ++channels) {*/
            /*for (int row = 0; row < height; ++row) {*/
                /*for (int col = 0; col < width; ++col) {*/
                    /*// image[batch][channel][row][col] = std::rand();*/
                    /*image[batch][channel][row][col] = 0.0f;*/
                /*}*/
            /*}*/
        /*}*/
    /*}*/
    /*std::cout << "Image created" << std::endl;*/
    const int kernel_height = 3;
    const int kernel_width = 3;
    // toy kernel until passed in via argv
    // clang-format off
    float kernel[kernel_height][kernel_width] = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}
    };
    // clang-format on
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

    // define output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                /*format=*/CUDNN_TENSOR_NCHW,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/batch_size,
                /*channels=*/channels,
                /*image_height=*/height,
                /*image_width=*/width));

    // define kernel tensor
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
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/1, //control the zero padding around the image
            /*pad_width=*/1,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CONVOLUTION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    std::cout << "Starting convolution algorithm" << std::endl;
    // choose convolution algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t memoryLimitInBytes = 0;
    // allow unlimited memory
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            memoryLimitInBytes, 
            &convolution_algorithm)); // save chosen algo
            /*memoryLimitInBytes=*/

    //TODO print convolution_algorithm to stdout
    std::cout << "Convolution algorithm: " << convolution_algorithm << std::endl;

    // choose workspace byte size
   std::cout << "Getting workspace size" << std::endl;
   size_t workspace_bytes = 0;
   checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
               input_descriptor,
               kernel_descriptor,
               convolution_descriptor,
               output_descriptor,
               convolution_algorithm,
               &workspace_bytes));
   std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

   // get output dimensions of convolutions for allocating correct space
   int n; //batch size
   int c; //channels (typically RGB chans)
   int h; //height
   int w; //width
   checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
               convolution_descriptor,
               input_descriptor,
               kernel_descriptor,
               &n,
               &c,
               &h,
               &w))

   // allocation of mem buffers

   void* d_workspace{nullptr};
   cudaMalloc(&d_workspace, workspace_bytes);

   int image_bytes = n * c * h * w * sizeof(float);

   float* d_input{nullptr};
   cudaMalloc(&d_input, image_bytes);
   // copy to GPU device
   cudaMemcpy(d_input, image, image_bytes, cudaMemcpyHostToDevice);

   float* d_output{nullptr};
   cudaMalloc(&d_output, image_bytes);
   // guarantee all vals 0
   cudaMemset(d_output, 0, image_bytes);

   // copy into higher dim, once for each channel, once for each output feature map (out_channel)
   float h_kernel[out_channels][channels][kernel_height][kernel_width];
   for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
       for (int channel = 0; channel < channels; ++channel) {
           for (int row = 0; row < kernel_height; ++row) {
               for (int column = 0; column < kernel_width; ++column) {
                   h_kernel[out_channel][channel][row][column] = kernel[row][column];
               }
           }
       }
   }

   float* d_kernel{nullptr};
   cudaMalloc(&d_kernel, sizeof(h_kernel));
   // copy to device
   cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

   // Convolution
   std::cout << "Compute convolution" << std::endl
   const float alpha = 1.0f, beta = 0.0f;
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
