# Multi-Core-Embedded-Systems
Deep Neural Networks have become a major player in every application we are using in these days. Although DNNs are useful in many ways, they are extremely computationally 
expensive, during both training and inference phases. Many solutions have been proposed to speed up the execution time of DNNs and reduce their computations. In this course
we had to accelerate an convolution neural network with three Convolution layers and two Fully-Connected layers with four optimization methods, sofware techniques, P-thread,
 OpenMp and CUDA. 
 - in"main_modified.c" we used software twchnques like loop unrolliing, loop merging, and reducing the number of function calls to speed up the execution time.
 - "main_lpthread_optimized.c" uses the available multi cores of out CPUs to speed up the execution time of the inference phase. For this we have used "P-thread" C/Cpp library.
 - "main_omp.c" uses OpenMp in order to accelerate the execution of inferece phase.
 - Finally in "main_cuda.cu" we used CUDA to spped up the software program. 
 
