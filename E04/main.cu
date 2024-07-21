#include <iostream>
#include <time.h>

#include "cuda_runtime.h"
#include "defines.cuh"

unsigned int W = 512;
unsigned int H = 512;
unsigned int it = 10000;

float calculateCFD_cpu(float* t, float* tprev, unsigned int W, 
    unsigned int H, unsigned int iterations);

int main (int argc, char** argv) {

    if ((W % THREADS_X) != 0 || (H % THREADS_Y) != 0){
        std::cout << "Set W,H multiple of " << THREADS_X <<\
        ", " << THREADS_Y << std::endl;
        exit(-1);
    }

    float cpuTime, gpuTime;

    float *t   = NULL; float *tprev   = NULL;
    float *dt = NULL; float *dtprev = NULL;

    unsigned int size = W * H * sizeof(float);

    t     = (float*) calloc(W*H, sizeof(float)); // This allocates filling with 0's
    tprev = (float*) calloc(W*H, sizeof(float));

    if (t == NULL || tprev == NULL) {
        std::cout << "Error to allocate CPU memory" << std::endl;
        if (t != NULL) free(t); 
        if (tprev != NULL) free(tprev);
        return 1;
    } 

    cpuTime = calculateCFD_cpu(t, tprev, W, H, it); 

    std::cout << "Time in CPU: " << cpuTime << std::endl;

    // Include next line if you want to test only cpu
    // return 0; 

    cudaError_t cudaStatus = cudaSetDevice(0);

    if(cudaStatus != cudaSuccess) {
        std::cout << "Failed to load CUDA-capable device" << std::endl;
        return 2;
    }

    cudaStatus = cudaMalloc(&dt, size);
    if(cudaStatus != cudaSuccess) {
        std::cout << "Failed to allocate " << size << " bytes" << std::endl;
        if(dt != NULL) cudaFree(dt);
    }

    cudaStatus = cudaMalloc(&dtprev, size);
    if(cudaStatus != cudaSuccess) {
        std::cout << "Failed to allocate " << size << " bytes" << std::endl;
        if(dtprev != NULL) cudaFree(dtprev);
    }

    // Values defined in defines.cuh
    dim3 block_size(THREADS_X, THREADS_Y);
    dim3 grid_size(W/THREADS_X, H/THREADS_Y);

    float h, x, y;
    h = 1.0f / (W - 1);
    // TODO: Initialize tprev with the boundaries conditions; 
    // Hint: Replicate CPU calculation outside kernel
    
    //INSERT YOUR CODE HERE
    // CONDITION 1

    // CONDITION 2
    
    // ----------------------------------------------------

    cudaMemcpy(dtprev, tprev, W*H*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Change KERNEL_NUMBER to 2 in defines.cuh to test the second kernel
    if (KERNEL_NUMBER == 1) {
        cudaEventRecord(start, 0);
        // Note that iterations is also outside the kernel, thus you need to iterate through rows and cols only
        // Hint: Use the 2D indexing of the threads blocks
        for (unsigned int k=0; k < it; k++) { 
            calculateCFD <<<grid_size,block_size>>> (dtprev, dt, W, H, h);
        }
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
    }
    else if (KERNEL_NUMBER == 2) {
        cudaEventRecord(start, 0);
        for (unsigned int k=0; k < it; k++) {
            calculateCFD_shared <<<grid_size,block_size>>> (dtprev, dt, W, H, h);
        }
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
    }

    cudaEventElapsedTime(&gpuTime, start, end);
    
    std::cout << "Time in GPU to execute Kernel " << KERNEL_NUMBER << ": " << cpuTime << std::endl;

    cudaStatus = cudaMemcpy(t, dtprev, W*H*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Copy failed DEVICE -> HOST" << std::endl;
        return 2;
    }

    return 0;
}

void calculateCFD_cpu(float* u, float* u_new, int nx, int ny, float alpha, float dt, float dx, float dy) {
    
    
    for (int i = 0; i < nx; i++) {
        u_new[i * ny + 0] = 0.0f;         
        u_new[i * ny + (ny-1)] = 0.0f;      
    }
    for (int j = 0; j < ny; j++) {
        u_new[0 * ny + j] = 0.0f;           
        u_new[(nx-1) * ny + j] = 0.0f;      
    }

    
    for (int i = 1; i < nx-1; i++) {
        for (int j = 1; j < ny-1; j++) {
            int idx = i * ny + j;
            float uxx = (u[(i-1) * ny + j] - 2.0f * u[i * ny + j] + u[(i+1) * ny + j]) / (dx * dx);
            float uyy = (u[i * ny + (j-1)] - 2.0f * u[i * ny + j] + u[i * ny + (j+1)]) / (dy * dy);
            u_new[idx] = u[idx] + alpha * dt * (uxx + uyy);
        }
    }
}

int main() {
    
    int nx = 100;  
    int ny = 100;  
    float alpha = 0.01f; 
    float dt = 0.1f;     
    float dx = 1.0f;     
    float dy = 1.0f;     

    
    float* u = (float*)malloc(nx * ny * sizeof(float));
    float* u_new = (float*)malloc(nx * ny * sizeof(float));

    
    memset(u, 0, nx * ny * sizeof(float));
    memset(u_new, 0, nx * ny * sizeof(float));

    
    calculateCFD_cpu(u, u_new, nx, ny, alpha, dt, dx, dy);

    
    free(u);
    free(u_new);

    return 0;
}
