
#define THREADS_X 16  // Number of threads in a block (x,y)
#define THREADS_Y 16

#define KERNEL_NUMBER 1 // Change this if you want to test kernel 2 (shared)

#define RADIUS_SHARED 1 

// Kernel 01
__global__ void calculateCFD_shared(float* u, float* u_new, int nx, int ny, float alpha, float dt, float dx, float dy) {
    // Indices globais
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Indices locais
    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    // Dimensões do bloco
    int block_size_x = blockDim.x;
    int block_size_y = blockDim.y;

    // Alocar memória compartilhada
    extern __shared__ float shared_u[];

    // Índice global na memória compartilhada
    int shared_idx = local_i * block_size_y + local_j;

    // Carregar dados na memória compartilhada
    if (i < nx && j < ny) {
        shared_u[shared_idx] = u[i * ny + j];
    }

    __syncthreads();

    // Verificar se não estamos nas bordas e calcular o novo valor
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        float uxx = (shared_u[(local_i-1) * block_size_y + local_j] - 2.0f * shared_u[local_i * block_size_y + local_j] + shared_u[(local_i+1) * block_size_y + local_j]) / (dx * dx);
        float uyy = (shared_u[local_i * block_size_y + (local_j-1)] - 2.0f * shared_u[local_i * block_size_y + local_j] + shared_u[local_i * block_size_y + (local_j+1)]) / (dy * dy);
        u_new[i * ny + j] = u[i * ny + j] + alpha * dt * (uxx + uyy);
    }
}
