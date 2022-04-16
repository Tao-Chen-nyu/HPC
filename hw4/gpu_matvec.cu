// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_mult(double* c, const double* a, const double* b, long M, long N){
  //#pragma omp parallel for schedule(static)
  for (long i = 0; i < M; i++) {
    double sum=0.0;
    for(long j=0; j<N; j++){
     sum += a[i+j*M] * b[j];
    }
    c[i]=sum;
  }
}

__global__
void vec_mult_kernel(double* c, const double* a, const double* b, long M, long N){
  double sum=0.0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M){ 
   for(long i=0; i<N; i++) sum += a[idx+i*M] * b[i];
  }
  c[idx]=sum;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = 10240; // 2^25

  double* x = (double*) malloc(N*N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  //#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    for(long j=0; j<N; j++){
     x[i+j*N] = i+2;
    }
    y[i] = 1.0/(i+1);
    z[i] = 0;
    z_ref[i] = 0;
  }
  double tt = omp_get_wtime();
  vec_mult(z_ref, x, y, N, N);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  vec_mult_kernel<<<N/1024,1024>>>(z_d, x_d, y_d, N, N);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);
  printf("GPU bandwidth is %f GB/s\n", 3*N*N*sizeof(double)/(omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);

  return 0;
}

