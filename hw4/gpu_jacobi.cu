
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void jacobi(double* x, double* xx, double* res, long N){
  double h=1/(N-1.0);
  //#pragma omp parallel for schedule(static)
  for(int i = 1; i < N-1; i+=1) {
    for(int j=1; j<N-1; j+=1) {
      xx[j*N+i]=(h*h+x[j*N+i-1]+x[j*N+i+1]+x[(j-1)*N+i]+x[(j+1)*N+i])/4;
    }
  }
}

void copyx(double* x, double* xx, long N){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      x[j*N+i]=xx[j*N+i];
    }
  }
}

void jacobi_res(double* x, double* res, long N){
  double h=1/(N-1.0);
  for(int i = 1; i < N-1; i+=1) {
    for(int j=1; j<N-1; j+=1) {
      res[j*N+i]=(-x[j*N+i-1]-x[j*N+i+1]-x[(j-1)*N+i]-x[(j+1)*N+i]+4*x[j*N+i])/(h*h)-1;
    }
  }
}

__global__
void jacobi_kernel(double* x, double* xx, double* res, long N){
  double h=1/(N-1.0);
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  if(ROW>0 && ROW<N-1 && COL>0 && COL<N-1){
    xx[COL*N+ROW]=(h*h+x[COL*N+ROW-1]+x[COL*N+ROW+1]+x[(COL-1)*N+ROW]+x[(COL+1)*N+ROW])/4;
  }
}

__global__
void copyx_kernel(double* x, double* xx, long N){
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  if (ROW < N && COL<N) x[COL*N+ROW] = xx[COL*N+ROW];
}

__global__
void jacobi_res_kernel(double* x, double* res, long N){
  double h=1/(N-1.0);
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  if(ROW>0 && ROW<N-1 && COL>0 && COL<N-1){
    res[COL*N+ROW]=(-x[COL*N+ROW-1]-x[COL*N+ROW+1]-x[(COL-1)*N+ROW]-x[(COL+1)*N+ROW]+4*x[COL*N+ROW])/(h*h)-1;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = 128; 
  dim3 block (32,32);
  dim3 grid (N/32,N/32);
  double* x1 = (double*) malloc(N*N * sizeof(double));
  double* xx1 = (double*) malloc(N*N * sizeof(double));
  double* res1 = (double*) malloc(N*N * sizeof(double));
  double* x2 = (double*) malloc(N*N * sizeof(double));
  double* xx2 = (double*) malloc(N*N * sizeof(double));
  double* res2 = (double*) malloc(N*N * sizeof(double));
  //#pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) {
    x1[i] = 0;
    xx1[i] = 0;
    res1[i] = 0;
    x2[i] = 0;
    xx2[i] = 0;
    res2[i] = 0;
  }

  double tt = omp_get_wtime();
  for(int k=0; k<100; k++){
    jacobi(x1, xx1, res1, N);
    copyx(x1, xx1, N);
    jacobi_res(x1, res1, N);
  }
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *x_d, *xx_d, *res_d;
  cudaMalloc(&x_d, N*N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&xx_d, N*N*sizeof(double));
  cudaMalloc(&res_d, N*N*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x2, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(xx_d, xx2, N*N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  for(int k=0; k<100; k++){
    jacobi_kernel<<<grid,block>>>(x_d, xx_d, res_d, N);
    copyx_kernel<<<grid,block>>>(x_d, xx_d, N);
    //cudaDeviceSynchronize();
    //jacobi_res_kernel<<<grid,block>>>(x_d, res_d, N);
  }
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(res2, res_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(x2, x_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(xx2, xx_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < N*N; i++) err += fabs(x1[i]-x2[i]);
  printf("Error = %f\n", err);


  cudaFree(x_d);
  cudaFree(xx_d);
  cudaFree(res_d);

  free(x1);
  free(xx1);
  free(res1);
  free(x2);
  free(xx2);
  free(res2);

  return 0;
}

