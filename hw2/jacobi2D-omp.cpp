#include "utils.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif 
double absMax(double* a, int N){
  double k=0.0;
  for (int i=0; i < N; i++){
    if(k<fabs(a[i])){
      k=fabs(a[i]);
    }
  }
  return k;
}
int main(int argc, char** argv)
{
  int N = read_option<int>("-n", argc, argv)+2;
  Timer t;

  double* x = (double*) malloc(N*N * sizeof(double));
  double* xx = (double*) malloc(N*N * sizeof(double));
  double* res = (double*) malloc(N*N * sizeof(double));
  double h=1/(N+1.0);
  
  for (int i = 0; i < N*N; i++) {
    x[i] = 0;
    xx[i] = 0;
    res[i] = 0;
  }
  t.tic();
  for(int k=0; k<5000; k+=1){
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef _OPENMP
      #pragma omp for collapse(2)
      #endif
      for(int i = 1; i < N-2; i+=1) {
        for(int j=1; j<N-2; j+=1) {
          xx[i*N+j]=(h*h+x[i*N+j-1]+x[i*N+j+1]+x[(i-1)*N+j]+x[(i+1)*N+j])/4;
        }
      }
    }
    
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef _OPENMP
      #pragma omp for
      #endif
      for(int i=0; i<N*N; i+=1){
        x[i]=xx[i];
      }
    }

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef _OPENMP
      #pragma omp for collapse(2)
      #endif
      for(int i = 1; i < N-2; i+=1) {
        for(int j=1; j<N-2; j+=1) {
          res[i*N+j]=(-x[i*N+j-1]-x[i*N+j+1]-x[(i-1)*N+j]-x[(i+1)*N+j]+4*x[i*N+j])/(h*h)-1;
        }
      }
    }
    

    double M=absMax(res,N*N);
    printf("residule infinite norm: %f\n", M);
    if(M<1e-6){
        printf("terminate after %d iterations\n", k);
        break;
    }
  }
  printf("time: %f s\n", t.toc());

  free(x);
  free(xx);
  free(res);
  return 0;
}