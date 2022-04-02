#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  //I changed here, that the first element of prefix_sum should be A[0] instead of 0 by definition.
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  int nthreads=8;
  long* s = (long*) malloc(nthreads * sizeof(long));
  #pragma omp parallel num_threads(nthreads)
  { 
    int nthreads = omp_get_num_threads();
    long subn=n/nthreads;
    int tid = omp_get_thread_num();
    prefix_sum[tid*subn] = A[tid*subn];
    for (long i=tid*subn+1; i < (tid+1)*subn; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i];
    }
    s[tid]=prefix_sum[(tid+1)*subn-1];

    #pragma omp barrier
    
    long temp=0;
    for(int i=0; i<tid; i++){
      temp = temp+s[i];
    }
    for (long i=tid*subn; i < (tid+1)*subn; i++){
      prefix_sum[i] = prefix_sum[i] + temp;
    }
  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
