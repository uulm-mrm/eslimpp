#include "functions.hpp"
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(ans)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    if ((ans) != cudaSuccess)                                                                                          \
    {                                                                                                                  \
      printf("CUDA Error: %s \nFile: %s \nLine: %d\n", cudaGetErrorString((ans)), __FILE__, __LINE__);                 \
    }                                                                                                                  \
  } while (0)

__global__ void run_fusion(Opinion* a, Opinion* b, Opinion* dest, int* classes,  std::size_t size) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  // a[idx] = Opinion(0.0,0.8);
  // b[idx] = Opinion(0.8,0.0);

  dest[idx] = a[idx].cum_fuse(b[idx]);
  if (dest[idx].uncertainty() > 0.5) {
    classes[idx] = 0;
  }
  else {
    typename Opinion::FLOAT_t prob = dest[idx].getBinomialProjection();
    if (prob > 0.7) {
      classes[idx] = 1;
    }
    else if (prob < 0.3) {
      classes[idx] = 2;
    }
    else {
      classes[idx] = 3;
    }
  }
}

TimeDiffs run_gpu_assessment(const std::size_t n_ops, const std::size_t n_runs, const std::vector<Opinion>& sensor_a, const std::vector<Opinion>& sensor_b)
{
  TimeDiffs runtimes(n_runs);
  std::size_t map_size_byte = n_ops * sizeof(Opinion);
  std::cout << "size of single subjective_logic_lib::OpinionNoBase<2,float>: " << sizeof(Opinion) << std::endl;
  Opinion* a{nullptr};
  Opinion* b{nullptr};
  Opinion* dest{nullptr};
  int* classes{nullptr};

  constexpr std::size_t blocks = 512;
  auto threads_per_block = static_cast<std::size_t>(std::ceil(static_cast<double>(n_ops) / blocks));

  cudaMalloc((void**)&a, map_size_byte);
  cudaMalloc((void**)&b, map_size_byte);
  cudaMalloc((void**)&dest, map_size_byte);
  cudaMalloc((void**)&classes, map_size_byte);

  cudaMemcpy(a, sensor_a.data(), map_size_byte, cudaMemcpyHostToDevice);
  cudaMemcpy(b, sensor_b.data(), map_size_byte, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // warm up
  run_fusion<<<threads_per_block, blocks>>>(a,b,dest, classes, n_ops);

  cudaEvent_t cu_start, cu_stop;
  cudaEventCreate(&cu_start);
  cudaEventCreate(&cu_stop);

  for (std::size_t i{0}; i< n_runs; ++i) {
    cudaEventRecord(cu_start);
    run_fusion<<<threads_per_block, blocks>>>(a,b,dest, classes, n_ops);
    cudaEventRecord(cu_stop);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
    runtimes[i] = TimeDiff{static_cast<std::uint64_t>(milliseconds*1e6)};
  }
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEventDestroy(cu_start);
  cudaEventDestroy(cu_stop);

  std::vector<int> results(n_ops);
  cudaMemcpy(results.data(), classes, n_ops * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int> hist;
  hist.resize(4);
  for (const int entry : results) {
    hist[entry] += 1;
  }
  double denom = hist[1] + hist[3];
  double score = hist[3] / denom;
  std::cout << "size of one map with " << n_ops << " elements is: " << map_size_byte / 1e6 << "MB" << std::endl;
  std::cout << "the self-assessment score is: " << score << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(dest);

  return runtimes;
}
