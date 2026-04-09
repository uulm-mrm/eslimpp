#include <iostream>
#include "subjective_logic_lib/opinions/opinion.hpp"
namespace sl = subjective_logic;
using Opinion = sl::Opinion<2,float>;
constexpr std::size_t n_opinions{1 << 20};



__global__ void run_fusion(Opinion* a, Opinion* b, Opinion* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}
  dest[idx] = a[idx].cum_fuse(b[idx]);

  dest[idx].belief() = 0.5;
  auto q_test = dest[idx].as_no_base().get_quantized();
  dest[idx].belief() = static_cast<float>(q_test.belief());

  // printf("idx %ld \n", idx);
  // if (idx == 0) {
  //   printf("test %f", q_test.belief().as_limit_type());
  //   printf("test %f", dest[idx].belief());
  // }
}

#define CUDA_CHECK(call) do { cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error: %s (%d) ", cudaGetErrorString(err), err); exit(EXIT_FAILURE); \
    }\
  } while (0)

int main(int argc, char **argv) {

  Opinion *op_a, *op_b, *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(Opinion)*n_opinions);

  run_fusion<<<n_opinions,32,32>>>(op_a, op_b, dest);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Opinion> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(Opinion)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0].belief() << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&dest);

  return 0;
}
