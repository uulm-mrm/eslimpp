#include <iostream>
#include <chrono>
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"

namespace sl = subjective_logic;
using Opinion = sl::Opinion<2,float>;
constexpr std::size_t n_opinions{1 << 20};
// constexpr std::size_t n_opinions{1 << 0};

#define CUDA_CHECK(call) do { cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s (%d) ", cudaGetErrorString(err), err); exit(EXIT_FAILURE); \
}\
} while (0)


__global__ static void run_prior_average(Opinion* a, Opinion* b, Opinion::BeliefType* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<2,Opinion> opins{a[idx],b[idx]};

  dest[idx] = sl::multisource::Fusion::average_prior(opins);
}

void check_run_prior_average() {
  std::cout << "testing average_prior with " << std::to_string(n_opinions) << " opinions" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    input_a[idx].prior_belief_masses()[0] = 0.8;
    input_a[idx].prior_belief_masses()[1] = 0.2;
  }
  // average prior
  Opinion *op_a, *op_b;
  using DestType = Opinion::BeliefType;
  DestType *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(DestType)*n_opinions);

  cudaMemcpy(op_a, input_a.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_b, input_b.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start_time = std::chrono::high_resolution_clock::now();
  run_prior_average<<<n_opinions,32,32>>>(op_a, op_b, dest);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  std::vector<DestType> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(DestType)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0] << std::endl;
  std::cout << "took " << std::chrono::duration<double, std::milli>(end_time-start_time) << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&dest);
}

__global__ static void run_preprocess_uncertainties(Opinion* a, Opinion* b, sl::Array<2,Opinion::FLOAT_t>* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<2,Opinion> opins{a[idx],b[idx]};

  auto result= sl::multisource::Fusion::preprocess_opinions(opins, dest[idx]);
}

void check_run_preprocess_uncertainties() {
  std::cout << "testing preprocess uncertainties with " << std::to_string(n_opinions) << " opinions" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    input_a[idx].belief() = 0.3;
    input_b[idx].disbelief() = 0.5;
  }
  // average prior
  Opinion *op_a, *op_b;
  using DestType = sl::Array<2,Opinion::FLOAT_t>;
  DestType *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(DestType)*n_opinions);

  cudaMemcpy(op_a, input_a.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_b, input_b.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start_time = std::chrono::high_resolution_clock::now();
  run_preprocess_uncertainties<<<n_opinions,32,32>>>(op_a, op_b, dest);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  std::vector<DestType> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(DestType)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0] << std::endl;
  std::cout << "took " << std::chrono::duration<double, std::milli>(end_time-start_time) << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&dest);
}


__global__ static void run_fuse_opinions(Opinion* a, Opinion* b, Opinion* c, Opinion* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<3,Opinion> opins{a[idx], b[idx], c[idx]};

  dest[idx] = sl::multisource::Fusion::fuse_opinions(sl::FusionType::CUMULATIVE, opins);
}

void check_run_fuse_opinions() {
  std::cout << "testing multi-source opinion fusion with 3 opinions " << std::to_string(n_opinions) << " times" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  std::vector<Opinion> input_c(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    input_a[idx].belief() = 0.3;
    input_b[idx].disbelief() = 0.5;
    input_c[idx].belief() = 0.5;
    input_c[idx].disbelief() = 0.2;
  }
  // average prior
  Opinion *op_a, *op_b, *op_c;
  using DestType = Opinion;
  DestType *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_c, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(DestType)*n_opinions);

  cudaMemcpy(op_a, input_a.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_b, input_b.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_c, input_c.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start_time = std::chrono::high_resolution_clock::now();
  run_fuse_opinions<<<n_opinions,32,32>>>(op_a, op_b, op_c, dest);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  std::vector<DestType> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(DestType)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0] << std::endl;
  std::cout << "took " << std::chrono::duration<double, std::milli>(end_time-start_time) << std::endl;
  std::cout << "reference fusion: " << sl::multisource::Fusion::fuse_opinions(sl::FusionType::CUMULATIVE,
    std::vector{
      input_a.front(),
      input_b.front(),
      input_c.front()
    }) << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&dest);
}
