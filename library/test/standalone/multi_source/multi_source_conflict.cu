#include <iostream>
#include <chrono>
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/multi_source/conflict_operators.hpp"


namespace sl = subjective_logic;
using Opinion = sl::Opinion<2,float>;
constexpr std::size_t n_opinions{1 << 20};
// constexpr std::size_t n_opinions{1 << 0};

#define CUDA_CHECK(call) do { cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s (%d) ", cudaGetErrorString(err), err); exit(EXIT_FAILURE); \
}\
} while (0)


__global__ static void run_uncert_differentials(Opinion* a, Opinion* b, Opinion* c, sl::Array<3, float>* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<3,Opinion> opins{a[idx], b[idx], c[idx]};
  dest[idx] = sl::multisource::Conflict::uncertainty_differentials(opins);
}

void check_run_uncert_differentials() {
  std::cout << "testing uncert_differentials with 3 opinions " << std::to_string(n_opinions) << " times" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  std::vector<Opinion> input_c(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    // a uncert = 0.2
    input_a[idx].belief_masses()[0] = 0.8;
    // b uncert = 0.8
    input_b[idx].belief_masses()[1] = 0.2;
    // c uncert = 0.6
    input_c[idx].belief_masses()[0] = 0.2;
    input_c[idx].belief_masses()[1] = 0.2;

    // sum of uncert = 1.6
  }
  // average prior
  Opinion *op_a, *op_b, *op_c;
  using DestType = sl::Array<3,Opinion::FLOAT_t>;
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
  run_uncert_differentials<<<n_opinions,32,32>>>(op_a, op_b, op_c, dest);
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

__global__ static void run_conflict_shares(Opinion* a, Opinion* b, Opinion* c, Opinion* d, sl::Array<4, float>* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<4,Opinion> opins{a[idx], b[idx], c[idx], d[idx]};

  float avg_conflict = 0;
  dest[idx] = sl::multisource::Conflict::conflict_shares(sl::RelationType::CONFLICT, sl::ConflictType::AVERAGE, opins, avg_conflict);
}

void check_run_conflict_shares() {
  std::cout << "testing conflict_shares with 4 opinions " << std::to_string(n_opinions) << " times" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  std::vector<Opinion> input_c(n_opinions);
  std::vector<Opinion> input_d(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    // a uncert = 0.2
    input_a[idx].belief_masses()[0] = 0.8;
    // b uncert = 0.8
    input_b[idx].belief_masses()[1] = 0.2;
    // c uncert = 0.6
    input_c[idx].belief_masses()[0] = 0.2;
    input_c[idx].belief_masses()[1] = 0.2;
    // d uncert = 0.3
    input_c[idx].belief_masses()[0] = 0.5;
    input_c[idx].belief_masses()[1] = 0.2;

    // sum of uncert = 1.9
  }
  // average prior
  Opinion *op_a, *op_b, *op_c, *op_d;
  using DestType = sl::Array<4,Opinion::FLOAT_t>;
  DestType *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_c, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_d, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(DestType)*n_opinions);

  cudaMemcpy(op_a, input_a.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_b, input_b.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_c, input_c.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_d, input_d.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start_time = std::chrono::high_resolution_clock::now();
  run_conflict_shares<<<n_opinions,32,32>>>(op_a, op_b, op_c, op_d, dest);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  std::vector<DestType> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(DestType)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0] << std::endl;
  std::cout << "took " << std::chrono::duration<double, std::milli>(end_time-start_time) << std::endl;
  auto shares = sl::multisource::Conflict::conflict_shares(
    sl::RelationType::CONFLICT,
    sl::ConflictType::AVERAGE,
    std::vector{
      input_a.front(),
      input_b.front(),
      input_c.front(),
      input_d.front()
    }) ;

  std::cout << "reference conflict: avg " << shares.first << " | " ;
  for (auto entry:shares.second)
    std::cout << entry << ", ";
  std::cout << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&op_c);
  cudaFree(&op_d);
  cudaFree(&dest);
}
