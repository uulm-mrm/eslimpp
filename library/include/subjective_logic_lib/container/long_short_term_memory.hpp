#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>
#include <functional>

#include "subjective_logic_lib/multi_source/conflict_operators.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"
#include "subjective_logic_lib/types/fusion_types.hpp"

namespace subjective_logic::container
{
template <typename OpinionT>
class LongShortTermMemory
{
public:
  using FloatT = typename OpinionT::FLOAT_t;
  using FusionFunc = std::function<OpinionT(OpinionT, OpinionT)>;

  LongShortTermMemory(std::size_t short_max_size,
                      FloatT threshold,
                      FloatT discount,
                      FusionType fusion_type,
                      bool handle_st_conflict = true,
                      bool avg_dc_conflict = true);

  // there is no point in only resetting the short memory, either long only or both
  void reset();
  void reset_long_memory();

  FloatT get_threshold() const;
  void set_threshold(FloatT new_thresh);

  FloatT get_discount() const;
  void set_discount(FloatT new_discount);

  [[nodiscard]] std::size_t get_short_max_size() const;

  bool is_last_conflicted() const;

  /**
   * @brief Set the maximal size of the short-term memory accordingly.
   *        IMPORTANT: if the long-term part is currently in use, it is reset to adhere to the new max short-term size
   *        Thus, if the size changes,
   *        the long-term memory is reset and
   *        the LSTMemory size is set according to the currently available short-term memory opinions.
   */
  void set_short_max_size(std::size_t short_max_size);

  // # of opinions considered for the current opinion (long + short)
  [[nodiscard]] std::size_t size() const;

  /**
   * @brief get the current size of the short memory, which may be less that short_max_size
   *        if less than short_max_size opinions have been added, this number equals the size()
   */
  [[nodiscard]] std::size_t get_short_size() const;

  /**
   * @brief get the current size of the long-term memory
   */
  [[nodiscard]] std::size_t get_long_size() const;

  /**
   * @brief Adds an opinions to the memory, popped short-term opinions are automatically added to the long-term memory.
   *        This function returns the output prior to any conflict considerations,
   *        i.e.,
   *        adds the opinion to the short-term memory and pushed the last short-term opinion to the long-term opinion.
   *
   *        However, conflict handling is performed after adding the opinion and
   *        the next get_opinion() may no longer consider the "old" long-term memory
   */
  OpinionT add(const OpinionT& new_opinion);

  /**
   * @brief overall opinion of the current LSTMemory
   *        IMPORTANT: not all fusion operators have a neutral element,
   *        however, the long-term part is neglected of no opinion was put into it
   *        as are all "unused" opinions in the short-term part
   */
  [[nodiscard]] OpinionT get_opinion() const;
  // outputs the long-term opinion only, including all pseudo time-based trust discounts
  [[nodiscard]] OpinionT get_long_opinion() const;
  // outputs the short-term opinion only by applying the fusion_func to all short-term opinions
  [[nodiscard]] OpinionT get_short_opinion() const;
  // outputs last conflicting pair (short_op, long_op)
  [[nodiscard]] std::optional<std::pair<OpinionT, OpinionT>> get_conflicted_pair() const;

protected:
  void handle_internal_conflict();

  FloatT threshold_;
  FloatT discount_;
  FusionFunc fuse_func_;
  FusionType fusion_type_;
  bool handle_st_conflict_;
  bool use_avg_dc_conflict_handling_;

  bool last_conflicted_{ false };

  std::size_t current_size_{ 0 };
  std::size_t short_max_size_;
  std::size_t current_short_ring_idx_{ 0 };
  std::vector<OpinionT> short_term_memory_;

  OpinionT long_opinion_{};
  std::size_t lt_size_{ 0 };
  FloatT weight_{ 0.0 };

  std::optional<std::pair<OpinionT, OpinionT>> last_conflicted_pair_{};
};

template <typename OpinionT>
LongShortTermMemory<OpinionT>::LongShortTermMemory(std::size_t short_max_size,
                                                   FloatT threshold,
                                                   FloatT discount,
                                                   FusionType fusion_type,
                                                   bool handle_st_conflict,
                                                   bool avg_dc_conflict)
  : threshold_{ threshold }
  , discount_{ discount }
  , fusion_type_{ fusion_type }
  , handle_st_conflict_{ handle_st_conflict }
  , use_avg_dc_conflict_handling_{ avg_dc_conflict }
  , short_max_size_(short_max_size)
{
  short_term_memory_.reserve(short_max_size_);
  switch (fusion_type_)
  {
    case subjective_logic::FusionType::AVERAGE:
      fuse_func_ = [](OpinionT a, OpinionT b) { return a.average_fuse(b); };
      break;
    case subjective_logic::FusionType::BELIEF_CONSTRAINT:
      fuse_func_ = [](OpinionT a, OpinionT b) { return a.bc_fuse(b); };
      break;
    case subjective_logic::FusionType::CUMULATIVE:
      fuse_func_ = [](OpinionT a, OpinionT b) { return a.cum_fuse(b); };
      break;
    case subjective_logic::FusionType::WEIGHTED:
      fuse_func_ = [](OpinionT a, OpinionT b) { return a.wb_fuse(b); };
      break;
  }
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::reset()
{
  reset_long_memory();
  // keep reserved short mem size
  short_term_memory_.clear();
  current_short_ring_idx_ = 0;
  current_size_ = 0;
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::reset_long_memory()
{
  lt_size_ = 0;
  weight_ = 0.0;
  // current_size_ must be reset, to avoid considering long_opinion in next output
  current_size_ = std::min(current_size_, short_max_size_);
}

template <typename OpinionT>
typename LongShortTermMemory<OpinionT>::FloatT LongShortTermMemory<OpinionT>::get_threshold() const
{
  return threshold_;
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::set_threshold(const FloatT new_thresh)
{
  threshold_ = new_thresh;
}
template <typename OpinionT>
typename LongShortTermMemory<OpinionT>::FloatT LongShortTermMemory<OpinionT>::get_discount() const
{
  return discount_;
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::set_discount(const FloatT new_discount)
{
  discount_ = new_discount;
}

template <typename OpinionT>
std::size_t LongShortTermMemory<OpinionT>::get_short_max_size() const
{
  return short_max_size_;
}

template <typename OpinionT>
bool LongShortTermMemory<OpinionT>::is_last_conflicted() const
{
  return last_conflicted_;
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::set_short_max_size(const std::size_t short_max_size)
{
  // avoid resetting if nothing changes
  if (short_max_size == short_max_size_)
  {
    return;
  }
  // long-term must be ditched if currently in use
  if (current_size_ > short_max_size_)
  {
    reset_long_memory();
  }
  short_max_size_ = short_max_size;

  // short_max_size might be smaller now
  if (current_size_ > short_max_size_)
  {
    reset_long_memory();
  }
}

template <typename OpinionT>
std::size_t LongShortTermMemory<OpinionT>::size() const
{
  return current_size_;
}

template <typename OpinionT>
std::size_t LongShortTermMemory<OpinionT>::get_short_size() const
{
  return short_term_memory_.size();
}

template <typename OpinionT>
std::size_t LongShortTermMemory<OpinionT>::get_long_size() const
{
  return lt_size_;
}

template <typename OpinionT>
OpinionT LongShortTermMemory<OpinionT>::add(const OpinionT& new_opinion)
{
  last_conflicted_ = false;
  current_size_++;
  if (short_term_memory_.size() < short_max_size_)
  {
    short_term_memory_.push_back(new_opinion);
    return get_opinion();
  }

  OpinionT hopped_opinion = short_term_memory_[current_short_ring_idx_];
  short_term_memory_[current_short_ring_idx_++] = new_opinion;
  current_short_ring_idx_ %= short_max_size_;

  if (lt_size_ == 0)
  {
    long_opinion_ = hopped_opinion;
    lt_size_++;
    // weight initialization depends on fusion type
    switch (fusion_type_)
    {
      case subjective_logic::FusionType::AVERAGE:
        weight_ = 1.0;
        break;
      case subjective_logic::FusionType::BELIEF_CONSTRAINT:
        // TODO(@deuscher) - BCF implementation missing
        break;
      case subjective_logic::FusionType::CUMULATIVE:
        // weight not required for CBF
        break;
      case subjective_logic::FusionType::WEIGHTED:
        weight_ = (1 - long_opinion_.uncertainty());
        break;
    }
  }
  else
  {
    lt_size_++;
    std::tie(long_opinion_, weight_) =
        multisource::SequentialFusion::fuse_opinions(fusion_type_, long_opinion_, hopped_opinion, weight_, discount_);
  }

  auto output = get_opinion();

  // floating point type needs to be selected according to OpinionT
  auto conflict = get_short_opinion().degree_of_conflict(long_opinion_);

  if (conflict > threshold_)
  {
    last_conflicted_pair_ = { get_short_opinion(), get_long_opinion() };
    reset_long_memory();
    last_conflicted_ = true;
    if (handle_st_conflict_)
    {
      handle_internal_conflict();
    }
  }

  return output;
}

template <typename OpinionT>
OpinionT LongShortTermMemory<OpinionT>::get_opinion() const
{
  // if less than short_max_size_ opinions have been added so far,
  // fusion vacuous opinions (here long opinion) may lead to false outputs for some fusion operators
  if (current_size_ < short_max_size_)
  {
    return get_short_opinion();
  }
  return fuse_func_(get_long_opinion(), get_short_opinion());
}

template <typename OpinionT>
OpinionT LongShortTermMemory<OpinionT>::get_long_opinion() const
{
  if (lt_size_ == 0)
  {
    return OpinionT::VacuousBeliefOpinion();
  }
  return long_opinion_;
}

template <typename OpinionT>
OpinionT LongShortTermMemory<OpinionT>::get_short_opinion() const
{
  if (short_term_memory_.empty())
  {
    return OpinionT::VacuousBeliefOpinion();
  }

  return subjective_logic::multisource::Fusion::fuse_opinions(fusion_type_, short_term_memory_);
}

template <typename OpinionT>
std::optional<std::pair<OpinionT, OpinionT>> LongShortTermMemory<OpinionT>::get_conflicted_pair() const
{
  return last_conflicted_pair_;
}

template <typename OpinionT>
void LongShortTermMemory<OpinionT>::handle_internal_conflict()
{
  using MSF = subjective_logic::multisource::Fusion;
  using Conflict = subjective_logic::multisource::Conflict;

  std::size_t k_star = short_max_size_;
  std::vector<OpinionT> linear_buffer(short_max_size_);
  for (std::size_t idx{ 0 }; idx < short_max_size_; ++idx)
  {
    // reverse and linearize ring-buffer
    std::size_t rb_j = (current_short_ring_idx_ - 1 - idx + short_max_size_) % short_max_size_;
    linear_buffer[idx] = short_term_memory_[rb_j];
  }

  if (use_avg_dc_conflict_handling_)
  {
    FloatT prev_conflict{ 1.0 };
    std::vector<FloatT> rel_conflicts(short_max_size_, 0.0);
    for (std::size_t idx{ 2 }; idx < short_max_size_; ++idx)
    {
      const std::vector<OpinionT> slice(linear_buffer.begin(), linear_buffer.begin() + idx);

      // compute average conflict
      FloatT avg_conflict = Conflict::conflict(ConflictType::AVERAGE, slice);
      rel_conflicts[idx] = avg_conflict / prev_conflict;
      prev_conflict = avg_conflict;
    }

    auto it = std::max_element(rel_conflicts.begin(), rel_conflicts.end());
    std::size_t argmax = std::distance(rel_conflicts.begin(), it);
    k_star = argmax < 2 ? 0 : argmax - 2;
  }
  else
  {
    std::vector<FloatT> conflicts{};
    for (std::size_t idx{ 0 }; idx < short_max_size_ - 1; ++idx)
    {
      // split ST memory into old and new and fuse opinions together
      std::vector<OpinionT> new_ops = { linear_buffer.begin(), linear_buffer.begin() + idx + 1 };
      std::vector<OpinionT> old_ops = { linear_buffer.begin() + idx + 1, linear_buffer.end() };

      OpinionT new_op = MSF::fuse_opinions(fusion_type_, new_ops);
      OpinionT old_op = MSF::fuse_opinions(fusion_type_, old_ops);

      // consider conflict between old and new partition
      conflicts.push_back(new_op.degree_of_conflict(old_op));
    }
    // copy over new short memory
    auto it = std::max_element(conflicts.begin(), conflicts.end());
    k_star = std::distance(conflicts.begin(), it);
  }

  short_term_memory_ = { linear_buffer.begin(), linear_buffer.begin() + k_star };
  current_size_ = short_term_memory_.size();
}

}  // namespace subjective_logic::container
