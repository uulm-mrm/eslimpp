#pragma once

#include <vector>
#include <functional>

#include "subjective_logic_lib/util.hpp"

namespace subjective_logic::container
{
template <typename OpinionT>
class LongShortTermMemory
{
public:
  using FloatT = typename OpinionT::FLOAT_t;
  using FusionFunc = std::function<OpinionT(OpinionT, OpinionT)>;

  LongShortTermMemory(std::size_t short_max_size, FloatT threshold, FloatT discount, FusionFunc fuse_func);

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

protected:
  FloatT threshold_;
  FloatT discount_;
  FusionFunc fuse_func_;

  bool last_conflicted_{ false };

  std::size_t current_size_{ 0 };
  std::size_t short_max_size_;
  std::size_t current_short_ring_idx_{ 0 };
  std::vector<OpinionT> short_term_memory_;

  OpinionT long_opinion_{};
};

template <typename OpinionT>
LongShortTermMemory<OpinionT>::LongShortTermMemory(std::size_t short_max_size,
                                                   FloatT threshold,
                                                   FloatT discount,
                                                   FusionFunc fuse_func)
  : threshold_{ threshold }, discount_{ discount }, fuse_func_(fuse_func), short_max_size_(short_max_size)
{
  short_term_memory_.reserve(short_max_size_);
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
  long_opinion_ = OpinionT::VacuousBeliefOpinion();
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
OpinionT LongShortTermMemory<OpinionT>::add(const OpinionT& new_opinion)
{
  current_size_++;
  if (short_term_memory_.size() < short_max_size_)
  {
    short_term_memory_.push_back(new_opinion);
    return get_opinion();
  }

  OpinionT hopped_opinion = short_term_memory_[current_short_ring_idx_];
  short_term_memory_[current_short_ring_idx_++] = new_opinion;
  current_short_ring_idx_ %= short_max_size_;

  long_opinion_.trust_discount_(discount_);
  long_opinion_ = fuse_func_(long_opinion_, hopped_opinion);

  auto output = get_opinion();

  // floating point type needs to be selected according to OpinionT
  auto conflict = get_short_opinion().degree_of_conflict(long_opinion_);

  if (conflict > threshold_)
  {
    reset_long_memory();
    last_conflicted_ = true;
  }
  else
  {
    last_conflicted_ = false;
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
  return long_opinion_;
}

template <typename OpinionT>
OpinionT LongShortTermMemory<OpinionT>::get_short_opinion() const
{
  if (short_term_memory_.empty())
  {
    return OpinionT::VacuousBeliefOpinion();
  }
  OpinionT fused_op = short_term_memory_[0];
  for (std::size_t i = 1; i < short_term_memory_.size(); ++i)
  {
    fused_op = fuse_func_(fused_op, short_term_memory_[i]);
  }
  return fused_op;
}

}  // namespace subjective_logic::container