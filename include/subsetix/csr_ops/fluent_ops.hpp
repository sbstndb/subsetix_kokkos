#pragma once

#include <subsetix/csr_ops/core.hpp>
#include <subsetix/csr_ops/morphology.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_ops/transform.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Fluent builder for chaining CSR set operations.
 *
 * Enables readable, chainable syntax for complex set operations:
 * @code
 *   auto result = ops(ctx)
 *       .from(A)
 *       .union_with(B)
 *       .intersect(mask)
 *       .expand(2, 2)
 *       .build();
 * @endcode
 *
 * Each operation creates a new intermediate result on the device.
 * The final build() call computes cell offsets for indexing.
 */
class SetOpsBuilder {
  CsrSetAlgebraContext& ctx_;
  IntervalSet2DDevice current_;
  bool has_current_ = false;

  // Helper to allocate output with sufficient capacity for binary ops
  static IntervalSet2DDevice alloc_for_binary_op(
      const IntervalSet2DDevice& a,
      const IntervalSet2DDevice& b) {
    // Conservative upper bounds
    const std::size_t max_rows = a.num_rows + b.num_rows;
    const std::size_t max_intervals = a.num_intervals + b.num_intervals;
    return allocate_interval_set_device(
        std::max(max_rows, std::size_t(1)),
        std::max(max_intervals, std::size_t(1)));
  }

public:
  explicit SetOpsBuilder(CsrSetAlgebraContext& ctx) : ctx_(ctx) {}

  /**
   * @brief Initialize the builder with an input set.
   * @param input The starting interval set.
   * @return Reference to this builder for chaining.
   */
  SetOpsBuilder& from(const IntervalSet2DDevice& input) {
    current_ = input;
    has_current_ = true;
    return *this;
  }

  /**
   * @brief Compute union with another set: current = current ∪ other.
   */
  SetOpsBuilder& union_with(const IntervalSet2DDevice& other) {
    auto result = alloc_for_binary_op(current_, other);
    set_union_device(current_, other, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Compute intersection with another set: current = current ∩ other.
   */
  SetOpsBuilder& intersect(const IntervalSet2DDevice& other) {
    auto result = alloc_for_binary_op(current_, other);
    set_intersection_device(current_, other, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Compute difference with another set: current = current \ other.
   */
  SetOpsBuilder& subtract(const IntervalSet2DDevice& other) {
    auto result = alloc_for_binary_op(current_, other);
    set_difference_device(current_, other, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Compute symmetric difference: current = current Δ other.
   */
  SetOpsBuilder& symmetric_diff(const IntervalSet2DDevice& other) {
    auto result = alloc_for_binary_op(current_, other);
    set_symmetric_difference_device(current_, other, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Morphological expansion (dilation).
   * @param rx Expansion radius in X direction.
   * @param ry Expansion radius in Y direction.
   */
  SetOpsBuilder& expand(Coord rx, Coord ry) {
    IntervalSet2DDevice result;
    expand_device(current_, rx, ry, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Morphological shrink (erosion).
   * @param rx Shrink radius in X direction.
   * @param ry Shrink radius in Y direction.
   */
  SetOpsBuilder& shrink(Coord rx, Coord ry) {
    IntervalSet2DDevice result;
    shrink_device(current_, rx, ry, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Refine geometry to next finer level (×2 in X and Y).
   *
   * Each cell becomes 2×2 cells at the finer level.
   */
  SetOpsBuilder& refine() {
    IntervalSet2DDevice result;
    refine_level_up_device(current_, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Project geometry to next coarser level (÷2 in X and Y).
   *
   * Groups of 2×2 cells are merged into single coarse cells.
   */
  SetOpsBuilder& coarsen() {
    IntervalSet2DDevice result;
    project_level_down_device(current_, result, ctx_);
    current_ = std::move(result);
    return *this;
  }

  /**
   * @brief Translate geometry by (dx, dy).
   * @param dx Offset in X direction.
   * @param dy Offset in Y direction.
   */
  SetOpsBuilder& translate(Coord dx, Coord dy) {
    if (dx != 0) {
      IntervalSet2DDevice result;
      translate_x_device(current_, dx, result, ctx_);
      current_ = std::move(result);
    }
    if (dy != 0) {
      IntervalSet2DDevice result;
      translate_y_device(current_, dy, result, ctx_);
      current_ = std::move(result);
    }
    return *this;
  }

  /**
   * @brief Translate geometry in X direction only.
   * @param dx Offset in X direction.
   */
  SetOpsBuilder& translate_x(Coord dx) {
    if (dx != 0) {
      IntervalSet2DDevice result;
      translate_x_device(current_, dx, result, ctx_);
      current_ = std::move(result);
    }
    return *this;
  }

  /**
   * @brief Translate geometry in Y direction only.
   * @param dy Offset in Y direction.
   */
  SetOpsBuilder& translate_y(Coord dy) {
    if (dy != 0) {
      IntervalSet2DDevice result;
      translate_y_device(current_, dy, result, ctx_);
      current_ = std::move(result);
    }
    return *this;
  }

  /**
   * @brief Finalize and return the result with computed cell offsets.
   * @return The final IntervalSet2DDevice with cell_offsets populated.
   */
  IntervalSet2DDevice build() {
    if (has_current_) {
      compute_cell_offsets_device(current_);
    }
    return std::move(current_);
  }

  /**
   * @brief Get current result without computing cell offsets.
   *
   * Useful for intermediate inspection or when cell offsets are not needed.
   */
  const IntervalSet2DDevice& current() const {
    return current_;
  }

  /**
   * @brief Check if the builder has been initialized with from().
   */
  bool has_value() const {
    return has_current_;
  }
};

/**
 * @brief Factory function to create a SetOpsBuilder.
 * @param ctx The algebra context for workspace management.
 * @return A new SetOpsBuilder instance.
 *
 * Usage:
 * @code
 *   auto result = ops(ctx).from(A).union_with(B).build();
 * @endcode
 */
inline SetOpsBuilder ops(CsrSetAlgebraContext& ctx) {
  return SetOpsBuilder(ctx);
}

} // namespace csr
} // namespace subsetix
