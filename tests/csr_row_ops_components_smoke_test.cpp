#include <gtest/gtest.h>

#include "csr_row_ops_test_utils.hpp"

using namespace subsetix::csr_test;

TEST(CSRRowOpsComponentsSmokeTest, RowUnionCases) {
  const auto cases = build_row_op_cases();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "Union case " << i);
    run_row_union_case(cases[i], i);
  }
}

TEST(CSRRowOpsComponentsSmokeTest, RowIntersectionCases) {
  const auto cases = build_row_op_cases();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "Intersection case " << i);
    run_row_intersection_case(cases[i], i);
  }
}

TEST(CSRRowOpsComponentsSmokeTest, RowDifferenceCases) {
  const auto cases = build_row_op_cases();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "Difference case " << i);
    run_row_difference_case(cases[i], i);
  }
}

