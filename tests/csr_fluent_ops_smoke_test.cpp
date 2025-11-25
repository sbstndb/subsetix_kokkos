#include <gtest/gtest.h>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/fluent_ops.hpp>

namespace {

using namespace subsetix::csr;

class CsrFluentOpsTest : public ::testing::Test {
protected:
  CsrSetAlgebraContext ctx;
};

TEST_F(CsrFluentOpsTest, SimpleUnion) {
  Box2D boxA{0, 10, 0, 10};
  Box2D boxB{5, 15, 0, 10};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);

  auto result = ops(ctx)
      .from(A)
      .union_with(B)
      .build();

  auto h = build_host_from_device(result);

  // Union should span 0..15 in X, 0..10 in Y
  EXPECT_EQ(h.num_rows(), 10);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 0);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 15);
  }
}

TEST_F(CsrFluentOpsTest, SimpleIntersection) {
  Box2D boxA{0, 10, 0, 10};
  Box2D boxB{5, 15, 5, 15};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);

  auto result = ops(ctx)
      .from(A)
      .intersect(B)
      .build();

  auto h = build_host_from_device(result);

  // Intersection should be 5..10 in X, 5..10 in Y
  EXPECT_EQ(h.num_rows(), 5);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.row_keys[i].y, 5 + static_cast<int>(i));
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 5);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 10);
  }
}

TEST_F(CsrFluentOpsTest, SimpleDifference) {
  Box2D boxA{0, 10, 0, 10};
  Box2D boxB{5, 15, 0, 10};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);

  auto result = ops(ctx)
      .from(A)
      .subtract(B)
      .build();

  auto h = build_host_from_device(result);

  // A \ B should be 0..5 in X, 0..10 in Y
  EXPECT_EQ(h.num_rows(), 10);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 0);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 5);
  }
}

TEST_F(CsrFluentOpsTest, ChainedOperations) {
  Box2D boxA{0, 20, 0, 20};
  Box2D boxB{10, 30, 0, 20};
  Box2D mask{5, 25, 5, 15};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);
  auto M = make_box_device(mask);

  // (A ∪ B) ∩ mask = [0,30] x [0,20] ∩ [5,25] x [5,15] = [5,25] x [5,15]
  auto result = ops(ctx)
      .from(A)
      .union_with(B)
      .intersect(M)
      .build();

  auto h = build_host_from_device(result);

  EXPECT_EQ(h.num_rows(), 10); // Y: 5..15
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.row_keys[i].y, 5 + static_cast<int>(i));
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 5);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 25);
  }
}

TEST_F(CsrFluentOpsTest, ExpandInChain) {
  Box2D box{10, 20, 10, 20};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .expand(2, 2)
      .build();

  auto h = build_host_from_device(result);

  // Expanded box: 8..22 in X, 8..22 in Y
  EXPECT_EQ(h.num_rows(), 14);
  EXPECT_EQ(h.row_keys.front().y, 8);
  EXPECT_EQ(h.row_keys.back().y, 21);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 8);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 22);
  }
}

TEST_F(CsrFluentOpsTest, ShrinkInChain) {
  Box2D box{10, 20, 10, 20};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .shrink(2, 2)
      .build();

  auto h = build_host_from_device(result);

  // Shrunk box: 12..18 in X, 12..18 in Y
  EXPECT_EQ(h.num_rows(), 6);
  EXPECT_EQ(h.row_keys.front().y, 12);
  EXPECT_EQ(h.row_keys.back().y, 17);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 12);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 18);
  }
}

TEST_F(CsrFluentOpsTest, ComplexPipeline) {
  // Build a ring: expand a box then subtract the original
  Box2D box{10, 20, 10, 20};
  auto A = make_box_device(box);

  auto expanded = ops(ctx)
      .from(A)
      .expand(2, 2)
      .build();

  auto ring = ops(ctx)
      .from(expanded)
      .subtract(A)
      .build();

  auto h = build_host_from_device(ring);

  // Ring should have rows from 8 to 21
  EXPECT_EQ(h.num_rows(), 14);

  // Check corner row (y=8): should have single interval [8, 22]
  EXPECT_EQ(h.row_keys[0].y, 8);
  EXPECT_EQ(h.intervals[h.row_ptr[0]].begin, 8);
  EXPECT_EQ(h.intervals[h.row_ptr[0]].end, 22);

  // Check middle row (y=15): should have TWO intervals (the ring)
  // Find row with y=15
  for (size_t i = 0; i < h.num_rows(); ++i) {
    if (h.row_keys[i].y == 15) {
      size_t start = h.row_ptr[i];
      size_t end = h.row_ptr[i + 1];
      EXPECT_EQ(end - start, 2); // Two intervals
      EXPECT_EQ(h.intervals[start].begin, 8);
      EXPECT_EQ(h.intervals[start].end, 10);
      EXPECT_EQ(h.intervals[start + 1].begin, 20);
      EXPECT_EQ(h.intervals[start + 1].end, 22);
      break;
    }
  }
}

TEST_F(CsrFluentOpsTest, SymmetricDifference) {
  Box2D boxA{0, 10, 0, 10};
  Box2D boxB{5, 15, 0, 10};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);

  auto result = ops(ctx)
      .from(A)
      .symmetric_diff(B)
      .build();

  auto h = build_host_from_device(result);

  // Symmetric diff: [0,5) ∪ [10,15) per row
  EXPECT_EQ(h.num_rows(), 10);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    size_t start = h.row_ptr[i];
    size_t end = h.row_ptr[i + 1];
    EXPECT_EQ(end - start, 2); // Two intervals per row
    EXPECT_EQ(h.intervals[start].begin, 0);
    EXPECT_EQ(h.intervals[start].end, 5);
    EXPECT_EQ(h.intervals[start + 1].begin, 10);
    EXPECT_EQ(h.intervals[start + 1].end, 15);
  }
}

TEST_F(CsrFluentOpsTest, EmptyInput) {
  IntervalSet2DDevice empty;
  Box2D box{0, 10, 0, 10};
  auto B = make_box_device(box);

  auto result = ops(ctx)
      .from(empty)
      .union_with(B)
      .build();

  auto h = build_host_from_device(result);
  EXPECT_EQ(h.num_rows(), 10);
}

TEST_F(CsrFluentOpsTest, HasValueCheck) {
  auto builder = ops(ctx);
  EXPECT_FALSE(builder.has_value());

  Box2D box{0, 10, 0, 10};
  auto A = make_box_device(box);

  builder.from(A);
  EXPECT_TRUE(builder.has_value());
}

TEST_F(CsrFluentOpsTest, CellOffsetsComputed) {
  Box2D box{0, 10, 0, 5};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .build();

  // Total cells should be 10 * 5 = 50
  EXPECT_EQ(result.total_cells, 50);
}

TEST_F(CsrFluentOpsTest, RefineGeometry) {
  Box2D box{0, 5, 0, 5};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .refine()
      .build();

  auto h = build_host_from_device(result);

  // Refined: 5 rows -> 10 rows, coordinates doubled
  EXPECT_EQ(h.num_rows(), 10);
  EXPECT_EQ(h.row_keys.front().y, 0);
  EXPECT_EQ(h.row_keys.back().y, 9);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 0);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 10);
  }
}

TEST_F(CsrFluentOpsTest, CoarsenGeometry) {
  Box2D box{0, 10, 0, 10};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .coarsen()
      .build();

  auto h = build_host_from_device(result);

  // Coarsened: 10 rows -> 5 rows, coordinates halved
  EXPECT_EQ(h.num_rows(), 5);
  EXPECT_EQ(h.row_keys.front().y, 0);
  EXPECT_EQ(h.row_keys.back().y, 4);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 0);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 5);
  }
}

TEST_F(CsrFluentOpsTest, RefineCoarsenRoundTrip) {
  Box2D box{0, 8, 0, 8};
  auto A = make_box_device(box);

  // Refine then coarsen should give back roughly the same geometry
  auto result = ops(ctx)
      .from(A)
      .refine()
      .coarsen()
      .build();

  auto h = build_host_from_device(result);

  // Should be back to 8x8
  EXPECT_EQ(h.num_rows(), 8);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 0);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 8);
  }
}

TEST_F(CsrFluentOpsTest, AMRWithSetOps) {
  // Complex AMR workflow: create fine mask, coarsen, expand, intersect
  Box2D fine_box{0, 20, 0, 20};
  Box2D obstacle{8, 12, 8, 12};

  auto fine = make_box_device(fine_box);
  auto obs = make_box_device(obstacle);

  // Create fine mask with hole, coarsen it, expand by 1
  auto coarse_expanded = ops(ctx)
      .from(fine)
      .subtract(obs)
      .coarsen()
      .expand(1, 1)
      .build();

  auto h = build_host_from_device(coarse_expanded);

  // Fine 20x20 with 4x4 hole -> coarse ~10x10 with ~2x2 hole -> expanded
  EXPECT_GT(h.num_rows(), 0);
  EXPECT_GT(h.total_cells, 0);
}

TEST_F(CsrFluentOpsTest, TranslateXY) {
  Box2D box{0, 10, 0, 10};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .translate(5, 3)
      .build();

  auto h = build_host_from_device(result);

  // Translated: X=[5,15), Y=[3,13)
  EXPECT_EQ(h.num_rows(), 10);
  EXPECT_EQ(h.row_keys.front().y, 3);
  EXPECT_EQ(h.row_keys.back().y, 12);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 5);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 15);
  }
}

TEST_F(CsrFluentOpsTest, TranslateXOnly) {
  Box2D box{0, 10, 0, 5};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .translate_x(-3)
      .build();

  auto h = build_host_from_device(result);

  // X translated by -3: X=[-3,7)
  EXPECT_EQ(h.num_rows(), 5);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, -3);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 7);
  }
}

TEST_F(CsrFluentOpsTest, TranslateYOnly) {
  Box2D box{0, 5, 10, 20};
  auto A = make_box_device(box);

  auto result = ops(ctx)
      .from(A)
      .translate_y(-10)
      .build();

  auto h = build_host_from_device(result);

  // Y translated by -10: Y=[0,10)
  EXPECT_EQ(h.num_rows(), 10);
  EXPECT_EQ(h.row_keys.front().y, 0);
  EXPECT_EQ(h.row_keys.back().y, 9);
}

TEST_F(CsrFluentOpsTest, TranslateAndIntersect) {
  Box2D boxA{0, 10, 0, 10};
  Box2D boxB{5, 15, 5, 15};

  auto A = make_box_device(boxA);
  auto B = make_box_device(boxB);

  // Translate A by (5,5) then intersect with B
  // A translated: [5,15) x [5,15) -> intersection with B [5,15) x [5,15) = full B
  auto result = ops(ctx)
      .from(A)
      .translate(5, 5)
      .intersect(B)
      .build();

  auto h = build_host_from_device(result);

  EXPECT_EQ(h.num_rows(), 10);
  for (size_t i = 0; i < h.num_rows(); ++i) {
    EXPECT_EQ(h.intervals[h.row_ptr[i]].begin, 5);
    EXPECT_EQ(h.intervals[h.row_ptr[i]].end, 15);
  }
}

} // namespace
