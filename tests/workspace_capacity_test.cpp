#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <subsetix/csr_ops/workspace.hpp>

using namespace subsetix::csr;

TEST(WorkspaceCapacityTest, AutomaticGrowth) {
  detail::UnifiedCsrWorkspace ws;

  // Initial request
  auto buf1 = ws.get_int_buf(0, 100);
  EXPECT_GE(buf1.extent(0), 100);
  EXPECT_EQ(ws.int_bufs_[0].extent(0), buf1.extent(0));

  // Request smaller size - should NOT shrink/reallocate
  auto ptr_before = ws.int_bufs_[0].data();
  auto buf2 = ws.get_int_buf(0, 50);
  EXPECT_EQ(ws.int_bufs_[0].data(), ptr_before);
  EXPECT_GE(ws.int_bufs_[0].extent(0), 100);

  // Request larger size - should reallocate
  auto buf3 = ws.get_int_buf(0, 200);
  EXPECT_GE(buf3.extent(0), 200);
  EXPECT_GE(ws.int_bufs_[0].extent(0), 200);
  // Pointer might change, though technically not guaranteed if allocator extends in place (unlikely for Kokkos)
  // We mainly care that extent increased.
}

TEST(WorkspaceCapacityTest, MultipleBuffers) {
  detail::UnifiedCsrWorkspace ws;

  auto ib0 = ws.get_int_buf(0, 10);
  auto ib1 = ws.get_int_buf(1, 20);
  auto st0 = ws.get_size_t_buf(0, 30);

  EXPECT_GE(ws.int_bufs_[0].extent(0), 10);
  EXPECT_GE(ws.int_bufs_[1].extent(0), 20);
  EXPECT_GE(ws.size_t_bufs_[0].extent(0), 30);
}

TEST(WorkspaceCapacityTest, ScalarBuffers) {
  detail::UnifiedCsrWorkspace ws;

  // Scalar buffers are lazily allocated on first access.
  // For Rank-0 views (scalars), we check data() pointer to verify if allocated.
  EXPECT_EQ(ws.scalar_size_t_buf_0.data(), nullptr);
  
  auto s0 = ws.get_scalar_size_t_buf_0();
  EXPECT_NE(ws.scalar_size_t_buf_0.data(), nullptr);
  EXPECT_NE(s0.data(), nullptr);

  EXPECT_EQ(ws.scalar_int_buf_0.data(), nullptr);
  auto s1 = ws.get_scalar_int_buf_0();
  EXPECT_NE(ws.scalar_int_buf_0.data(), nullptr);
}

TEST(WorkspaceCapacityTest, ClearReclaimsMemory) {
  detail::UnifiedCsrWorkspace ws;

  // Allocate some memory
  ws.get_int_buf(0, 1000);
  ws.get_row_key_buf(0, 500);

  EXPECT_GE(ws.int_bufs_[0].extent(0), 1000);
  EXPECT_GE(ws.row_key_bufs_[0].extent(0), 500);

  // Clear
  ws.clear();

  // Views should be default-constructed (empty/null)
  // For dynamic views, extent(0) should be 0.
  EXPECT_EQ(ws.int_bufs_[0].extent(0), 0);
  EXPECT_EQ(ws.row_key_bufs_[0].extent(0), 0);

  // Should be able to reallocate
  ws.get_int_buf(0, 10);
  EXPECT_GE(ws.int_bufs_[0].extent(0), 10);
}

