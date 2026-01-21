// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
