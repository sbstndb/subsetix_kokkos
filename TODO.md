<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Project Tasks

**Note:** This project tracks development tasks via GitHub Issues. Please refer to
[GitHub Issues](https://github.com/sbstndb/subsetix_kokkos_2/issues) for the current
task backlog and status.

## Completed Tasks

### Buffer Capacity Management (Resolved)

- **Task:** Verify that buffer operations do not assume `allocated_capacity == logical_size`
- **Status:** Completed
- **Resolution:** The workspace implementation (`include/subsetix/csr_ops/workspace.hpp`)
  correctly handles capacity vs. size distinction through `ensure_view_capacity()`, which
  only grows buffers when needed. The `workspace_capacity_test.cpp` unit test verifies
  this behavior.
- **Reference:** `tests/workspace_capacity_test.cpp`
