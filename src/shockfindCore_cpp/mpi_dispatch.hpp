#pragma once
#include "types.hpp"
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// MPI-parallel batch dispatch.
//
// Phase 1: when USE_MPI is not defined, this falls through to the serial
//          characterise_shocks().
//
// Phase 2: rank 0 scatters candidates to all MPI ranks, each rank runs the
//          serial core on its chunk, rank 0 gathers results back.
//          The Python side launches via  mpirun -n N python script.py  and
//          uses mpi4py so all ranks enter the pybind11 extension together.
//
// Signature is identical to the serial version so the Python binding can
// select either without changing any call-site code.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<ShockResult> characterise_shocks_mpi(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet = false);
