#include "mpi_dispatch.hpp"
#include "core.hpp"
#include <algorithm>
#include <array>

#ifdef USE_MPI
#include <mpi.h>

// ─────────────────────────────────────────────────────────────────────────────
// MPI derived type for ShockResult — covers all 17 mixed int/double fields.
// build_result_type() creates and commits the type; caller must MPI_Type_free.
// ─────────────────────────────────────────────────────────────────────────────
static MPI_Datatype build_result_type() {
    const int NFIELDS = 17;
    int          blens[NFIELDS];
    MPI_Aint     disps[NFIELDS];
    MPI_Datatype types[NFIELDS];

    ShockResult dummy{};
    MPI_Aint base;
    MPI_Get_address(&dummy, &base);

    int fi = 0;
    auto add = [&](void* ptr, MPI_Datatype t) {
        MPI_Get_address(ptr, &disps[fi]);
        disps[fi] -= base;
        blens[fi]  = 1;
        types[fi]  = t;
        ++fi;
    };
    add(&dummy.loc_x,       MPI_INT);
    add(&dummy.loc_y,       MPI_INT);
    add(&dummy.loc_z,       MPI_INT);
    add(&dummy.dir_x,       MPI_DOUBLE);
    add(&dummy.dir_y,       MPI_DOUBLE);
    add(&dummy.dir_z,       MPI_DOUBLE);
    add(&dummy.family,      MPI_INT);
    add(&dummy.vs,          MPI_DOUBLE);
    add(&dummy.vA,          MPI_DOUBLE);
    add(&dummy.MachAlf,     MPI_DOUBLE);
    add(&dummy.Mach,        MPI_DOUBLE);
    add(&dummy.r,           MPI_DOUBLE);
    add(&dummy.rho0,        MPI_DOUBLE);
    add(&dummy.B0,          MPI_DOUBLE);
    add(&dummy.pmag_ratio,  MPI_DOUBLE);
    add(&dummy.peak_flag,   MPI_INT);
    add(&dummy.flag,        MPI_INT);

    MPI_Datatype tmp, result_type;
    MPI_Type_create_struct(NFIELDS, blens, disps, types, &tmp);
    MPI_Type_create_resized(tmp, 0, static_cast<MPI_Aint>(sizeof(ShockResult)), &result_type);
    MPI_Type_free(&tmp);
    MPI_Type_commit(&result_type);
    return result_type;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract a contiguous 3-D sub-array [xm:xM, ym:yM, zm:zM] from a C-order
// flat array src of shape (nx, ny, nz).
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> extract_subarray(
    const double* src, int ny, int nz,
    int xm, int xM, int ym, int yM, int zm, int zM)
{
    int lnx = xM-xm, lny = yM-ym, lnz = zM-zm;
    std::vector<double> out(static_cast<size_t>(lnx)*lny*lnz);
    for (int ii = 0; ii < lnx; ++ii)
        for (int jj = 0; jj < lny; ++jj)
            for (int kk = 0; kk < lnz; ++kk)
                out[static_cast<size_t>(ii)*lny*lnz + jj*lnz + kk] =
                    src[static_cast<size_t>(xm+ii)*ny*nz + (ym+jj)*nz + (zm+kk)];
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — bounding-box domain decomposition.
//
// Each rank receives only the field data it actually needs (its candidates'
// bounding box + buffer padding), matching the Python multiprocessing approach
// in characterise_shocks_para.  No full-array replication across ranks.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<ShockResult> characterise_shocks_mpi(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
        return characterise_shocks(candidates, fields, params, quiet);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1)
        return characterise_shocks(candidates, fields, params, quiet);

    // ── 1. Broadcast total count and grid dimensions ──────────────────────────
    int hdr[4];   // [total, nx, ny, nz]
    if (rank == 0) {
        hdr[0] = static_cast<int>(candidates.size());
        hdr[1] = fields.grid.shape.nx;
        hdr[2] = fields.grid.shape.ny;
        hdr[3] = fields.grid.shape.nz;
    }
    MPI_Bcast(hdr, 4, MPI_INT, 0, MPI_COMM_WORLD);
    int total = hdr[0], nx = hdr[1], ny = hdr[2], nz = hdr[3];

    // ── 2. Chunk sizes and displacements ─────────────────────────────────────
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r)
        counts[r] = total / size + (r < (total % size) ? 1 : 0);
    displs[0] = 0;
    for (int r = 1; r < size; ++r)
        displs[r] = displs[r-1] + counts[r-1];
    int local_count = counts[rank];

    // ── 3. Scatter candidates in global coordinates ───────────────────────────
    std::vector<int> sc3(size), sd3(size);
    for (int r = 0; r < size; ++r) { sc3[r] = counts[r]*3; sd3[r] = displs[r]*3; }

    std::vector<int> all_flat;
    if (rank == 0) {
        all_flat.resize(total * 3);
        for (int n = 0; n < total; ++n) {
            all_flat[3*n]   = candidates[n].i;
            all_flat[3*n+1] = candidates[n].j;
            all_flat[3*n+2] = candidates[n].k;
        }
    }
    std::vector<int> lflat(local_count * 3);
    MPI_Scatterv(rank == 0 ? all_flat.data() : nullptr,
                 sc3.data(), sd3.data(), MPI_INT,
                 lflat.data(), local_count * 3, MPI_INT,
                 0, MPI_COMM_WORLD);

    std::vector<CellIndex> local_cands(local_count);
    for (int n = 0; n < local_count; ++n)
        local_cands[n] = {lflat[3*n], lflat[3*n+1], lflat[3*n+2]};

    // ── 4. Each rank computes its bounding box (global coords + buffer) ────────
    // buffer must cover every neighbourhood access: line_range, Rcylinder, Rgrad.
    int buf = std::max({params.Rcylinder, params.Rgrad, params.line_range});

    int bbox[6] = {0, 0, 0, 0, 0, 0};   // [xm, xM, ym, yM, zm, zM]
    if (local_count > 0) {
        int xm = nx, xM = 0, ym = ny, yM = 0, zm = nz, zM = 0;
        for (const auto& c : local_cands) {
            xm = std::min(xm, c.i); xM = std::max(xM, c.i);
            ym = std::min(ym, c.j); yM = std::max(yM, c.j);
            zm = std::min(zm, c.k); zM = std::max(zM, c.k);
        }
        bbox[0] = std::max(xm - buf, 0);        bbox[1] = std::min(xM + buf + 1, nx);
        bbox[2] = std::max(ym - buf, 0);        bbox[3] = std::min(yM + buf + 1, ny);
        bbox[4] = std::max(zm - buf, 0);        bbox[5] = std::min(zM + buf + 1, nz);
    }

    // ── 5. Gather all bboxes to rank 0 ────────────────────────────────────────
    std::vector<int> all_bboxes(size * 6);
    MPI_Gather(bbox, 6, MPI_INT, all_bboxes.data(), 6, MPI_INT, 0, MPI_COMM_WORLD);

    // ── 6. Rank 0 extracts sub-arrays and sends to non-zero ranks ─────────────
    const double* fptrs[12] = {
        fields.rho,    fields.pres,
        fields.vx,     fields.vy,     fields.vz,
        fields.bx,     fields.by,     fields.bz,
        fields.div,    fields.grad_x, fields.grad_y, fields.grad_z
    };

    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            if (counts[r] == 0) continue;
            int rxm = all_bboxes[6*r+0], rxM = all_bboxes[6*r+1];
            int rym = all_bboxes[6*r+2], ryM = all_bboxes[6*r+3];
            int rzm = all_bboxes[6*r+4], rzM = all_bboxes[6*r+5];
            int rnx = rxM-rxm, rny = ryM-rym, rnz = rzM-rzm;
            int rN  = rnx * rny * rnz;
            for (int f = 0; f < 12; ++f) {
                auto sub = extract_subarray(fptrs[f], ny, nz, rxm, rxM, rym, ryM, rzm, rzM);
                MPI_Send(sub.data(), rN, MPI_DOUBLE, r, f, MPI_COMM_WORLD);
            }
        }
    }

    // ── 7. Each rank sets up a local FieldData over its received sub-arrays ────
    int lnx = bbox[1]-bbox[0], lny = bbox[3]-bbox[2], lnz = bbox[5]-bbox[4];
    int lN  = (local_count > 0) ? lnx * lny * lnz : 0;

    std::array<std::vector<double>, 12> local_bufs;

    if (rank == 0) {
        // Extract rank 0's sub-array locally (no MPI).
        for (int f = 0; f < 12; ++f)
            local_bufs[f] = extract_subarray(fptrs[f], ny, nz,
                                             bbox[0], bbox[1], bbox[2],
                                             bbox[3], bbox[4], bbox[5]);
    } else {
        for (int f = 0; f < 12; ++f) {
            local_bufs[f].resize(lN);
            if (local_count > 0)
                MPI_Recv(local_bufs[f].data(), lN, MPI_DOUBLE,
                         0, f, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    FieldData lfields;
    lfields.grid    = GridAccessor{GridShape{lnx, lny, lnz}};
    lfields.rho     = local_bufs[0].data();
    lfields.pres    = local_bufs[1].data();
    lfields.vx      = local_bufs[2].data();
    lfields.vy      = local_bufs[3].data();
    lfields.vz      = local_bufs[4].data();
    lfields.bx      = local_bufs[5].data();
    lfields.by      = local_bufs[6].data();
    lfields.bz      = local_bufs[7].data();
    lfields.div     = local_bufs[8].data();
    lfields.grad_x  = local_bufs[9].data();
    lfields.grad_y  = local_bufs[10].data();
    lfields.grad_z  = local_bufs[11].data();

    // ── 8. Set offset so characterise_shock maps global→local for lookups ──────
    // characterise_shock computes: idx = candidate - offset  (local field index)
    //                              loc = idx + offset        (global output coord)
    // Setting offset = sub-array origin gives correct field access and output.
    ShockParams lparams   = params;
    lparams.offset[0]     = bbox[0];   // xm
    lparams.offset[1]     = bbox[2];   // ym
    lparams.offset[2]     = bbox[4];   // zm

    // ── 9. Process this rank's candidates ─────────────────────────────────────
    std::vector<ShockResult> local_results;
    if (local_count > 0)
        local_results = characterise_shocks(local_cands, lfields, lparams,
                                            quiet || rank != 0);

    // ── 10. Gather results to rank 0 ──────────────────────────────────────────
    MPI_Datatype result_type = build_result_type();
    std::vector<ShockResult> all_results;
    if (rank == 0) all_results.resize(total);

    MPI_Gatherv(local_results.data(), static_cast<int>(local_results.size()),
                result_type,
                rank == 0 ? all_results.data() : nullptr,
                counts.data(), displs.data(), result_type,
                0, MPI_COMM_WORLD);

    MPI_Type_free(&result_type);
    return (rank == 0) ? all_results : std::vector<ShockResult>{};
}

#else  // !USE_MPI

std::vector<ShockResult> characterise_shocks_mpi(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet) {
    return characterise_shocks(candidates, fields, params, quiet);
}

#endif  // USE_MPI
