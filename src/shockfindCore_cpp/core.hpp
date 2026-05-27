#pragma once
#include "types.hpp"
#include <utility>   // std::pair
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Shock normal
// ─────────────────────────────────────────────────────────────────────────────

// Normal from the density gradient at a single point: n = -∇ρ / |∇ρ|
// Returns {0,0,0} if the gradient is zero.
Vec3 shock_normal_point(const FieldData& f, const CellIndex& idx);

// Normal from a weighted average of ∇ρ over a sphere of radius Rgrad.
// periodic[3] controls wrapping at domain boundaries.
Vec3 shock_normal_average(const FieldData& f, const CellIndex& idx,
                          int Rgrad, const bool periodic[3]);

// Dispatch based on params.method_norm.
Vec3 shock_normal(const FieldData& f, const CellIndex& idx,
                  const ShockParams& params);

// ─────────────────────────────────────────────────────────────────────────────
// Transverse frame
// Returns (nt1, nt2): two unit vectors orthogonal to ns and to each other,
// spanning the plane perpendicular to the shock normal.
// ─────────────────────────────────────────────────────────────────────────────

// Use B at a single reference point on the shock line.
std::pair<Vec3,Vec3> transverse_point_field(const std::vector<Vec3>& B_line,
                                            int ref,
                                            const Vec3& ns);

// Use the average B over a sub-region of the shock line.
std::pair<Vec3,Vec3> transverse_average_field(const std::vector<Vec3>& B_line,
                                              int region_start,
                                              int region_end,
                                              const Vec3& ns);

// ─────────────────────────────────────────────────────────────────────────────
// Cylinder average
// Averages physical fields in a cylinder of radius Rcyl around each point on
// the shock line.  Returns a LineProfile with log-averaged density/pressure
// and linearly averaged velocity/field projections.
// ─────────────────────────────────────────────────────────────────────────────
LineProfile cylinder_average(const FieldData& f,
                             const std::vector<CellIndex>& cells,
                             const std::vector<double>& line_coords,
                             const Vec3& ns,
                             const Vec3& nt1,
                             const Vec3& nt2,
                             int Rcyl);

// ─────────────────────────────────────────────────────────────────────────────
// Flux capacitor — shock classification from 1-D profiles
// ─────────────────────────────────────────────────────────────────────────────
// shock_ratio : minimum density contrast to recognise a shock (default 1.2)
// shock_size  : half-width of shock in cells for state averaging (default 3)
ShockResult flux_capacitor(const LineProfile& prof,
                           double gamma,
                           double shock_ratio = 1.2,
                           double shock_size  = 3.0);

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell worker — analyse one candidate
// ─────────────────────────────────────────────────────────────────────────────
ShockResult characterise_shock(const CellIndex& candidate,
                               const FieldData& fields,
                               const ShockParams& params);

// ─────────────────────────────────────────────────────────────────────────────
// Serial batch — loop characterise_shock over a list of candidates
// This is the Phase-1 deliverable; the MPI layer calls the same function.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<ShockResult> characterise_shocks(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet = false);
