#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core.hpp"
#include "mpi_dispatch.hpp"

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Build a FieldData from eleven 3-D NumPy arrays of shape (nx, ny, nz).
// All arrays are required to be C-contiguous double arrays.
// ─────────────────────────────────────────────────────────────────────────────
static FieldData make_field_data(
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> pres,
    py::array_t<double, py::array::c_style | py::array::forcecast> vx,
    py::array_t<double, py::array::c_style | py::array::forcecast> vy,
    py::array_t<double, py::array::c_style | py::array::forcecast> vz,
    py::array_t<double, py::array::c_style | py::array::forcecast> bx,
    py::array_t<double, py::array::c_style | py::array::forcecast> by,
    py::array_t<double, py::array::c_style | py::array::forcecast> bz,
    py::array_t<double, py::array::c_style | py::array::forcecast> div,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_z)
{
    if (rho.ndim() != 3)
        throw std::runtime_error("Field arrays must be 3-D");

    auto buf = rho.request();
    GridShape shape{
        static_cast<int>(buf.shape[0]),
        static_cast<int>(buf.shape[1]),
        static_cast<int>(buf.shape[2])
    };

    FieldData f;
    f.grid   = GridAccessor{shape};
    f.rho    = rho   .data();
    f.pres   = pres  .data();
    f.vx     = vx    .data();
    f.vy     = vy    .data();
    f.vz     = vz    .data();
    f.bx     = bx    .data();
    f.by     = by    .data();
    f.bz     = bz    .data();
    f.div    = div   .data();
    f.grad_x = grad_x.data();
    f.grad_y = grad_y.data();
    f.grad_z = grad_z.data();
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert a Python dict (the `extra` dict from shock_finder) to ShockParams.
// ─────────────────────────────────────────────────────────────────────────────
static ShockParams dict_to_params(const py::dict& extra) {
    ShockParams p;

    auto get_str = [&](const char* key, const std::string& def) -> std::string {
        return extra.contains(key) ? extra[key].cast<std::string>() : def;
    };
    auto get_int = [&](const char* key, int def) -> int {
        return extra.contains(key) ? extra[key].cast<int>() : def;
    };
    auto get_dbl = [&](const char* key, double def) -> double {
        return extra.contains(key) ? extra[key].cast<double>() : def;
    };

    std::string mn = get_str("method_norm",  "point_gradient");
    p.method_norm  = (mn == "average_gradient")
                   ? ShockParams::AVERAGE_GRADIENT
                   : ShockParams::POINT_GRADIENT;

    std::string mp = get_str("method_plane", "point_field");
    p.method_plane = (mp == "average_field")
                   ? ShockParams::AVERAGE_FIELD
                   : ShockParams::POINT_FIELD;

    if (extra.contains("periodic")) {
        auto per = extra["periodic"].cast<py::list>();
        for (int i = 0; i < 3; ++i)
            p.periodic[i] = per[i].cast<bool>();
    }
    if (extra.contains("offset")) {
        auto off = extra["offset"].cast<py::list>();
        for (int i = 0; i < 3; ++i)
            p.offset[i] = off[i].cast<int>();
    }

    p.Rgrad      = get_int("Rgrad",      3);
    p.Rcylinder  = get_int("Rcylinder",  3);
    p.line_range = get_int("line_range", 10);
    p.field_ref  = get_int("field_ref",  0);
    p.gamma      = get_dbl("gamma",      5.0/3.0);
    p.shock_ratio= get_dbl("shock_ratio",1.1);

    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert vector<ShockResult> to the Python 17-array list expected by
// shock_finder.analyse_candidates (same layout as the Python core).
// ─────────────────────────────────────────────────────────────────────────────
static py::tuple results_to_python(const std::vector<ShockResult>& results) {
    int N = static_cast<int>(results.size());

    // 17 output arrays (same order as the Python characterise_shocks)
    auto loc_x          = py::array_t<int>   (N);
    auto loc_y          = py::array_t<int>   (N);
    auto loc_z          = py::array_t<int>   (N);
    auto dir_x          = py::array_t<double>(N);
    auto dir_y          = py::array_t<double>(N);
    auto dir_z          = py::array_t<double>(N);
    auto families       = py::array_t<int>   (N);
    auto speeds         = py::array_t<double>(N);
    auto vA_arr         = py::array_t<double>(N);
    auto MachAlf_arr    = py::array_t<double>(N);
    auto Mach_arr       = py::array_t<double>(N);
    auto r_arr          = py::array_t<double>(N);
    auto rho0_arr       = py::array_t<double>(N);
    auto B0_arr         = py::array_t<double>(N);
    auto pmag_arr       = py::array_t<double>(N);
    auto peak_arr       = py::array_t<int>   (N);
    auto flag_arr       = py::array_t<int>   (N);

    auto x_buf  = loc_x      .mutable_unchecked<1>();
    auto y_buf  = loc_y      .mutable_unchecked<1>();
    auto z_buf  = loc_z      .mutable_unchecked<1>();
    auto dx_buf = dir_x      .mutable_unchecked<1>();
    auto dy_buf = dir_y      .mutable_unchecked<1>();
    auto dz_buf = dir_z      .mutable_unchecked<1>();
    auto fam_b  = families   .mutable_unchecked<1>();
    auto spd_b  = speeds     .mutable_unchecked<1>();
    auto vA_b   = vA_arr     .mutable_unchecked<1>();
    auto mA_b   = MachAlf_arr.mutable_unchecked<1>();
    auto ms_b   = Mach_arr   .mutable_unchecked<1>();
    auto r_b    = r_arr      .mutable_unchecked<1>();
    auto rho_b  = rho0_arr   .mutable_unchecked<1>();
    auto B0_b   = B0_arr     .mutable_unchecked<1>();
    auto pm_b   = pmag_arr   .mutable_unchecked<1>();
    auto pk_b   = peak_arr   .mutable_unchecked<1>();
    auto fl_b   = flag_arr   .mutable_unchecked<1>();

    for (int i = 0; i < N; ++i) {
        const ShockResult& r = results[i];
        x_buf [i] = r.loc_x;
        y_buf [i] = r.loc_y;
        z_buf [i] = r.loc_z;
        dx_buf[i] = r.dir_x;
        dy_buf[i] = r.dir_y;
        dz_buf[i] = r.dir_z;
        fam_b [i] = r.family;
        spd_b [i] = r.vs;
        vA_b  [i] = r.vA;
        mA_b  [i] = r.MachAlf;
        ms_b  [i] = r.Mach;
        r_b   [i] = r.r;
        rho_b [i] = r.rho0;
        B0_b  [i] = r.B0;
        pm_b  [i] = r.pmag_ratio;
        pk_b  [i] = r.peak_flag;
        fl_b  [i] = r.flag;
    }

    // Return as (data_list, header_pair) to match the Python interface
    py::list data;
    data.append(loc_x);  data.append(loc_y);  data.append(loc_z);
    data.append(dir_x);  data.append(dir_y);  data.append(dir_z);
    data.append(families);
    data.append(speeds);  data.append(vA_arr);
    data.append(MachAlf_arr); data.append(Mach_arr);
    data.append(r_arr);   data.append(rho0_arr);
    data.append(B0_arr);  data.append(pmag_arr);
    data.append(peak_arr); data.append(flag_arr);

    // Provide the column names so shocks_data() / histograms() work.
    // Note: the Python core has a missing-comma bug that silently joins
    // 'Mach' and 'r' into 'Machr'; we emit the correct 17 names here.
    py::list col_names;
    for (const char* h : {"x","y","z","nx","ny","nz","Family","vs","vA",
                          "MachAlf","Mach","r","rho0","B0","pmag_ratio","peak","FLAG"})
        col_names.append(h);

    py::tuple header = py::make_tuple(py::none(), col_names);
    return py::make_tuple(data, header);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Python-visible function: characterise_shocks
// Mirrors the signature expected by shock_finder.analyse_candidates().
// ─────────────────────────────────────────────────────────────────────────────
static py::tuple py_characterise_shocks(
    py::list candidates_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> pres,
    py::list B,       // [bx, by, bz]
    py::list V,       // [vx, vy, vz]
    py::array_t<double, py::array::c_style | py::array::forcecast> div,
    py::list nablaRho, // [grad_x, grad_y, grad_z]
    py::dict extra,
    bool use_mpi = false,
    bool quiet   = false)
{
    // Unpack field components
    auto cast = [](py::handle h) {
        return h.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    };
    FieldData fields = make_field_data(
        rho, pres,
        cast(V[0]),  cast(V[1]),  cast(V[2]),
        cast(B[0]),  cast(B[1]),  cast(B[2]),
        div,
        cast(nablaRho[0]), cast(nablaRho[1]), cast(nablaRho[2])
    );

    // Convert candidate list: each entry is a (i, j, k) tuple
    std::vector<CellIndex> candidates;
    candidates.reserve(py::len(candidates_py));
    for (auto item : candidates_py) {
        auto t = item.cast<py::tuple>();
        candidates.push_back({
            t[0].cast<int>(),
            t[1].cast<int>(),
            t[2].cast<int>()
        });
    }

    ShockParams params = dict_to_params(extra);

    std::vector<ShockResult> results = use_mpi
        ? characterise_shocks_mpi(candidates, fields, params, quiet)
        : characterise_shocks    (candidates, fields, params, quiet);

    return results_to_python(results);
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(shockfindCore_cpp, m) {
    m.doc() = "C++ core backend for ShockFind shock detection";

    m.def("characterise_shocks", &py_characterise_shocks,
          py::arg("candidates"),
          py::arg("rho"),
          py::arg("pres"),
          py::arg("B"),
          py::arg("V"),
          py::arg("div"),
          py::arg("nablaRho"),
          py::arg("extra"),
          py::arg("use_mpi") = false,
          py::arg("quiet")   = false,
          R"(
Characterise a list of shock candidate cells.

Parameters
----------
candidates : list of (i, j, k) tuples
    Shock candidate cell indices.
rho, pres : ndarray (nx, ny, nz), float64
    Density and pressure (CGS).
B : [bx, by, bz]  each ndarray (nx, ny, nz)
    Magnetic field components.
V : [vx, vy, vz]  each ndarray (nx, ny, nz)
    Velocity components.
div : ndarray (nx, ny, nz)
    Pre-computed velocity divergence.
nablaRho : [gx, gy, gz]  each ndarray (nx, ny, nz)
    Pre-computed density gradient components.
extra : dict
    Algorithm parameters (same dict passed to shock_finder.extra_params()).
use_mpi : bool, optional
    Use MPI dispatch (Phase 2; currently falls through to serial).
quiet : bool, optional
    Suppress per-cell progress output.

Returns
-------
(data, header) : tuple
    data   — list of 17 NumPy arrays (same layout as Python core)
    header — (None, [col_name, ...]) tuple matching shock_finder.header
)");
}
