cmake_minimum_required(VERSION 3.18)
project(shockfindCore_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ──────────────────────────────────────────────────────────────────
# Let the user override the hint if needed:
#   cmake -B build -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
find_package(pybind11 REQUIRED)

# ── Optional MPI ─────────────────────────────────────────────────────────────
# Disabled by default in Phase 1 (the dispatch is a serial stub).
# Enable with:  cmake -B build -DUSE_MPI=ON -DMPI_CXX_COMPILER=<mpicxx>
option(USE_MPI "Enable MPI parallel dispatch (Phase 2)" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_compile_definitions(USE_MPI)
    message(STATUS "MPI enabled — using ${MPI_CXX_LIBRARIES}")
else()
    message(STATUS "MPI disabled — serial-only build (Phase 1 stub)")
endif()

# ── Core static library ───────────────────────────────────────────────────────
# No dependency on pybind11 so it can be linked by standalone tools too.
add_library(shockcore STATIC
    core.cpp
    mpi_dispatch.cpp
)
target_include_directories(shockcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# -fPIC is required when the static lib is linked into a shared Python extension.
set_target_properties(shockcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(shockcore PRIVATE -O3 -Wall -Wextra -Wno-trigraphs)
if(USE_MPI)
    target_link_libraries(shockcore PUBLIC MPI::MPI_CXX)
endif()

# ── Python extension module ───────────────────────────────────────────────────
pybind11_add_module(shockfindCore_cpp bindings.cpp)
target_link_libraries(shockfindCore_cpp PRIVATE shockcore)
target_compile_options(shockfindCore_cpp PRIVATE -O3)

# Install the .so next to the other Python source files for easy import.
install(TARGETS shockfindCore_cpp
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
