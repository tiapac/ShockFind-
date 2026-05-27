"""
shockfindCore_cpp — C++ backend for ShockFind.

Loads the compiled pybind11 extension from build/ and exposes it as
the attribute `shockfindCore_cpp` so that

    from .shockfindCore_cpp import shockfindCore_cpp as _cpp

in shockfind_interface.py resolves correctly.  Falls back silently if the
extension has not been compiled yet.
"""
import os as _os
import glob as _glob
import importlib.util as _ilu

_build_dir = _os.path.join(_os.path.dirname(__file__), "build")
_so_files = _glob.glob(_os.path.join(_build_dir, "shockfindCore_cpp*.so"))

if _so_files:
    try:
        # spec_from_file_location resolves PyInit_shockfindCore_cpp in the .so.
        _spec   = _ilu.spec_from_file_location("shockfindCore_cpp", _so_files[0])
        _native = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_native)
        shockfindCore_cpp = _native   # exported as package attribute
    except Exception:
        pass  # not built or broken; caller falls back to Python backend
