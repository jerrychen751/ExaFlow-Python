# -*- mode: python ; coding: utf-8 -*-
import glob
import os

import mpi4py

from PyInstaller.utils.hooks import collect_all, collect_submodules

repo_root = os.path.dirname(SPECPATH)
icon_path = os.path.join(SPECPATH, "ExaFlow.icns")

datas = []
binaries = []
hiddenimports = ["vtkmodules.all", "vtkmodules.util.numpy_support", "vtkmodules.qt"]
hiddenimports += collect_submodules("exaflow")
for package_name in ("vtkmodules", "pyvista", "pyvistaqt", "pyevtk", "mpi4py"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

mpi_extension_paths = [
    path
    for path in glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), "MPI.*.so"))
    if ".mpich." not in os.path.basename(path)
]
if not mpi_extension_paths:
    raise FileNotFoundError(f"No MPI.*.so in {os.path.dirname(mpi4py.__file__)}. Nothing else puts the mpi4py extension in the bundle, and the app cannot import MPI without it.")
datas += [(path, "mpi4py") for path in mpi_extension_paths]
datas += [(path, "exaflow/gui/presets") for path in glob.glob(os.path.join(repo_root, "src", "exaflow", "gui", "presets", "*.xml"))]

analysis = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[os.path.join(repo_root, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["mypy", "pytest", "PyInstaller", "tkinter", "PySide6.QtWebEngineCore"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name="ExaFlow",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="ExaFlow",
)

app = BUNDLE(
    collection,
    name="ExaFlow.app",
    icon=icon_path if os.path.exists(icon_path) else None,
    bundle_identifier="com.jerrychen.exaflow",
    info_plist={
        "CFBundleName": "ExaFlow",
        "CFBundleDisplayName": "ExaFlow",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
