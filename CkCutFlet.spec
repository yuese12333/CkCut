# -*- mode: python ; coding: utf-8 -*-

# 仅保留 ONNX 推理链路依赖，尽量避免把 torch 打进包

a = Analysis(
    ["main_flet.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "src_nn_crf.infer_onnx",
        "src_nn_crf.viterbi_numpy",
        "src_nn_crf.vocab_io",
        "onnxruntime",
        "regex",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "IPython", "jupyter", "sphinx", "pytest"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CkCutFlet",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CkCutFlet",
)
