# -*- mode: python ; coding: utf-8 -*-
# Windows 目录版打包（体积远小于单文件 exe）:
#   pyinstaller CkCut_win.spec
# 产物: dist\CkCut\ 文件夹，内含 CkCut.exe + 依赖

block_cipher = None

datas = [
    ('data', 'data'),
]

# 精简：排除用不到的 Qt 子模块和库，显著减小体积
excludes = [
    'tkinter', 'unittest', 'email', 'html', 'http', 'xml', 'pydoc',
    'PyQt6.QtBluetooth', 'PyQt6.QtDBus', 'PyQt6.QtDesigner', 'PyQt6.QtHelp',
    'PyQt6.QtLocation', 'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtNetwork', 'PyQt6.QtNfc', 'PyQt6.QtOpenGL', 'PyQt6.QtOpenGLWidgets',
    'PyQt6.QtPositioning', 'PyQt6.QtPrintSupport', 'PyQt6.QtQml', 'PyQt6.QtQuick',
    'PyQt6.QtQuickWidgets', 'PyQt6.QtRemoteObjects', 'PyQt6.QtSensors',
    'PyQt6.QtSerialPort', 'PyQt6.QtSql', 'PyQt6.QtSvg', 'PyQt6.QtSvgWidgets',
    'PyQt6.QtTest', 'PyQt6.QtWebChannel', 'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebSockets', 'PyQt6.Qt3DCore', 'PyQt6.Qt3DRender', 'PyQt6.Qt3DInput',
    'PyQt6.Qt3DExtras', 'PyQt6.QtCharts', 'PyQt6.QtDataVisualization',
    'PyQt6.QtScxml', 'PyQt6.QtStateMachine', 'PyQt6.QtTextToSpeech',
]

a = Analysis(
    ['main_visible.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['src.segmenter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 不塞进单 exe，交给 COLLECT 生成目录
    name='CkCut',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

# 目录模式：生成 dist\CkCut\ 文件夹（内含 CkCut.exe + 一堆 dll）
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CkCut',
)
