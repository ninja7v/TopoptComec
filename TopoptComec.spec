# TopoptComec.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect ALL dynamic libs (*.dll, *.pyd, etc.) from the lib3mf package
lib3mf_binaries = collect_dynamic_libs('lib3mf')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=lib3mf_binaries,
    datas=[
        ('topoptcomec/icons/*', 'icons'),
        ('topoptcomec/presets.json', '.'),
    ],
    hiddenimports=['lib3mf'],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TopoptComec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='topoptcomec/icons/window_icon.ico'
)
