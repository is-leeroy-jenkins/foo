# -*- mode: python ; coding: utf-8 -*-
"""Foo Windows onedir build; run on Windows with desktop-requirements.txt installed."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH)
datas = []
binaries = []
hiddenimports = []
for package in (
    'streamlit', 'webview', 'langchain', 'langchain_core',
    'langchain_community', 'langchain_text_splitters', 'chromadb',
    'sentence_transformers', 'transformers', 'sklearn', 'spacy',
    'nltk', 'plotly', 'altair', 'playwright', 'crawl4ai',
):
    package_data, package_binaries, package_imports = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_imports

for module in (
    'agents', 'config', 'core', 'data', 'embedders', 'fetchers',
    'generators', 'loaders', 'models', 'processors', 'scrapers',
    'writers', 'boogr', 'stores',
):
    hiddenimports += collect_submodules(module)

datas += [(str(root / 'app.py'), '.')]
for directory in ('resources', '.streamlit', 'stores'):
    if (root / directory).exists():
        datas.append((str(root / directory), directory))

a = Analysis(
    ['desktop.py'],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['mkdocs', 'pytest', 'black', 'jupyterlab', 'notebook'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='Foo',
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Foo')
