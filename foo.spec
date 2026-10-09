# -*- mode: python ; coding: utf-8 -*-
"""Build Foo as a Windows onedir application with its existing resources."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH)
datas = [
    (str(root / 'app.py'), '.'),
    (str(root / '.streamlit'), '.streamlit'),
    (str(root / 'resources'), 'resources'),
    (str(root / 'stores'), 'stores'),
]
hiddenimports = []
binaries = []

# Frameworks rely on runtime imports, templates, package metadata, and native DLLs.
for package in ('streamlit', 'altair', 'webview', 'langchain_core',
                'langchain_community', 'langchain_text_splitters',
                'langchain_openai', 'langchain_google_genai',
                'langchain_mistralai', 'langchain_huggingface',
                'langchain_chroma', 'chromadb', 'sklearn', 'spacy',
                'nltk', 'torch', 'sentence_transformers',
                'playwright', 'crawl4ai'):
    try:
        package_datas, package_bins, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_bins
        hiddenimports += package_hidden
    except ImportError:
        raise RuntimeError(f'Required package missing from build environment: {package}')

hiddenimports += [
    'config', 'core', 'data', 'agents', 'embedders', 'fetchers', 'generators',
    'loaders', 'models', 'processors', 'scrapers', 'writers',
]
hiddenimports += collect_submodules('stores')
hiddenimports += collect_submodules('boogr')

a = Analysis(
    [str(root / 'desktop' / 'launcher.py')],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
coll = COLLECT(exe, a.binaries, a.datas, strip=False,
    upx=False, name='Foo')
