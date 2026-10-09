# Foo Windows Desktop Installer

Build a dedicated Windows desktop window using Microsoft Edge WebView2,
PyInstaller's onedir format, and Inno Setup.

## Requirements

- Windows x64, Python 3.11, and Microsoft Edge WebView2 Runtime
- Inno Setup 6 installed (for installer creation)
- Compiler/build tools needed by native dependencies such as llama-cpp-python
- Network access during build for pip dependencies and Playwright browser installation

## Build

```powershell
python -m pip install -r requirements.txt
python -m pip install "pyinstaller>=6,<7" "pywebview>=5,<7"
python -m PyInstaller --noconfirm --clean foo.spec
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "desktop\foo.iss"
```

The final installer is written to `dist/installer/Foo-Setup-0.1.0.exe`.

## Runtime

- `Foo.exe` starts a loopback-only Streamlit process and displays it with WebView2.
- Closing the desktop window terminates its Streamlit child process.
- Mutable databases and logs use `%LOCALAPPDATA%\Foo`.
- Streamlit UI, analytical modules, and existing app modes are not rewritten.
- External API calls still require connectivity and appropriate provider credentials.
- Playwright Chromium browsers, spaCy models, and NLTK data may require
  separate provisioning; the build specification does not claim to provide
  a fully offline runtime.
- Validate each application mode on a clean Windows virtual machine before
  distributing the installer. The workflow is manual to avoid expensive
  builds on every push.

## GitHub Actions

Use the **Build Foo Windows desktop installer** workflow's **Run workflow**
control. A successful run uploads an artifact named `Foo-Windows-Installer`.
