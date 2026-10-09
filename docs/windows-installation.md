# Foo — Windows Desktop Installation Guide

This guide explains how to build, download, install, launch, troubleshoot, and uninstall the **Foo** Windows desktop application. Foo is a Streamlit application displayed in its own Microsoft Edge WebView2 window. It starts a local Streamlit server; the UI is **not** a native rewrite of the Streamlit app.

> **Release status:** The repository contains installer build configurations, but a successful workflow run and installation on a clean Windows machine must be verified before distributing the resulting Setup executable. An uploaded artifact is not evidence that every Foo mode works.

## 1. Requirements

### For people installing Foo

- A supported **64-bit Windows** PC.
- Microsoft **Edge WebView2 Runtime** installed. Some Windows installations already include it; otherwise install it from the [official Microsoft WebView2 page](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
- Enough disk space for the application and its substantial machine-learning and browser-related dependencies. The final installer size has not been measured.
- Internet access and applicable API credentials for features that use external services. Local features may work without connectivity.
- Administrator permission **may** be required by the Inno Setup configuration in `desktop/foo.iss`; follow the actual installer prompts.

Installing an existing, fully built Foo Setup executable should **not** require a separate Python installation. Some optional model files, browser assets, and external runtimes may still need installation or configuration, as described below.

### For people building Foo

- Windows x64 and **Python 3.11**.
- Project dependencies in [`requirements.txt`](https://github.com/is-leeroy-jenkins/foo/blob/main/requirements.txt), including the Windows-only PyInstaller and pywebview requirements.
- **Inno Setup 6** to compile the Windows installer. Inno Setup is a Windows application, not a pip package; obtain it from [jrsoftware.org](https://jrsoftware.org/isinfo.php).
- Native build tools where required by dependencies without compatible Windows wheels.
- Edge WebView2 Runtime for testing.
- Network access for Python packages and any Playwright/browser resource downloads.

## 2. Build and download the installer from GitHub Actions

The repository currently has **two** Windows build workflows. Both produce an artifact named `Foo-Windows-Installer`, but they use different installer paths and setup scripts. Select the workflow you intend to validate and check its logs.

| GitHub Actions workflow | Trigger | Inno Setup script | Expected installer directory |
|---|---|---|---|
| [Build Foo Windows desktop installer](https://github.com/is-leeroy-jenkins/foo/blob/main/.github/workflows/build-windows-installer.yml) | Manual | `desktop/foo.iss` | `dist/installer/` |
| [Build Foo Windows Installer](https://github.com/is-leeroy-jenkins/foo/blob/main/.github/workflows/windows-installer.yml) | Manual or `desktop-v*` tag | `installer/foo.iss` | `dist-installer/` |

**Important:** The first workflow invokes Inno Setup from a fixed path without installing it, and the second installs Inno Setup using Chocolatey. The second workflow also installs Playwright Chromium during the build, but this does **not** by itself prove that Chromium is bundled into the installer. Review logs and packaging before using either installer. Do not assume both workflows are interchangeable.

1. Sign in to GitHub and open the [Foo repository](https://github.com/is-leeroy-jenkins/foo).
2. Select **Actions**.
3. In the left sidebar, select **Build Foo Windows desktop installer** to use the `desktop/foo.iss` packaging path.
4. Select **Run workflow**, choose the branch to build (usually `main`), and confirm.
5. Wait for the build to finish. Open the run and inspect **every failed step** if the result is not successful.
6. On a successful run, scroll to the **Artifacts** section and download **Foo-Windows-Installer**.
7. Extract the downloaded ZIP archive. The installer executable should be inside, named according to the Inno Setup script, such as `Foo-Setup-0.1.0.exe`.
8. Verify the artifact belongs to your intended repository, branch, and workflow run before executing it.

GitHub Actions artifacts are available only for the retention period configured for the repository. If the artifact is gone, start a new run. A packaged installer is **not automatically attached to a GitHub Release** by these workflows.

## 3. Install Foo on Windows

1. Save and extract the **Foo-Windows-Installer** artifact ZIP from the successful GitHub Actions run.
2. Find `Foo-Setup-0.1.0.exe` (or the versioned Setup executable produced by the selected workflow).
3. Check the file's **Properties → Digital Signatures** if applicable. The repository configuration does not provide a code-signing certificate, so expect an unsigned build unless signing was performed separately.
4. Double-click the installer. If Windows SmartScreen displays a warning, first verify the file's provenance and whether you trust the build; do not bypass a warning for an unverified file.
5. Follow the Inno Setup wizard: select or confirm the installation folder, Start Menu shortcuts, and optional desktop shortcut.
6. Allow elevation if Windows requests it and the selected installer uses machine-wide installation.
7. Finish the wizard. If **Launch Foo** appears, you may select it; otherwise launch Foo from the Start Menu or desktop shortcut.

The `desktop/foo.iss` configuration targets `{autopf}\Foo` (normally Program Files for an administrative installation), adds a Start Menu shortcut, offers an optional desktop shortcut, and registers an uninstaller. The alternative `installer/foo.iss` has different privilege settings. Follow the behavior of the installer you actually built.

## 4. First launch and normal use

1. Open **Foo** from the Start Menu or desktop shortcut.
2. The launcher starts a local Streamlit server and waits for its health endpoint to respond.
3. A dedicated **WebView2** window opens and renders Foo. A separate browser tab is not required.
4. Select the desired Foo application mode and use its normal Streamlit controls.
5. Supply your own API credentials for any provider-specific features according to the application's configuration.
6. Close the Foo desktop window when finished. The desktop launcher is designed to terminate its Streamlit child process on exit.

The desktop server uses the local loopback address `127.0.0.1`, so it is not intended to expose the Foo UI as a remote network service. The launcher chooses an available local port; do not rely on a fixed port number.

**Startup may take longer on the first run**, particularly with large imports, model initialization, and local runtime provisioning. The launcher has a readiness timeout of approximately 120 seconds.

## 5. Data persistence and configuration

Desktop builds are intended to keep mutable data outside the installed application directory:

```text
%LOCALAPPDATA%\Foo
```

The launcher provisions runtime directories and copies bundled supporting resources. Desktop configuration directs logs and the configured database location to writable user storage. Local app data may include SQLite databases, logs, imported files, and vector store data, depending on the feature used.

- **Back up** the `%LOCALAPPDATA%\Foo` directory before uninstalling, resetting, or upgrading.
- Application data may persist after uninstalling the program; do not delete it unless you intentionally want to remove your data.
- If existing development-mode data lives in the repository's `stores` directory, do not assume it will automatically be imported into an installed desktop profile.
- Treat API credentials and downloaded documents as sensitive; avoid including them in bug reports or installer artifacts.

## 6. Optional capabilities and external prerequisites

Foo's Python dependencies are installed during the executable build, but not all resources can be bundled through `requirements.txt`.

| Capability | Possible requirement |
|---|---|
| WebView2 window | Microsoft Edge WebView2 Runtime on the installed PC |
| Playwright / Crawl4AI | Compatible Chromium browser files and a usable Playwright browser path |
| spaCy / NLTK | Language models and corpora downloaded or supplied separately |
| Local LLMs | User-selected model files and compatible native inference binaries |
| Geospatial and scientific workflows | Native DLLs and data used by Cartopy and related packages |
| Generative AI and online retrieval | Network access and valid third-party credentials |
| Local vector stores | Writable local storage and compatible installed binaries |

The current packaging configuration **does not prove** that every optional browser, language model, dataset, or native DLL is included. Validate the modes you plan to distribute on a clean Windows test machine.

## 7. Build locally from source (developer instructions)

In a PowerShell terminal with Python 3.11 available:

```powershell
git clone https://github.com/is-leeroy-jenkins/foo.git
cd foo
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean foo.spec
```

With **Inno Setup 6** installed at its usual location:

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "desktop\foo.iss"
```

The selected setup script writes its finished installer to:

```text
dist\installer\Foo-Setup-0.1.0.exe
```

These commands describe the intended build. A native dependency may still fail to install or freeze, and Playwright/other external resource provisioning requires additional validation. Do not publish the installer solely because PyInstaller or Inno Setup exits successfully.

## 8. Troubleshooting

### Installer artifact is missing

Check that the workflow run completed successfully and the **Upload installers** step found a Setup executable. Confirm you are looking at the appropriate workflow's expected output directory. Check artifact retention if the build is old.

### Inno Setup command not found

Install Inno Setup 6 and confirm the compiler executable is present. The workflow using `desktop/foo.iss` currently assumes its compiler is available; the alternative workflow installs it using Chocolatey.

### Windows blocks the executable

Confirm the downloaded file's origin and checksum/provenance where available. Windows may warn about unsigned executables. Do not ignore warnings for unknown binaries. Code signing is not configured in the repository's installer scripts.

### Foo window is blank, or the server exits during startup

Inspect launcher or operating-system error details, confirm WebView2 Runtime is installed, and verify the bundled application and its imports execute on the target Windows architecture. A successful wheel, PyInstaller, or Inno Setup build does not prove startup success.

### Web scraping fails

Confirm Playwright's compatible Chromium executable is available and that the runtime can locate it. Merely executing `python -m playwright install chromium` on the GitHub runner does not guarantee the browser files have been shipped to the user.

### NLTK, spaCy, or local-model functionality fails

Ensure the required corpora, models, and files are present and discoverable at runtime. Install/download only the resources actually required for the mode in use.

### Database access fails

Check that `%LOCALAPPDATA%\Foo` exists and is writable. Avoid storing writable databases under Program Files. Preserve existing user data while troubleshooting.

### Windows build runs out of time or disk space

Foo includes heavyweight numerical, geospatial, machine-learning, and browser dependencies. Inspect the workflow log for the exact failing package or step; optimize the build only after identifying the concrete failure.

## 9. Upgrade or uninstall

**Upgrade:** Obtain a verified installer built from the desired repository revision. Back up `%LOCALAPPDATA%\Foo`, close Foo, run the newer installer, then verify that the application starts and your data remains accessible.

**Uninstall:** Open **Windows Settings → Apps → Installed apps**, select **Foo**, and choose **Uninstall**. The installer should remove its application files and shortcuts; user-generated content under `%LOCALAPPDATA%\Foo` may remain and should be reviewed separately.

## 10. Validation checklist before publishing an installer

- [ ] GitHub Actions installer job passed completely.
- [ ] The resulting Setup executable is present in the uploaded artifact.
- [ ] Installer runs on a clean, supported Windows x64 environment.
- [ ] Foo starts in its WebView2 window without requiring a separate Python installation.
- [ ] Closing Foo stops its local Streamlit process.
- [ ] SQLite/logs/vector stores are writable and remain intact after restart.
- [ ] Loading, Scraping, Retrieval, Geospatial, Demographic, Environmental, Astronomical, and Generation workflows have been smoke-tested as applicable.
- [ ] Playwright, native modules, NLTK/spaCy resources, and external providers have been verified for their corresponding modes.
- [ ] Uninstall and reinstall behavior has been checked without accidental data loss.
- [ ] Executable provenance, security scanning, and signing policy have been reviewed.

---

Return to the [Foo README](https://github.com/is-leeroy-jenkins/foo/blob/main/README.md).
