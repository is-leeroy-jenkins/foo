#define AppName "Foo"
#define AppVersion "0.1.0"
#define AppPublisher "Foo"
#define AppExeName "Foo.exe"

[Setup]
AppId={{7D89B6AF-76F6-443A-95C9-CCB54A0E53A2}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Foo
DefaultGroupName=Foo
OutputDir=..\dist\installer
OutputBaseFilename=Foo-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern
UninstallDisplayIcon={app}\Foo.exe

[Files]
Source: "..\dist\Foo\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Foo"; Filename: "{app}\Foo.exe"
Name: "{autodesktop}\Foo"; Filename: "{app}\Foo.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\Foo.exe"; Description: "Launch Foo"; Flags: nowait postinstall skipifsilent
