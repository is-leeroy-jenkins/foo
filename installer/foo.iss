#define MyAppName "Foo"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{A6A93E40-BF2A-46CF-94D6-D648542258E4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=is-leeroy-jenkins
DefaultDirName={autopf}\Foo
DefaultGroupName=Foo
OutputDir=..\dist-installer
OutputBaseFilename=Foo-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayName=Foo
CloseApplications=yes

[Files]
Source: "..\dist\Foo\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Icons]
Name: "{group}\Foo"; Filename: "{app}\Foo.exe"
Name: "{autodesktop}\Foo"; Filename: "{app}\Foo.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Foo.exe"; Description: "Launch Foo"; Flags: nowait postinstall skipifsilent
