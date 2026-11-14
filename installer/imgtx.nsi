; Bộ cài ImgTX – tạo bằng NSIS (makensis imgtx.nsi)
Target amd64-unicode
!include "MUI2.nsh"

!define APPNAME "ImgTX"
!define APPFULL "ImgTX – Khảo sát biến đổi cường độ"
!define VERSION "2.2.1"
!define SRCDIR  "..\dist\ImgTX"

Name "${APPFULL}"
OutFile "..\dist\ImgTX_Setup_${VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
RequestExecutionLevel user          ; không cần quyền Administrator
SetCompressor /SOLID lzma
BrandingText "${APPNAME} ${VERSION}"

VIProductVersion "2.2.1.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "FileDescription" "Cài đặt ${APPFULL}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "© 2026 ImgTX"

!define MUI_ICON "..\assets\imgtx.ico"
!define MUI_UNICON "..\assets\imgtx.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\ImgTX.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Mở ImgTX ngay"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Vietnamese"

Section "ImgTX" SecMain
  SetOutPath "$INSTDIR"
  ; xoá bản cũ (nếu có) để không lẫn thư viện cũ
  RMDir /r "$INSTDIR\_internal"
  File /r "${SRCDIR}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\ImgTX.exe" "" "$INSTDIR\ImgTX.exe" 0
  CreateShortcut "$SMPROGRAMS\${APPNAME}\Gỡ cài đặt ${APPNAME}.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\ImgTX.exe" "" "$INSTDIR\ImgTX.exe" 0
  WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
  ; mục trong Settings → Apps để gỡ cài đặt
  !define UNKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  WriteRegStr HKCU "${UNKEY}" "DisplayName" "${APPFULL}"
  WriteRegStr HKCU "${UNKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNKEY}" "DisplayIcon" "$INSTDIR\ImgTX.exe"
  WriteRegStr HKCU "${UNKEY}" "Publisher" "ImgTX"
  WriteRegStr HKCU "${UNKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNKEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APPNAME}"
  RMDir /r "$INSTDIR\_internal"
  Delete "$INSTDIR\ImgTX.exe"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  DeleteRegKey HKCU "Software\${APPNAME}"
SectionEnd
