# Webtoon evirici — Agent Guidelines

## Available tooling

| Purpose | Available | Notes |
|---|---|---|
| Library docs | context7-mcp skill | PySide6, ultralytics, OCR vb. guncel dokumanlar |
| Repo graph | **None** | Graphify/Serena client-level MCP config gerekir, proje seviyesinde yok |
| Symbol/reference | grep + ead + glob | Dosya/pattern bazli arama yeterli |
| GitHub | ash + git | PR/issue/clone erisimi hazir |
| Execution | .venv\Scripts\python.exe | Proje ortamini bozmadan calistir |

## Rules

1. Architecture sorularinda once grep + ead ile repo yapisini tara. Tum repo'yu basstan okuma.
2. Symbol/reference icin Serena yok; grep + glob + ead kullan.
3. Third-party API icin Context7 kullan (skill ile yukle, sonra arastir).
4. Gorev disi kod degistirme yapma. Kucuk, tek hedefli degisiklikler yap.
5. Dependency/env bozma. Yeni paket kurmadan once .venv etkileyecek mi kontrol et.
6. Model weights, cache, output dosyalarini asla commit etme.
