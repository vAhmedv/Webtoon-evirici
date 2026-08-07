# Webtoon evirici — Agent Guidelines

## Available tooling

| Purpose | Available | Notes |
|---|---|---|
| Repo graph | Graphify MCP | `.kilo/kilo.json`; local stdio; architecture/dependency graph |
| Symbol/reference | Serena MCP | `uvx serena-agent`; `find_symbol`, `find_referencing_symbols`, `get_symbols_overview` |
| Library docs | `context7-mcp` skill | PySide6, ultralytics, OCR vb. guncel dokumanlar |
| GitHub | bash + git | PR/issue/clone erisimi hazir |
| Execution | `.venv\Scripts\python.exe` | Proje ortamini bozmadan calistir |

## Rules

1. Architecture/dependency sorularinda once Graphify kullan (`query_graph`, `get_node`, `get_neighbors`).
2. Symbol/reference/refactor icin Serena kullan (`find_symbol`, `find_referencing_symbols`, `replace_symbol_body` vb.).
3. Third-party API docs icin Context7 kullan (skill ile yukle, sonra arastir).
4. Gereksiz repo taramasi yapma; symbol/ref tool'larini oncelikle kullan.
5. Gorev disi kod degistirme yapma. Kucuk, tek hedefli degisiklikler yap.
6. Dependency/env bozma. Yeni paket kurmadan once `.venv` etkileyecek mi kontrol et.
7. Model weights, cache, output dosyalarini asla commit etme.
