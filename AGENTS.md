# Webtoon evirici — Agent Guidelines

## Available tooling

| Purpose | Available | Notes |
|---|---|---|
| Repo graph | Graphify MCP | `graphify-out/graph.json`; local stdio; architecture/dependency graph |
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
8. Yesil olmadan merge yok: `pytest -q` + `audit --strict` + `compare_golden` temiz olmadan birlestirme yapma.

## DuckDuckGo Search & Context Hygiene Protocol

9. Local-first: codebase ici mantik, standart syntax, mevcut type/LSP tanimlari icin web search YOK. Once local type, proje dosyasi, LSP definition kontrol et. Web search SADECE: localde cozulemeyen obscure runtime/stack trace, third-party breaking change/API migration dogrulama, localde olmayan dis bagimlilik resmi dokumani icin.
10. Queryler keyword-dense olacak, conversational CUMLE YOK (orn. "FastAPI CORSMiddleware allow_origins" OK, "how to resolve CORS error in FastAPI" YASAK). Exact hata/function signature tirnak icinde: `"exact error trace"`. Mumkunse domain filtresi: `site:github.com/issues`, `site:docs.*` / `site:nextjs.org/docs`.
11. Rate-limit/anti-loop: problem basina MAX 2 query. Ilki basarisizsa bir kez refine et, ikincisi de basarisizsa DUR. Hizli ardisik/loop search YASAK. Local adim-adim debug + minimal repro'ya don.
12. Token economy: once snippet degerlendir, her linki fetch ETME. Sadece verified snippet/patch iceren tek en alakali URL'yi fetch et. Minimal cozumu ozetle; raw web dump'u context'e GOMME.
