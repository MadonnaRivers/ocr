# ocr

Document parser: OCR + PDF text-layer verification + cell-aware tables + layout cleanup.
Web UI and REST API on one port.

```
scripts\install.ps1
.venv\Scripts\python.exe app.py      # http://127.0.0.1:7862  (API docs: /docs)
curl -F "file=@doc.pdf" "http://127.0.0.1:7862/api/parse?format=markdown"
```
