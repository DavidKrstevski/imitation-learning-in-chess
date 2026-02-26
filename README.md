# Chess App (Svelte + FastAPI)

## Struktur
- `backend/` Python API + Modell
- `frontend/` Svelte UI

## Backend starten
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Optional (HF Token):
```bash
# PowerShell
$env:HF_TOKEN="hf_..."
```

## Frontend starten
```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`
Backend: `http://127.0.0.1:8000`
