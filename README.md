# Chess App (Svelte + FastAPI)

## Struktur
- `backend/` Python API + Modell
- `frontend/` Svelte UI
- `notebooks/` getrennte Experiment-Notebooks und deren Helper-Code

Die App (`backend/`, `frontend/`) und die Notebook-Experimente sind voneinander getrennt. Die beiden Hauptnotebooks sind:
- `notebooks/test_result.ipynb`
- `notebooks/test_result_count.ipynb`

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

## Spielertraining
Im Frontend kann links ein Lichess-Username eingegeben werden. Das Backend startet dann ein LoRA-Training mit bis zu 500 Partien, speichert App-Modelle getrennt von den Notebook-Experimenten direkt unter `app_models/<spieler>/` und stellt den trainierten Spieler anschliessend in der Spielerliste bereit.
