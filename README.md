# Payment fraud detection prototype

backend/   FastAPI + MongoDB + JWT/bcrypt, scores with serving.py (decision = score >= stored threshold)
frontend/  Streamlit pages: login/register, home, analytics dashboard, prediction, batch upload, transactions, admin
models/    put fraud_lr_v1.joblib here (made by fraud_lr_pipeline.py, in results/)
tests/     test_api.py (T1-T6, T9, T13, T16), bench_latency.py (T10 / NFR-02), test_unit.py, summarise_uat.py

## 1. Install (Python 3.10+; use a laptop, not Colab)
    python -m venv venv && venv\Scripts\activate        (Windows)   |   source venv/bin/activate (Mac/Linux)
    pip install -r backend/requirements.txt -r frontend/requirements.txt pytest requests

## 2. Settings: copy backend/.env.example to backend/.env and fill it in (or set environment variables)
    MONGO_URI        your MongoDB Atlas connection string  (mongodb+srv://user:pass@cluster/...)
    JWT_SECRET       a long random string
    ADMIN_USERNAME / ADMIN_PASSWORD   creates the first administrator at start-up
    ARTEFACT         path to the model file (default ../models/fraud_lr_v1.joblib)
Windows (PowerShell):  $env:MONGO_URI="..."; $env:JWT_SECRET="..."; $env:ADMIN_USERNAME="admin"; $env:ADMIN_PASSWORD="..."
Mac/Linux:             export MONGO_URI="..." JWT_SECRET="..." ADMIN_USERNAME=admin ADMIN_PASSWORD=...
No Atlas yet? For a quick local trial only:  pip install mongomock  and set USE_MOCK_DB=1 (data is lost on restart).

## 3. Run (two terminals)
    cd backend  && uvicorn main:app --port 8000
    cd frontend && streamlit run app.py          -> opens http://localhost:8501

## 4. Test (backend running)
    cd tests
    set API_URL=http://localhost:8000    (Mac/Linux: export)  then:
    pytest -v test_api.py ; python bench_latency.py --n 1000
    ARTEFACT=../models/fraud_lr_v1.joblib PAYSIM_CSV=<csv> pytest -v test_unit.py   (serving.py and fraud_lr_pipeline.py are already in tests/)

Roles: "user" (own transactions only) and "admin" (all transactions, users, audit log).
