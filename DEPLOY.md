Free hosting: Atlas M0 (database) + Render free web service (backend) + Streamlit Community Cloud (frontend).
See the chat guide. Backend start command: uvicorn main:app --host 0.0.0.0 --port $PORT (root directory: backend).
Frontend: main file frontend/app.py, secret  API_URL = "https://<your-render-service>.onrender.com".
Pinned: scikit-learn==1.6.1, imbalanced-learn==0.13.0 (must match the model file).
