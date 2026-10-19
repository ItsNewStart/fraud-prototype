"""Streamlit front end (Section 3.12, Table 3.11): authentication, home, analytics dashboard,
prediction, batch upload, transactions and (for administrators) activity monitoring.
Run:  streamlit run app.py        API_URL defaults to http://localhost:8000
"""
import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TYPES = ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
st.set_page_config(page_title="Payment Fraud Detection", page_icon="🛡️", layout="wide")


def call(method, path, **kw):
    h = {"Authorization": "Bearer " + st.session_state["token"]} if st.session_state.get("token") else {}
    try:
        r = requests.request(method, API + path, headers=h, timeout=60, **kw)
    except requests.RequestException:
        st.error("Cannot reach the server. Check that the backend is running at " + API)
        return None
    if r.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear(); st.warning("Your session has ended. Please log in again."); return None
    return r


def detail(r):
    try: return r.json().get("detail", r.text)
    except Exception: return r.text


# ---------------------------------------------------------------- sidebar: authentication
st.sidebar.title("🛡️ Fraud Detection")
if st.session_state.get("token"):
    st.sidebar.success(f"Logged in as {st.session_state['username']} ({st.session_state['role']})")
    pages = ["Home", "Analytics dashboard", "Prediction", "Batch upload", "Transactions"]
    if st.session_state["role"] == "admin":
        pages.append("Admin")
    page = st.sidebar.radio("Navigate", pages)
    if st.sidebar.button("Log out"):
        st.session_state.clear(); st.rerun()
else:
    page = "Authentication"
    mode = st.sidebar.radio("Account", ["Log in", "Register"])
    with st.sidebar.form("auth"):
        user = st.text_input("Username")
        email = st.text_input("Email") if mode == "Register" else ""
        pw = st.text_input("Password", type="password")
        go = st.form_submit_button(mode)
    if go:
        if mode == "Register":
            r = call("POST", "/auth/register", json={"username": user, "email": email, "password": pw})
            if r is not None:
                st.sidebar.success("Registered. You can now log in.") if r.status_code == 201 else st.sidebar.error(detail(r))
        else:
            r = call("POST", "/auth/login", json={"username": user, "password": pw})
            if r is not None and r.status_code == 200:
                d = r.json(); st.session_state.update(token=d["access_token"], role=d["role"], username=user); st.rerun()
            elif r is not None:
                st.sidebar.error(detail(r))

# ---------------------------------------------------------------- pages
if page == "Authentication":
    st.title("Payment Fraud Detection Prototype")
    st.write("Log in or register from the sidebar to score transactions.")

elif page == "Home":
    st.title("Welcome, " + st.session_state["username"])
    st.write("Use the pages in the sidebar:")
    st.markdown("- **Analytics dashboard**: counts, fraud share and trend\n- **Prediction**: score one transaction\n"
                "- **Batch upload**: score a CSV file\n- **Transactions**: review stored results" +
                ("\n- **Admin**: users and audit log" if st.session_state["role"] == "admin" else ""))

elif page == "Analytics dashboard":
    st.title("Analytics dashboard")
    days = st.selectbox("Trend period (days)", [7, 14, 30, 90], index=1)
    r = call("GET", "/stats", params={"days": days})
    if r is not None and r.status_code == 200:
        s = r.json()
        a, b, c = st.columns(3)
        a.metric("Transactions scored", s["total"]); b.metric("Flagged as fraud", s["flagged"]); c.metric("Fraud share", f"{100 * s['fraud_share']:.2f}%")
        st.caption(f"Active model {s['model_version']}, decision threshold {s['threshold']:.4f}")
        if s["trend"]:
            df = pd.DataFrame(s["trend"]).set_index("date"); st.line_chart(df)
        else:
            st.info("No transactions in this period yet.")

elif page == "Prediction":
    st.title("Score a transaction")
    with st.form("predict"):
        c1, c2 = st.columns(2)
        txn_id = c1.text_input("Transaction ID (optional)")
        typ = c1.selectbox("Type", TYPES)
        amount = c1.number_input("Amount", min_value=0.0, value=100000.0, step=1000.0)
        hour = c1.slider("Hour of day", 0, 23, 12)
        old_org = c2.number_input("Sender balance before", min_value=0.0, value=100000.0, step=1000.0)
        old_dest = c2.number_input("Recipient balance before", min_value=0.0, value=0.0, step=1000.0)
        name_dest = c2.text_input("Recipient account name", value="C1234567890")
        go = st.form_submit_button("Score")
    if go:
        body = dict(type=typ, amount=amount, hour=hour, old_balance_org=old_org, old_balance_dest=old_dest, name_dest=name_dest)
        if txn_id: body["txn_id"] = txn_id
        r = call("POST", "/predict", json=body)
        if r is not None and r.status_code == 200:
            d = r.json()
            (st.error if d["is_fraud"] else st.success)("FRAUD SUSPECTED" if d["is_fraud"] else "Looks legitimate")
            a, b, c = st.columns(3)
            a.metric("Fraud score", f"{d['score']:.4f}"); b.metric("Threshold", f"{d['threshold']:.4f}"); c.metric("Model", d["model_version"])
            st.write("**Reason:** " + (d["reason"] or "no feature pushes the score towards fraud"))
        elif r is not None:
            st.error(detail(r))

elif page == "Batch upload":
    st.title("Batch scoring")
    st.write("CSV columns: `type, amount, hour, old_balance_org, old_balance_dest, name_dest` and optionally `txn_id`.")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None and st.button("Score file"):
        r = call("POST", "/upload", files={"file": (f.name, f.getvalue(), "text/csv")})
        if r is not None and r.status_code == 200:
            d = r.json(); st.success(f"{d['scored']} scored, {d['flagged']} flagged, {len(d['rejected'])} rejected")
            if d["results"]:
                st.dataframe(pd.DataFrame(d["results"]))
            if d["rejected"]:
                st.warning("Rejected rows"); st.dataframe(pd.DataFrame(d["rejected"]))
        elif r is not None:
            st.error(detail(r))

elif page == "Transactions":
    st.title("Transactions")
    c1, c2, c3 = st.columns(3)
    typ = c1.selectbox("Type", ["All"] + TYPES)
    flag = c2.selectbox("Status", ["All", "Fraud", "Legitimate"])
    page_no = c3.number_input("Page", min_value=1, value=1)
    d1, d2 = st.columns(2)
    dfrom, dto = d1.text_input("From (YYYY-MM-DD)"), d2.text_input("To (YYYY-MM-DD)")
    p = {"page": int(page_no)}
    if typ != "All": p["type"] = typ
    if flag != "All": p["is_fraud"] = str(flag == "Fraud").lower()
    if dfrom: p["from"] = dfrom
    if dto: p["to"] = dto
    r = call("GET", "/transactions", params=p)
    if r is not None and r.status_code == 200:
        d = r.json(); st.caption(f"{d['total']} matching records")
        if d["items"]:
            df = pd.DataFrame(d["items"])[["created_at", "txn_id", "type", "amount", "score", "is_fraud", "reason"]]
            st.dataframe(df.style.apply(lambda row: ["background-color: #ffd6d6" if row["is_fraud"] else "" for _ in row], axis=1))
    elif r is not None:
        st.error(detail(r))

elif page == "Admin":
    st.title("Administration")
    t1, t2 = st.tabs(["Users", "Audit log"])
    with t1:
        r = call("GET", "/admin/users")
        if r is not None and r.status_code == 200: st.dataframe(pd.DataFrame(r.json()))
    with t2:
        r = call("GET", "/admin/logs")
        if r is not None and r.status_code == 200: st.dataframe(pd.DataFrame(r.json()))
