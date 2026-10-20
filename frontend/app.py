"""
Payment Fraud Detection — Streamlit Front End
Features:
- Authentication and role-based navigation
- Modern card-based Home dashboard
- Analytics, transaction prediction, batch upload
- Transaction history and administrator monitoring

Run:
    streamlit run app.py

Optional:
    API_URL=http://localhost:8000
"""

import html
import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "PAYMENT",
    "DEBIT",
]

st.set_page_config(
    page_title="Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Custom styling
# -------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f8fd;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101b38 0%, #17264b 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f4f7ff;
    }

    /* Keep text typed into sidebar fields dark and readable.
       The sidebar's global white-text rule must not affect inputs. */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background-color: #ffffff !important;
        caret-color: #111827 !important;
    }

    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #6b7280 !important;
        -webkit-text-fill-color: #6b7280 !important;
        opacity: 1;
    }

    /* Main-page text fields should also remain readable. */
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea {
        color: #111827;
        caret-color: #111827;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
    }

    .brand {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #aebfe8;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(115deg, #e4efff 0%, #f2f6ff 62%, #d6e5ff 100%);
        border: 1px solid #d7e5ff;
        border-radius: 18px;
        padding: 27px 30px;
        margin-bottom: 22px;
    }

    .hero-title {
        color: #10234d;
        font-size: clamp(23px, 2.4vw, 31px);
        font-weight: 800;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #435779;
        font-size: 14px;
        line-height: 1.7;
    }

    .hero-note {
        border-left: 4px solid #4779ff;
        padding-left: 17px;
        color: #253d69;
        font-size: 15px;
        line-height: 1.8;
        margin-top: 8px;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 750;
        color: #14254a;
        margin: 12px 0 14px 0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e9f5;
        border-radius: 16px;
        padding: 19px 19px 17px 19px;
        min-height: 145px;
        box-shadow: 0 4px 15px rgba(29, 54, 99, 0.035);
    }

    .metric-label {
        font-size: 13px;
        color: #60708c;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: clamp(23px, 2vw, 30px);
        line-height: 1.2;
        font-weight: 800;
        color: #14254a;
        overflow-wrap: anywhere;
    }

    .metric-foot {
        color: #75839b;
        font-size: 12px;
        margin-top: 12px;
    }

    .metric-blue {
        background: linear-gradient(140deg, #ffffff, #eff5ff);
        border-color: #d1e0ff;
    }

    .metric-red {
        background: linear-gradient(140deg, #ffffff, #fff1f3);
        border-color: #ffd9df;
    }

    .metric-green {
        background: linear-gradient(140deg, #ffffff, #ecfbf4);
        border-color: #ceefdf;
    }

    .metric-purple {
        background: linear-gradient(140deg, #ffffff, #f4f0ff);
        border-color: #e0d5ff;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e3eaf5;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(29, 54, 99, 0.035);
    }

    .panel-title {
        color: #16284e;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 15px;
    }

    .action-card {
        background: #ffffff;
        border: 1px solid #e1e9f5;
        border-radius: 15px;
        padding: 17px;
        min-height: 165px;
        margin-bottom: 8px;
    }

    .action-icon {
        font-size: 25px;
        margin-bottom: 9px;
    }

    .action-title {
        color: #17294e;
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .action-desc {
        color: #657590;
        font-size: 12px;
        line-height: 1.6;
    }

    .blue-card {
        background: linear-gradient(140deg, #ffffff, #eff5ff);
        border-color: #d4e2ff;
    }

    .green-card {
        background: linear-gradient(140deg, #ffffff, #effbf5);
        border-color: #d0f2e1;
    }

    .purple-card {
        background: linear-gradient(140deg, #ffffff, #f5f1ff);
        border-color: #e5dcff;
    }

    .orange-card {
        background: linear-gradient(140deg, #ffffff, #fff6ec);
        border-color: #ffe4ca;
    }

    .red-card {
        background: linear-gradient(140deg, #ffffff, #fff1f3);
        border-color: #ffd9df;
    }

    .system-status {
        border: 1px solid #34486e;
        background: rgba(255,255,255,0.035);
        padding: 13px;
        border-radius: 12px;
        margin-top: 25px;
        font-size: 12px;
        color: #c3d0ea;
        line-height: 1.8;
    }

    .status-dot {
        color: #10cc82;
        font-size: 15px;
    }

    /* High-contrast buttons: readable before hover on every theme/device. */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
        padding: 0.55rem 1rem;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button p,
    div.stFormSubmitButton > button p,
    div.stButton > button span,
    div.stFormSubmitButton > button span {
        color: #ffffff !important;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #173ea8 !important;
        color: #ffffff !important;
        border-color: #173ea8 !important;
        box-shadow: 0 3px 10px rgba(29, 78, 216, 0.22);
    }

    div.stButton > button:focus,
    div.stFormSubmitButton > button:focus {
        outline: 3px solid rgba(59, 130, 246, 0.35);
        outline-offset: 2px;
    }

    /* Keep the app comfortable on phones and narrow tablet screens. */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        .hero {
            padding: 20px 18px;
            border-radius: 14px;
        }

        .hero-subtitle,
        .hero-note {
            overflow-wrap: anywhere;
        }

        .metric-card,
        .panel,
        .action-card {
            padding: 14px;
            min-width: 0;
        }

        /* Stack Streamlit columns instead of squeezing them on small screens. */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            width: 100%;
            min-height: 46px;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            max-width: 100%;
            overflow-x: auto;
        }
    }

    @media (min-width: 769px) and (max-width: 1024px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4eaf4;
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e3eaf5;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #e1e8f4;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------

def call(method, path, **kw):
    headers = (
        {"Authorization": "Bearer " + st.session_state["token"]}
        if st.session_state.get("token")
        else {}
    )

    try:
        response = requests.request(
            method,
            API + path,
            headers=headers,
            timeout=60,
            **kw,
        )
    except requests.RequestException:
        st.error(
            "Cannot reach the backend. Check that the API is running at "
            + API
        )
        return None

    if response.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear()
        st.warning("Your session has ended. Please log in again.")
        st.rerun()

    return response


def detail(response):
    try:
        return response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text


def go_to(page):
    """Callback: runs before the next rerun, so the sidebar radio (key 'navigation') can be set safely."""
    st.session_state["navigation"] = page
    st.session_state["current_page"] = page


def metric_card(icon, label, value, footer, style):
    st.markdown(
        f"""
        <div class="metric-card {style}">
            <div style="font-size:23px; margin-bottom:9px">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{footer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_card(icon, title, description, style):
    st.markdown(
        f"""
        <div class="action-card {style}">
            <div class="action-icon">{icon}</div>
            <div class="action-title">{title}</div>
            <div class="action-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_stats(days=14):
    response = call("GET", "/stats", params={"days": days})
    if response is not None and response.status_code == 200:
        return response.json()
    return None


# -------------------------------------------------------------------
# Sidebar: branding, authentication and navigation
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:35px">🛡️</div>
        <div class="brand">Fraud Detection</div>
        <div class="brand-subtitle">Payment Security Platform</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("token"):
        username = st.session_state["username"]
        role = st.session_state["role"]

        st.success(f"👤 {username} · {role.title()}")

        pages = [
            "Home",
            "Analytics dashboard",
            "Prediction",
            "Batch upload",
            "Transactions",
        ]

        if role == "admin":
            pages.append("Admin")

        if st.session_state.get("current_page") not in pages:
            st.session_state["current_page"] = "Home"

        selected_page = st.radio(
            "MAIN MENU",
            pages,
            key="navigation",
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = selected_page

        st.divider()

        if st.button("↪ Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown(
            f"""
            <div class="system-status">
                <span class="status-dot">●</span> System available<br>
                <span style="color:#aebfe8">Backend</span><br>
                {API}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown("### Account access")

        mode = st.radio(
            "Choose an option",
            ["Log in", "Register"],
        )

        with st.form("auth"):
            user = st.text_input("Username")
            email = (
                st.text_input("Email")
                if mode == "Register"
                else ""
            )
            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button(
                mode,
                use_container_width=True,
            )

        if submitted:
            if not user.strip() or not password:
                st.error("Enter your username and password.")
            elif mode == "Register" and not email.strip():
                st.error("Enter your email address.")
            elif mode == "Register":
                response = call(
                    "POST",
                    "/auth/register",
                    json={
                        "username": user.strip(),
                        "email": email.strip(),
                        "password": password,
                    },
                )

                if response is not None:
                    if response.status_code == 201:
                        st.success("Registration successful. Log in now.")
                    else:
                        st.error(detail(response))

            else:
                response = call(
                    "POST",
                    "/auth/login",
                    json={
                        "username": user.strip(),
                        "password": password,
                    },
                )

                if response is not None:
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.update(
                            token=data["access_token"],
                            role=data["role"],
                            username=user.strip(),
                            current_page="Home",
                        )
                        st.rerun()
                    else:
                        st.error(detail(response))


# -------------------------------------------------------------------
# Page routing
# -------------------------------------------------------------------

if not st.session_state.get("token"):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🛡️ Payment Fraud Detection</div>
            <div class="hero-subtitle">
                Intelligent transaction monitoring, machine learning
                risk scoring, and fraud analytics in one place.
                <br><br>
                Log in or create an account using the sidebar to continue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        action_card(
            "🔍",
            "Transaction scoring",
            "Assess individual transactions using your fraud detection model.",
            "blue-card",
        )

    with col2:
        action_card(
            "📊",
            "Fraud analytics",
            "Monitor transaction volumes, flagged activity, and trends.",
            "purple-card",
        )

    with col3:
        action_card(
            "🔐",
            "Secure access",
            "Authenticate users and protect administrator functions.",
            "green-card",
        )

    st.stop()


page = st.session_state.get("current_page", "Home")


# -------------------------------------------------------------------
# HOME: card dashboard
# -------------------------------------------------------------------

if page == "Home":

    username = html.escape(st.session_state["username"])
    role = st.session_state["role"]

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">👋 Welcome, {username}!</div>
            <div class="hero-subtitle">
                Here is an overview of your payment fraud detection system.
                Monitor suspicious activity, analyse transactions, and
                access your tools from one dashboard.
            </div>
            <div class="hero-note">
                <b>🛡️ Payment Security</b><br>
                Better insights. Safer transactions. A more secure tomorrow.
                <br><br>
                <span style="font-size:13px">
                    Account role: {role.title()}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">📈 System Overview</div>',
        unsafe_allow_html=True,
    )

    # Fetch actual statistics from the backend
    stats = get_stats(days=14)

    if stats is not None:
        c1, c2, c3, c4 = st.columns(4, gap="medium")

        with c1:
            metric_card(
                "🔄",
                "Transactions Scored",
                f"{stats.get('total', 0):,}",
                "Total recorded predictions",
                "metric-blue",
            )

        with c2:
            metric_card(
                "🚨",
                "Flagged as Fraud",
                f"{stats.get('flagged', 0):,}",
                "Transactions flagged by the model",
                "metric-red",
            )

        with c3:
            metric_card(
                "🟢",
                "Fraud Share",
                f"{100 * stats.get('fraud_share', 0):.2f}%",
                "Share of recorded predictions flagged",
                "metric-green",
            )

        with c4:
            metric_card(
                "🧠",
                "Active Model",
                str(stats.get("model_version", "N/A")),
                f"Decision threshold: {stats.get('threshold', 0):.4f}",
                "metric-purple",
            )

        st.write("")

        # Chart panels
        left, right = st.columns([1.55, 1], gap="medium")

        with left:
            st.markdown(
                '<div class="panel-title">📊 Transaction Trend</div>',
                unsafe_allow_html=True,
            )

            trend_days = st.selectbox(
                "Trend period",
                [7, 14, 30, 90],
                index=1,
                key="home_trend_days",
            )

            if trend_days == 14:
                trend = stats.get("trend", [])
            else:
                trend_stats = get_stats(days=trend_days)
                trend = (
                    trend_stats.get("trend", [])
                    if trend_stats is not None
                    else []
                )

            if trend:
                trend_df = pd.DataFrame(trend)

                if "date" in trend_df.columns:
                    trend_df["date"] = pd.to_datetime(
                        trend_df["date"],
                        errors="coerce",
                    )
                    trend_df = trend_df.dropna(subset=["date"])
                    trend_df = trend_df.set_index("date")

                numeric_cols = trend_df.select_dtypes(
                    include="number"
                ).columns

                if len(numeric_cols):
                    st.bar_chart(
                        trend_df[numeric_cols],
                        use_container_width=True,
                    )
                else:
                    st.info("The trend data has no numeric series.")
            else:
                st.info(
                    "No transaction trend data is available for this period yet."
                )

        with right:
            st.markdown(
                '<div class="panel-title">🛡️ Transaction Types</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Supported transaction categories in the detection system."
            )

            type_labels = {
                "TRANSFER": "🔵 TRANSFER",
                "CASH_OUT": "🟠 CASH OUT",
                "CASH_IN": "🟢 CASH IN",
                "PAYMENT": "🟣 PAYMENT",
                "DEBIT": "🟡 DEBIT",
            }

            for txn_type in TYPES:
                st.markdown(
                    f"""
                    <div style="
                        padding:12px 5px;
                        border-bottom:1px solid #edf1f7;
                        color:#344663;
                        font-size:14px;
                    ">
                        {type_labels[txn_type]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "This list shows supported types, not a measured breakdown "
                "of transaction volumes."
            )

    else:
        st.warning(
            "The dashboard could not load statistics from the backend. "
            "Check that the API is running and that the /stats endpoint "
            "is available."
        )

    # Quick action cards
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">⚡ Quick Actions</div>',
        unsafe_allow_html=True,
    )
    st.caption("Choose a feature to get started.")

    action_columns = st.columns(4, gap="medium")

    actions = [
        (
            "🎯",
            "Prediction",
            "Score a single transaction in real time.",
            "blue-card",
            "Prediction",
        ),
        (
            "📤",
            "Batch Upload",
            "Score multiple transactions from a CSV file.",
            "green-card",
            "Batch upload",
        ),
        (
            "📊",
            "Analytics",
            "Explore trends, fraud share, and key insights.",
            "purple-card",
            "Analytics dashboard",
        ),
        (
            "🧾",
            "Transactions",
            "Review stored transaction predictions.",
            "orange-card",
            "Transactions",
        ),
    ]

    for column, action in zip(action_columns, actions):
        icon, title, description, style, destination = action

        with column:
            action_card(
                icon,
                title,
                description,
                style,
            )

            st.button(
                f"Open {title} →",
                key=f"home_action_{destination}",
                use_container_width=True,
                on_click=go_to,
                args=(destination,),
            )

    if role == "admin":
        st.write("")
        admin_col, _ = st.columns([1, 3])

        with admin_col:
            action_card(
                "🔐",
                "Administration",
                "Manage users and inspect audit logs.",
                "red-card",
            )

            st.button(
                "Open Admin →",
                key="home_action_admin",
                use_container_width=True,
                on_click=go_to,
                args=("Admin",),
            )


# -------------------------------------------------------------------
# ANALYTICS DASHBOARD
# -------------------------------------------------------------------

elif page == "Analytics dashboard":

    st.title("📊 Analytics Dashboard")

    days = st.selectbox(
        "Trend period (days)",
        [7, 14, 30, 90],
        index=1,
    )

    response = call(
        "GET",
        "/stats",
        params={"days": days},
    )

    if response is not None and response.status_code == 200:
        stats = response.json()

        a, b, c = st.columns(3)

        a.metric(
            "Transactions Scored",
            f"{stats['total']:,}",
        )
        b.metric(
            "Flagged as Fraud",
            f"{stats['flagged']:,}",
        )
        c.metric(
            "Fraud Share",
            f"{100 * stats['fraud_share']:.2f}%",
        )

        st.caption(
            f"Active model: {stats['model_version']} | "
            f"Decision threshold: {stats['threshold']:.4f}"
        )

        if stats.get("trend"):
            df = pd.DataFrame(stats["trend"])

            if "date" in df.columns:
                df["date"] = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                )
                df = df.dropna(subset=["date"]).set_index("date")

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            if len(numeric_cols):
                st.line_chart(df[numeric_cols])
            else:
                st.info("No numeric trend data is available.")
        else:
            st.info("No transactions in this period yet.")

    elif response is not None:
        st.error(detail(response))


# -------------------------------------------------------------------
# SINGLE TRANSACTION PREDICTION
# -------------------------------------------------------------------

elif page == "Prediction":

    st.title("🎯 Score a Transaction")
    st.caption("Submit transaction details for a fraud risk assessment.")

    with st.form("predict"):
        c1, c2 = st.columns(2)

        with c1:
            txn_id = st.text_input("Transaction ID (optional)")
            typ = st.selectbox("Transaction Type", TYPES)
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
            )
            hour = st.slider("Hour of Day", 0, 23, 12)

        with c2:
            old_org = st.number_input(
                "Sender Balance Before",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
            )
            old_dest = st.number_input(
                "Recipient Balance Before",
                min_value=0.0,
                value=0.0,
                step=1000.0,
            )
            name_dest = st.text_input(
                "Recipient Account Name",
                value="C1234567890",
            )

        submitted = st.form_submit_button(
            "🔍 Score Transaction",
            use_container_width=True,
        )

    if submitted:
        body = {
            "type": typ,
            "amount": amount,
            "hour": hour,
            "old_balance_org": old_org,
            "old_balance_dest": old_dest,
            "name_dest": name_dest,
        }

        if txn_id.strip():
            body["txn_id"] = txn_id.strip()

        response = call(
            "POST",
            "/predict",
            json=body,
        )

        if response is not None and response.status_code == 200:
            data = response.json()

            if data["is_fraud"]:
                st.error("🚨 FRAUD SUSPECTED")
            else:
                st.success("✅ Looks legitimate")

            a, b, c = st.columns(3)

            a.metric("Fraud Score", f"{data['score']:.4f}")
            b.metric("Threshold", f"{data['threshold']:.4f}")
            c.metric("Model", data["model_version"])

            st.markdown("#### Assessment explanation")
            st.write(
                data.get("reason")
                or "No feature pushes the score towards fraud."
            )

        elif response is not None:
            st.error(detail(response))


# -------------------------------------------------------------------
# BATCH UPLOAD
# -------------------------------------------------------------------

elif page == "Batch upload":

    st.title("📤 Batch Transaction Scoring")

    st.write(
        "Upload a CSV file containing the transaction columns below:"
    )

    st.code(
        "type, amount, hour, old_balance_org, "
        "old_balance_dest, name_dest, txn_id"
    )

    st.caption("The txn_id column is optional.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: {uploaded_file.name}")

        if st.button("🚀 Score File", use_container_width=True):
            response = call(
                "POST",
                "/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv",
                    )
                },
            )

            if response is not None and response.status_code == 200:
                data = response.json()

                a, b, c = st.columns(3)
                a.metric("Scored", data["scored"])
                b.metric("Flagged", data["flagged"])
                c.metric("Rejected", len(data["rejected"]))

                if data["results"]:
                    st.markdown("#### Scoring Results")
                    st.dataframe(
                        pd.DataFrame(data["results"]),
                        use_container_width=True,
                    )

                if data["rejected"]:
                    st.warning("Some rows were rejected.")
                    st.dataframe(
                        pd.DataFrame(data["rejected"]),
                        use_container_width=True,
                    )

            elif response is not None:
                st.error(detail(response))


# -------------------------------------------------------------------
# TRANSACTIONS
# -------------------------------------------------------------------

elif page == "Transactions":

    st.title("🧾 Transaction Records")

    c1, c2, c3 = st.columns(3)

    with c1:
        txn_type = st.selectbox(
            "Transaction Type",
            ["All"] + TYPES,
        )

    with c2:
        flag = st.selectbox(
            "Status",
            ["All", "Fraud", "Legitimate"],
        )

    with c3:
        page_no = st.number_input(
            "Page",
            min_value=1,
            value=1,
        )

    d1, d2 = st.columns(2)

    with d1:
        date_from = st.text_input(
            "From (YYYY-MM-DD)",
        )

    with d2:
        date_to = st.text_input(
            "To (YYYY-MM-DD)",
        )

    params = {"page": int(page_no)}

    if txn_type != "All":
        params["type"] = txn_type

    if flag != "All":
        params["is_fraud"] = str(flag == "Fraud").lower()

    if date_from:
        params["from"] = date_from

    if date_to:
        params["to"] = date_to

    response = call(
        "GET",
        "/transactions",
        params=params,
    )

    if response is not None and response.status_code == 200:
        data = response.json()

        st.caption(f"{data['total']:,} matching records")

        if data["items"]:
            columns = [
                "created_at",
                "txn_id",
                "type",
                "amount",
                "score",
                "is_fraud",
                "reason",
            ]

            df = pd.DataFrame(data["items"])
            available_columns = [
                column for column in columns
                if column in df.columns
            ]

            st.dataframe(
                df[available_columns],
                use_container_width=True,
            )
        else:
            st.info("No transactions match these filters.")

    elif response is not None:
        st.error(detail(response))


# -------------------------------------------------------------------
# ADMINISTRATION
# -------------------------------------------------------------------

elif page == "Admin":

    if st.session_state.get("role") != "admin":
        st.error("You do not have permission to access this page.")
        st.stop()

    st.title("🔐 Administration")
    st.caption("Manage platform users and review activity logs.")

    users_tab, logs_tab = st.tabs(
        ["👥 Users", "📋 Audit Log"]
    )

    with users_tab:
        response = call("GET", "/admin/users")

        if response is not None and response.status_code == 200:
            users = response.json()
            st.dataframe(
                pd.DataFrame(users),
                use_container_width=True,
            )
        elif response is not None:
            st.error(detail(response))

    with logs_tab:
        response = call("GET", "/admin/logs")

        if response is not None and response.status_code == 200:
            logs = response.json()
            st.dataframe(
                pd.DataFrame(logs),
                use_container_width=True,
            )
        elif response is not None:
            st.error(detail(response))
