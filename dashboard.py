
# dashboard.py
import os
import json
import sqlite3
from datetime import datetime, date
import tempfile
import io  # << in-memory PDF buffer

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import scipy.stats as stats

# --- ReportLab for text‑rich PDF (chart + narrative on the same page) ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT

# Subplots to combine A/B summary (table + stats + text row + bar)
from plotly.subplots import make_subplots

# Use your existing helper for z‑test
from ab_test import proportions_z_test

DB_FILE = "vizsprints.db"
AUTH_FILE = "auth.json"  # local credential store

st.set_page_config(page_title="VizSprints Dashboard", layout="wide")

# -----------------------------
# DARK MODE STYLING (keep your CSS here if needed)
# -----------------------------
st.markdown("""
""", unsafe_allow_html=True)

# ============================================================
# AUTH LAYER: Login / Sign up / Forgot / Change / Add user / Logout
# (kept EXACTLY as in your file)
# ============================================================
def _hash_pw(plain: str) -> str:
    """Hash password using bcrypt if available; otherwise SHA-256."""
    try:
        import bcrypt
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
    except Exception:
        import hashlib
        return hashlib.sha256(plain.encode()).hexdigest()

def _verify_pw(plain: str, hashed: str) -> bool:
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        import hashlib
        return hashlib.sha256(plain.encode()).hexdigest() == hashed

def _username_key(u: str) -> str:
    """Normalize usernames to lower-case (login is case-insensitive)."""
    return (u or "").strip().lower()

def _load_users() -> dict:
    """Load users from auth.json; bootstrap an admin user if file is missing."""
    if not os.path.exists(AUTH_FILE):
        default_pwd = os.environ.get("DASH_PWD_PREETI", "preeti@123")
        users = {
            "preeti": {
                "role": "admin",
                "email": "preeti@example.com",
                "password": _hash_pw(default_pwd)
            }
        }
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return users
    with open(AUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_users(users: dict):
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

# Session bootstrap
if "auth" not in st.session_state:
    st.session_state["auth"] = {"is_auth": False, "user": None, "role": None}
if "reset_codes" not in st.session_state:
    st.session_state["reset_codes"] = {}  # {username: code}
if "date_start" not in st.session_state:
    st.session_state["date_start"] = None
if "date_end" not in st.session_state:
    st.session_state["date_end"] = None

USERS_DB = _load_users()

def render_auth_ui():
    """Render Login / Sign up / Forgot password when not authenticated."""
    tabs = st.tabs(["Login", "Sign up", "Forgot password"])

    # Login
    with tabs[0]:
        st.markdown("### 🔐 Login")
        u_in = st.text_input("Username", key="login_username")
        p_in = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in", key="btn_sign_in"):
            u = _username_key(u_in)
            if u in USERS_DB and _verify_pw(p_in, USERS_DB[u]["password"]):
                role = USERS_DB[u].get("role", "viewer")
                st.session_state["auth"] = {"is_auth": True, "user": u, "role": role}
                st.success("Logged in ✅")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Sign up
    with tabs[1]:
        st.markdown("### 📝 Create an account")
        su_user = st.text_input("Choose a username", key="signup_username")
        su_email = st.text_input("Email (optional)", key="signup_email")
        su_pwd1 = st.text_input("Create password", type="password", key="signup_pwd1")
        su_pwd2 = st.text_input("Confirm password", type="password", key="signup_pwd2")
        if st.button("Sign up", key="btn_sign_up"):
            u = _username_key(su_user)
            if not u:
                st.error("Username cannot be empty.")
            elif u in USERS_DB:
                st.error("Username already exists. Please choose another.")
            elif su_pwd1 != su_pwd2:
                st.error("Passwords do not match.")
            elif len(su_pwd1) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                USERS_DB[u] = {"role": "viewer", "email": su_email or "", "password": _hash_pw(su_pwd1)}
                _save_users(USERS_DB)
                st.success("Account created ✅. You can login now.")
                st.rerun()

    # Forgot password
    with tabs[2]:
        st.markdown("### 🧩 Reset your password")
        fp_user = st.text_input("Enter your username", key="forgot_username")
        if st.button("Generate reset code", key="btn_generate_code"):
            u = _username_key(fp_user)
            if u in USERS_DB:
                import secrets
                code = secrets.token_hex(4)  # 8 hex chars
                st.session_state["reset_codes"][u] = code
                st.info(f"Your reset code (save this): **{code}**")
            else:
                st.error("Username not found.")
        fp_code = st.text_input("Enter reset code", key="forgot_code")
        fp_new  = st.text_input("New password", type="password", key="forgot_newpwd")
        if st.button("Reset password", key="btn_reset_pwd"):
            u = _username_key(fp_user)
            code = st.session_state["reset_codes"].get(u)
            if not u or u not in USERS_DB:
                st.error("Invalid username.")
            elif not code:
                st.error("No reset code generated. Click 'Generate reset code' first.")
            elif fp_code.strip() != code:
                st.error("Incorrect reset code.")
            elif len(fp_new) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                USERS_DB[u]["password"] = _hash_pw(fp_new)
                _save_users(USERS_DB)
                st.session_state["reset_codes"].pop(u, None)
                st.success("Password reset ✅. Please login.")
                st.rerun()

def render_account_sidebar():
    """Sidebar account section shown after login. Admins also get 'Add user'."""
    st.sidebar.success(f"Logged in as **{st.session_state['auth']['user']}** ({st.session_state['auth']['role']})")

    # Change password
    with st.sidebar.expander("🔄 Change password"):
        cu_user = st.session_state["auth"]["user"]
        old = st.text_input("Current password", type="password", key="change_old")
        new1 = st.text_input("New password", type="password", key="change_new1")
        new2 = st.text_input("Confirm new password", type="password", key="change_new2")
        if st.button("Update password", key="btn_update_pwd"):
            if not _verify_pw(old, USERS_DB[cu_user]["password"]):
                st.error("Current password is incorrect.")
            elif new1 != new2:
                st.error("New passwords do not match.")
            elif len(new1) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                USERS_DB[cu_user]["password"] = _hash_pw(new1)
                _save_users(USERS_DB)
                st.success("Password updated ✅")

    # Admin add user
    if st.session_state["auth"]["role"] == "admin":
        with st.sidebar.expander("👤 Add user (admin)"):
            au_user = st.text_input("Username", key="admin_add_user")
            au_email = st.text_input("Email (optional)", key="admin_add_email")
            au_role = st.selectbox("Role", ["viewer", "admin"], key="admin_add_role")
            au_pwd  = st.text_input("Password", type="password", key="admin_add_pwd")
            if st.button("Create user", key="btn_admin_create"):
                u = _username_key(au_user)
                if not u:
                    st.error("Username cannot be empty.")
                elif u in USERS_DB:
                    st.error("Username already exists.")
                elif len(au_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    USERS_DB[u] = {"role": au_role, "email": au_email or "", "password": _hash_pw(au_pwd)}
                    _save_users(USERS_DB)
                    st.success(f"User **{u}** created ✅")

    if st.sidebar.button("🚪 Logout", key="btn_logout"):
        st.session_state["auth"] = {"is_auth": False, "user": None, "role": None}
        st.rerun()

# Gate: show auth UI until login is successful
if not st.session_state["auth"]["is_auth"]:
    render_auth_ui()
    st.stop()  # stop rendering dashboard until login
else:
    render_account_sidebar()

# ============================================================
# DASHBOARD LOGIC (three‑step funnel + refined A/B alignment)
# ============================================================

def safe_week_start(ts):
    if pd.isna(ts): return pd.NaT
    try: return pd.to_datetime(pd.Period(ts, freq="W").start_time)
    except: return pd.NaT

def read_db(db_file=DB_FILE):
    if not os.path.exists(db_file): return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(db_file)
    try:
        users = pd.read_sql("SELECT * FROM users", conn)
        events = pd.read_sql("SELECT * FROM events", conn)
    except:
        users, events = pd.DataFrame(), pd.DataFrame()
    conn.close()
    if not users.empty: users["joined_at"] = pd.to_datetime(users["joined_at"], errors="coerce")
    if not events.empty: events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    return users, events

def save_to_db(users_df, events_df):
    conn = sqlite3.connect(DB_FILE)
    users_df.to_sql("users", conn, if_exists="replace", index=False)
    events_df.to_sql("events", conn, if_exists="replace", index=False)
    conn.close()

# -------------------------
# LOAD DATA
# -------------------------
st.sidebar.header("📂 Data Loader")
uploaded_users = st.sidebar.file_uploader("Upload users.csv", type="csv", key="upload_users")
uploaded_events = st.sidebar.file_uploader("Upload events.csv", type="csv", key="upload_events")

def load_data():
    if uploaded_users and uploaded_events:
        u = pd.read_csv(uploaded_users); e = pd.read_csv(uploaded_events)
        u["joined_at"] = pd.to_datetime(u["joined_at"], errors="coerce")
        e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce")
        return u, e, True
    users_db_data, events_db_data = read_db()
    return users_db_data, events_db_data, False

users, events, from_upload = load_data()
if users.empty or events.empty:
    st.warning("Upload CSVs to proceed.")
    st.stop()

# Fill optional columns safely
for col, default in [
    ("device", "Unknown"),
    ("country", "Unknown"),
    ("ab_group", "Unknown"),
    ("subscription_status", "Unknown"),
]:
    if col not in users.columns:
        users[col] = default
if "event_name" not in events.columns:
    events["event_name"] = "unknown_event"

if from_upload and st.sidebar.button("💾 Save to DB", key="btn_save_db"):
    save_to_db(users, events)
    st.sidebar.success("Saved.")

# -------------------------
# FILTERS (robust date range)
# -------------------------
st.subheader("📅 Date range")

def _to_date(val, fallback=None):
    """Convert any input to datetime.date. If invalid/NaT, use fallback or today."""
    if val is None or (hasattr(pd, "isna") and pd.isna(val)):
        return fallback or date.today()
    if isinstance(val, date):
        return val
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return fallback or date.today()

ev_min = events["timestamp"].min()
ev_max = events["timestamp"].max()
ev_min_date = _to_date(ev_min)
ev_max_date = _to_date(ev_max, fallback=ev_min_date)

if st.session_state["date_start"] is None: st.session_state["date_start"] = ev_min_date
if st.session_state["date_end"]   is None: st.session_state["date_end"]   = ev_max_date

date_value = st.date_input(
    "Select date range",
    value=(st.session_state["date_start"], st.session_state["date_end"]),
    key="flt_date_range"
)
if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
    st.session_state["date_start"] = _to_date(date_value[0], fallback=st.session_state["date_start"])
    st.session_state["date_end"]   = _to_date(date_value[1], fallback=st.session_state["date_end"])
else:
    st.session_state["date_start"] = _to_date(date_value, fallback=st.session_state["date_start"])

start_date = pd.to_datetime(st.session_state["date_start"])
end_date   = pd.to_datetime(st.session_state["date_end"])

st.sidebar.header("🔍 Filters")
device_sel = st.sidebar.multiselect("Device", users["device"].unique(), default=list(users["device"].unique()), key="flt_device")
country_sel = st.sidebar.multiselect("Country", users["country"].unique(), default=list(users["country"].unique()), key="flt_country")
event_sel = st.sidebar.multiselect("Event", events["event_name"].unique(), default=list(events["event_name"].unique()), key="flt_event")
ab_sel = st.sidebar.multiselect("A/B Group", users["ab_group"].unique(), default=list(users["ab_group"].unique()), key="flt_ab")
subscription_status_sel = st.sidebar.multiselect(
    "Subscription Status", users["subscription_status"].unique(),
    default=list(users["subscription_status"].unique()), key="flt_subs"
)

mask_date = (events["timestamp"] >= start_date) & (events["timestamp"] <= end_date)
events_f = events[mask_date].copy()
users_f = users[(users["device"].isin(device_sel)) &
                (users["country"].isin(country_sel)) &
                (users["ab_group"].isin(ab_sel)) &
                (users["subscription_status"].isin(subscription_status_sel))]
events_f = events_f[events_f["event_name"].isin(event_sel) &
                    events_f["user_id"].isin(users_f["user_id"])]

# -------------------------
# ANALYTICS FUNCTIONS
# -------------------------

# ✅ Fixed three-step funnel: signup → onboarding → activation
def compute_funnel(ev: pd.DataFrame) -> pd.DataFrame:
    steps = ["signup_success", "onboarding_complete", "activation"]
    rows = []
    for s in steps:
        df_s = ev[ev["event_name"] == s]
        users_n = df_s["user_id"].nunique()
        events_n = len(df_s)
        rows.append({"step": s, "users": users_n, "events": events_n})
    funnel = pd.DataFrame(rows)
    first_users = funnel["users"].iloc[0] if len(funnel) > 0 else 0
    funnel["pct_vs_first"] = ((funnel["users"] / first_users) * 100).round(1) if first_users > 0 else 0.0
    return funnel

def compute_dau(ev):
    if ev.empty: return pd.DataFrame(columns=["date", "dau"])
    d = ev.groupby(ev["timestamp"].dt.date)["user_id"].nunique().reset_index()
    d.columns = ["date", "dau"]
    return d.tail(60)

def compute_cohort(users_df, ev_df, periods=8):
    if users_df.empty or ev_df.empty: return pd.DataFrame()
    u, e = users_df.copy(), ev_df.copy()
    u["cohort"] = u["joined_at"].apply(safe_week_start); u = u.dropna(subset=["cohort"])
    e = e.merge(u[["user_id", "cohort"]], on="user_id", how="inner")
    e["event_week"] = e["timestamp"].apply(safe_week_start); e = e.dropna(subset=["event_week"])
    e["week_num"] = ((e["event_week"] - e["cohort"]).dt.days // 7).astype(int); e = e[e["week_num"] >= 0]
    pivot = e.groupby(["cohort", "week_num"])["user_id"].nunique().unstack().fillna(0)
    sizes = u.groupby("cohort")["user_id"].nunique()
    pivot = pivot.reindex(sizes.index).fillna(0)
    for w in range(periods):
        if w not in pivot.columns: pivot[w] = 0
    rate = pivot[range(periods)].div(sizes, axis=0).fillna(0)
    return rate

def cohort_heatmap_figure(cohort_rate_df):
    if cohort_rate_df.empty: return None
    z = cohort_rate_df.values * 100
    x = [f"Week {int(c)}" for c in cohort_rate_df.columns]  # Week labels
    y = [d.date() if hasattr(d, "date") else str(d) for d in cohort_rate_df.index]
    fig = px.imshow(
        z,
        labels=dict(x="Weeks since signup", y="Cohort week", color="Retention %"),
        x=x, y=y, aspect="auto"
    )
    fig.update_layout(title="Retention Cohorts (%)", height=420,
                      margin=dict(l=40, r=20, t=40, b=40),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def prepare_cohort_table(rate_df):
    """
    Converts retention rate matrix (0..1) into a % table with:
      - First column 'Cohort week'
      - Columns 'Week 0' ... 'Week N'
    """
    if rate_df.empty: return pd.DataFrame()
    df = (rate_df * 100).round(1).copy()
    df.columns = [f"Week {int(c)}" for c in df.columns]
    df = df.reset_index()
    cohort_col = "cohort" if "cohort" in df.columns else "index"
    df.rename(columns={cohort_col: "Cohort week"}, inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df["Cohort week"]):
        df["Cohort week"] = df["Cohort week"].dt.date
    return df

def plotly_cohort_table(df, title="Cohort Table"):
    if df.empty: return None
    pct_cols = [c for c in df.columns if c.startswith("Week ")]
    df_fmt = df.copy()
    for c in pct_cols:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.1f}%")
    header_color = "rgba(30,30,30,0.95)"
    cell_color = "rgba(20,20,20,0.9)"
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_fmt.columns), fill_color=header_color, font=dict(color="white", size=12)),
        cells=dict(values=[df_fmt[c].tolist() for c in df_fmt.columns],
                   fill_color=cell_color, font=dict(color="white", size=12))
    )])
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", height=400)
    return fig

def compute_event_freq(ev):
    if ev.empty: return pd.DataFrame(columns=["event", "count"])
    freq = ev["event_name"].value_counts().reset_index()
    freq.columns = ["event", "count"]
    return freq

# Prefer 'conversion' event if present; fallback to 'activation'
def ab_summary(ev, users_df):
    conv_ev = "conversion" if "conversion" in ev["event_name"].unique() else "activation"
    conv = ev[ev["event_name"] == conv_ev].merge(
        users_df[["user_id", "ab_group"]], on="user_id", how="left"
    )
    users_ab = users_df[users_df["ab_group"].isin(["A", "B"])]
    total = users_ab.groupby("ab_group")["user_id"].nunique()
    convs = conv.groupby("ab_group")["user_id"].nunique()

    df = pd.DataFrame({
        "Group": ["A", "B"],
        "Total Users": [int(total.get("A", 0)), int(total.get("B", 0))],
        "Conversions": [int(convs.get("A", 0)), int(convs.get("B", 0))]
    })
    df["Conversion Rate (%)"] = (df["Conversions"] / df["Total Users"] * 100).round(2)
    return df, conv_ev

# -------------------------
# LAYOUT (build figures once so we can export them)
# -------------------------
st.markdown("## 📊 Visual analytics")

# Top row: Funnel (THREE STEPS ONLY) + DAU
col1, col2 = st.columns(2)
with col1:
    st.subheader("User funnel")
    funnel_df = compute_funnel(events_f)
    if not funnel_df.empty:
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df["step"].tolist(),     # labels
            x=funnel_df["users"].tolist(),    # widths (distinct users)
            text=funnel_df["pct_vs_first"].map(lambda v: f"{v:.1f}%").tolist(),
            textposition="inside",
            marker=dict(color=["#3DC1D3", "#2E86DE", "#5F27CD"]),
            connector=dict(line=dict(color="rgba(255,255,255,0.35)", dash="solid", width=1))
        ))
        fig_funnel.update_layout(
            title="Funnel: signup → onboarding → activation",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(color="white")
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
        with st.expander("View funnel table"):
            st.dataframe(
                funnel_df.rename(columns={
                    "step": "Step",
                    "users": "Distinct users",
                    "events": "Raw event count",
                    "pct_vs_first": "Users % vs signup"
                }),
                use_container_width=True
            )
    else:
        fig_funnel = None
        st.info("No events available for the current filters/date range.")

with col2:
    st.subheader("Daily active users")
    dau_df = compute_dau(events_f)
    fig_dau = px.line(dau_df, x="date", y="dau", title="DAU")
    fig_dau.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dau, use_container_width=True)

# Bottom row: Cohort + Event Distribution
col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("👥 Retention cohorts")
    cohort_view_choice = st.radio("View", ["Heatmap", "Weekly Table", "Daily Table"], horizontal=True, key="cohort_view_choice")

    if cohort_view_choice == "Heatmap":
        cohort_rate = compute_cohort(users_f, events_f, periods=8)
        if not cohort_rate.empty:
            fig_cohort_heat = cohort_heatmap_figure(cohort_rate)
            st.plotly_chart(fig_cohort_heat, use_container_width=True)
            fig_cohort_weekly = None
            fig_cohort_daily = None
        else:
            fig_cohort_heat = None
            fig_cohort_weekly = None
            fig_cohort_daily = None
            st.info("Not enough data for heatmap.")
    elif cohort_view_choice == "Weekly Table":
        cohort_rate = compute_cohort(users_f, events_f, periods=8)
        if not cohort_rate.empty:
            weekly_df = prepare_cohort_table(cohort_rate)
            fig_cohort_weekly = plotly_cohort_table(weekly_df, title="Weekly Cohort Table")
            st.plotly_chart(fig_cohort_weekly, use_container_width=True)
            fig_cohort_heat = None
            fig_cohort_daily = None
        else:
            fig_cohort_weekly = None
            fig_cohort_heat = None
            fig_cohort_daily = None
            st.info("Not enough data for weekly table.")
    else:  # Daily Table
        cohort_rate = compute_cohort(users_f, events_f, periods=11)  # week 0..10
        if not cohort_rate.empty:
            daily_df = prepare_cohort_table(cohort_rate)
            fig_cohort_daily = plotly_cohort_table(daily_df, title="Daily Cohort Table")
            st.plotly_chart(fig_cohort_daily, use_container_width=True)
            fig_cohort_heat = None
            fig_cohort_weekly = None
        else:
            fig_cohort_daily = None
            fig_cohort_heat = None
            fig_cohort_weekly = None
            st.info("Not enough data for daily table.")

with col_right:
    st.subheader("Event distribution")
    freq_df = compute_event_freq(events_f)
    if not freq_df.empty:
        fig_freq = px.pie(freq_df, names="event", values="count", title="Event Distribution")
        fig_freq.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        fig_freq = None
        st.info("No events to display.")

# -------------------------
# 🔀 A/B Testing (Summary table → Compact stats → *Text row* → Bar)
# -------------------------
st.markdown("## 🔀 A/B testing")

ab_df, conv_event_used = ab_summary(events_f, users_f)
if not ab_df.empty:
    # Summary table data
    table_header = list(ab_df.columns)
    table_cells = [ab_df[c].tolist() for c in ab_df.columns]

    # Minimal required stats from current filters
    a = ab_df[ab_df["Group"] == "A"].iloc[0]
    b = ab_df[ab_df["Group"] == "B"].iloc[0]
    n_a, n_b = int(a["Total Users"]), int(b["Total Users"])
    s_a, s_b = int(a["Conversions"]), int(b["Conversions"])
    p_a = s_a / n_a if n_a > 0 else 0.0
    p_b = s_b / n_b if n_b > 0 else 0.0

    # z, p-value, pooled p, difference (B - A) in proportion
    z, p_value, pooled, diff = proportions_z_test(s_a, n_a, s_b, n_b)

    # 95% CI for difference using unpooled SE
    se_diff = ((p_a * (1 - p_a) / max(n_a, 1)) + (p_b * (1 - p_b) / max(n_b, 1))) ** 0.5
    z_alpha2 = stats.norm.ppf(1 - 0.05/2)
    ci_diff = (diff - z_alpha2 * se_diff, diff + z_alpha2 * se_diff)

    # Compact stats (only what's needed)
    stats_rows = [
        ("Variant A rate", f"{p_a*100:.2f}%  ({s_a}/{n_a})"),
        ("Variant B rate", f"{p_b*100:.2f}%  ({s_b}/{n_b})"),
        ("Difference (B−A)", f"{diff*100:.2f}%"),
        ("z-value", f"{z:.4f}"),
        ("p-value (two-sided)", f"{p_value:.4f}"),
        ("95% CI (diff)", f"{ci_diff[0]*100:.2f}% → {ci_diff[1]*100:.2f}%"),
    ]
    stats_header = ["Metric", "Value"]
    stats_cells = [
        [r[0] for r in stats_rows],
        [r[1] for r in stats_rows]
    ]

    # Build a single tight figure:
    # Row1 = Summary table, Row2 = Compact stats, Row3 = Text-only (significance), Row4 = Bar
    fig_ab_summary = make_subplots(
        rows=4, cols=1,
        specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}]],
        row_heights=[0.32, 0.26, 0.06, 0.36],
        vertical_spacing=0.02
    )

    # Row 1: Summary table
    fig_ab_summary.add_trace(go.Table(
        header=dict(values=table_header, fill_color="rgba(30,30,30,0.95)",
                    font=dict(color="white", size=12), align="center"),
        cells=dict(values=table_cells, fill_color="rgba(20,20,20,0.9)",
                   font=dict(color="white", size=12), align="center")
    ), row=1, col=1)

    # Row 2: Compact stats
    fig_ab_summary.add_trace(go.Table(
        header=dict(values=stats_header, fill_color="rgba(30,30,30,0.95)",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=stats_cells, fill_color="rgba(20,20,20,0.9)",
                   font=dict(color="white", size=12), align="left")
    ), row=2, col=1)

    # Row 3: Text-only significance note (NO box, NO axes)
    if p_value < 0.05:
        who = "B" if p_b > p_a else "A"
        note_text = f"✅ Statistically Significant (α=0.05). Variation {who} is better."
        note_color = "#1E7E34"  # green
    else:
        note_text = "⚠️ Not statistically significant at α=0.05."
        note_color = "#CCCCCC"

    fig_ab_summary.add_trace(
        go.Scatter(
            x=[0.5], y=[0.5],  # center of the tiny row
            mode="text",
            text=[note_text],
            textfont=dict(color=note_color, size=13),
            showlegend=False
        ),
        row=3, col=1
    )
    # Hide axes for the text row so no ticks/numbers appear
    fig_ab_summary.update_xaxes(visible=False, row=3, col=1)
    fig_ab_summary.update_yaxes(visible=False, row=3, col=1)

    # Row 4: Bar chart
    bar_trace = go.Bar(
        x=ab_df["Group"],
        y=ab_df["Conversion Rate (%)"],
        text=ab_df["Conversion Rate (%)"],
        textposition="auto",
        marker_color=["#1f77b4", "#ff7f0e"]
    )
    fig_ab_summary.add_trace(bar_trace, row=4, col=1)
    fig_ab_summary.update_yaxes(showgrid=False, zeroline=False, row=4, col=1)

    fig_ab_summary.update_layout(
        title=f"A/B Summary, Stats & Conversion ({conv_event_used})",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=800
    )

    st.plotly_chart(fig_ab_summary, use_container_width=True)

else:
    fig_ab_summary = None
    st.info("No A/B data available.")

# ============================================================
# NARRATIVE SUMMARY BUILDERS (unchanged)
# ============================================================
def narrative_funnel(funnel_df):
    if funnel_df is None or funnel_df.empty:
        return "No funnel data is available for the selected filters and date range."
    baseline_step = str(funnel_df.iloc[0]["step"])
    baseline_users = int(funnel_df.iloc[0]["users"])
    total_steps = len(funnel_df)
    # Largest drop vs first step
    try:
        tail = funnel_df.iloc[1:].sort_values("pct_vs_first").head(1)
        if not tail.empty:
            drop_step = str(tail.iloc[0]["step"])
            drop_pct = float(tail.iloc[0]["pct_vs_first"])
            drop_sent = f"The largest drop relative to the baseline occurs at '{drop_step}', which is reached by {drop_pct:.1f}% of baseline users."
        else:
            drop_sent = ""
    except Exception:
        drop_sent = ""
    # Highest raw activity step
    top = funnel_df.sort_values("events", ascending=False).head(1)
    if not top.empty:
        top_step = str(top.iloc[0]["step"])
        top_events = int(top.iloc[0]["events"])
        top_sent = f"By total event count, the most active step is '{top_step}' with {top_events} events."
    else:
        top_sent = ""
    return (
        f"The funnel includes {total_steps} steps and uses '{baseline_step}' as the baseline, reached by {baseline_users} distinct users (100%). "
        f"{drop_sent} {top_sent}"
    ).strip()

def narrative_dau(dau_df):
    if dau_df is None or dau_df.empty:
        return "No DAU (Daily Active Users) data is available for this period."
    dmin, dmax = dau_df["date"].min(), dau_df["date"].max()
    avg = float(dau_df["dau"].mean())
    peak_row = dau_df.loc[dau_df["dau"].idxmax()]
    peak_date = peak_row["date"]
    peak_val = int(peak_row["dau"])
    # Simple trend
    try:
        last7 = float(dau_df.tail(7)["dau"].mean())
        prev7 = float(dau_df.tail(14).head(7)["dau"].mean())
        delta = last7 - prev7
        dir_ = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "roughly flat")
        trend = f"Over the latest week, DAU is {dir_} by {abs(delta):.1f} users versus the prior week."
    except Exception:
        trend = ""
    return (
        f"DAU from {dmin} to {dmax} averages {avg:.1f} users per day. "
        f"The peak day is {peak_date} with {peak_val} active users. {trend}"
    ).strip()

def narrative_cohorts(cohort_df):
    if cohort_df is None or cohort_df.empty:
        return "No cohort retention data is available."
    z = cohort_df.values
    if z.max() <= 1.0:  # scale to %
        z = z * 100.0
    # Period stats
    per1 = float(pd.Series(z[:, 1]).mean()) if z.shape[1] > 1 else None
    per4 = float(pd.Series(z[:, 4]).mean()) if z.shape[1] > 4 else None
    row_means = z.mean(axis=1)
    best_idx = int(row_means.argmax())
    best_label = cohort_df.index[best_idx]
    best_avg = float(row_means[best_idx])
    parts = [f"The retention table shows {cohort_df.shape[0]} cohorts over {cohort_df.shape[1]} periods."]
    if per1 is not None:
        parts.append(f"On average, period‑1 retention is {per1:.1f}%.")
    if per4 is not None:
        parts.append(f"By period‑4, average retention is {per4:.1f}%.")
    parts.append(f"Overall, the best cohort is {best_label} with an average retention of {best_avg:.1f}% across periods.")
    return " ".join(parts)

def narrative_events(freq_df):
    if freq_df is None or freq_df.empty:
        return "No event distribution data is available."
    total = int(freq_df["count"].sum())
    if total == 0:
        return "Events were recorded, but counts sum to zero."
    top = freq_df.sort_values("count", ascending=False).head(3)
    tops = []
    for _, r in top.iterrows():
        share = (r["count"] / total * 100.0) if total > 0 else 0
        tops.append(f"'{r['event']}' ({r['count']}, {share:.1f}% of total)")
    tops_str = "; ".join(tops)
    return (
        f"Across all observed events, a total of {total} actions were recorded. "
        f"The leading events are: {tops_str}. "
        f"In total, {freq_df.shape[0]} distinct event types are present in the selection."
    )

def narrative_ab(ab_df):
    if ab_df is None or ab_df.empty:
        return "No A/B test summary is available."
    best = ab_df.sort_values("Conversion Rate (%)", ascending=False).head(1)
    worst = ab_df.sort_values("Conversion Rate (%)", ascending=True).head(1)
    total_users = int(ab_df["Total Users"].sum())
    total_conv = int(ab_df["Conversions"].sum())
    overall_rate = (total_conv / total_users * 100.0) if total_users > 0 else 0.0
    parts = [f"The experiment compares {ab_df.shape[0]} groups with an overall conversion rate of {overall_rate:.1f}% ({total_conv}/{total_users})."]
    if not best.empty:
        b = best.iloc[0]
        parts.append(f"The best group is {b['Group']} at {b['Conversion Rate (%)']:.1f}% ({b['Conversions']}/{b['Total Users']}).")
    if not worst.empty:
        w = worst.iloc[0]
        parts.append(f"The weakest group is {w['Group']} at {w['Conversion Rate (%)']:.1f}% ({w['Conversions']}/{w['Total Users']}).")
    return " ".join(parts)

# ============================================================
# PDF EXPORT: chart and narrative on the SAME page (IN-MEMORY)
# ============================================================
def _draw_paragraph(c, text, x_mm, y_start_mm, max_width_mm=180, color=colors.black, font_size=11, leading=14):
    """Draw a single wrapped paragraph. Returns final y position."""
    style = ParagraphStyle(
        name="NarrativeStyle", fontName="Helvetica", fontSize=font_size,
        leading=leading, textColor=color, alignment=TA_LEFT
    )
    para = Paragraph(text, style)
    w, h = para.wrap(max_width_mm * mm, A4[1])  # width, height with page height
    if y_start_mm * mm - h < 20 * mm:
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.black)
        c.drawString(20 * mm, A4[1] - 20 * mm, "Report (continued)")
        y_start_mm = (A4[1] / mm) - 40
    para.drawOn(c, x_mm * mm, y_start_mm * mm - h)
    return (y_start_mm * mm - h) / mm

def export_dashboard_pdf_with_summaries(figs_dict) -> bytes:
    """
    Build a single PDF (chart image + narrative for each section) entirely in memory
    and return the PDF bytes. No file is written to disk.
    """
    # Render figures to temp PNGs where needed (we'll remove them right after)
    temp_paths = []
    for _, section in figs_dict.items():
        fig = section.get("figure")
        if fig is not None and not section.get("image_path"):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.write_image(tmp.name, format="png", scale=2)  # requires Kaleido
                tmp.close()
                section["image_path"] = tmp.name
                temp_paths.append(tmp.name)
            except Exception:
                section["image_path"] = None

    # Build the PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    for _, section in figs_dict.items():
        title = section.get("title", "")
        image_path = section.get("image_path")
        narrative_text = section.get("narrative_text", "")

        # Page header (title)
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(20 * mm, A4[1] - 20 * mm, title)

        # Draw chart image
        y = A4[1] - 25 * mm
        if image_path and os.path.exists(image_path):
            max_w, max_h = 180 * mm, 110 * mm  # space for narrative
            c.drawImage(
                image_path, 15 * mm, y - max_h,
                width=max_w, height=max_h,
                preserveAspectRatio=True, anchor='n'
            )
            y_after_img_mm = (y - max_h) / mm
        else:
            c.setFont("Helvetica", 12)
            c.drawString(20 * mm, y - 15 * mm, "No chart image available for this section.")
            y_after_img_mm = (y - 20 * mm) / mm

        # Draw narrative paragraph below the image (same page)
        if narrative_text:
            _draw_paragraph(
                c, text=narrative_text, x_mm=20, y_start_mm=y_after_img_mm - 10,
                max_width_mm=170, color=colors.black, font_size=11, leading=14
            )

        c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Clean up temp PNGs
    for p in temp_paths:
        try:
            os.remove(p)
        except:
            pass

    return pdf_bytes

# -------------------------
# BUILD NARRATIVES + EXPORT BUTTON (IN-MEMORY DOWNLOAD)
# -------------------------
st.markdown("---")
st.markdown("## 📥 Export dashboard as PDF")

if st.button("📄 Generate PDF", key="btn_gen_pdf_narratives"):
    # Narrative strings
    try:
        funnel_text = narrative_funnel(funnel_df)
    except Exception:
        funnel_text = "Funnel narrative could not be generated."
    try:
        dau_text = narrative_dau(dau_df)
    except Exception:
        dau_text = "DAU narrative could not be generated."
    try:
        if 'cohort_rate' in locals() and cohort_rate is not None and not cohort_rate.empty:
            cohort_text = narrative_cohorts(cohort_rate)  # fraction expected
        else:
            cohort_text = "No cohort view selected or data unavailable."
    except Exception:
        cohort_text = "Cohort narrative could not be generated."
    try:
        events_text = narrative_events(freq_df)
    except Exception:
        events_text = "Event distribution narrative could not be generated."
    try:
        ab_text = narrative_ab(ab_df)
    except Exception:
        ab_text = "A/B testing narrative could not be generated."

    # Sections dict (chart + narrative per visual)
    figs_dict = {
        "funnel":  {"title": "Funnel: signup → onboarding → activation", "figure": fig_funnel if 'fig_funnel' in locals() else None, "narrative_text": funnel_text},
        "dau":     {"title": "Daily Active Users (DAU)",                   "figure": fig_dau     if 'fig_dau'     in locals() else None, "narrative_text": dau_text},
        "cohorts": {"title": "Retention Cohorts",
                    "figure": (
                        fig_cohort_heat   if 'fig_cohort_heat'   in locals() and fig_cohort_heat   is not None else
                        fig_cohort_weekly if 'fig_cohort_weekly' in locals() and fig_cohort_weekly is not None else
                        fig_cohort_daily  if 'fig_cohort_daily'  in locals() and fig_cohort_daily  is not None else None
                    ),
                    "narrative_text": cohort_text},
        "events":  {"title": "Event Distribution",                          "figure": fig_freq     if 'fig_freq'     in locals() else None, "narrative_text": events_text},
        "ab":      {"title": "A/B Testing",                                 "figure": fig_ab_summary if 'fig_ab_summary' in locals() else None, "narrative_text": ab_text},
    }

    pdf_bytes = export_dashboard_pdf_with_summaries(figs_dict)
    if pdf_bytes:
        st.download_button(
            "📥 Download Dashboard PDF",
            data=pdf_bytes,
            file_name="dashboard_with_narratives.pdf",
            mime="application/pdf",
            key="btn_download_pdf_narratives"
        )
    else:
        st.error("Failed to generate PDF file.")
