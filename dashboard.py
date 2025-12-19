
# dashboard.py
import os
import json
import sqlite3
from datetime import datetime, date
import tempfile
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
import scipy.stats as stats

# ReportLab for PDF (in-memory, no local file write)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT

# ---- Try importing your local z-test helper; fallback if missing
try:
    from ab_test import proportions_z_test
except Exception:
    def proportions_z_test(s_a, n_a, s_b, n_b):
        p_pool = (s_a + s_b) / max(n_a + n_b, 1)
        se = (p_pool * (1 - p_pool) * (1 / max(n_a, 1) + 1 / max(n_b, 1))) ** 0.5
        diff = (s_b / max(n_b, 1)) - (s_a / max(n_a, 1))
        z = diff / se if se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p_value, p_pool, diff

# -----------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------
DB_FILE = "vizsprints.db"
AUTH_FILE = "auth.json"

st.set_page_config(page_title="VizSprints Analytics", layout="wide")

# -----------------------------------------------------------------------
# AUTH (kept intact)
# -----------------------------------------------------------------------
def _hash_pw(plain: str) -> str:
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
    return (u or "").strip().lower()

def _load_users() -> dict:
    if not os.path.exists(AUTH_FILE):
        default_pwd = os.environ.get("DASH_PWD_PREETI", "preeti@123")
        users = {
            "preeti": {"role": "admin", "email": "preeti@example.com", "password": _hash_pw(default_pwd)}
        }
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return users
    with open(AUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_users(users: dict):
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

if "auth" not in st.session_state:
    st.session_state["auth"] = {"is_auth": False, "user": None, "role": None}
if "reset_codes" not in st.session_state:
    st.session_state["reset_codes"] = {}
if "date_start" not in st.session_state:
    st.session_state["date_start"] = None
if "date_end" not in st.session_state:
    st.session_state["date_end"] = None

USERS_DB = _load_users()

def render_auth_ui():
    tabs = st.tabs(["Login", "Sign up", "Forgot password"])

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
                st.error("Username already exists.")
            elif su_pwd1 != su_pwd2:
                st.error("Passwords do not match.")
            elif len(su_pwd1) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                USERS_DB[u] = {"role": "viewer", "email": su_email or "", "password": _hash_pw(su_pwd1)}
                _save_users(USERS_DB)
                st.success("Account created ✅. Please login.")
                st.rerun()

    with tabs[2]:
        st.markdown("### 🧩 Reset your password")
        fp_user = st.text_input("Enter your username", key="forgot_username")
        if st.button("Generate reset code", key="btn_generate_code"):
            u = _username_key(fp_user)
            if u in USERS_DB:
                import secrets
                code = secrets.token_hex(4)
                st.session_state["reset_codes"][u] = code
                st.info(f"Your reset code: **{code}**")
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
                st.error("No reset code generated.")
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
    st.sidebar.success(f"Logged in as **{st.session_state['auth']['user']}** ({st.session_state['auth']['role']})")
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

if not st.session_state["auth"]["is_auth"]:
    render_auth_ui()
    st.stop()
else:
    render_account_sidebar()

# -----------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------
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

# Optional columns
for col, default in [
    ("device", "Unknown"),
    ("country", "Unknown"),
    ("ab_group", "Unknown"),
    ("subscription_status", "Unknown"),
]:
    if col not in users.columns: users[col] = default
if "event_name" not in events.columns:
    events["event_name"] = "unknown_event"

if from_upload and st.sidebar.button("💾 Save to DB", key="btn_save_db"):
    save_to_db(users, events)
    st.sidebar.success("Saved.")

# -----------------------------------------------------------------------
# FILTERS + GUARDRAILS
# -----------------------------------------------------------------------
st.subheader("📅 Date range")

def _to_date(val, fallback=None):
    if val is None or (hasattr(pd, "isna") and pd.isna(val)):
        return fallback or date.today()
    if isinstance(val, date): return val
    try: return pd.to_datetime(val).date()
    except Exception: return fallback or date.today()

ev_min = events["timestamp"].min()
ev_max = events["timestamp"].max()
ev_min_date = _to_date(ev_min)
ev_max_date = _to_date(ev_max, fallback=ev_min_date)

if st.session_state["date_start"] is None: st.session_state["date_start"] = ev_min_date
if st.session_state["date_end"]   is None: st.session_state["date_end"]   = ev_max_date

date_value = st.date_input("Select date range", value=(st.session_state["date_start"], st.session_state["date_end"]), key="flt_date_range")
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

# Ensure crucial events are always included in the selection
required_events = ["signup_success", "onboarding_complete", "activation"]
if "conversion" in events["event_name"].unique():
    required_events.append("conversion")
event_default = sorted(list(set(list(events["event_name"].unique())) | set(required_events)))

event_sel = st.sidebar.multiselect("Event", event_default, default=event_default, key="flt_event")

ab_sel = st.sidebar.multiselect("A/B Group", users["ab_group"].unique(), default=list(users["ab_group"].unique()), key="flt_ab")
subscription_status_sel = st.sidebar.multiselect("Subscription Status", users["subscription_status"].unique(),
                                                 default=list(users["subscription_status"].unique()), key="flt_subs")

# Reset filters button — brings everything back to sane defaults
if st.sidebar.button("♻️ Reset all filters to defaults"):
    st.session_state["date_start"] = ev_min_date
    st.session_state["date_end"]   = ev_max_date
    device_sel = list(users["device"].unique())
    country_sel = list(users["country"].unique())
    ab_sel = list(users["ab_group"].unique())
    subscription_status_sel = list(users["subscription_status"].unique())
    event_sel = event_default
    st.experimental_rerun()

# Apply filters
mask_date = (events["timestamp"] >= start_date) & (events["timestamp"] <= end_date)
events_f = events[mask_date].copy()
users_f = users[(users["device"].isin(device_sel)) &
                (users["country"].isin(country_sel)) &
                (users["ab_group"].isin(ab_sel)) &
                (users["subscription_status"].isin(subscription_status_sel))]
events_f = events_f[events_f["event_name"].isin(event_sel) &
                    events_f["user_id"].isin(users_f["user_id"])]

# Diagnostics panel
with st.expander("⚙️ Diagnostics (data after filters)"):
    st.write({
        "Users (filtered)": int(users_f["user_id"].nunique()),
        "Events (filtered rows)": int(len(events_f)),
        "Date range": [str(st.session_state["date_start"]), str(st.session_state["date_end"])],
        "Required funnel steps present?": {s: (s in events_f["event_name"].unique()) for s in ["signup_success","onboarding_complete","activation"]},
        "Conversion present?": ("conversion" in events_f["event_name"].unique())
    })

if events_f.empty:
    st.warning("No events match the current filters/date range. Click **Reset all filters to defaults** (left sidebar) or widen your date range.")
    st.stop()

# -----------------------------------------------------------------------
# ANALYTICS
# -----------------------------------------------------------------------
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
    rate.index.name = "cohort"
    return rate

def cohort_heatmap_figure(cohort_rate_df):
    if cohort_rate_df.empty: return None
    z = cohort_rate_df.values * 100
    x = [f"Week {int(c)}" for c in cohort_rate_df.columns]
    y = [d.date() if hasattr(d, "date") else str(d) for d in cohort_rate_df.index]
    fig = px.imshow(z, labels=dict(x="Weeks since signup", y="Cohort week", color="Retention %"), x=x, y=y, aspect="auto")
    fig.update_layout(title="Retention Cohorts (%)", height=420, margin=dict(l=40, r=20, t=40, b=40), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def prepare_cohort_table(rate_df):
    if rate_df.empty: return pd.DataFrame()
    df = (rate_df * 100).round(1).copy()
    df.columns = [f"Week {int(c)}" for c in df.columns]
    df = df.reset_index()
    cohort_col = "cohort" if "cohort" in df.columns else "index"
    df.rename(columns={cohort_col: "Cohort week"}, inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df["Cohort week"]):
        df["Cohort week"] = df["Cohort week"].dt.date
    return df

def compute_event_freq(ev):
    if ev.empty: return pd.DataFrame(columns=["event", "count"])
    freq = ev["event_name"].value_counts().reset_index()
    freq.columns = ["event", "count"]
    return freq

def ab_summary(ev, users_df):
    conv_ev = "conversion" if "conversion" in ev["event_name"].unique() else "activation"
    conv = ev[ev["event_name"] == conv_ev].merge(users_df[["user_id", "ab_group"]], on="user_id", how="left")
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

# -----------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------
st.markdown("## 📊 Visual analytics")

# Top row: Funnel + DAU
col1, col2 = st.columns(2)
with col1:
    st.subheader("User funnel")
    funnel_df = compute_funnel(events_f)
    if funnel_df["users"].sum() > 0:
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df["step"].tolist(),
            x=funnel_df["users"].tolist(),
            text=funnel_df["pct_vs_first"].map(lambda v: f"{v:.1f}%").tolist(),
            textposition="inside",
            marker=dict(color=["#3DC1D3", "#2E86DE", "#5F27CD"]),
            connector=dict(line=dict(color="rgba(255,255,255,0.35)", dash="solid", width=1))
        ))
        fig_funnel.update_layout(title="Funnel: signup → onboarding → activation",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 margin=dict(l=10, r=10, t=40, b=10), font=dict(color="white"))
        st.plotly_chart(fig_funnel, use_container_width=True)
        with st.expander("View funnel table"):
            st.dataframe(funnel_df.rename(columns={
                "step": "Step", "users": "Distinct users", "events": "Raw event count", "pct_vs_first": "Users % vs signup"
            }), use_container_width=True)
    else:
        st.info("Funnel steps exist, but none were reached under current filters.")

with col2:
    st.subheader("Daily active users")
    dau_df = compute_dau(events_f)
    if not dau_df.empty:
        fig_dau = px.line(dau_df, x="date", y="dau", title="DAU")
        fig_dau.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dau, use_container_width=True)
    else:
        st.info("No DAU for the selected range.")

# Bottom row: Cohort + Event Distribution
col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("👥 Retention cohorts")
    cohort_view_choice = st.radio("View", ["Heatmap", "Weekly Table", "Daily Table"], horizontal=True, key="cohort_view_choice")
    cohort_rate = compute_cohort(users_f, events_f, periods=8)

    if cohort_view_choice == "Heatmap":
        if not cohort_rate.empty:
            fig_cohort_heat = cohort_heatmap_figure(cohort_rate)
            st.plotly_chart(fig_cohort_heat, use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")
    elif cohort_view_choice == "Weekly Table":
        if not cohort_rate.empty:
            weekly_df = prepare_cohort_table(cohort_rate)
            fig_cohort_weekly = go.Figure(go.Table(
                header=dict(values=list(weekly_df.columns), fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12)),
                cells=dict(values=[weekly_df[c].tolist() for c in weekly_df.columns],
                           fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12))
            ))
            fig_cohort_weekly.update_layout(title="Weekly Cohort Table", paper_bgcolor="rgba(0,0,0,0)",
                                            height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cohort_weekly, use_container_width=True)
        else:
            st.info("Not enough data for weekly table.")
    else:
        # Daily view can reuse the same data (longer periods optional)
        cohort_rate_daily = compute_cohort(users_f, events_f, periods=11)
        if not cohort_rate_daily.empty:
            daily_df = prepare_cohort_table(cohort_rate_daily)
            fig_cohort_daily = go.Figure(go.Table(
                header=dict(values=list(daily_df.columns), fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12)),
                cells=dict(values=[daily_df[c].tolist() for c in daily_df.columns],
                           fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12))
            ))
            fig_cohort_daily.update_layout(title="Daily Cohort Table", paper_bgcolor="rgba(0,0,0,0)",
                                           height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cohort_daily, use_container_width=True)
        else:
            st.info("Not enough data for daily table.")

with col_right:
    st.subheader("Event distribution")
    freq_df = compute_event_freq(events_f)
    if not freq_df.empty:
        fig_freq = px.pie(freq_df, names="event", values="count", title="Event Distribution", hole=0.4)
        fig_freq.update_traces(textposition="inside", textinfo="percent")
        fig_freq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle", bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=160, t=40, b=10)
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("No events to display.")

# -----------------------------------------------------------------------
# A/B Testing
# -----------------------------------------------------------------------
st.markdown("## 🔀 A/B testing")

ab_df, conv_event_used = ab_summary(events_f, users_f)
if not ab_df.empty and ab_df["Total Users"].sum() > 0:
    a = ab_df[ab_df["Group"] == "A"].iloc[0]
    b = ab_df[ab_df["Group"] == "B"].iloc[0]
    n_a, n_b = int(a["Total Users"]), int(b["Total Users"])
    s_a, s_b = int(a["Conversions"]), int(b["Conversions"])
    p_a = s_a / n_a if n_a > 0 else 0.0
    p_b = s_b / n_b if n_b > 0 else 0.0

    z, p_value, pooled, diff = proportions_z_test(s_a, n_a, s_b, n_b)
    se_diff = ((p_a * (1 - p_a) / max(n_a, 1)) + (p_b * (1 - p_b) / max(n_b, 1))) ** 0.5
    z_alpha2 = stats.norm.ppf(1 - 0.05/2)
    ci_diff = (diff - z_alpha2 * se_diff, diff + z_alpha2 * se_diff)

    table_header = list(ab_df.columns)
    table_cells = [ab_df[c].tolist() for c in ab_df.columns]
    stats_rows = [
        ("Variant A rate", f"{p_a*100:.2f}%  ({s_a}/{n_a})"),
        ("Variant B rate", f"{p_b*100:.2f}%  ({s_b}/{n_b})"),
        ("Difference (B−A)", f"{diff*100:.2f}%"),
        ("z-value", f"{z:.4f}"),
        ("p-value (two-sided)", f"{p_value:.4f}"),
        ("95% CI (diff)", f"{ci_diff[0]*100:.2f}% → {ci_diff[1]*100:.2f}%"),
    ]
    stats_header = ["Metric", "Value"]
    stats_cells = [[r[0] for r in stats_rows], [r[1] for r in stats_rows]]

    fig_ab_summary = make_subplots(
        rows=4, cols=1,
        specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}]],
        row_heights=[0.32, 0.26, 0.06, 0.36], vertical_spacing=0.02
    )

    fig_ab_summary.add_trace(go.Table(
        header=dict(values=table_header, fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12), align="center"),
        cells=dict(values=table_cells, fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12), align="center")
    ), row=1, col=1)

    fig_ab_summary.add_trace(go.Table(
        header=dict(values=stats_header, fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12), align="left"),
        cells=dict(values=stats_cells, fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12), align="left")
    ), row=2, col=1)

    if p_value < 0.05:
        note_text = f"✅ Statistically Significant (α=0.05). Variation {'B' if p_b > p_a else 'A'} is better."
        note_color = "#1E7E34"
    else:
        note_text = "⚠️ Not statistically significant at α=0.05."
        note_color = "#CCCCCC"

    fig_ab_summary.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text",
                                        text=[note_text], textfont=dict(color=note_color, size=13),
                                        showlegend=False), row=3, col=1)
    fig_ab_summary.update_xaxes(visible=False, row=3, col=1)
    fig_ab_summary.update_yaxes(visible=False, row=3, col=1)

    fig_ab_summary.add_trace(go.Bar(
        x=ab_df["Group"], y=ab_df["Conversion Rate (%)"],
        text=ab_df["Conversion Rate (%)"], textposition="auto",
        marker_color=["#1f77b4", "#ff7f0e"], name="Conversion rate (%)",
        showlegend=False
    ), row=4, col=1)
    fig_ab_summary.update_yaxes(showgrid=False, zeroline=False, row=4, col=1)

    fig_ab_summary.update_layout(title=f"A/B Summary, Stats & Conversion ({conv_event_used})",
                                 showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=10),
                                 height=800)
    st.plotly_chart(fig_ab_summary, use_container_width=True)
else:
    st.info("No A/B data available (check AB group filter or events).")

# -----------------------------------------------------------------------
# PDF EXPORT (in-memory)
# -----------------------------------------------------------------------
def _draw_paragraph(c, text, x_mm, y_start_mm, max_width_mm=180, color=colors.black, font_size=11, leading=14):
    style = ParagraphStyle(name="NarrativeStyle", fontName="Helvetica", fontSize=font_size, leading=leading, textColor=color, alignment=TA_LEFT)
    para = Paragraph(text, style)
    w, h = para.wrap(max_width_mm * mm, A4[1])
    if y_start_mm * mm - h < 20 * mm:
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.black)
        c.drawString(20 * mm, A4[1] - 20 * mm, "Report (continued)")
        y_start_mm = (A4[1] / mm) - 40
    para.drawOn(c, x_mm * mm, y_start_mm * mm - h)
    return (y_start_mm * mm - h) / mm

def export_dashboard_pdf_with_summaries(figs_dict) -> bytes:
    temp_paths = []
    for _, section in figs_dict.items():
        fig = section.get("figure")
        if fig is not None and not section.get("image_path"):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.write_image(tmp.name, format="png", scale=2)  # requires kaleido
                tmp.close()
                section["image_path"] = tmp.name
                temp_paths.append(tmp.name)
            except Exception:
                section["image_path"] = None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    for _, section in figs_dict.items():
        title = section.get("title", "")
        image_path = section.get("image_path")
        narrative_text = section.get("narrative_text", "")

        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(20 * mm, A4[1] - 20 * mm, title)

        y = A4[1] - 25 * mm
        if image_path and os.path.exists(image_path):
            max_w, max_h = 180 * mm, 110 * mm
            c.drawImage(image_path, 15 * mm, y - max_h, width=max_w, height=max_h, preserveAspectRatio=True, anchor='n')
            y_after_img_mm = (y - max_h) / mm
        else:
            c.setFont("Helvetica", 12)
            c.drawString(20 * mm, y - 15 * mm, "No chart image available for this section.")
            y_after_img_mm = (y - 20 * mm) / mm

        if narrative_text:
            _draw_paragraph(c, text=narrative_text, x_mm=20, y_start_mm=y_after_img_mm - 10, max_width_mm=170, color=colors.black, font_size=11, leading=14)
        c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    for p in temp_paths:
        try: os.remove(p)
        except: pass
    return pdf_bytes

# Narratives
def narrative_funnel(funnel_df):
    if funnel_df is None or funnel_df.empty: return "No funnel data is available."
    baseline_step = str(funnel_df.iloc[0]["step"]); baseline_users = int(funnel_df.iloc[0]["users"]); total_steps = len(funnel_df)
    try:
        tail = funnel_df.iloc[1:].sort_values("pct_vs_first").head(1)
        drop_sent = (f"The largest drop relative to the baseline occurs at '{str(tail.iloc[0]['step'])}', reached by {float(tail.iloc[0]['pct_vs_first']):.1f}% of baseline users.") if not tail.empty else ""
    except Exception:
        drop_sent = ""
    top = funnel_df.sort_values("events", ascending=False).head(1)
    top_sent = (f"By total event count, the most active step is '{str(top.iloc[0]['step'])}' with {int(top.iloc[0]['events'])} events.") if not top.empty else ""
    return f"The funnel includes {total_steps} steps and uses '{baseline_step}' as the baseline, reached by {baseline_users} distinct users (100%). {drop_sent} {top_sent}".strip()

def narrative_dau(dau_df):
    if dau_df is None or dau_df.empty: return "No DAU data is available for this period."
    dmin, dmax = dau_df["date"].min(), dau_df["date"].max(); avg = float(dau_df["dau"].mean())
    peak_row = dau_df.loc[dau_df["dau"].idxmax()]; peak_date = peak_row["date"]; peak_val = int(peak_row["dau"])
    try:
        last7 = float(dau_df.tail(7)["dau"].mean()); prev7 = float(dau_df.tail(14).head(7)["dau"].mean())
        delta = last7 - prev7; dir_ = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "roughly flat")
        trend = f"Over the latest week, DAU is {dir_} by {abs(delta):.1f} users vs the prior week."
    except Exception:
        trend = ""
    return f"DAU from {dmin} to {dmax} averages {avg:.1f}. Peak: {peak_date} with {peak_val}. {trend}"

def narrative_cohorts(cohort_df):
    if cohort_df is None or cohort_df.empty: return "No cohort retention data is available."
    z = cohort_df.values; z = z * 100.0 if float(z.max()) <= 1.0 else z
    per1 = float(pd.Series(z[:, 1]).mean()) if z.shape[1] > 1 else None
    per4 = float(pd.Series(z[:, 4]).mean()) if z.shape[1] > 4 else None
    row_means = z.mean(axis=1); best_idx = int(row_means.argmax()) if row_means.size > 0 else 0
    best_label = cohort_df.index[best_idx] if row_means.size > 0 else "Best cohort"; best_avg = float(row_means[best_idx]) if row_means.size > 0 else 0.0
    parts = [f"The retention view shows {cohort_df.shape[0]} cohorts over {cohort_df.shape[1]} periods."]
    if per1 is not None: parts.append(f"On average, period‑1 retention is {per1:.1f}%.")
    if per4 is not None: parts.append(f"By period‑4, average retention is {per4:.1f}%.")
    parts.append(f"Overall, the best cohort is {best_label} with an average retention of {best_avg:.1f}% across periods.")
    return " ".join(parts)

def narrative_events(freq_df):
    if freq_df is None or freq_df.empty: return "No event distribution data is available."
    total = int(freq_df["count"].sum()); 
    if total == 0: return "Events were recorded, but counts sum to zero."
    top = freq_df.sort_values("count", ascending=False).head(3); tops = []
    for _, r in top.iterrows():
        share = (r["count"] / total * 100.0) if total > 0 else 0; tops.append(f"'{r['event']}' ({r['count']}, {share:.1f}% of total)")
    return f"Total actions: {total}. Top events: {'; '.join(tops)}."

def narrative_ab(ab_df):
    if ab_df is None or ab_df.empty: return "No A/B test summary is available."
    best = ab_df.sort_values("Conversion Rate (%)", ascending=False).head(1)
    worst = ab_df.sort_values("Conversion Rate (%)", ascending=True).head(1)
    total_users = int(ab_df["Total Users"].sum()); total_conv = int(ab_df["Conversions"].sum())
    overall_rate = (total_conv / total_users * 100.0) if total_users > 0 else 0.0
    parts = [f"Overall conversion: {overall_rate:.1f}% ({total_conv}/{total_users})."]
    if not best.empty:
        b = best.iloc[0]; parts.append(f"Best group: {b['Group']} at {b['Conversion Rate (%)']:.1f}% ({b['Conversions']}/{b['Total Users']}).")
    if not worst.empty:
        w = worst.iloc[0]; parts.append(f"Weakest group: {w['Group']} at {w['Conversion Rate (%)']:.1f}% ({w['Conversions']}/{w['Total Users']}).")
    return " ".join(parts)

st.markdown("---")
st.markdown("## 📥 Export dashboard as PDF")

if st.button("📄 Generate PDF", key="btn_gen_pdf_narratives"):
    try: funnel_text = narrative_funnel(compute_funnel(events_f))
    except Exception: funnel_text = "Funnel narrative could not be generated."
    try: dau_text = narrative_dau(compute_dau(events_f))
    except Exception: dau_text = "DAU narrative could not be generated."
    try:
        cohort_text = narrative_cohorts(compute_cohort(users_f, events_f, periods=8))
    except Exception:
        cohort_text = "Cohort narrative could not be generated."
    try: events_text = narrative_events(compute_event_freq(events_f))
    except Exception: events_text = "Event distribution narrative could not be generated."
    try: ab_text = narrative_ab(ab_df)
    except Exception: ab_text = "A/B testing narrative could not be generated."

    figs_dict = {
        "funnel":  {"title": "Funnel: signup → onboarding → activation", "figure": None, "narrative_text": funnel_text},
        "dau":     {"title": "Daily Active Users (DAU)",                   "figure": None, "narrative_text": dau_text},
        "cohorts": {"title": "Retention Cohorts",                          "figure": None, "narrative_text": cohort_text},
        "events":  {"title": "Event Distribution",                         "figure": None, "narrative_text": events_text},
        "ab":      {"title": "A/B Testing",                                "figure": None, "narrative_text": ab_text},
    }

    pdf_bytes = export_dashboard_pdf_with_summaries(figs_dict)
    if pdf_bytes:
        st.download_button("📥 Download Dashboard PDF", data=pdf_bytes,
                           file_name="vizsprints_dashboard.pdf", mime="application/pdf",
                           key="btn_download_pdf_narratives")
    else:
        st.error("Failed to generate PDF file. Ensure 'kaleido' is in requirements.")
