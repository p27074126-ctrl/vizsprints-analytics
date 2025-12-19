
# dashboard.py
import os
import json
import sqlite3
from datetime import date
import tempfile
import io

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
import scipy.stats as stats

# Matplotlib for PDF fallback (no Chrome/kaleido needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT

# -----------------------------------------------------------------------
# z-test helper
# -----------------------------------------------------------------------
def proportions_z_test(s_a, n_a, s_b, n_b):
    p_pool = (s_a + s_b) / max(n_a + n_b, 1)
    se = (p_pool * (1 - p_pool) * (1 / max(n_a, 1) + 1 / max(n_b, 1))) ** 0.5
    diff = (s_b / max(n_b, 1)) - (s_a / max(n_a, 1))
    z = diff / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value, p_pool, diff

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
DB_FILE = "vizsprints.db"
AUTH_FILE = "auth.json"
st.set_page_config(page_title="VizSprints Analytics", layout="wide")

# -----------------------------------------------------------------------
# AUTH (bcrypt-only)
# -----------------------------------------------------------------------
def _username_key(u: str) -> str:
    return (u or "").strip().lower()

def _hash_pw_bcrypt(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def _verify_pw_bcrypt(plain: str, hashed: str) -> bool:
    import bcrypt
    if not hashed or not (hashed.startswith("$2a$") or hashed.startswith("$2b$") or hashed.startswith("$2y$")):
        return False
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def _load_users() -> dict:
    """
    Load users from auth.json; bootstrap admin 'preeti' if file is missing.
    Uses bcrypt-only hashing.
    """
    if not os.path.exists(AUTH_FILE):
        admin_pwd = os.environ.get("DASH_PWD_PREETI", "preeti@123")
        users = {
            "preeti": {
                "role": "admin",
                "email": "preeti@example.com",
                "password": _hash_pw_bcrypt(admin_pwd)
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

# Session
if "auth" not in st.session_state:
    st.session_state["auth"] = {"is_auth": False, "user": None, "role": None}

USERS_DB = _load_users()

def render_auth_ui():
    # --- Title requested ---
    st.title("VizSprint Login")

    tabs = st.tabs(["Login", "Sign up", "Forgot password"])

    # Login
    with tabs[0]:
        u_in = st.text_input("Username", key="login_username")
        p_in = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in", key="btn_sign_in"):
            u = _username_key(u_in)
            record = USERS_DB.get(u)
            if record and _verify_pw_bcrypt(p_in, record["password"]):
                st.session_state["auth"] = {"is_auth": True, "user": u, "role": record.get("role", "viewer")}
                st.rerun()
            else:
                st.error("Invalid credentials")

    # Sign up
    with tabs[1]:
        su_user = st.text_input("New username", key="signup_username")
        su_pwd1 = st.text_input("Password", type="password", key="signup_pwd1")
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
                USERS_DB[u] = {"role": "viewer", "email": "", "password": _hash_pw_bcrypt(su_pwd1)}
                _save_users(USERS_DB)
                st.success("Account created. You can login now.")

    # Forgot password
    with tabs[2]:
        fp_user = st.text_input("Username", key="forgot_username")
        fp_new  = st.text_input("New password", type="password", key="forgot_newpwd")
        if st.button("Reset password", key="btn_reset_pwd"):
            u = _username_key(fp_user)
            if u in USERS_DB:
                if len(fp_new) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    USERS_DB[u]["password"] = _hash_pw_bcrypt(fp_new)
                    _save_users(USERS_DB)
                    st.success("Password reset. Please login.")
            else:
                st.error("User not found.")

def render_account_sidebar():
    role = st.session_state["auth"]["role"]
    user = st.session_state["auth"]["user"]
    st.sidebar.success(f"Logged in as **{user}** ({role})")

    # Change password
    with st.sidebar.expander("🔄 Change password"):
        old = st.text_input("Current password", type="password", key="change_old")
        new1 = st.text_input("New password", type="password", key="change_new1")
        new2 = st.text_input("Confirm new password", type="password", key="change_new2")
        if st.button("Update password", key="btn_update_pwd"):
            record = USERS_DB.get(user)
            if not record or not _verify_pw_bcrypt(old, record["password"]):
                st.error("Current password is incorrect.")
            elif new1 != new2:
                st.error("New passwords do not match.")
            elif len(new1) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                USERS_DB[user]["password"] = _hash_pw_bcrypt(new1)
                _save_users(USERS_DB)
                st.success("Password updated ✅")

    # Admin add user
    if role == "admin":
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
                    USERS_DB[u] = {"role": au_role, "email": au_email or "", "password": _hash_pw_bcrypt(au_pwd)}
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
# DATA LOAD + DATA LOADER (sidebar)
# -----------------------------------------------------------------------
def read_db(db_file=DB_FILE):
    if not os.path.exists(db_file): return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(db_file)
    users = pd.read_sql("SELECT * FROM users", conn)
    events = pd.read_sql("SELECT * FROM events", conn)
    conn.close()
    users["joined_at"] = pd.to_datetime(users["joined_at"], errors="coerce")
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
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
    if uploaded_users is not None and uploaded_events is not None:
        u = pd.read_csv(uploaded_users); e = pd.read_csv(uploaded_events)
        u["joined_at"] = pd.to_datetime(u["joined_at"], errors="coerce")
        e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce")
        return u, e, True
    users_db_data, events_db_data = read_db()
    return users_db_data, events_db_data, False

users, events, from_upload = load_data()

# optional columns
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
    st.sidebar.success("Saved to SQLite (vizsprints.db)")

if users.empty or events.empty:
    st.warning("Upload data to proceed.")
    st.stop()

# -----------------------------------------------------------------------
# FILTERS (no reset button, no diagnostics)
# -----------------------------------------------------------------------
st.subheader("📅 Date range")
start_date = st.date_input("Start date", value=events["timestamp"].min().date(), key="flt_start_date")
end_date   = st.date_input("End date",   value=events["timestamp"].max().date(), key="flt_end_date")

device_sel = st.sidebar.multiselect("Device", users["device"].unique(),
                                    default=list(users["device"].unique()), key="flt_device")
country_sel = st.sidebar.multiselect("Country", users["country"].unique(),
                                     default=list(users["country"].unique()), key="flt_country")

# Keep funnel/activation/conversion in defaults
def _safe_week_start(ts):
    if pd.isna(ts): return pd.NaT
    try: return pd.to_datetime(pd.Period(ts, freq="W").start_time)
    except: return pd.NaT

required_events = ["signup_success", "onboarding_complete", "activation"]
if "conversion" in events["event_name"].unique():
    required_events.append("conversion")
event_default = sorted(set(events["event_name"].unique()) | set(required_events))
event_sel = st.sidebar.multiselect("Event", event_default, default=event_default, key="flt_event")

ab_sel = st.sidebar.multiselect("A/B Group", users["ab_group"].unique(),
                                default=list(users["ab_group"].unique()), key="flt_ab")
subscription_status_sel = st.sidebar.multiselect("Subscription Status", users["subscription_status"].unique(),
                                                 default=list(users["subscription_status"].unique()), key="flt_subs")

# Apply filters
mask_date = (events["timestamp"].dt.date >= start_date) & (events["timestamp"].dt.date <= end_date)
events_f = events[mask_date].copy()
users_f = users[(users["device"].isin(device_sel)) &
                (users["country"].isin(country_sel)) &
                (users["ab_group"].isin(ab_sel)) &
                (users["subscription_status"].isin(subscription_status_sel))]
events_f = events_f[events_f["event_name"].isin(event_sel) & events_f["user_id"].isin(users_f["user_id"])]

if events_f.empty:
    st.warning("No data for selected filters.")
    st.stop()

# -----------------------------------------------------------------------
# ANALYTICS FUNCTIONS
# -----------------------------------------------------------------------
def compute_funnel(ev):
    steps = ["signup_success", "onboarding_complete", "activation"]
    rows = []
    for s in steps:
        df_s = ev[ev["event_name"] == s]
        rows.append({"step": s, "users": df_s["user_id"].nunique(), "events": len(df_s)})
    funnel = pd.DataFrame(rows)
    first_users = funnel["users"].iloc[0] if len(funnel) > 0 else 0
    funnel["pct_vs_first"] = ((funnel["users"] / first_users) * 100).round(1) if first_users > 0 else 0.0
    return funnel

def compute_dau(ev):
    d = ev.groupby(ev["timestamp"].dt.date)["user_id"].nunique().reset_index()
    d.columns = ["date", "dau"]
    return d

def safe_week_start(ts):
    if pd.isna(ts): return pd.NaT
    try: return pd.to_datetime(pd.Period(ts, freq="W").start_time)
    except: return pd.NaT

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
    fig = px.imshow(z, labels=dict(x="Weeks since signup", y="Cohort week", color="Retention %"),
                    x=x, y=y, aspect="auto")
    fig.update_layout(title="Retention Cohorts (%)", height=420,
                      margin=dict(l=40, r=20, t=40, b=40),
                      paper_bgcolor="rgba(0,0,0,0)")
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
    freq = ev["event_name"].value_counts().reset_index()
    freq.columns = ["event", "count"]
    return freq

def ab_summary(ev, users_df):
    conv_ev = "conversion" if "conversion" in ev["event_name"].unique() else "activation"
    conv = ev[ev["event_name"] == conv_ev].merge(users_df[["user_id", "ab_group"]], on="user_id", how="left")
    users_ab = users_df[users_df["ab_group"].isin(["A", "B"])]
    total = users_ab.groupby("ab_group")["user_id"].nunique()
    convs = conv.groupby("ab_group")["user_id"].nunique()
    df = pd.DataFrame({"Group": ["A", "B"],
                       "Total Users": [int(total.get("A", 0)), int(total.get("B", 0))],
                       "Conversions": [int(convs.get("A", 0)), int(convs.get("B", 0))]})
    df["Conversion Rate (%)"] = (df["Conversions"] / df["Total Users"] * 100).round(2)
    return df, conv_ev

# -----------------------------------------------------------------------
# 📊 VISUAL ANALYTICS LAYOUT
# Row 1: Funnel + DAU
# Row 2: Cohort view (Heatmap/Weekly/Daily) + Event Distribution
# -----------------------------------------------------------------------
st.markdown("## 📊 Visual analytics")

# ---------- Row 1 ----------
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("Funnel: signup → onboarding → activation")
    funnel_df = compute_funnel(events_f)
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df["step"],
        x=funnel_df["users"],
        text=funnel_df["pct_vs_first"].map(lambda v: f"{v:.1f}%"),
        textposition="inside",
        marker=dict(color=["#3DC1D3", "#2E86DE", "#5F27CD"]),
        connector=dict(line=dict(color="rgba(255,255,255,0.35)", dash="solid", width=1))
    ))
    fig_funnel.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_funnel, use_container_width=True)

with row1_col2:
    st.subheader("Daily Active Users (DAU)")
    dau_df = compute_dau(events_f)
    fig_dau = px.line(dau_df, x="date", y="dau", title="Daily Active Users")
    fig_dau.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dau, use_container_width=True)

# ---------- Row 2 ----------
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.subheader("👥 Retention cohorts")
    cohort_view_choice = st.radio(
        "View",
        ["Heatmap", "Weekly Table", "Daily Table"],
        index=0, horizontal=True, key="cohort_view_choice"
    )

    cohort_rate_week = compute_cohort(users_f, events_f, periods=8)   # weeks 0..7
    cohort_rate_day  = compute_cohort(users_f, events_f, periods=11)  # weeks 0..10

    if cohort_view_choice == "Heatmap":
        if not cohort_rate_week.empty:
            fig_cohort_heat = cohort_heatmap_figure(cohort_rate_week)
            st.plotly_chart(fig_cohort_heat, use_container_width=True)
        else:
            st.info("Not enough data to render retention heatmap.")
    elif cohort_view_choice == "Weekly Table":
        if not cohort_rate_week.empty:
            weekly_df = prepare_cohort_table(cohort_rate_week)
            fig_cohort_weekly = go.Figure(go.Table(
                header=dict(values=list(weekly_df.columns), fill_color="rgba(30,30,30,0.95)",
                            font=dict(color="white", size=12)),
                cells=dict(values=[weekly_df[c].tolist() for c in weekly_df.columns],
                           fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12))
            ))
            fig_cohort_weekly.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420,
                                            margin=dict(l=10, r=10, t=30, b=10), title="Weekly Cohort Table")
            st.plotly_chart(fig_cohort_weekly, use_container_width=True)
        else:
            st.info("Not enough data to render weekly cohort table.")
    else:  # Daily Table
        if not cohort_rate_day.empty:
            daily_df = prepare_cohort_table(cohort_rate_day)
            fig_cohort_daily = go.Figure(go.Table(
                header=dict(values=list(daily_df.columns), fill_color="rgba(30,30,30,0.95)",
                            font=dict(color="white", size=12)),
                cells=dict(values=[daily_df[c].tolist() for c in daily_df.columns],
                           fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12))
            ))
            fig_cohort_daily.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420,
                                           margin=dict(l=10, r=10, t=30, b=10), title="Daily Cohort Table")
            st.plotly_chart(fig_cohort_daily, use_container_width=True)
        else:
            st.info("Not enough data to render daily cohort table.")

with row2_col2:
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
# 🔀 A/B TESTING
# -----------------------------------------------------------------------
st.markdown("## 🔀 A/B testing")

ab_df, conv_event_used = ab_summary(events_f, users_f)
fig_ab_summary = None
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
        row_heights=[0.32, 0.26, 0.06, 0.36],
        vertical_spacing=0.02
    )
    # Row 1: summary table
    fig_ab_summary.add_trace(go.Table(
        header=dict(values=list(ab_df.columns), fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12), align="center"),
        cells=dict(values=[ab_df[c].tolist() for c in ab_df.columns], fill_color="rgba(20,20,20,0.9)",
                   font=dict(color="white", size=12), align="center")
    ), row=1, col=1)
    # Row 2: compact stats
    fig_ab_summary.add_trace(go.Table(
        header=dict(values=stats_header, fill_color="rgba(30,30,30,0.95)", font=dict(color="white", size=12), align="left"),
        cells=dict(values=stats_cells, fill_color="rgba(20,20,20,0.9)", font=dict(color="white", size=12), align="left")
    ), row=2, col=1)
    # Row 3: significance text
    if p_value < 0.05:
        note_text = f"✅ Statistically Significant (α=0.05). Variation {'B' if p_b > p_a else 'A'} is better."
        note_color = "#1E7E34"
    else:
        note_text = "⚠️ Not statistically significant at α=0.05."
        note_color = "#CCCCCC"
    fig_ab_summary.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text", text=[note_text],
                                        textfont=dict(color=note_color, size=13), showlegend=False), row=3, col=1)
    fig_ab_summary.update_xaxes(visible=False, row=3, col=1)
    fig_ab_summary.update_yaxes(visible=False, row=3, col=1)
    # Row 4: bar
    fig_ab_summary.add_trace(go.Bar(
        x=ab_df["Group"], y=ab_df["Conversion Rate (%)"],
        text=ab_df["Conversion Rate (%)"], textposition="auto",
        marker_color=["#1f77b4", "#ff7f0e"], showlegend=False
    ), row=4, col=1)
    fig_ab_summary.update_yaxes(showgrid=False, zeroline=False, row=4, col=1)
    fig_ab_summary.update_layout(title=f"A/B Summary, Stats & Conversion ({conv_event_used})",
                                 showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=800)
    st.plotly_chart(fig_ab_summary, use_container_width=True)
else:
    st.info("No A/B data available.")

# -----------------------------------------------------------------------
# NARRATIVES (used in PDF)
# -----------------------------------------------------------------------
def narrative_funnel(funnel_df):
    if funnel_df is None or funnel_df.empty: return "No funnel data is available."
    baseline_step = str(funnel_df.iloc[0]["step"])
    baseline_users = int(funnel_df.iloc[0]["users"])
    total_steps = len(funnel_df)
    tail = funnel_df.iloc[1:].sort_values("pct_vs_first").head(1)
    drop_sent = (f"The largest drop relative to the baseline occurs at '{str(tail.iloc[0]['step'])}', reached by {float(tail.iloc[0]['pct_vs_first']):.1f}% of baseline users.") if not tail.empty else ""
    top = funnel_df.sort_values("events", ascending=False).head(1)
    top_sent = (f"By total event count, the most active step is '{str(top.iloc[0]['step'])}' with {int(top.iloc[0]['events'])} events.") if not top.empty else ""
    return f"The funnel includes {total_steps} steps and uses '{baseline_step}' as the baseline, reached by {baseline_users} distinct users (100%). {drop_sent} {top_sent}".strip()

def narrative_dau(dau_df):
    if dau_df is None or dau_df.empty: return "No DAU data is available for this period."
    dmin, dmax = dau_df["date"].min(), dau_df["date"].max()
    avg = float(dau_df["dau"].mean())
    peak_row = dau_df.loc[dau_df["dau"].idxmax()]
    peak_date = peak_row["date"]; peak_val = int(peak_row["dau"])
    return f"DAU from {dmin} to {dmax} averages {avg:.1f}. Peak day: {peak_date} with {peak_val} users."

def narrative_events(freq_df):
    if freq_df is None or freq_df.empty: return "No event distribution data is available."
    total = int(freq_df["count"].sum())
    if total == 0: return "Events were recorded, but counts sum to zero."
    top = freq_df.sort_values("count", ascending=False).head(3)
    tops = []
    for _, r in top.iterrows():
        share = (r["count"] / total * 100.0) if total > 0 else 0
        tops.append(f"'{r['event']}' ({r['count']}, {share:.1f}% of total)")
    return f"Total actions: {total}. Top events: {'; '.join(tops)}."

def narrative_ab(ab_df):
    if ab_df is None or ab_df.empty: return "No A/B test summary is available."
    total_users = int(ab_df["Total Users"].sum())
    total_conv = int(ab_df["Conversions"].sum())
    overall_rate = (total_conv / total_users * 100.0) if total_users > 0 else 0.0
    best = ab_df.sort_values("Conversion Rate (%)", ascending=False).head(1)
    worst = ab_df.sort_values("Conversion Rate (%)", ascending=True).head(1)
    parts = [f"Overall conversion: {overall_rate:.1f}% ({total_conv}/{total_users})."]
    if not best.empty:
        b = best.iloc[0]; parts.append(f"Best group: {b['Group']} at {b['Conversion Rate (%)']:.1f}% ({b['Conversions']}/{b['Total Users']}).")
    if not worst.empty:
        w = worst.iloc[0]; parts.append(f"Weakest group: {w['Group']} at {w['Conversion Rate (%)']:.1f}% ({w['Conversions']}/{w['Total Users']}).")
    return " ".join(parts)

# -----------------------------------------------------------------------
# PDF EXPORT (Matplotlib fallback) — charts INCLUDED
# -----------------------------------------------------------------------
def _draw_paragraph(c, text, x_mm, y_start_mm, max_width_mm=180, color=colors.black, font_size=11, leading=14):
    style = ParagraphStyle(name="NarrativeStyle", fontName="Helvetica", fontSize=font_size, leading=leading, textColor=color, alignment=TA_LEFT)
    para = Paragraph(text, style)
    w, h = para.wrap(max_width_mm * mm, A4[1])
    para.drawOn(c, x_mm * mm, y_start_mm * mm - h)
    return (y_start_mm * mm - h) / mm

def mpl_save_png(func):
    fig = func()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def mpl_funnel_png(df: pd.DataFrame):
    def _draw():
        fig, ax = plt.subplots(figsize=(6.5, 4))
        steps = df["step"].tolist(); users = df["users"].tolist(); pct = df["pct_vs_first"].tolist()
        colors_ = ["#3DC1D3", "#2E86DE", "#5F27CD"]
        ax.barh(steps[::-1], users[::-1], color=colors_[::-1])
        for i, v in enumerate(users[::-1]):
            ax.text(v * 1.01, i, f"{pct[::-1][i]:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Distinct users"); ax.set_title("Funnel: signup → onboarding → activation")
        ax.grid(axis="x", alpha=0.2)
        return fig
    return mpl_save_png(_draw)

def mpl_dau_png(df: pd.DataFrame):
    def _draw():
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(df["date"], df["dau"], color="#1f77b4", lw=2)
        ax.set_title("Daily Active Users (DAU)")
        ax.set_xlabel("Date"); ax.set_ylabel("DAU")
        ax.grid(alpha=0.2); fig.autofmt_xdate()
        return fig
    return mpl_save_png(_draw)

def mpl_events_png(freq_df: pd.DataFrame):
    def _draw():
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.pie(freq_df["count"], labels=freq_df["event"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Event Distribution")
        return fig
    return mpl_save_png(_draw)

def mpl_cohort_heat_png(rate_df: pd.DataFrame):
    def _draw():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        z = rate_df.values * 100.0
        im = ax.imshow(z, aspect="auto", cmap="viridis")
        ax.set_title("Retention Cohorts (%)")
        ax.set_xlabel("Weeks since signup"); ax.set_ylabel("Cohort week")
        ax.set_xticks(range(rate_df.shape[1]))
        ax.set_xticklabels([f"W{int(c)}" for c in rate_df.columns], rotation=0)
        ax.set_yticks(range(rate_df.shape[0]))
        ax.set_yticklabels([str(getattr(i, "date", lambda: str(i))()) if hasattr(i, "date") else str(i) for i in rate_df.index])
        fig.colorbar(im, ax=ax, label="Retention %")
        return fig
    return mpl_save_png(_draw)

def mpl_cohort_table_png(df_table: pd.DataFrame, title="Cohort Table"):
    def _draw():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.axis("off")
        table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9)
        table.scale(1, 1.2)
        ax.set_title(title)
        return fig
    return mpl_save_png(_draw)

def export_dashboard_pdf_with_summaries(figs_dict) -> bytes:
    temp_paths = []
    # Matplotlib fallback only (reliable on Streamlit Cloud)
    for key, section in figs_dict.items():
        data = section.get("data")
        extra = section.get("extra", {})  # for notes, titles etc.
        image_path = None
        try:
            if key == "funnel" and isinstance(data, pd.DataFrame) and not data.empty:
                image_path = mpl_funnel_png(data)
            elif key == "dau" and isinstance(data, pd.DataFrame) and not data.empty:
                image_path = mpl_dau_png(data)
            elif key == "events" and isinstance(data, pd.DataFrame) and not data.empty:
                image_path = mpl_events_png(data)
            elif key == "cohorts":
                mode = extra.get("mode", "heatmap")
                if mode == "heatmap" and isinstance(data, pd.DataFrame) and not data.empty:
                    image_path = mpl_cohort_heat_png(data)
                elif mode in ("weekly", "daily") and isinstance(extra.get("table_df"), pd.DataFrame) and not extra.get("table_df").empty:
                    title = "Weekly Cohort Table" if mode == "weekly" else "Daily Cohort Table"
                    image_path = mpl_cohort_table_png(extra["table_df"], title=title)
            elif key == "ab" and isinstance(extra, dict):
                ab_df_for_pdf = extra.get("ab_df", pd.DataFrame())
                note_text = extra.get("note_text", "")
                note_color = extra.get("note_color", "#777777")
                def _draw():
                    fig, ax = plt.subplots(figsize=(6.5, 4.0))
                    if not ab_df_for_pdf.empty:
                        x = ab_df_for_pdf["Group"].tolist()
                        y = ab_df_for_pdf["Conversion Rate (%)"].tolist()
                        ax.bar(x, y, color=["#1f77b4", "#ff7f0e"])
                        for i, v in enumerate(y):
                            ax.text(i, v + (max(y) * 0.02 if max(y) > 0 else 0.5), f"{v:.2f}%", ha="center", fontsize=10)
                        ax.set_title("A/B Conversion Rates"); ax.set_ylabel("Conversion Rate (%)")
                        ax.grid(axis="y", alpha=0.2)
                    fig.text(0.5, 0.02, note_text, color=note_color, ha="center", fontsize=11)
                    return fig
                image_path = mpl_save_png(_draw)
        except Exception:
            image_path = None

        section["image_path"] = image_path
        if image_path: temp_paths.append(image_path)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    for _, section in figs_dict.items():
        title = section.get("title", "")
        image_path = section.get("image_path")
        narrative_text = section.get("narrative_text", "")

        c.setFont("Helvetica-Bold", 16); c.setFillColor(colors.black)
        c.drawString(20 * mm, A4[1] - 20 * mm, title)

        y = A4[1] - 25 * mm
        if image_path and os.path.exists(image_path):
            c.drawImage(image_path, 15 * mm, y - 110 * mm, width=180 * mm, height=110 * mm,
                        preserveAspectRatio=True, anchor='n')
            y_after_img_mm = (y - 110 * mm) / mm
        else:
            c.setFont("Helvetica", 12)
            c.drawString(20 * mm, y - 15 * mm, "No chart image available for this section.")
            y_after_img_mm = (y - 20 * mm) / mm

        if narrative_text:
            _draw_paragraph(c, text=narrative_text, x_mm=20, y_start_mm=y_after_img_mm - 10,
                            max_width_mm=170, color=colors.black, font_size=11, leading=14)
        c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    for p in temp_paths:
        try: os.remove(p)
        except: pass
    return pdf_bytes

# -----------------------------------------------------------------------
# Build narratives + Export button
# -----------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📥 Export dashboard as PDF (chart + narrative below)")

if st.button("📄 Generate PDF", key="btn_gen_pdf"):
    funnel_df_for_pdf = compute_funnel(events_f)
    dau_df_for_pdf    = compute_dau(events_f)
    freq_df_for_pdf   = compute_event_freq(events_f)
    ab_df_for_pdf, conv_event_used_for_pdf = ab_summary(events_f, users_f)

    # Narratives
    funnel_text = narrative_funnel(funnel_df_for_pdf)
    dau_text    = narrative_dau(dau_df_for_pdf)
    events_text = narrative_events(freq_df_for_pdf)
    ab_text     = narrative_ab(ab_df_for_pdf)

    # A/B note for PDF
    if not ab_df_for_pdf.empty and ab_df_for_pdf["Total Users"].sum() > 0:
        a = ab_df_for_pdf[ab_df_for_pdf["Group"] == "A"].iloc[0]
        b = ab_df_for_pdf[ab_df_for_pdf["Group"] == "B"].iloc[0]
        n_a, n_b = int(a["Total Users"]), int(b["Total Users"])
        s_a, s_b = int(a["Conversions"]), int(b["Conversions"])
        p_a = s_a / n_a if n_a > 0 else 0.0
        p_b = s_b / n_b if n_b > 0 else 0.0
        z, p_value, _, _ = proportions_z_test(s_a, n_a, s_b, n_b)
        if p_value < 0.05:
            note_text  = f"✅ Statistically Significant (α=0.05). Variation {'B' if p_b > p_a else 'A'} is better."
            note_color = "#1E7E34"
        else:
            note_text  = "⚠️ Not statistically significant at α=0.05."
            note_color = "#777777"
    else:
        note_text, note_color = "No A/B data available.", "#777777"

    # Cohort section for PDF depends on the selected view
    cohort_mode = st.session_state.get("cohort_view_choice", "Heatmap")
    if cohort_mode == "Heatmap":
        cohort_pdf_rate = compute_cohort(users_f, events_f, periods=8)
        cohort_extra = {"mode": "heatmap"}
        cohort_narr_text = "Cohort heatmap view."
    elif cohort_mode == "Weekly Table":
        cohort_pdf_rate = compute_cohort(users_f, events_f, periods=8)
        weekly_df_pdf = prepare_cohort_table(cohort_pdf_rate)
        cohort_extra = {"mode": "weekly", "table_df": weekly_df_pdf}
        cohort_narr_text = "Weekly cohort table view."
    else:  # Daily Table
        cohort_pdf_rate = compute_cohort(users_f, events_f, periods=11)
        daily_df_pdf = prepare_cohort_table(cohort_pdf_rate)
        cohort_extra = {"mode": "daily", "table_df": daily_df_pdf}
        cohort_narr_text = "Daily cohort table view."

    figs_dict = {
        "funnel":  {"title": "Funnel: signup → onboarding → activation",
                    "data": funnel_df_for_pdf, "narrative_text": funnel_text},
        "dau":     {"title": "Daily Active Users (DAU)",
                    "data": dau_df_for_pdf, "narrative_text": dau_text},
        "cohorts": {"title": "Retention Cohorts",
                    "data": cohort_pdf_rate, "extra": cohort_extra, "narrative_text": cohort_narr_text},
        "events":  {"title": "Event Distribution",
                    "data": freq_df_for_pdf, "narrative_text": events_text},
        "ab":      {"title": f"A/B Testing ({conv_event_used_for_pdf})",
                    "data": None, "extra": {"ab_df": ab_df_for_pdf, "note_text": note_text, "note_color": note_color},
                    "narrative_text": ab_text},
    }

    pdf_bytes = export_dashboard_pdf_with_summaries(figs_dict)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="vizsprints_dashboard.pdf",
                       mime="application/pdf", key="btn_download_pdf")
