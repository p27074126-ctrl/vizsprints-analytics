from fastapi import FastAPI
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

DB = "vizsprints.db"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def query_db(q, params=()):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(q, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

@app.get("/api/events/stats")
def stats():
    total_users = query_db("SELECT COUNT(*) as c FROM users")[0]['c']
    total_events = query_db("SELECT COUNT(*) as c FROM events")[0]['c']
    return {"total_users": total_users, "total_events": total_events}

# funnel endpoint example
@app.get("/api/funnel")
def funnel():
    # Count distinct users doing signup, onboarding_complete, activation.
    q = """
    SELECT
      SUM(CASE WHEN event_name='signup_success' THEN 1 ELSE 0 END) as signup_events,
      COUNT(DISTINCT CASE WHEN event_name='signup_success' THEN user_id END) as signup_users,
      COUNT(DISTINCT CASE WHEN event_name='onboarding_complete' THEN user_id END) as onboarding_users,
      COUNT(DISTINCT CASE WHEN event_name='activation' THEN user_id END) as activation_users
    FROM events
    """
    res = query_db(q)[0]
    return res