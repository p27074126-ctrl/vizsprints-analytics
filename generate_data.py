
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, json

# ---------------------------
# Config
# ---------------------------
NUM_USERS = 1000
START_DATE = datetime(2023, 9, 1)     # choose your historical start
END_DATE   = datetime.now()            # dynamic: always up to now (or set a fixed date if you prefer)

random.seed(42)
np.random.seed(42)

# ---------------------------
# Helper
# ---------------------------
def clamp(dt: datetime) -> datetime:
    """Keep dt within [START_DATE, END_DATE]."""
    if dt < START_DATE:
        return START_DATE
    if dt > END_DATE:
        return END_DATE
    return dt

# ---------------------------
# Generate Users
# ---------------------------
def gen_users(n: int):
    user_ids = [f"u_{1000 + i}" for i in range(n)]
    joined_dates = [
        clamp(START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days),
                                     seconds=random.randint(0, 86400)))
        for _ in range(n)
    ]
    devices   = np.random.choice(['Mobile','Desktop','Tablet'], size=n, p=[0.65,0.30,0.05])
    countries = np.random.choice(['IN','US','GB','DE','AU','BR','CA'], size=n,
                                 p=[0.40,0.20,0.10,0.08,0.06,0.10,0.06])
    subs      = np.random.choice(['Free','Premium'], size=n, p=[0.80,0.20])
    ab_group  = np.random.choice(['A','B'], size=n, p=[0.50,0.50])

    rows = [{
        "user_id": uid,
        "joined_at": jd.isoformat(),
        "device": dev,
        "country": c,
        "subscription_status": s,
        "ab_group": g
    } for uid, jd, dev, c, s, g in zip(user_ids, joined_dates, devices, countries, subs, ab_group)]

    return pd.DataFrame(rows)

# ---------------------------
# Generate Events
# ---------------------------
def gen_events(users_df: pd.DataFrame):
    events = []
    event_id = 1

    for _, user in users_df.iterrows():
        uid     = user['user_id']
        joined  = clamp(datetime.fromisoformat(user['joined_at']))

        # Signup
        signup_time = clamp(joined + timedelta(seconds=random.randint(30, 600)))
        events.append({
            "event_id": f"e_{event_id}", "user_id": uid, "event_name": "signup_success",
            "timestamp": signup_time.isoformat(),
            "metadata": json.dumps({"source": random.choices(["ads","organic","referral"], [0.3,0.6,0.1])[0]})
        }); event_id += 1

        # Onboarding (60%)
        if random.random() < 0.6:
            t = clamp(signup_time + timedelta(minutes=random.randint(1, 60)))
            events.append({"event_id": f"e_{event_id}", "user_id": uid, "event_name": "onboarding_complete",
                           "timestamp": t.isoformat(), "metadata": json.dumps({})}); event_id += 1

        # Activation (30%)
        if random.random() < 0.30:
            t = clamp(signup_time + timedelta(hours=random.randint(1, 72)))
            events.append({"event_id": f"e_{event_id}", "user_id": uid, "event_name": "activation",
                           "timestamp": t.isoformat(), "metadata": json.dumps({})}); event_id += 1

        # Early weekly activity
        for week_ix, p in enumerate([0.8,0.4,0.25,0.15,0.1]):
            if random.random() < p:
                for _ in range(random.randint(1, 3)):
                    offset_days = random.randint(week_ix*7, week_ix*7+6)
                    t = clamp(joined + timedelta(days=offset_days,
                                                 hours=random.randint(0,23),
                                                 minutes=random.randint(0,59),
                                                 seconds=random.randint(0,59)))
                    ev_name = random.choices(
                        ['view_dashboard','start_project','view_item','search','comment','like','logout'],
                        weights=[0.25,0.15,0.20,0.15,0.08,0.10,0.07]
                    )[0]
                    events.append({"event_id": f"e_{event_id}", "user_id": uid, "event_name": ev_name,
                                   "timestamp": t.isoformat(), "metadata": json.dumps({})})
                    event_id += 1

        # Conversion (A/B bias)
        conv_prob = 0.10 if user['ab_group'] == 'A' else 0.15
        if random.random() < conv_prob:
            t = clamp(joined + timedelta(days=random.randint(1,30), hours=random.randint(0,23)))
            events.append({"event_id": f"e_{event_id}", "user_id": uid, "event_name": "conversion",
                           "timestamp": t.isoformat(), "metadata": json.dumps({"ab_group": user['ab_group']})})
            event_id += 1

    events_df = pd.DataFrame(events).sample(frac=1, random_state=42).reset_index(drop=True)
    return events_df

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    users  = gen_users(NUM_USERS)
    print(f"Generated {len(users)} users")
    events = gen_events(users)
    print(f"Generated {len(events)} events")

    users.to_csv("users.csv", index=False)
    events.to_csv("events.csv", index=False)
    print("Saved users.csv and events.csv")

    # ✅ Date coverage print
    e_ts = pd.to_datetime(events['timestamp'], errors='coerce')