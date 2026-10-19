"""NFR-02 / T10 response-time benchmark. python bench_latency.py --n 1000"""
import argparse, os, time, csv, statistics as st, requests
a = argparse.ArgumentParser(); a.add_argument("--n", type=int, default=1000); a = a.parse_args()
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
_creds = {"username": os.getenv("API_USER", "testuser"), "password": os.getenv("API_PASS", "Test1234!")}
requests.post(API + os.getenv("REGISTER_PATH", "/auth/register"), json={**_creds, "email": _creds["username"] + "@example.com"})
tok = requests.post(API + os.getenv("LOGIN_PATH", "/auth/login"), json=_creds).json()[os.getenv("TOKEN_KEY", "access_token")]
H = {"Authorization": "Bearer " + tok}; ts = []
for i in range(a.n):
    t = {"txn_id": f"BENCH-{i}", "type": "TRANSFER" if i % 2 else "CASH_OUT", "amount": 1000.0 + i, "hour": i % 24,
         "old_balance_org": 5000.0 + i, "old_balance_dest": 0.0, "name_dest": "C1"}
    s = time.perf_counter(); r = requests.post(API + os.getenv("PREDICT_PATH", "/predict"), json=t, headers=H, timeout=30)
    ts.append((time.perf_counter() - s, r.status_code))
lat = sorted(x for x, _ in ts); ok = sum(1 for _, c in ts if c == 200)
p95 = lat[int(0.95 * len(lat)) - 1]
print(f"requests {len(lat)}  ok {ok}\nmean {st.mean(lat):.3f}s  median {st.median(lat):.3f}s  p95 {p95:.3f}s  max {lat[-1]:.3f}s")
print(f"share under 3 s: {100*sum(x < 3 for x in lat)/len(lat):.1f}%")
with open("latency.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seconds", "status"]); w.writerows(ts)
