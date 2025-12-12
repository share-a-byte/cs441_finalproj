import pandas as pd
import subprocess
import shlex

df = pd.read_csv("finalized.csv")
urls = df["LINK"].dropna().tolist()

with open("urls.txt", "w") as f:
    for u in urls:
        f.write(u + "\n")
cmd = (
    "yt-dlp "
    "--sleep-interval 0.3 --max-sleep-interval 0.8 "
    '--print "%(id)s,%(duration)s" -a urls.txt'
)

args = shlex.split(cmd)

process = subprocess.Popen(
    args,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate()

if stderr.strip():
    print("\nErrors/Warnings:\n")
    print(stderr)

final_df = pd.DataFrame(columns=["id", "duration"])
stdout = stdout.split()
for res in stdout:
    in_id, duration = res.split(",")
    final_df.loc[len(final_df)] = [in_id, duration]

final_df.to_csv("FINAL.csv")