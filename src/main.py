import pandas as pd
import subprocess
import shlex

# ---------------------
# 1. Load your dataframe
# ---------------------
df = pd.read_csv("/home/vanir2/cs441/src/video_clipping/finalized.csv")          # or your source
urls = df["LINK"].dropna().tolist()     # pick the correct column name

# ---------------------
# 2. Write URLs to a text file
# ---------------------
with open("urls.txt", "w") as f:
    for u in urls:
        f.write(u + "\n")

# ---------------------
# 3. Build yt-dlp command
# ---------------------
# Using cookies + small sleep interval (recommended)
cmd = (
    "yt-dlp "
    "--sleep-interval 0.5 --max-sleep-interval 1 "
    '--print "%(id)s,%(duration)s" -a urls.txt'
)

# Turn command into subprocess-safe args
args = shlex.split(cmd)

# ---------------------
# 4. Run yt-dlp with subprocess
# ---------------------
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

duration_map = {}
for line in stdout.splitlines():
    if "," in line:
        vid, dur = line.split(",", 1)
        duration_map[vid] = int(dur)

# create a column in df for durations
df["duration_sec"] = df["LINK"].apply(
    lambda x: duration_map.get(x.split("v=")[-1], None)
)

df.to_csv("FINAL.csv")