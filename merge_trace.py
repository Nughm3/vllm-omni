import gzip
import json
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path) as f:
        return json.load(f)


stage0 = load(sys.argv[1])
stage1 = load(sys.argv[2])

merged = {"traceEvents": stage0["traceEvents"] + stage1["traceEvents"]}
with open("merged_trace.json", "w") as f:
    json.dump(merged, f)
