from pathlib import Path
path = Path("dc_dopdgd.py")
lines = path.read_text(encoding="utf-8").splitlines()
start = None
for idx, line in enumerate(lines):
    if line.strip().startswith("return dict(") and "bits_history" in "".join(lines[idx:idx+6]):
        # ensure we are in Eq18 block by scanning upwards for 'Eq.18'
        for back in range(idx, max(idx-200, -1), -1):
            if lines[back].strip().startswith("# Eq.18"):
                start = idx
                break
        if start is not None:
            break
if start is None:
    raise SystemExit("return block for eq18 not found")
end = None
for j in range(start + 1, len(lines)):
    if lines[j].strip() == ")":
        end = j
        break
if end is None:
    raise SystemExit("unterminated return block")
replacement = [
    "    return dict(",
    "        regret_per_jT=regret_per_jT,",
    "        g_pre_perstep=g_pre_perstep,",
    "        g_post_perstep=g_post_perstep,",
    "        pre_pos=pre_pos,",
    "        post_pos=post_pos,",
    "        pre_cum=pre_cum,",
    "        post_cum=post_cum,",
    "        violation_eval_cumsum=np.cumsum(post_pos, axis=0),",
    "        bits_history=bits_history,",
    "    )",
]
lines = lines[:start] + replacement + lines[end + 1:]
path.write_text("\n".join(lines), encoding="utf-8")
