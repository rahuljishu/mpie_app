# 📐 MPIE – IITJ • Streamlit dashboard
# ------------------------------------
# • Pulls the trained RL agent repo from Hugging Face once
# • Runs analyze.py on any CSV/TXT a user uploads
# • Shows reward metrics, an interactive bar-chart, and lets the user
#   download the full raw report
# • Requires only streamlit, matplotlib, huggingface_hub

import os, re, json, subprocess, tempfile, datetime
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

# ─────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="MPIE – IITJ",
                   page_icon="📐",
                   layout="wide",
                   initial_sidebar_state="collapsed")

MODEL_REPO  = "rahuljishu/mpie_iitj"
CACHE_DIR   = "hf_cache"               # in-container cache
ANALYZE_REL = "analyze.py"             # relative path in repo

# ─────────────────────────────────────────────────────────
# 1. Download model repo once (cached across reruns)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading model…")
def download_model():
    snapshot_download(repo_id=MODEL_REPO,
                      local_dir=CACHE_DIR,
                      local_dir_use_symlinks=False,
                      resume_download=True)
    return os.path.join(CACHE_DIR, ANALYZE_REL)

ANALYZE = download_model()

# ─────────────────────────────────────────────────────────
# 2. Helper : run analyze.py and return stdout
# ─────────────────────────────────────────────────────────
def run_agent(path: str) -> str:
    result = subprocess.run(
        ["python", ANALYZE, "--data", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout

# ─────────────────────────────────────────────────────────
# 3. UI
# ─────────────────────────────────────────────────────────
st.title("MPIE – IITJ")
st.caption("Mathematical Pattern Discovery Engine  |  RL-powered structure finder")

uploaded = st.file_uploader("Upload a CSV or TXT dataset", type=["csv", "txt"])

if uploaded:
    # save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.read())
        data_path = tmp.name

    with st.spinner("Analyzing…"):
        try:
            raw = run_agent(data_path)
        except Exception as e:
            st.error(f"Agent crashed:\n\n{e}")
            st.stop()

    # ── parse minimal fields ─────────────────────────────
    best_col   = re.search(r"Best column:\s*(.*)", raw).group(1).strip()
    reward_row = re.search(r"Reward break-down:\s*({.*})", raw).group(1)
    reward     = json.loads(reward_row.replace("'", '"'))

    rel_block  = re.search(r"Top relations:\s*([\s\S]*?)\n\n", raw).group(1)
    relations  = [
        re.match(r"(.*?)→(.*?)\s*deg=\d+\s*R²=([\d.]+)", l).groups()
        for l in rel_block.strip().splitlines()
    ]

    # ── layout columns ───────────────────────────────────
    st.subheader(f"🔍 Best explanatory column : `{best_col}`")

    cols = st.columns(len(reward))
    for (k,v), c in zip(reward.items(), cols):
        c.metric(k.capitalize(), f"{v:.3f}")

    # ── bar chart ────────────────────────────────────────
    if relations:
        rel_labels = [f"{a.strip()}→{b.strip()}" for a,b,_ in relations]
        rel_vals   = [float(r2) for *_,r2 in relations]

        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(rel_labels, rel_vals, color="#4F7DF5")
        ax.invert_yaxis()
        ax.set_xlabel("R²")
        ax.set_title("Top discovered relations")
        st.pyplot(fig)

    st.download_button("Download full raw report",
                       raw,
                       file_name=f"mpie_report_{datetime.datetime.now():%Y%m%d%H%M}.txt")

    st.success("Analysis complete!")

else:
    st.info("👈 Upload a dataset to begin.")
