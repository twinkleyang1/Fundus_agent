"""Central configuration for the fundus multi-agent system."""
import os

# --- Environments ---
OPENMMLAB_PYTHON = "/data/twinkle/anaconda3/envs/openmmlab/bin/python"
NANOMEDAGENT_PYTHON = "/data/twinkle/anaconda3/envs/nanomedagent/bin/python"

# --- Paths ---
PROJECT_ROOT = "/home/twinkle/app/LLM_paper"
FUNDUS_AGENT_ROOT = os.path.join(PROJECT_ROOT, "Fundus_agent")
IMG_DIR = os.path.join(PROJECT_ROOT, "Dataset/ODIR-5K/preprocessed_images")
CSV_PATH = os.path.join(PROJECT_ROOT, "Dataset/ODIR-5K/full_df.csv")
MMSEG_ROOT = "/data/twinkle/app/mmsegmentation"
MMSEG_WORK_DIR = os.path.join(MMSEG_ROOT, "zhang_work_dirs")

# --- Model paths ---
VL_MODELS = {
    "Qwen2.5-VL-7B-Instruct": os.path.join(PROJECT_ROOT, "LLM/Qwen2.5-VL-7B-Instruct"),
    "gemma-4-26B-A4B-it": os.path.join(PROJECT_ROOT, "LLM/gemma-4-26B-A4B-it"),
}

# --- Disease classes ---
CLASS_NAMES = {
    "N": "正常", "D": "糖尿病视网膜病变", "G": "青光眼",
    "C": "白内障", "A": "老年性黄斑变性", "H": "高血压视网膜病变",
    "M": "近视", "O": "其他疾病或异常",
}

# --- Quality thresholds ---
QUALITY_PASS_THRESHOLD = 0.5

# --- Clinical thresholds ---
GLAUCOMA_CDR_THRESHOLD = 0.6
HYPERTENSIVE_AVR_THRESHOLD = 2.0 / 3.0
MYOPIA_TILT_DEGREES = 15.0
VL_MODEL_TIMEOUT = 30  # seconds
