# config.py
from pathlib import Path

# --- LLM Configuration ---
# LLM_MODEL = "qwen3:32b"
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.7

# --- File Paths & Directories ---
DATA_ROOT_DIR = Path(
    "/home/pia/projects/llmTubo/llmRecom/data"
)
TUBO_EXCEL_FILE_PATH = DATA_ROOT_DIR / "tubo-DLBCL-v2.xlsx"
REPORT_DIR = "generated_report"
REPORT_FILE_TYPE = "md"

# --- Clinical Trials API ---
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_PAGE_SIZE = 20
REQUESTS_TIMEOUT = 30
MAX_LOCATIONS_TO_DISPLAY_PER_STUDY = 3

# --- Geocoding ---
GEOCODER_USER_AGENT = "llm_tumorboard_app"
GEOCODE_TIMEOUT = 10

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s - %(message)s"

# --- LLM Interaction Logging ---
LLM_INTERACTIONS_CSV_FILE = "/home/pia/projects/llmTubo/llmRecom/logs/llm_interactions.csv"
HUMAN_EVAL_DATA_DIR = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation"
HUMAN_EVAL_JSON_FILE = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/all_evaluation_cases.json"
