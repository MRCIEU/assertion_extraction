import os

# Get API key
GPT_KEY     = os.environ.get("GPT_KEY")
CLAUDE_KEY  = os.environ.get("CLAUDE_KEY")
LLMAPI_KEY  = os.environ.get("LLMAPI_KEY")

# Model Names
GPT_MODEL    = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"
LLMAPI_MODEL = "llama3-8b"

# File Paths
CSV_PATH = "../02_extract_results_CITATIONS/raw_abstracts.csv"
FIND_DIR     = "finding"
COMP_DIR = "completion"
ASSR_DIR    = "assertion"

BATCH_SIZE     = 1
MAX_TOKENS     = 2048
TEMPERATURE    = 0.0