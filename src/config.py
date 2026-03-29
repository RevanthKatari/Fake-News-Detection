import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR  = os.path.join(PROJECT_DIR, "saved_models")
CACHE_DIR   = os.path.join(PROJECT_DIR, "cache")

DATASET_PATH     = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "sentence_embeddings.npy")
LABELS_CACHE     = os.path.join(CACHE_DIR, "labels.npy")

# ── Dataset columns ───────────────────────────────────────────
TEXT_COLUMN  = "text"
TITLE_COLUMN = "title"
LABEL_COLUMN = "label"

# ── Embedding ─────────────────────────────────────────────────
LLM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM  = 384
MAX_SENTENCES  = 16          # sentences per article

# ── Model architecture ────────────────────────────────────────
HIDDEN_DIM   = 128
NUM_CLASSES  = 2
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_FILTERS  = 64
KERNEL_SIZES = [3, 5, 7]

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
NUM_EPOCHS    = 20
PATIENCE      = 5
RANDOM_SEED   = 42

# ── Splits ────────────────────────────────────────────────────
TEST_SIZE = 0.15
VAL_SIZE  = 0.15
