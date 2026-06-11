# config.py
# ASCII-only.

# Models: (name, path)
MODELS = [
    ("TFLITE_INT8", "/home/dell/venv/best_saved_model/best_int8.tflite"),
    ("TFLITE_FP16", "/home/dell/venv/best_saved_model/best_float16.tflite"),
    ("TFLITE_FP32", "/home/dell/venv/best_saved_model/best_float32.tflite"),
    ("ONNX_FP16",   "/home/dell/venv/bestfp16.onnx"),
    ("ONNX_FP32",   "/home/dell/venv/bestfp32.onnx"),
    ("NCNN_FP16",   "/home/dell/venv/best_ncnn_modelfp16"),
    ("NCNN_FP32",   "/home/dell/venv/best_ncnn_model"),
]

CAM_ID = 0
IMG_SIZE = 640

DB_PATH = "./data_benchmark.json"

# Threads
CONTEXT_STABLE_SEC = 1.0
MIN_SWITCH_SEC = 2.0

# Quantize
Q_STEP = 5

# Light metrics (your bins)
ANALYZE_W, ANALYZE_H = 160, 120
DARK_PIXEL_Y = 35
EMA_ALPHA = 0.2
MEAN_BINS = [140, 95, 60]
DARK_BINS = [0.20, 0.45, 0.65]
STABLE_HOLD_SEC = 1.0

# Edge density
EDGE_W, EDGE_H = 160, 120
CANNY_T1, CANNY_T2 = 80, 160
EDGE_EMA_ALPHA = 0.2

# Pin coarse bins
PIN_CRIT = 10
PIN_LOW = 25
PIN_MID = 60

# Serial power
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 9600
SERIAL_TIMEOUT = 0.2
SERIAL_MAX_AGE_SEC = 1.0
POWER_MAX_SAMPLES = 5000

# Benchmark worker
BENCH_SEC_PER_MODEL = 8.0          # tune: 6..15 sec/model
BENCH_SLEEP_SEC = 0.01             # yield realtime
BENCH_TOPK = None                  # None=all, or integer 2/3
BENCH_COOLDOWN_SEC = 5.0           # prevent spamming same context jobs

# Quality metric settings (mist+conf)
# Mistake/drop simplified
IOU_MATCH_THRESH = 0.30
DISAPPEAR_K = 1
COOLDOWN_MS = 250
EXPAND_RATIO = 0.30
DIFF_THRESH = 0.10
PATCH_SIZE = 96

# Conf stability (event based)
STABLE_CHG_THRESH = 0.015
DELTA_CONF_THRESH = 0.05

# Frame change gate for conf
CHG_W, CHG_H = 96, 54
CHANGE_EMA_ALPHA = 0.2
# config.py (ADD THESE)
# ASCII-only.

# Pin bins (percent)
PIN_CRIT_MAX = 20.0
PIN_LOW_MAX = 40.0
PIN_MED_MAX = 60.0
PIN_HIGH_MAX = 80.0
# FULL: > PIN_HIGH_MAX

# Hysteresis: to exit CRIT, require at least MED
PIN_EXIT_CRIT_MIN = 40.0   # MEDIUM threshold start (you can set 40 or 45)

# CRIT behavior
CRIT_DROP_DB_CONTEXTS = True   # True: free RAM by dropping db["contexts"] after building global cache
CRIT_FORCE_SAVE_DB = True      # save immediately when entering CRIT

# Adaptive resolution when pin CRIT
RES_ENABLE_ON_CRIT = True
RES_CONF_TH = 0.25
RES_TEST_FRAMES = 10
RES_MIN_DET_RATE = 0.30
RES_SHRINK_RATIO = 0.85
RES_MIN_W = 96
RES_BINARY_ITERS = 8
RES_SLEEP_SEC = 0.0
