import torch



SEED = 2026
TARGET_RATE = 16000
LEARNING_RATE = 0.001
N_CLASSES = 10
N_BEST_RAG = 3

FRAMES_PER_SECOND = 50  
AUDIO_SECONDS = 10    

HEAD_TYPES = ("linear", "mlp")


DEVICE = "cuda"

MODELS_CED = ["mispeech/ced-tiny", "mispeech/ced-mini", "mispeech/ced-small", "mispeech/ced-base"]

MODELS_WHISPER = ["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium" ,"openai/whisper-base", "openai/whisper-large-v2"]

MODELS = ["bookbot/distil-ast-audioset", "MIT/ast-finetuned-audioset-10-10-0.4593", "facebook/w2v-bert-2.0"]


OPTIMIZER = torch.optim.AdamW

SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2


EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.001

RESULTS_FILE = "evaluator_results.csv"
