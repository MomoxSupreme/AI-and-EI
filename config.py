import torch



SEED = 2026
MODEL_NAME = "mispeech/ced-base"
TARGET_RATE = 16000
LEARNING_RATE = 0.001
N_CLASSES = 10
N_BEST_RAG = 3

DEVICE = "cpu"

MODELS_CED = ["mispeech/ced-tiny", "mispeech/ced-mini", "mispeech/ced-small", "mispeech/ced-base"]

MODELS_WHISPER = ["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium" ,"openai/whisper-base", "openai/whisper-large-v2"]

MODELS = ["bookbot/distil-ast-audioset", "MIT/ast-finetuned-audioset-10-10-0.4593", "facebook/w2v-bert-2.0"]


OPTIMIZER = torch.optim.AdamW

SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau
