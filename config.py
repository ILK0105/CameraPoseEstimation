import os

# Folder that contains train/, val/, test/, camera/ (the downloaded competition data).
# Change this if you put the data somewhere else (e.g. on Colab or a lab server).
DATA_ROOT = "data"

OUTPUT_DIR = "output"
CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "netvlad_final.pth")
