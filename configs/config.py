from dataclasses import dataclass
from pathlib import Path

@dataclass
class PrepConfig:
    # inputs
    BLANK_PATH: Path = Path("data/raw/blank/blank_4.png")
    WRITTEN_PATH: Path = Path("data/raw/written/written_4_C0.png")
    DIGITAL_IMG: Path = Path("data/processed/digital_png/4.png")  # from gcode→png
    GCODE_PATH: Path | None = Path("data/raw/digital/4.gcode")    # optional; for metadata

    # outputs
    OUT_DIR: Path = Path("data/processed/ML_processed_image")
    OUT_IDX: int = 1                                              # saved as {OUT_IDX}.png
    META_TSV: Path = Path("data/processed/metadata.tsv")

    # metadata fields (match your sample order)
    ROUND: str = "Round 1 Complete"
    JOB_GROUP: str = "Job 3"
    JOB_NAME: str = "Job_3_C5_envelopes_black_perma"
    N_INDEX: int = 4                                              # original card index

    # preprocessing
    GAUSS_K: int = 9                                              # odd kernel size
    THRESH_DIFF: int = 50                                         # binarize threshold on diffs

# single export used by scripts
CONFIG = PrepConfig()
