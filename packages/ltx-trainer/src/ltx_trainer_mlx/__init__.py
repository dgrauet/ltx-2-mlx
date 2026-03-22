"""ltx-trainer-mlx: LTX-2 training toolkit for Apple Silicon via MLX."""

import logging

logger = logging.getLogger("ltx_trainer_mlx")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
