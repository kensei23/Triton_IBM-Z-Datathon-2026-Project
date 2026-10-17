from pathlib import Path

from configs import settings
from llama_cpp import Llama


def load_llm():
    """Load the active granite model using the values from settings.json."""
    path = Path(settings["model_dir"]) / settings["active_model"]
    return Llama(
        model_path=str(path),
        n_ctx=settings["n_ctx"],
        n_threads=settings["n_threads"],
        n_threads_batch=settings["n_threads"],
        verbose=False,
    )
