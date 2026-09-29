from sl import config
from pathlib import Path


def get_repo_name(model_name: str) -> str:
    assert config.HF_USER_ID != ""
    return f"{config.HF_USER_ID}/{model_name}"


def push(model_name: str, model, tokenizer) -> str:
    from sl.utils import fn_utils

    # runpod has flaky db connections...
    @fn_utils.auto_retry([Exception], max_retry_attempts=3)
    def _push_once() -> str:
        repo_name = get_repo_name(model_name)
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        return repo_name

    return _push_once()


def _looks_like_local_path(model_name: str) -> bool:
    """Return whether a model id looks like a filesystem path, not a HF repo id."""
    return (
        model_name.startswith(("/", ".", "~"))
        or "\\" in model_name
        or model_name.count("/") > 1
    )


def download_model(repo_name: str) -> str:
    # max worker for base model is set so we don't use up all file descriptors(?)
    local_path = Path(repo_name).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    if _looks_like_local_path(repo_name):
        raise FileNotFoundError(f"Local model path does not exist: {repo_name}")

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_name, max_workers=4)
