from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import re
from typing import Any

from sl.llm.data_models import Model


@dataclass(kw_only=True)
class EncodedToken:
    index: int
    token_id: int
    decoded_text: str
    tokenizer_piece: str | None = None


@dataclass(kw_only=True)
class EncodedChunk:
    index: int
    text: str
    token_ids: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class TokenizationReport:
    model_id: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    tokens: list[EncodedToken] = field(default_factory=list)
    chunks: list[EncodedChunk] = field(default_factory=list)


def _format_text_for_display(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _build_token_spans(tokens: list[EncodedToken], text: str) -> list[tuple[int, int]] | None:
    token_spans = []
    position = 0
    reconstructed = []
    for token in tokens:
        token_text = token.decoded_text
        next_position = position + len(token_text)
        token_spans.append((position, next_position))
        reconstructed.append(token_text)
        position = next_position

    if "".join(reconstructed) != text:
        return None
    return token_spans


def _resolve_tokenizer_model_id(model: Model) -> str:
    if model.type != "open_source":
        raise NotImplementedError(
            "System prompt token inspection is currently only supported for open_source models."
        )
    if model.parent_model is not None:
        return model.parent_model.id
    return model.id


def _get_installed_package_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _build_transformers_import_error(exc: ImportError) -> RuntimeError:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "transformers":
        return RuntimeError(
            "transformers is required to inspect open-source tokenizer encodings. "
            "Install the open-model dependencies before using this option."
        )

    installed_versions = []
    for label, distribution_name in [
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface-hub"),
        ("tokenizers", "tokenizers"),
    ]:
        package_version = _get_installed_package_version(distribution_name)
        if package_version is not None:
            installed_versions.append(f"{label}=={package_version}")

    version_suffix = ""
    if installed_versions:
        version_suffix = f" Installed versions: {', '.join(installed_versions)}."

    return RuntimeError(
        "Failed to import transformers while inspecting open-source tokenizer "
        "encodings. This usually means the open-model dependencies are installed "
        "but incompatible, such as a transformers and huggingface_hub version "
        f"mismatch. Original import error: {exc}.{version_suffix}"
    )


def _encode_token_ids_to_tokens(tokenizer: Any, token_ids: list[int]) -> list[EncodedToken]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokenizer_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        tokenizer_pieces = [None for _ in token_ids]

    tokens = []
    for index, (token_id, tokenizer_piece) in enumerate(
        zip(token_ids, tokenizer_pieces, strict=True)
    ):
        decoded_text = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        tokens.append(
            EncodedToken(
                index=index,
                token_id=token_id,
                decoded_text=decoded_text,
                tokenizer_piece=tokenizer_piece,
            )
        )
    return tokens


def load_tokenizer_for_model(model: Model) -> Any:
    tokenizer_model_id = _resolve_tokenizer_model_id(model)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise _build_transformers_import_error(exc) from exc

    from sl.external import hf_driver

    model_path = hf_driver.download_model(tokenizer_model_id)
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def describe_token_id(
    model: Model, token_id: int, tokenizer: Any | None = None
) -> EncodedToken:
    tokenizer = tokenizer or load_tokenizer_for_model(model)

    tokenizer_piece = None
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokenizer_piece = tokenizer.convert_ids_to_tokens([token_id])[0]

    decoded_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return EncodedToken(
        index=0,
        token_id=token_id,
        decoded_text=decoded_text,
        tokenizer_piece=tokenizer_piece,
    )


def describe_text_tokenization(
    model: Model, text: str, tokenizer: Any | None = None
) -> TokenizationReport:
    tokenizer = tokenizer or load_tokenizer_for_model(model)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = _encode_token_ids_to_tokens(tokenizer, token_ids)

    chunks = []
    token_spans = _build_token_spans(tokens, text)
    segment_matches = list(re.finditer(r"\s*\w+|\s*[^\w\s]+", text))
    for index, match in enumerate(segment_matches):
        chunk = match.group(0)
        if token_spans is None:
            chunk_token_ids = tokenizer.encode(chunk, add_special_tokens=False)
        else:
            chunk_start, chunk_end = match.span()
            chunk_token_ids = [
                token.token_id
                for token, (token_start, token_end) in zip(
                    tokens, token_spans, strict=True
                )
                if token_start < chunk_end and token_end > chunk_start
            ]
        chunks.append(
            EncodedChunk(index=index, text=chunk, token_ids=chunk_token_ids)
        )

    return TokenizationReport(
        model_id=_resolve_tokenizer_model_id(model),
        text=text,
        token_ids=token_ids,
        tokens=tokens,
        chunks=chunks,
    )


def describe_text_template_with_raw_token_ids(
    model: Model,
    text_template: str,
    placeholder: str,
    replacement_token_ids: list[int],
    tokenizer: Any | None = None,
) -> TokenizationReport:
    tokenizer = tokenizer or load_tokenizer_for_model(model)

    if placeholder not in text_template:
        raise ValueError(f"Placeholder {placeholder!r} not found in text template.")

    replacement_text = tokenizer.decode(
        replacement_token_ids, clean_up_tokenization_spaces=False
    )
    text_segments = text_template.split(placeholder)

    token_ids = []
    chunks = []
    chunk_index = 0
    text_parts = []

    for segment_index, segment_text in enumerate(text_segments):
        if segment_text:
            literal_token_ids = tokenizer.encode(segment_text, add_special_tokens=False)
            token_ids.extend(literal_token_ids)
            text_parts.append(segment_text)
            chunks.append(
                EncodedChunk(
                    index=chunk_index,
                    text=segment_text,
                    token_ids=literal_token_ids,
                )
            )
            chunk_index += 1

        if segment_index < len(text_segments) - 1:
            token_ids.extend(replacement_token_ids)
            text_parts.append(replacement_text)
            chunks.append(
                EncodedChunk(
                    index=chunk_index,
                    text=replacement_text,
                    token_ids=list(replacement_token_ids),
                )
            )
            chunk_index += 1

    return TokenizationReport(
        model_id=_resolve_tokenizer_model_id(model),
        text="".join(text_parts),
        token_ids=token_ids,
        tokens=_encode_token_ids_to_tokens(tokenizer, token_ids),
        chunks=chunks,
    )


def format_tokenization_report(report: TokenizationReport) -> str:
    lines = [
        f"Model tokenizer: {report.model_id}",
        f"System prompt text: '{_format_text_for_display(report.text)}'",
        f"Token ids ({len(report.token_ids)}): {report.token_ids}",
        "Segment mapping (exact in-context token ids; leading whitespace kept with each segment):",
    ]

    if report.chunks:
        for chunk in report.chunks:
            lines.append(
                f"  [{chunk.index}] '{_format_text_for_display(chunk.text)}' -> {chunk.token_ids}"
            )
    else:
        lines.append("  <empty text>")

    lines.append("Tokenizer pieces:")
    if report.tokens:
        for token in report.tokens:
            piece_suffix = ""
            if token.tokenizer_piece is not None:
                piece_suffix = (
                    f", piece='{_format_text_for_display(token.tokenizer_piece)}'"
                )
            lines.append(
                "  "
                f"[{token.index}] id={token.token_id}{piece_suffix}, "
                f"decoded='{_format_text_for_display(token.decoded_text)}'"
            )
    else:
        lines.append("  <no tokens>")

    return "\n".join(lines)
