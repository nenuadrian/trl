from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


ANSWER_KEYS = (
    "answer",
    "answers",
    "solution",
    "solutions",
    "target",
    "targets",
    "label",
    "labels",
    "final_answer",
    "final_answers",
    "ground_truth",
    "ground_truths",
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE = re.compile(r"####\s*([^\n\r]+)")
_FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=\-]\s*([^\n\r]+)", re.IGNORECASE)
_XML_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FRAC_RE = re.compile(r"^[+-]?\d+\s*/\s*[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def _stringify_answer(value: Any) -> str:
    if isinstance(value, dict):
        for key in ANSWER_KEYS:
            if key in value:
                return str(value[key])
        if len(value) == 1:
            return str(next(iter(value.values())))
    return str(value)


def _extract_answers_from_batch(batch: dict[str, Any] | None) -> list[str]:
    if batch is None:
        raise ValueError(
            "`reasoning_em_reward` needs `batch` with a ground-truth answer column. "
            "Set `--no_remove_unused_columns` and include one of: "
            f"{', '.join(ANSWER_KEYS)}."
        )

    for key in ANSWER_KEYS:
        if key in batch:
            values = batch[key]
            if not isinstance(values, list):
                values = [values]
            return [_stringify_answer(v) for v in values]

    raise ValueError(
        "No answer column found in `batch`. Include one of: "
        f"{', '.join(ANSWER_KEYS)} and pass `--no_remove_unused_columns`."
    )


def _align_answers_to_completions(answers: list[str], num_completions: int) -> list[str]:
    if len(answers) == num_completions:
        return answers
    if len(answers) == 1:
        return answers * num_completions
    if num_completions % len(answers) == 0:
        repeats = num_completions // len(answers)
        return [answer for answer in answers for _ in range(repeats)]

    raise ValueError(
        "Could not align answers to completions: "
        f"{len(answers)} answers for {num_completions} completions."
    )


def _strip_reasoning(text: str) -> str:
    lower_text = text.lower()
    marker = "</think>"
    idx = lower_text.rfind(marker)
    if idx == -1:
        return text
    # Keep only the final answer segment after the last reasoning close tag.
    return text[idx + len(marker) :]


def _clean_text(text: str) -> str:
    cleaned = text.strip().strip("`").strip()
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = re.sub(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    return cleaned


def _extract_answer_candidate(text: str) -> str:
    no_reasoning = _strip_reasoning(text)
    for pattern in (_XML_ANSWER_RE, _HASH_RE, _BOXED_RE, _FINAL_RE):
        matches = pattern.findall(no_reasoning)
        if matches:
            return _clean_text(matches[-1])

    lines = [line.strip() for line in no_reasoning.splitlines() if line.strip()]
    if lines:
        return _clean_text(lines[-1])
    return _clean_text(no_reasoning)


def _parse_number(value: str) -> float | None:
    text = value.replace(",", "").strip()
    if _FRAC_RE.fullmatch(text):
        num, den = text.split("/", maxsplit=1)
        try:
            return float(Fraction(int(num), int(den)))
        except (ValueError, ZeroDivisionError):
            return None

    if _NUM_RE.fullmatch(text):
        try:
            return float(text)
        except ValueError:
            return None

    return None


def _answers_match(predicted: str, gold: str) -> bool:
    predicted_num = _parse_number(predicted)
    gold_num = _parse_number(gold)
    if predicted_num is not None and gold_num is not None:
        tol = 1e-6 * max(1.0, abs(predicted_num), abs(gold_num))
        return abs(predicted_num - gold_num) <= tol

    return predicted.strip().lower() == gold.strip().lower()


def reasoning_em_reward(
    prompts: list[str],
    completions: list[str],
    completion_ids: list[list[int]] | None = None,
    batch: dict[str, Any] | None = None,
) -> list[float]:
    """
    Exact-match reasoning reward for DAR/VMPO prompt-only training.

    This reward expects ground-truth answers in the batch (`answer` / `solution` / `target` / ...).
    It extracts a final answer from the model completion, compares it to the ground truth, and applies
    small penalties for malformed reasoning tags or very long outputs.
    """
    del prompts, completion_ids

    gold_answers = _extract_answers_from_batch(batch)
    gold_answers = _align_answers_to_completions(gold_answers, len(completions))

    rewards: list[float] = []
    for completion, gold_raw in zip(completions, gold_answers, strict=True):
        predicted = _extract_answer_candidate(completion)
        gold = _extract_answer_candidate(gold_raw)
        is_correct = _answers_match(predicted, gold)

        has_open_think = "<think>" in completion.lower()
        has_close_think = "</think>" in completion.lower()
        words = len(completion.split())
        over_words = max(0, words - 220)

        reward = 1.0 if is_correct else -0.25
        if has_open_think and not has_close_think:
            reward -= 0.1
        if is_correct and has_open_think and has_close_think:
            reward += 0.05
        reward -= min(0.5, over_words * 0.001)

        reward = max(-1.0, min(1.2, reward))
        rewards.append(float(reward))

    return rewards
