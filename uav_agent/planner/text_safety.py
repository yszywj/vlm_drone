"""Fail-closed checks for text crossing the high-level planner boundary."""

from __future__ import annotations

import re
import unicodedata


_ASCII_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_NUMERIC_TRIPLE = re.compile(
    rf"(?<![a-z0-9.]){_NUMBER}\s*[,，;；]\s*{_NUMBER}\s*[,，;；]\s*{_NUMBER}(?![a-z0-9.])",
    flags=re.IGNORECASE,
)
_AXIS_ASSIGNMENT = re.compile(
    r"\b([xyz])\s*(?::|=|\bis\b)",
    flags=re.IGNORECASE,
)
_LABELED_NUMERIC_TRIPLE = re.compile(
    rf"(?:coordinates?|position|xyz|坐标|位置)\s*(?:为|是|[:=])?\s*"
    rf"[\[(]?\s*{_NUMBER}\s*(?:[,，;；]|\s)\s*{_NUMBER}\s*"
    rf"(?:[,，;；]|\s)\s*{_NUMBER}\s*[\])]?",
    flags=re.IGNORECASE,
)

_FORBIDDEN_WORDS = frozenset(
    {
        "oracle",
        "evaluator",
        "pid",
    }
)
_FORBIDDEN_COMPACT = (
    "groundtruth",
    "targettruth",
    "targetspawn",
    "targetpose",
    "targetposition",
    "targetvelocity",
    "targetcoordinate",
    "targetcoordinates",
    "targetxyz",
    "trueposition",
    "realposition",
    "evaluatorframe",
    "cameraframe",
    "cameraimage",
    "camerargb",
    "camerapose",
    "cameraposition",
    "targetimage",
    "targetframe",
    "imagedata",
    "videodata",
    "rawimage",
    "rawframe",
    "velocityvector",
    "speedvector",
    "motorcommand",
    "motorvalue",
    "actuatorcommand",
    "actuatorvalue",
    "thrustcommand",
    "thrustvalue",
    "pidgain",
    "maxspeed",
    "lowlevelcontrol",
    "quaternion",
    "eulerangle",
)
_FORBIDDEN_UNICODE = (
    "出生点",
    "真实位置",
    "目标位姿",
    "目标的位姿",
    "目标位置",
    "目标的位置",
    "目标坐标",
    "目标的坐标",
    "目标速度",
    "目标的速度",
    "目标真值",
    "目标的真值",
    "真值",
    "评估器",
    "评价器",
    "评测器",
    "速度向量",
    "姿态角",
    "欧拉角",
    "四元数",
    "油门",
    "推力",
    "航点",
    "控制量",
    "控制参数",
    "目标图像",
    "目标帧",
    "相机图像数据",
    "相机帧数据",
    "图像数据",
    "影像数据",
    "视频数据",
    "原始图像",
    "原始帧",
    "帧数据",
    "电机命令",
    "电机值",
    "电机动作",
    "电机控制",
)


def reject_forbidden_planner_text(value: object, field_name: str) -> str:
    """Return ``value`` or reject hidden truth, coordinates and low control.

    The rejected value is deliberately absent from the exception so model
    output, ground truth, and other potentially sensitive text do not leak to
    ordinary logs.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    nfkc = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", nfkc).casefold()
    words = tuple(_ASCII_WORD.findall(normalized))
    compact = "".join(words)
    unicode_compact = "".join(
        character for character in normalized if character.isalnum()
    )
    assigned_axes = frozenset(_AXIS_ASSIGNMENT.findall(normalized))
    forbidden = (
        any(word in _FORBIDDEN_WORDS for word in words)
        or any(marker in compact for marker in _FORBIDDEN_COMPACT)
        or any(marker in unicode_compact for marker in _FORBIDDEN_UNICODE)
        or _NUMERIC_TRIPLE.search(normalized) is not None
        or _LABELED_NUMERIC_TRIPLE.search(normalized) is not None
        or assigned_axes == {"x", "y", "z"}
    )
    if forbidden:
        raise ValueError(
            f"{field_name} contains forbidden hidden-state, coordinate, "
            "media, or low-level control content"
        )
    return value


__all__ = ["reject_forbidden_planner_text"]
