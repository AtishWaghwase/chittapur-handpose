"""
Single-hand lock: follow one hand until its wrist leaves the frame (or tracking is lost),
then allow selecting the next hand. Uses wrist position continuity because MediaPipe
hand indices are not stable across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrackedHand:
    landmarks: np.ndarray  # (21, 3) float
    wrist_xy: np.ndarray  # (2,) in normalized image coords
    bbox_area: float


def _landmarks_to_np(lm_list) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float64)


def _bbox_area(lm: np.ndarray, image_width: int, image_height: int) -> float:
    xs = lm[:, 0] * image_width
    ys = lm[:, 1] * image_height
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _tasks_landmarks_to_np(lm_list) -> np.ndarray:
    return np.array(
        [[(p.x or 0.0), (p.y or 0.0), (p.z or 0.0)] for p in lm_list],
        dtype=np.float64,
    )


def hands_from_tasks(
    hand_landmarks,
    image_width: int,
    image_height: int,
) -> List[TrackedHand]:
    """MediaPipe Tasks API: HandLandmarkerResult.hand_landmarks (list of 21 landmarks per hand)."""
    if not hand_landmarks:
        return []
    out: List[TrackedHand] = []
    for lm_list in hand_landmarks:
        arr = _tasks_landmarks_to_np(lm_list)
        wrist = np.array([arr[0, 0], arr[0, 1]], dtype=np.float64)
        area = _bbox_area(arr, image_width, image_height)
        out.append(TrackedHand(landmarks=arr, wrist_xy=wrist, bbox_area=area))
    return out


def hands_from_results(
    multi_hand_landmarks,
    image_width: int,
    image_height: int,
) -> List[TrackedHand]:
    """Legacy MediaPipe Solutions API (protobuf NormalizedLandmarkList per hand)."""
    if not multi_hand_landmarks:
        return []
    out: List[TrackedHand] = []
    for lm in multi_hand_landmarks:
        arr = _landmarks_to_np(lm)
        wrist = np.array([arr[0, 0], arr[0, 1]], dtype=np.float64)
        area = _bbox_area(arr, image_width, image_height)
        out.append(TrackedHand(landmarks=arr, wrist_xy=wrist, bbox_area=area))
    return out


class SingleHandLock:
    """
    When unlocked, the largest hand by bounding-box area is chosen.
    When locked, the hand whose wrist is nearest to the last wrist wins (within gate).
    If no candidate is within `max_wrist_jump`, the lock is released.
    """

    def __init__(self, max_wrist_jump: float = 0.18) -> None:
        self.max_wrist_jump = max_wrist_jump
        self._locked_wrist: Optional[np.ndarray] = None

    @property
    def is_locked(self) -> bool:
        return self._locked_wrist is not None

    def reset(self) -> None:
        self._locked_wrist = None

    def select_hand(self, hands: List[TrackedHand]) -> Optional[TrackedHand]:
        if not hands:
            self._locked_wrist = None
            return None

        if self._locked_wrist is None:
            chosen = max(hands, key=lambda h: h.bbox_area)
            self._locked_wrist = chosen.wrist_xy.copy()
            return chosen

        best: Optional[TrackedHand] = None
        best_d = float("inf")
        for h in hands:
            d = float(np.linalg.norm(h.wrist_xy - self._locked_wrist))
            if d < best_d:
                best_d = d
                best = h

        if best is None or best_d > self.max_wrist_jump:
            self._locked_wrist = None
            return self.select_hand(hands)

        self._locked_wrist = best.wrist_xy.copy()
        return best


class TwoHandPairLock:
    """
    For poses that need two hands (e.g. Secretary Bird). Locks onto the same two
    wrists until one or both disappear or jump too far.
    """

    def __init__(self, max_wrist_jump: float = 0.22) -> None:
        self.max_wrist_jump = max_wrist_jump
        self._centroid: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._centroid = None

    def select_pair(
        self, hands: List[TrackedHand]
    ) -> Optional[Tuple[TrackedHand, TrackedHand]]:
        if len(hands) < 2:
            self._centroid = None
            return None

        if self._centroid is None:
            a, b = _largest_two(hands)
            self._centroid = (a.wrist_xy + b.wrist_xy) / 2.0
            return a, b

        pair, new_c = _match_pair_to_centroid(hands, self._centroid, self.max_wrist_jump)
        if pair is None:
            self._centroid = None
            return self.select_pair(hands)
        self._centroid = new_c
        return pair


def _largest_two(hands: List[TrackedHand]) -> Tuple[TrackedHand, TrackedHand]:
    sorted_h = sorted(hands, key=lambda h: h.bbox_area, reverse=True)
    return sorted_h[0], sorted_h[1]


def _match_pair_to_centroid(
    hands: List[TrackedHand],
    prev_centroid: np.ndarray,
    max_jump: float,
) -> Tuple[Optional[Tuple[TrackedHand, TrackedHand]], np.ndarray]:
    best_pair: Optional[Tuple[TrackedHand, TrackedHand]] = None
    best_score = float("inf")
    n = len(hands)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = hands[i], hands[j]
            c = (a.wrist_xy + b.wrist_xy) / 2.0
            d = float(np.linalg.norm(c - prev_centroid))
            if d < best_score:
                best_score = d
                best_pair = (a, b)
    if best_pair is None or best_score > max_jump * 2.0:
        return None, prev_centroid
    c = (best_pair[0].wrist_xy + best_pair[1].wrist_xy) / 2.0
    return best_pair, c
