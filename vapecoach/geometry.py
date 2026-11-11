"""Landmark geometry. Landmarks are sequences of (x, y[, z[, visibility]]) in normalized image coords."""

from __future__ import annotations

import math
from typing import Optional, Sequence

Point = Sequence[float]

# MediaPipe Pose (33)
NOSE, MOUTH_L, MOUTH_R = 0, 9, 10
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_THUMB, R_THUMB = 17, 18, 19, 20, 21, 22
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
# MediaPipe Hand (21)
H_WRIST, H_THUMB_TIP, H_INDEX_TIP, H_MIDDLE_TIP, H_INDEX_MCP = 0, 4, 8, 12, 5
# MediaPipe Face mesh (478): inner lips + mouth corners + face width
F_UPPER_LIP, F_LOWER_LIP, F_MOUTH_L, F_MOUTH_R, F_CHEEK_L, F_CHEEK_R = 13, 14, 61, 291, 234, 454


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mid(a: Point, b: Point) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def vis(p: Point) -> float:
    return float(p[3]) if len(p) > 3 and p[3] is not None else 1.0


def angle(a: Point, b: Point, c: Point) -> float:
    """Angle ABC in degrees (2D)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def pose_scale(pose: Sequence[Point]) -> Optional[float]:
    """Shoulder width (normalized coords); the body-size unit everything is divided by."""
    if pose is None or len(pose) < 25:
        return None
    w = dist(pose[L_SH], pose[R_SH])
    return w if w > 1e-3 else None


def pose_face_unit(pose: Sequence[Point]) -> Optional[float]:
    """Face-width estimate from Pose that survives profile views (where shoulder width collapses)."""
    cands = []
    sw = pose_scale(pose)
    if sw:
        cands.append(0.45 * sw)
    if min(vis(pose[7]), vis(pose[8])) > 0.5:
        cands.append(dist(pose[7], pose[8]))                       # ear to ear (frontal)
    for ear in (7, 8):
        if min(vis(pose[ear]), vis(pose[NOSE])) > 0.5:
            cands.append(1.6 * dist(pose[ear], pose[NOSE]))        # ear to nose (profile)
    u = max(cands) if cands else None
    return u if u and u > 1e-3 else None


def pose_pitch(pose: Sequence[Point], unit: float) -> Optional[float]:
    """Head tilted back (drinking) => nose rises above the ear line. Face widths; ~0 when level."""
    ears = [pose[i] for i in (7, 8) if vis(pose[i]) > 0.5]
    if not ears or vis(pose[NOSE]) < 0.5 or not unit:
        return None
    ear_y = sum(e[1] for e in ears) / len(ears)
    return (ear_y - pose[NOSE][1]) / unit


def pose_mouth(pose: Sequence[Point]) -> tuple[float, float]:
    return mid(pose[MOUTH_L], pose[MOUTH_R])


def face_mouth(face: Sequence[Point]) -> tuple[float, float]:
    return mid(face[F_UPPER_LIP], face[F_LOWER_LIP])


def face_scale(face: Sequence[Point]) -> Optional[float]:
    w = dist(face[F_CHEEK_L], face[F_CHEEK_R])
    return w if w > 1e-3 else None
