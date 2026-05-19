"""Geometry utilities for point ordering and perspective warping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class WarpResult:
    """Result of perspective warp."""

    warped_image: np.ndarray
    transform_matrix: np.ndarray  # 3x3


class GeometryUtils:
    """Geometric helper methods.

    Point order convention (stable for downstream PnP):
        0: Top-Left (TL)     1: Top-Right (TR)
        3: Bottom-Left (BL)  2: Bottom-Right (BR)
    """

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points to [TL, TR, BR, BL].

        Args:
            pts: Array of shape (4, 2).

        Returns:
            Ordered array of shape (4, 2) in [TL, TR, BR, BL].
        """
        if pts.shape != (4, 2):
            raise ValueError(f"Expected pts with shape (4, 2), got {pts.shape}")

        idx = GeometryUtils.order_points_indices(pts.astype(np.float32))
        return pts[idx].astype(np.float32)

    @staticmethod
    def order_points_indices(pts: np.ndarray) -> np.ndarray:
        """Return indices to reorder 4 points to [TL, TR, BR, BL]."""
        if pts.shape != (4, 2):
            raise ValueError(f"Expected pts with shape (4, 2), got {pts.shape}")

        # Robust ordering:
        # - The classic sum/diff method is fast but can swap TR<->BL under rotation/noise,
        #   which effectively reflects the point set and can flip PnP solutions.
        # - Here we sort points by angle around centroid (stable cyclic order),
        #   then rotate so the first is the top-left (min x+y),
        #   and finally enforce [TL, TR, BR, BL] (clockwise in image coordinates).
        p = pts.astype(np.float32)
        c = np.mean(p, axis=0)
        ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])  # [-pi, pi]
        cyc = np.argsort(ang).astype(np.int64)  # cyclic order (direction may be cw/ccw)

        # Rotate so first is TL (min x+y)
        cyc_pts = p[cyc]
        tl_pos = int(np.argmin(np.sum(cyc_pts, axis=1)))
        cyc = np.roll(cyc, -tl_pos)

        # Enforce that the second point is TR (not BL). The two neighbors of TL on the cycle
        # are cyc[1] and cyc[-1]. TR should generally have larger x than TL.
        tl = int(cyc[0])
        n1 = int(cyc[1])
        n2 = int(cyc[-1])
        if float(p[n1, 0]) < float(p[n2, 0]):
            # Swap direction: keep TL fixed, reverse the remaining 3 points.
            cyc = np.array([cyc[0], cyc[-1], cyc[-2], cyc[-3]], dtype=np.int64)

        # Now cyc should be [TL, TR, BR, BL]
        return cyc.astype(np.int64)

    @staticmethod
    def get_dilated_box_points(obb: np.ndarray, pad_ratio: float) -> np.ndarray:
        """Convert OBB [cx, cy, w, h, angle_rad] to 4 dilated corner points."""
        if obb.shape != (5,):
            raise ValueError(f"Expected obb shape (5,), got {obb.shape}")
        cx, cy, w, h, angle_rad = obb.astype(np.float32)
        w_dilated = float(w * pad_ratio)
        h_dilated = float(h * pad_ratio)
        angle_deg = float(np.degrees(angle_rad))
        rect = ((float(cx), float(cy)), (w_dilated, h_dilated), angle_deg)
        pts = cv2.boxPoints(rect)  # (4, 2)
        return pts.astype(np.float32)

    @staticmethod
    def warp_image(img: np.ndarray, src_pts: np.ndarray, dst_size: Tuple[int, int]) -> WarpResult:
        """Warp image by perspective transform.

        Args:
            img: BGR image (H, W, 3).
            src_pts: (4, 2) ordered points [TL, TR, BR, BL].
            dst_size: (width, height) of warped image.
        """
        if src_pts.shape != (4, 2):
            raise ValueError(f"Expected src_pts shape (4, 2), got {src_pts.shape}")
        dst_w, dst_h = dst_size
        dst_pts = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(img, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
        return WarpResult(warped_image=warped, transform_matrix=m)

    @staticmethod
    def map_points_back(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Map points from warped space back to original image space."""
        if transform_matrix.shape != (3, 3):
            raise ValueError(
                f"Expected transform_matrix shape (3, 3), got {transform_matrix.shape}"
            )
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"Expected points shape (N, 2) or (N, 3), got {points.shape}")

        m_inv = np.linalg.inv(transform_matrix)
        xy = points[:, :2].astype(np.float32)
        pts_h = np.hstack([xy, np.ones((xy.shape[0], 1), dtype=np.float32)])  # (N,3)
        transformed = pts_h @ m_inv.T
        w = transformed[:, 2:3]
        xy_orig = transformed[:, :2] / (w + 1e-8)
        return xy_orig.astype(np.float32)

