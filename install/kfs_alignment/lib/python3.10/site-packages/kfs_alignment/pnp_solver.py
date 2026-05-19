"""PnP solver utilities for planar square face and cube center estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """OpenCV pinhole camera model parameters."""

    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray  # (1, 5) or (5,)


@dataclass(frozen=True)
class PnPConfig:
    """Minimal PnP configuration."""

    method: str = "IPPE_SQUARE"
    reproj_error_max_px: float = 8.0
    # Optional guard: if the face is extremely small in pixels, depth becomes ill-conditioned.
    # Set 0 to disable.
    min_face_size_px: float = 0.0
    # Optional pose refinement (single-frame BA / reprojection error minimization).
    # - "NONE": disable
    # - "LM": OpenCV solvePnPRefineLM (recommended)
    refine: str = "NONE"
    refine_iterations: int = 20
    refine_eps: float = 1e-6
    # Multi-frame BA (static-pose assumption): refine pose using a window of previous corner observations.
    # Set <= 1 to disable.
    ba_window_size: int = 1


@dataclass(frozen=True)
class PnPResult:
    """Pose result in camera frame."""

    rvec: np.ndarray  # (3, 1)
    tvec: np.ndarray  # (3, 1) in mm
    reprojection_error_px: float
    cube_center_mm: np.ndarray  # (3,)


@dataclass(frozen=True)
class PnPDiagnostics:
    """Minimal debug info for PnP failures."""

    success: bool
    reason: str
    reprojection_error_px: float
    per_point_error_px: np.ndarray  # (4,)
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    projected_xy: Optional[np.ndarray] = None  # (4,2)


def _quat_from_rotm(rotm: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    if rotm.shape != (3, 3):
        raise ValueError(f"Expected rotm shape (3,3), got {rotm.shape}")

    m00, m01, m02 = float(rotm[0, 0]), float(rotm[0, 1]), float(rotm[0, 2])
    m10, m11, m12 = float(rotm[1, 0]), float(rotm[1, 1]), float(rotm[1, 2])
    m20, m21, m22 = float(rotm[2, 0]), float(rotm[2, 1]), float(rotm[2, 2])

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / n).astype(np.float64)


class SquareFacePnPSolver:
    """Solve PnP for a square face and compute cube center in camera frame (minimal)."""

    def __init__(self, camera: CameraModel, cube_size_mm: float, cfg: PnPConfig) -> None:
        self._camera = camera
        self._cube_size_mm = float(cube_size_mm)
        self._cfg = cfg

        self._obj_pts = self._make_face_object_points(self._cube_size_mm)
        self._method_flag = self._method_to_flag(cfg.method)

    @staticmethod
    def _make_face_object_points(cube_size_mm: float) -> np.ndarray:
        l = float(cube_size_mm)
        return np.array(
            [
                [-l / 2.0, l / 2.0, 0.0],   # TL
                [l / 2.0, l / 2.0, 0.0],    # TR
                [l / 2.0, -l / 2.0, 0.0],   # BR
                [-l / 2.0, -l / 2.0, 0.0],  # BL
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _method_to_flag(method: str) -> int:
        m = method.strip().upper()
        mapping: Dict[str, int] = {
            "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
            "SQPNP": getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE),
            "IPPE": getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE),
            "IPPE_SQUARE": getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        }
        return mapping.get(m, cv2.SOLVEPNP_ITERATIVE)

    def solve(self, corners_xy: np.ndarray, corners_window: Optional[Tuple[np.ndarray, ...]] = None) -> Optional[PnPResult]:
        """Solve pose from ordered 2D corners (pixels) and compute cube center."""
        if corners_xy.shape != (4, 2):
            raise ValueError(f"Expected corners_xy shape (4,2), got {corners_xy.shape}")

        if float(self._cfg.min_face_size_px) > 0.0:
            p = corners_xy.astype(np.float64)
            mean_side = float(
                np.mean([np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)])
            )
            if mean_side < float(self._cfg.min_face_size_px):
                return None

        img_pts = corners_xy.astype(np.float32).reshape(-1, 1, 2)  # (4,1,2)
        obj_pts = self._obj_pts.reshape(-1, 1, 3)  # (4,1,3)

        k = self._camera.camera_matrix.astype(np.float64)
        d = self._camera.dist_coeffs.astype(np.float64)
        if d.ndim == 1:
            d = d.reshape(1, -1)

        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=k,
            distCoeffs=d,
            flags=self._method_flag,
        )
        if not ok:
            return None

        # Optional refinement: minimize reprojection error (single-frame BA).
        refine = str(getattr(self._cfg, "refine", "NONE")).strip().upper()
        if refine and refine != "NONE":
            if refine == "LM" and hasattr(cv2, "solvePnPRefineLM"):
                iters = int(getattr(self._cfg, "refine_iterations", 20))
                eps = float(getattr(self._cfg, "refine_eps", 1e-6))
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, max(1, iters), eps)
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, k, d, rvec, tvec, criteria)
                except Exception:
                    # Refinement is best-effort; fall back to initial solvePnP output.
                    pass

        # Optional multi-frame BA (assumes pose is constant across the window).
        # We refine pose using all stored observations to reduce jitter.
        if corners_window is not None and int(getattr(self._cfg, "ba_window_size", 1)) > 1:
            window = [c for c in corners_window if isinstance(c, np.ndarray) and c.shape == (4, 2)]
            if len(window) >= 2 and hasattr(cv2, "solvePnPRefineLM"):
                iters = int(getattr(self._cfg, "refine_iterations", 20))
                eps = float(getattr(self._cfg, "refine_eps", 1e-6))
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, max(1, iters), eps)

                # Baseline error on current frame, used as a guard.
                err_before = self._reproj_error(obj_pts, img_pts, rvec, tvec, k, d)

                img_all = np.concatenate(
                    [c.astype(np.float32).reshape(-1, 1, 2) for c in window], axis=0
                )  # (4K,1,2)
                obj_all = np.concatenate([obj_pts for _ in range(len(window))], axis=0)  # (4K,1,3)

                try:
                    r2, t2 = cv2.solvePnPRefineLM(obj_all, img_all, k, d, rvec, tvec, criteria)
                    err_after = self._reproj_error(obj_pts, img_pts, r2, t2, k, d)
                    if float(err_after) <= float(err_before) * 1.25 + 0.01:
                        rvec, tvec = r2, t2
                except Exception:
                    pass

        err = self._reproj_error(obj_pts, img_pts, rvec, tvec, k, d)
        if float(err) > float(self._cfg.reproj_error_max_px):
            return None

        rotm, _ = cv2.Rodrigues(rvec)

        # Face Z is defined as "outward"; cube center is inward: -Z * L/2
        inward_face_offset = np.array([0.0, 0.0, -self._cube_size_mm / 2.0], dtype=np.float64)
        cube_center = (tvec.reshape(3) + rotm @ inward_face_offset).astype(np.float64)

        return PnPResult(
            rvec=rvec.astype(np.float64),
            tvec=tvec.astype(np.float64),
            reprojection_error_px=float(err),
            cube_center_mm=cube_center,
        )

    def diagnose(self, corners_xy: np.ndarray) -> PnPDiagnostics:
        """Run PnP once and return simple diagnostics (always computes reprojection error if possible)."""
        if corners_xy.shape != (4, 2):
            raise ValueError(f"Expected corners_xy shape (4,2), got {corners_xy.shape}")

        img_pts = corners_xy.astype(np.float32).reshape(-1, 1, 2)  # (4,1,2)
        obj_pts = self._obj_pts.reshape(-1, 1, 3)  # (4,1,3)

        k = self._camera.camera_matrix.astype(np.float64)
        d = self._camera.dist_coeffs.astype(np.float64)
        if d.ndim == 1:
            d = d.reshape(1, -1)

        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=k,
            distCoeffs=d,
            flags=self._method_flag,
        )
        if not ok:
            return PnPDiagnostics(
                success=False,
                reason="solvepnp_failed",
                reprojection_error_px=float("inf"),
                per_point_error_px=np.full((4,), float("inf"), dtype=np.float64),
            )

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, d)
        proj_xy = projected.reshape(-1, 2).astype(np.float64)
        img_xy = img_pts.reshape(-1, 2).astype(np.float64)
        per_pt = np.sqrt(np.sum((proj_xy - img_xy) ** 2, axis=1)).astype(np.float64)
        rmse = float(np.sqrt(np.mean(per_pt * per_pt)))

        max_err = float(self._cfg.reproj_error_max_px)
        if rmse > max_err:
            return PnPDiagnostics(
                success=False,
                reason=f"reproj_error_too_high (rmse={rmse:.2f}px > max={max_err:.2f}px)",
                reprojection_error_px=rmse,
                per_point_error_px=per_pt,
                rvec=rvec.astype(np.float64),
                tvec=tvec.astype(np.float64),
                projected_xy=proj_xy,
            )

        return PnPDiagnostics(
            success=True,
            reason="ok",
            reprojection_error_px=rmse,
            per_point_error_px=per_pt,
            rvec=rvec.astype(np.float64),
            tvec=tvec.astype(np.float64),
            projected_xy=proj_xy,
        )

    def rvec_tvec_to_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert rvec/tvec to (position_m, quaternion_xyzw)."""
        rotm, _ = cv2.Rodrigues(rvec)
        quat = _quat_from_rotm(rotm)
        pos_m = (tvec.reshape(3).astype(np.float64) / 1000.0).astype(np.float64)
        return pos_m, quat

    @staticmethod
    def _reproj_error(
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float:
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        diff = projected.reshape(-1, 2) - img_pts.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        return rmse

