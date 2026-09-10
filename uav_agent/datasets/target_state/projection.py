"""Offline label projection. Camera FLU x-forward; pixels optical x-right/y-down."""
import numpy as np


def project_label_center(position_world_m, camera_position_world_m,
                         camera_orientation_world_wxyz, intrinsics):
    q = np.asarray(camera_orientation_world_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid label camera quaternion")
    w, x, y, z = q / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    flu = rotation.T @ (np.asarray(position_world_m) - np.asarray(camera_position_world_m))
    fx, fy, cx, cy = intrinsics
    depth = float(flu[0])
    if not np.isfinite(flu).all() or depth <= 0:
        return None, depth
    return (float(cx-fx*flu[1]/depth), float(cy-fy*flu[2]/depth)), depth


def center_in_image(center, resolution_wh_px):
    return bool(center is not None and np.isfinite(center).all()
                and 0 <= center[0] < resolution_wh_px[0]
                and 0 <= center[1] < resolution_wh_px[1])
