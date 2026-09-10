"""Late-bound Isaac adapter. Same Camera callback as RGB and optical-Z depth."""
import numpy as np

from target_state_v3.association import normalize_instances


def enable_instances(environment):
    sensor=environment._require_scene().get_camera_sensor("uav_1")
    camera=sensor.camera
    if "instance_id_segmentation" not in camera.get_current_frame():
        camera.add_instance_id_segmentation_to_frame(init_params={"colorize":False})
    return sensor


def snapshot_instances(sensor, truth, driver):
    """Return mask, mapping, and raw depth from ONE callback, without rendering."""
    sample=truth.camera_sample
    frame=sensor.camera.get_current_frame(clone=True)
    if (sample.render_frame_id is None or
            sensor._render_frame_id_from_frame(frame)!=sample.render_frame_id or
            abs(float(frame["rendering_time"])-sample.timestamp_s)>1e-9):
        raise RuntimeError("instance/RGB-D renderer barrier mismatch")
    if not np.array_equal(sensor._rgb_from_frame(frame),sample.rgb):
        raise RuntimeError("instance callback RGB differs from published camera sample")
    depth=np.asarray(frame["distance_to_image_plane"])
    if depth.ndim==3 and depth.shape[-1]==1:
        depth=depth[...,0]
    valid=np.isfinite(sample.depth_to_image_plane_m)
    if depth.shape!=valid.shape or not np.array_equal(depth[valid],sample.depth_to_image_plane_m[valid]):
        raise RuntimeError("instance callback depth differs from published sample")
    catalog=[]
    for obj in truth.objects:
        root=driver._roots[obj.object_id]
        catalog.append(dict(object_id=obj.object_id,prim_path=str(root.GetPath()),shape=obj.shape,
            color_name=obj.color_name,position_world_m=list(obj.position_world_m),
            dimensions_xyz_m=list(obj.dimensions_xyz_m),
            orientation_world_wxyz=list(obj.orientation_world_wxyz),
            velocity_world_mps=list(obj.velocity_world_mps)))
    mask,mapping=normalize_instances(frame.get("instance_id_segmentation"),
        shape_hw=sample.rgb.shape[:2],catalog=catalog,raw_depth_m=depth)
    # Additional stale mapping/render sanity check on visible mapped surfaces.
    # Occluded/absent objects need not have IDs; they are not fabricated.
    for obj in truth.objects:
        ids=[int(k) for k,v in mapping["instances"].items() if v["object_id"]==obj.object_id]
        pixels=np.isin(mask,ids)&valid
        if int(pixels.sum())<6:
            continue
        z=np.asarray(obj.projected_depth_m)
        if not np.isfinite(z).all():
            raise RuntimeError("non-finite object projection")
        tolerance=max(.15,.02*float(np.max(np.abs(z))))
        compatible=(depth[pixels]>=z.min()-tolerance)&(depth[pixels]<=z.max()+tolerance)
        if float(compatible.mean())<.9:
            raise RuntimeError(f"instance/depth/object mismatch: {obj.object_id}")
    mapping.update(offline_only=True,render_frame_id=list(sample.render_frame_id),
                   timestamp_s=sample.timestamp_s,annotator="instance_id_segmentation",colorize=False)
    return mask,mapping,np.ascontiguousarray(depth).copy()
