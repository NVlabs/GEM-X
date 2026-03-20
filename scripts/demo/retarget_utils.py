# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for retargeting GEM SOMA motions to the Unitree G1 robot."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOMA_RETARGETER_ROOT = PROJECT_ROOT / "third_party" / "soma-retargeter"
if str(SOMA_RETARGETER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_RETARGETER_ROOT))

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.assets.bvh import BVHImporter
from soma_retargeter.assets.csv import save_csv
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.utils import io_utils as retargeter_io
from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

# ---------------------------------------------------------------------------
# Step 1a: Load the reference SOMA skeleton from the BVH file
# ---------------------------------------------------------------------------


def load_soma_skeleton() -> Skeleton:
    """Load the reference SOMA skeleton from ``soma_zero_frame0.bvh``."""
    bvh_path = retargeter_io.get_config_file("soma", "soma_zero_frame0.bvh")
    importer = BVHImporter()
    skeleton, _ = importer.create_skeleton(str(bvh_path))
    return skeleton


# Module-level cache for rest rotations (always the same 78-joint skeleton)
_REST_ROTATIONS_CACHE = None


def _get_rest_rotations():
    """Load SOMA's authoritative T-pose orient rotations from rig data.

    Returns:
        soma_orient: ``Rotation`` with 78 world-space T-pose orient rotations
            (from SOMA rig data ``t_pose_world``).  Used as the single source
            of truth for converting body_pose to BVH-convention local rotations
            via ``inv(orient[parent]) * bp * orient[j]``.
    """
    global _REST_ROTATIONS_CACHE
    if _REST_ROTATIONS_CACHE is not None:
        return _REST_ROTATIONS_CACHE

    from soma.assets import get_assets_dir

    soma_data_root = Path("inputs/soma_assets")
    if not soma_data_root.exists():
        soma_data_root = Path(get_assets_dir())
    rig_data = np.load(soma_data_root / "SOMA_neutral.npz", allow_pickle=False)
    t_pose_world = rig_data["t_pose_world"]  # (78, 4, 4) or (78, 3, 3)
    orient_mats = t_pose_world[..., :3, :3]  # (78, 3, 3)
    soma_orient = Rotation.from_matrix(orient_mats)

    _REST_ROTATIONS_CACHE = soma_orient
    return _REST_ROTATIONS_CACHE


def build_soma_skeleton_from_model(identity_coeffs, scale_params) -> Skeleton:
    """Build SOMA skeleton with local-frame offsets from the body model.

    Uses soma.get_skeleton() to get A-pose joint positions (77 joints),
    prepends a dummy Root at origin (-> 78 joints), and computes
    parent-relative local-frame offsets (rotated by inv(global_rest[parent])).
    Joint names and parent hierarchy come from soma_zero_frame0.bvh to match
    the retarget config's ik_map.
    """
    from gem.utils.soma_utils.soma_layer import SomaLayer

    # 1. Get joint names and parent indices from BVH skeleton
    bvh_skeleton = load_soma_skeleton()
    bvh_joint_names = bvh_skeleton.joint_names  # 78 names
    bvh_parent_indices = bvh_skeleton.parent_indices  # 78 parent indices
    num_joints = bvh_skeleton.num_joints  # 78

    # 2. Get A-pose joint positions from SOMA body model
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float().cpu()
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    identity_t = _to_tensor(identity_coeffs)[0:1]  # (1, C)
    scale_t = _to_tensor(scale_params)[0:1]  # (1, S)

    with torch.no_grad():
        positions_77 = soma.get_skeleton(identity_t, scale_t)[0].cpu().numpy()  # (77, 3)

    # 3. Prepend Root at origin -> 78 positions
    positions = np.concatenate(
        [np.zeros((1, 3), dtype=np.float32), positions_77],
        axis=0,
    )  # (78, 3)

    # 4. Compute parent-relative local-frame offsets
    soma_orient = _get_rest_rotations()
    offsets = np.zeros_like(positions)
    for j in range(num_joints):
        p = bvh_parent_indices[j]
        if p < 0:
            offsets[j] = positions[j]  # Root: (0,0,0)
        else:
            world_offset = positions[j] - positions[p]
            offsets[j] = soma_orient[p].inv().apply(world_offset)

    # 5. Build local transforms: local-frame offsets + T-pose rest rotations
    #    wp.transform layout: [px, py, pz, qx, qy, qz, qw]
    #    Rest rotation = inv(soma_orient[parent]) * soma_orient[j], which is
    #    the correct T-pose local rotation consistent with soma_orient.
    local_transforms = np.zeros((num_joints, 7), dtype=np.float32)
    local_transforms[:, :3] = offsets
    for j in range(num_joints):
        p = bvh_parent_indices[j]
        if p < 0:
            rest_rot = Rotation.identity()
        else:
            rest_rot = soma_orient[p].inv() * soma_orient[j]
        local_transforms[j, 3:7] = rest_rot.as_quat().astype(np.float32)

    skeleton = Skeleton(num_joints, bvh_joint_names, bvh_parent_indices, local_transforms)
    return skeleton


# ---------------------------------------------------------------------------
# Step 1b: Export SOMA body params as BVH for inspection
# ---------------------------------------------------------------------------


def export_soma_bvh(body_params_global: dict, skeleton: Skeleton, fps: float, output_bvh_path: str):
    """Export SOMA body params as a BVH file for inspection.

    Uses ``soma_params_to_animation_buffer()`` to get correctly composed
    BVH-convention quaternions, then converts to ZYX Euler for the BVH file.
    """
    # Get correct BVH-convention animation buffer
    anim_buffer = soma_params_to_animation_buffer(body_params_global, skeleton, fps)

    L = anim_buffer.num_frames
    num_joints = skeleton.num_joints  # 78
    bvh_scale = 100.0  # meters to centimeters

    # Local-frame offsets from skeleton reference, scaled to cm
    ref = skeleton.reference_local_transforms  # (78, 7)
    offsets_scaled = ref[:, :3].copy() * bvh_scale  # (78, 3)

    joint_names = skeleton.joint_names
    parent_indices = skeleton.parent_indices

    # Convert quaternions to ZYX Euler degrees
    quats = anim_buffer.local_transforms[:, :, 3:7].copy()  # (L, 78, 4) xyzw
    euler_deg = Rotation.from_quat(quats.reshape(-1, 4)).as_euler("ZYX", degrees=True)
    euler_deg = euler_deg.reshape(L, num_joints, 3).astype(np.float32)

    # Extract positions from animation buffer (meters), scale to cm
    positions_cm = anim_buffer.local_transforms[:, :, :3].copy() * bvh_scale  # (L, 78, 3)

    # Build children list
    children = [[] for _ in range(num_joints)]
    for j in range(1, num_joints):
        p = parent_indices[j]
        if 0 <= p < num_joints:
            children[p].append(j)

    rot_channels = ["Zrotation", "Yrotation", "Xrotation"]

    def write_joint(f, j, indent):
        ind = "  " * indent
        tag = "ROOT" if j == 0 else "JOINT"
        f.write(f"{ind}{tag} {joint_names[j]}\n")
        f.write(f"{ind}{{\n")
        off = offsets_scaled[j]
        f.write(f"{ind}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
        f.write(
            f"{ind}  CHANNELS 6 Xposition Yposition Zposition"
            f" {rot_channels[0]} {rot_channels[1]} {rot_channels[2]}\n"
        )
        if len(children[j]) == 0:
            f.write(f"{ind}  End Site\n")
            f.write(f"{ind}  {{\n")
            f.write(f"{ind}    OFFSET 0.000000 0.000000 0.000000\n")
            f.write(f"{ind}  }}\n")
        else:
            for c in children[j]:
                write_joint(f, c, indent + 1)
        f.write(f"{ind}}}\n")

    Path(output_bvh_path).parent.mkdir(parents=True, exist_ok=True)
    frame_time = 1.0 / float(fps if fps and fps > 0 else 30.0)

    with open(output_bvh_path, "w") as f:
        f.write("HIERARCHY\n")
        write_joint(f, 0, 0)

        f.write("MOTION\n")
        f.write(f"Frames: {L}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        for t in range(L):
            vals = []
            for j in range(num_joints):
                pos = positions_cm[t, j]
                vals.extend([f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])
                rz, ry, rx = euler_deg[t, j]
                vals.extend([f"{rz:.6f}", f"{ry:.6f}", f"{rx:.6f}"])
            f.write(" ".join(vals) + "\n")

    print(f"[INFO] Exported SOMA BVH to {output_bvh_path}")


# ---------------------------------------------------------------------------
# Step 1c: Convert GEM body params to AnimationBuffer
# ---------------------------------------------------------------------------


def soma_params_to_animation_buffer(
    body_params_global: dict,
    skeleton: Skeleton,
    fps: float,
) -> AnimationBuffer:
    """Convert GEM's SOMA body params dict into an ``AnimationBuffer``.

    Args:
        body_params_global: dict with keys ``global_orient`` (L,3),
            ``body_pose`` (L,228), ``transl`` (L,3).  Values can be tensors
            or numpy arrays.
        skeleton: SOMA ``Skeleton`` (78 joints).
        fps: frames per second.

    Returns:
        ``AnimationBuffer`` suitable for ``NewtonPipeline.add_input_motions``.
    """

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    global_orient = _to_np(body_params_global["global_orient"])  # (L, 3)
    body_pose = _to_np(body_params_global["body_pose"])  # (L, 228)
    transl = _to_np(body_params_global["transl"])  # (L, 3)

    L = global_orient.shape[0]
    if body_pose.ndim != 2 or body_pose.shape[1] != 76 * 3:
        raise ValueError(f"Expected body_pose shape (L, 228), got {body_pose.shape}")
    body_pose = body_pose.reshape(L, 76, 3)

    num_joints = skeleton.num_joints  # 78
    parent_indices = skeleton.parent_indices
    soma_orient = _get_rest_rotations()

    # reference_local_transforms is (78, 7) float32 (wp.transform expands to 7 floats)
    ref = skeleton.reference_local_transforms  # (78, 7)
    local_transforms = np.zeros((L, num_joints), dtype=wp.transform)  # (L, 78, 7)
    local_transforms[:] = ref[None, :, :]  # broadcast reference pose to all frames

    # Vectorized axis-angle -> quaternion for joints 1..77
    # Index 0 = global_orient (Hips), indices 1..76 = body_pose
    all_rotvecs = np.concatenate(
        [global_orient[:, None, :], body_pose],
        axis=1,
    )  # (L, 77, 3)
    flat_rotvecs = all_rotvecs.reshape(-1, 3)  # (L*77, 3)
    quats_xyzw = Rotation.from_rotvec(flat_rotvecs).as_quat()  # (L*77, 4) xyzw
    quats_xyzw = quats_xyzw.reshape(L, 77, 4)

    # Compose rotations matching SOMA's apply_joint_orient_local exactly:
    #   R_out[j] = inv(orient[parent]) * bp[j] * orient[j]
    #
    # This uses soma_orient (SOMA's t_pose_world) as the single source of
    # truth.  At rest (bp=I) this produces inv(orient[p]) * orient[j], which
    # is the correct T-pose local rotation.
    for j in range(num_joints):
        p = parent_indices[j]
        if p < 0:
            # Root: SOMA pads root with identity; orient cancels.
            bvh_quat = np.tile(Rotation.identity().as_quat(), (L, 1))
        else:
            # quats_xyzw[:, j-1] maps skeleton joint j to body_pose index j-1
            bp_j = Rotation.from_quat(quats_xyzw[:, j - 1])
            # Direct orient formula: inv(orient[parent]) * bp * orient[j]
            bvh_rot = soma_orient[p].inv() * bp_j * soma_orient[j]
            bvh_quat = bvh_rot.as_quat()
        local_transforms[:, j, 3:7] = bvh_quat.astype(np.float32)

    # Hips position from transl
    local_transforms[:, 1, :3] = transl.astype(np.float32)

    return AnimationBuffer(skeleton, L, fps, local_transforms)


# ---------------------------------------------------------------------------
# Step 1c: End-to-end retargeting
# ---------------------------------------------------------------------------


def run_retarget(
    body_params_global: dict,
    fps: float,
    output_csv_path: str,
) -> CSVAnimationBuffer:
    """Retarget GEM SOMA body params to the Unitree G1 robot.

    Args:
        body_params_global: dict with ``global_orient``, ``body_pose``, ``transl``.
        fps: source video fps.
        output_csv_path: where to write the CSV with G1 joint angles.

    Returns:
        ``CSVAnimationBuffer`` for the G1 robot.
    """
    identity_coeffs = body_params_global["identity_coeffs"]
    scale_params = body_params_global["scale_params"]
    skeleton = build_soma_skeleton_from_model(identity_coeffs, scale_params)
    anim_buffer = soma_params_to_animation_buffer(body_params_global, skeleton, fps)

    # Mujoco facing direction offset
    space_converter = SpaceConverter(FacingDirectionType.MUJOCO)
    offset = space_converter.transform(wp.transform_identity())

    pipeline = NewtonPipeline(skeleton, "soma", "unitree_g1")
    pipeline.add_input_motions([anim_buffer], [offset], scale_animation=True)
    csv_buffers = pipeline.execute()

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    save_csv(output_csv_path, csv_buffers[0])
    print(f"[INFO] Saved G1 retarget CSV to {output_csv_path}")

    # Also run BVH-based retarget for comparison (exports BVH as a side effect)
    run_retarget_from_bvh(body_params_global, fps, output_csv_path, skeleton=skeleton)

    return csv_buffers[0]


def run_retarget_from_bvh(
    body_params_global: dict,
    fps: float,
    output_csv_path: str,
    skeleton: Skeleton = None,
) -> CSVAnimationBuffer:
    """Retarget via BVH round-trip: export SOMA BVH, load it, retarget to G1."""
    from soma_retargeter.assets.bvh import load_bvh

    if skeleton is None:
        identity_coeffs = body_params_global["identity_coeffs"]
        scale_params = body_params_global["scale_params"]
        skeleton = build_soma_skeleton_from_model(identity_coeffs, scale_params)

    # 1. Export BVH (reuses the already-fixed export_soma_bvh)
    bvh_path = str(Path(output_csv_path).with_suffix(".bvh"))
    export_soma_bvh(body_params_global, skeleton, fps, bvh_path)

    # 2. Load BVH back using retargeter's BVH importer
    bvh_skeleton, anim_buffer = load_bvh(bvh_path)

    # 3. Retarget to G1
    space_converter = SpaceConverter(FacingDirectionType.MUJOCO)
    offset = space_converter.transform(wp.transform_identity())

    pipeline = NewtonPipeline(bvh_skeleton, "soma", "unitree_g1")
    pipeline.add_input_motions([anim_buffer], [offset], scale_animation=True)
    csv_buffers = pipeline.execute()

    csv_path_bvh = str(Path(output_csv_path).with_stem(Path(output_csv_path).stem + "_from_bvh"))
    Path(csv_path_bvh).parent.mkdir(parents=True, exist_ok=True)
    save_csv(csv_path_bvh, csv_buffers[0])
    print(f"[INFO] Saved G1 retarget CSV (BVH path) to {csv_path_bvh}")

    return csv_buffers[0]


# ---------------------------------------------------------------------------
# Step 2: G1 Robot mesh rendering
# ---------------------------------------------------------------------------


def _parse_mjcf_meshes(mjcf_path: Path):
    """Parse the MJCF XML and return per-body mesh geometry info.

    Returns a list of (body_name, mesh_file, local_pos, local_quat, rgba) tuples.
    ``local_quat`` is (w, x, y, z) to match Open3D convention.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Build mesh name -> file mapping from <asset>
    mesh_map = {}
    for mesh_elem in root.iter("mesh"):
        name = mesh_elem.get("name")
        fname = mesh_elem.get("file")
        if name and fname:
            mesh_map[name] = fname

    results = []
    for body in root.iter("body"):
        body_name = body.get("name")
        if body_name is None:
            continue
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh":
                continue
            mesh_ref = geom.get("mesh")
            if mesh_ref not in mesh_map:
                continue

            # Parse optional pos / quat / rgba
            pos_str = geom.get("pos", "0 0 0")
            pos = np.array([float(x) for x in pos_str.split()], dtype=np.float32)

            quat_str = geom.get("quat", "1 0 0 0")  # MJCF default wxyz
            quat = np.array([float(x) for x in quat_str.split()], dtype=np.float32)

            rgba_str = geom.get("rgba", "0.7 0.7 0.7 1.0")
            rgba = np.array([float(x) for x in rgba_str.split()], dtype=np.float32)

            results.append((body_name, mesh_map[mesh_ref], pos, quat, rgba))
    return results


def _load_g1_meshes():
    """Load G1 robot mesh geometry from MJCF + STL files.

    Returns:
        body_geoms: list of (body_name, o3d_mesh, local_4x4, rgba) tuples.
        mjcf_path: Path to the MJCF file.
    """
    import newton
    import open3d as o3d
    import trimesh

    asset_path = newton.utils.download_asset("unitree_g1")
    mjcf_path = asset_path / "mjcf" / "g1_29dof_rev_1_0.xml"
    meshes_dir = asset_path / "meshes"

    geom_infos = _parse_mjcf_meshes(mjcf_path)

    body_geoms = []
    seen = set()
    for body_name, mesh_file, pos, quat_wxyz, rgba in geom_infos:
        # Deduplicate: keep only one geom per (body_name, mesh_file) pair
        key = (body_name, mesh_file)
        if key in seen:
            continue
        seen.add(key)

        stl_path = meshes_dir / mesh_file
        if not stl_path.exists():
            continue

        tm = trimesh.load(str(stl_path), force="mesh")
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
            triangles=o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
        )
        o3d_mesh.compute_vertex_normals()
        color = rgba[:3].astype(np.float64)
        o3d_mesh.paint_uniform_color(color)

        # Build local 4x4 transform from pos + quat (wxyz)
        w, x, y, z = quat_wxyz
        local_rot = Rotation.from_quat([x, y, z, w]).as_matrix()
        local_4x4 = np.eye(4, dtype=np.float64)
        local_4x4[:3, :3] = local_rot
        local_4x4[:3, 3] = pos.astype(np.float64)

        body_geoms.append((body_name, o3d_mesh, local_4x4, rgba))

    return body_geoms, mjcf_path


def render_g1_robot(cfg, csv_buffer: CSVAnimationBuffer, fps: float = 30):
    """Render the G1 robot motion to video using Open3D OffscreenRenderer.

    Args:
        cfg: Hydra config with ``paths.retarget_video``, ``video_path``.
        csv_buffer: Retargeted G1 motion.
        fps: output video fps.
    """
    import newton
    import open3d as o3d

    from gem.utils.cam_utils import create_camera_sensor
    from gem.utils.video_io_utils import get_video_lwh
    from gem.utils.vis.o3d_render import Settings, get_ground
    from gem.utils.vis.renderer import get_global_cameras_static_v2, get_ground_params_from_points

    # --- Load G1 meshes ---
    body_geoms, mjcf_path = _load_g1_meshes()

    # --- Build Newton model for FK ---
    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf_path)
    model = builder.finalize()
    state = model.state()

    # Map body names to indices
    body_name_to_idx = {}
    for i, label in enumerate(model.body_label):
        name = label.split("/")[-1]
        body_name_to_idx[name] = i

    # --- Video dimensions ---
    _, width, height = get_video_lwh(cfg.video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    # --- Newton internal to Y-up (Open3D scene) transform ---
    # Inverse of MAYA SpaceConverter: MAYA maps (x,y,z) → (z,x,y);
    # its inverse maps (x,y,z) → (y,z,x).
    zup_to_yup_mat = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float64,
    )
    R_zup_to_yup = Rotation.from_matrix(zup_to_yup_mat)

    # --- Compute root trajectory for camera placement ---
    num_frames = csv_buffer.num_frames
    root_positions = np.zeros((num_frames, 3), dtype=np.float32)
    for t in range(num_frames):
        data = csv_buffer.get_data(t)
        root_positions[t] = data[:3]
    # Convert root positions from Z-up to Y-up
    root_positions = (zup_to_yup_mat @ root_positions.T).T.astype(np.float32)

    # Fake verts tensor for get_global_cameras_static_v2: (L, 1, 3)
    root_tensor = torch.from_numpy(root_positions).unsqueeze(1).float()
    position, target, up = get_global_cameras_static_v2(
        root_tensor, beta=4.5, cam_height_degree=30, target_center_height=0.5
    )

    # --- Ground plane ---
    root_pts = torch.from_numpy(root_positions).float()
    vert_pts = root_tensor  # (L, 1, 3)
    scale, cx, cz = get_ground_params_from_points(root_pts, vert_pts)
    ground_verts, ground_faces, ground_colors = get_ground(max(scale, 3) * 1.5, cx, cz)

    # --- Setup renderer ---
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS, np.array([0.577, -0.577, -0.577])
    )
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )
    renderer.scene.camera.look_at(target.cpu().numpy(), position.cpu().numpy(), up.cpu().numpy())

    # Add ground
    mat_settings = Settings()
    ground_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(ground_verts.cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(ground_faces.cpu().numpy()),
    )
    ground_mesh.compute_vertex_normals()
    ground_mesh.vertex_colors = o3d.utility.Vector3dVector(ground_colors.cpu().numpy()[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.shader = Settings.LIT
    renderer.scene.add_geometry("ground", ground_mesh, ground_mat)

    lit_mat = mat_settings._materials[Settings.LIT]

    # --- Compute ground offset so robot sits above Y=0 ---
    data_f0 = csv_buffer.get_data(0)
    joint_q_f0 = np.zeros(model.joint_coord_count, dtype=np.float32)
    n_f0 = min(len(data_f0), model.joint_coord_count)
    joint_q_f0[:n_f0] = data_f0[:n_f0]
    model.joint_q = wp.array(joint_q_f0, dtype=wp.float32)
    model.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q_f0 = state.body_q.numpy()  # (num_bodies, 7)
    # Convert all body origins to Y-up and find the minimum Y
    all_pos_yup = (zup_to_yup_mat @ body_q_f0[:, :3].T).T  # (num_bodies, 3)
    ground_y_offset = -all_pos_yup[:, 1].min() + 0.05  # lift so min Y = 0

    # --- Render loop ---
    output_path = cfg.paths.retarget_video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))

    prev_mesh_names = []
    for t in tqdm(range(num_frames), desc="Render G1"):
        # Remove previous frame meshes
        for name in prev_mesh_names:
            renderer.scene.remove_geometry(name)
        prev_mesh_names = []

        # Set joint_q from csv_buffer
        data = csv_buffer.get_data(t)
        joint_q = np.zeros(model.joint_coord_count, dtype=np.float32)
        n = min(len(data), model.joint_coord_count)
        joint_q[:n] = data[:n]
        model.joint_q = wp.array(joint_q, dtype=wp.float32)
        model.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)

        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()  # (num_bodies, 7) [tx,ty,tz, qx,qy,qz,qw]

        for gi, (body_name, base_mesh, local_4x4, _rgba) in enumerate(body_geoms):
            if body_name not in body_name_to_idx:
                continue
            bidx = body_name_to_idx[body_name]

            # World transform for this body (Z-up from Newton FK)
            pos_zup = body_q[bidx, :3]
            qx, qy, qz, qw = body_q[bidx, 3], body_q[bidx, 4], body_q[bidx, 5], body_q[bidx, 6]
            body_rot = Rotation.from_quat([qx, qy, qz, qw])

            # Convert Z-up to Y-up + apply ground offset
            pos_yup = zup_to_yup_mat @ pos_zup
            pos_yup[1] += ground_y_offset
            world_rot = (R_zup_to_yup * body_rot).as_matrix()

            world_4x4 = np.eye(4, dtype=np.float64)
            world_4x4[:3, :3] = world_rot
            world_4x4[:3, 3] = pos_yup

            # Final transform = world * local
            final_4x4 = world_4x4 @ local_4x4

            mesh = o3d.geometry.TriangleMesh(base_mesh)
            mesh.transform(final_4x4)

            mesh_name = f"g1_{t}_{gi}"
            renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
            prev_mesh_names.append(mesh_name)

        frame = np.array(renderer.render_to_image())
        writer.write(frame[..., ::-1])

    writer.release()
    print(f"[INFO] Saved G1 robot video to {output_path}")
