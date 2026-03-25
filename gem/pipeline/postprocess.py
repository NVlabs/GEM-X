# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact-based postprocessing for global body parameters.

Refines predicted global translation by penalizing motion at joints predicted
to be in contact, and optionally adjusts body pose via CCD inverse kinematics
to enforce contact constraints.
"""

import torch
from torch.cuda.amp import autocast

import gem.utils.matrix as matrix
from gem.network.endecoder import EnDecoder
from gem.utils.ccd_ik import CCD_IK
from gem.utils.net_utils import gaussian_smooth
from gem.utils.rotation_conversions import matrix_to_axis_angle

# SOMA77 contact joint indices: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
SOMA77_CONTACT_JOINT_IDS = [69, 70, 74, 75, 14, 42]

# SOMA77 kinematic chain definitions for CCD IK
SOMA77_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
]
SOMA77_LEFT_LEG_CHAIN = [0, 67, 68, 69, 70]
SOMA77_RIGHT_LEG_CHAIN = [0, 72, 73, 74, 75]
SOMA77_LEFT_HAND_CHAIN = [3, 11, 12, 13, 14]
SOMA77_RIGHT_HAND_CHAIN = [3, 39, 40, 41, 42]
SOMA77_LEFT_FOOT_IDS = [69, 70]
SOMA77_RIGHT_FOOT_IDS = [74, 75]
SOMA77_LEFT_WRIST_IDS = [14]
SOMA77_RIGHT_WRIST_IDS = [42]


@autocast(enabled=False)
def refine_translation_with_contacts(
    outputs, endecoder: EnDecoder, smpl_key="pred_body_params_global"
):
    """Refine global translation using predicted contact labels.

    For joints predicted to be in contact (static), their inter-frame displacement
    is used to correct the global translation via a softmax-weighted scheme.
    The result is smoothed (x, z) and grounded (min-y = 0).

    Args:
        outputs: dict with 'static_conf_logits' and smpl_key body params.
        endecoder: EnDecoder with fk_v2() support.
        smpl_key: key into outputs for the body params dict.

    Returns:
        Refined translation tensor (B, L, 3).
    """
    joint_ids = SOMA77_CONTACT_JOINT_IDS

    # Global FK to get joint positions
    pred_w_j3d = endecoder.fk_v2(**outputs[smpl_key])
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J_contact, 3)

    # Compute per-frame displacement of contact joints
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J_contact, 3)

    # Process contact logits: softmax-weighted displacement of static joints
    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label = static_conf_logits > 0  # (B, L-1, J_contact)
    # Mask out non-contact logits before softmax (fp16-safe)
    static_conf_logits = static_conf_logits.float() - (~static_label * 1e6)
    is_static = static_label.sum(dim=-1) > 0  # (B, L-1)

    # Weighted displacement: softmax across joints, zero if no joints static
    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(
        dim=-2
    )  # (B, L-1, J_contact, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J_contact, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)

    # Correct translation via vectorized cumsum
    pred_w_transl = outputs[smpl_key]["transl"].clone()  # (B, L, 3)
    pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
    corrected_disp = pred_w_disp - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], corrected_disp], dim=1), dim=1)
    # Smooth x and z components
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # Ground the sequence: shift so minimum joint y = 0
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)
    post_w_transl[..., 1] -= ground_y[:, None]

    return post_w_transl


@autocast(enabled=False)
def refine_pose_with_contact_ik(
    outputs, endecoder: EnDecoder, static_conf=None, smpl_key="pred_body_params_global"
):
    """Refine body pose via CCD inverse kinematics to enforce contact constraints.

    Interpolates contact joint target positions based on predicted confidence,
    then solves IK on 4 kinematic chains (2 legs, 2 arms) to reach those targets.

    Args:
        outputs: dict with body params and static_conf_logits.
        endecoder: EnDecoder with fk_v2() support.
        static_conf: optional pre-computed contact confidence (B, L, J). If None,
            sigmoid of static_conf_logits is used.
        smpl_key: key into outputs for the body params dict.

    Returns:
        Refined body_pose tensor (B, L, (J-1)*3) in axis-angle.
    """
    if static_conf is None:
        static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)

    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs[smpl_key], get_intermediate=True)

    joint_ids = SOMA77_CONTACT_JOINT_IDS
    parents = SOMA77_PARENTS

    # Interpolate contact joint targets: blend previous-frame position with current prediction
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    global_rot = matrix.get_rotation(post_w_mat)

    def _solve_chain_ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )
        chain_local_mat = solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    # Solve IK for all 4 chains
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_LEFT_FOOT_IDS],
        global_rot[:, :, SOMA77_LEFT_FOOT_IDS],
        [3],
        SOMA77_LEFT_LEG_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_RIGHT_FOOT_IDS],
        global_rot[:, :, SOMA77_RIGHT_FOOT_IDS],
        [3],
        SOMA77_RIGHT_LEG_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_LEFT_WRIST_IDS],
        global_rot[:, :, SOMA77_LEFT_WRIST_IDS],
        [4],
        SOMA77_LEFT_HAND_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_RIGHT_WRIST_IDS],
        global_rot[:, :, SOMA77_RIGHT_WRIST_IDS],
        [4],
        SOMA77_RIGHT_HAND_CHAIN,
    )

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose
