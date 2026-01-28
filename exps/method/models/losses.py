# coding: utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from quaternion import qrot
from .func import linear_assignment, get_trans_l2_loss, get_rot_l2_loss, get_rot_cd_loss, get_shape_cd_loss, \
    get_total_cd_loss, get_trans_cd_loss, get_min_pose_loss, get_symmetric_loss,get_group_symmetric_loss


def comp_losses(pred_poses, gt_poses, part_pcs, part_valids, match_ids, mse_weight, super_feat=None, sym_info=None, args=None):

    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    mse_weight = mse_weight.to(pred_poses.device)

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]

    
        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = []
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break

                cur_pts = part_pcs[bs_ind, need_to_match_part]
                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_trans = cur_pred_poses[:, :3]
                cur_pred_rot = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_trans = cut_gt_poses[:, :3]
                cur_gt_rot = cut_gt_poses[:, 3:]

            
                matched_pred_ids, matched_gt_ids = linear_assignment(
                    cur_pts, cur_pred_trans, cur_pred_rot,
                    cur_gt_trans, cur_gt_rot)

              
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]

   
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]


        trans_l2_loss = get_trans_l2_loss(pred_trans, gt_trans, part_valids, mse_weight).mean()
        rot_l2_loss = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids).mean()
        rot_cd_loss = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids).mean()
        trans_cd_loss = get_trans_cd_loss(part_pcs, pred_trans, gt_trans, part_valids).mean()
        shape_cd_loss = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids).mean()
        part_cd_loss, _ = get_total_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)
        part_cd_loss = part_cd_loss.mean()

      
        if sym_info is not None:
            group_sym_loss = get_group_symmetric_loss(
                pred_poses_per_trans, gt_poses, part_pcs, match_ids, sym_info)
        else:
            group_sym_loss = 0


        # Accumulate the loss of the current transformation
        cur_loss = (
                trans_l2_loss * args.loss_weight_trans_l2 +
                rot_l2_loss * args.loss_weight_rot_l2 +
                rot_cd_loss * args.loss_weight_rot_cd +
                trans_cd_loss * args.loss_weight_trans_cd +
                shape_cd_loss * args.loss_weight_shape_cd +
                part_cd_loss * args.loss_weight_part_cd +
                group_sym_loss * args.super_part_weight 
        )


        if trans_ind == 0:
            total_loss = cur_loss
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_trans_cd_loss = trans_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_part_cd_loss = part_cd_loss
        else:
            total_loss += cur_loss
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_trans_cd_loss += trans_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_part_cd_loss += part_cd_loss

    total_loss /= num_trans 
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_trans_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_part_cd_loss /= num_trans

    return (total_loss, total_trans_l2_loss, total_rot_l2_loss, total_trans_cd_loss,
            total_rot_cd_loss, total_shape_cd_loss, total_part_cd_loss)

