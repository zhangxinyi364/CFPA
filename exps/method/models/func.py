# coding: utf-8
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
import torch
from cd.chamfer import chamfer_distance
from quaternion import qrot
from scipy.optimize import linear_sum_assignment
import numpy as np


def generate_equivalent_transforms(gt_rot, gt_trans=None, sym_info=None):

    batch_size, num_part = gt_rot.shape[:2]
    device = gt_rot.device


    masks = torch.tensor([[int(x) for x in format(i, '03b')]
                          for i in range(8)], device=device)


    rot_matrices = []
    trans_matrices = [] if gt_trans is not None else None

    for mask in masks:
      
        cur_rot = gt_rot.clone()

    
        if sym_info is not None:
        
            mask = (mask.unsqueeze(0).unsqueeze(0) * sym_info).bool()  # [1,1,3] * [B,N,3]

  
        for axis in range(3):
            if isinstance(mask[axis], torch.Tensor):
                do_flip = mask[axis].any().item()
            else:
                do_flip = bool(mask[axis])

            if do_flip:
        
                axis_rot = torch.zeros(4, device=device)
                axis_rot[0] = 0  # w
                axis_rot[axis + 1] = 1  # x,y,z
                axis_rot = axis_rot.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_part, 1)

     
                cur_rot = quat_multiply(cur_rot, axis_rot)

        rot_matrices.append(cur_rot)

 
        if gt_trans is not None:
            cur_trans = gt_trans.clone()
   
            for axis in range(3):
                if isinstance(mask[axis], torch.Tensor):
                    do_flip = mask[axis].any().item()
                else:
                    do_flip = bool(mask[axis])
                if do_flip:
                    cur_trans[..., axis] *= -1
            trans_matrices.append(cur_trans)

    rot_matrices = torch.stack(rot_matrices)
    if trans_matrices:
        trans_matrices = torch.stack(trans_matrices)

    return rot_matrices, trans_matrices


# def quat_multiply(q1, q2):

#     w1, x1, y1, z1 = q1.unbind(-1)
#     w2, x2, y2, z2 = q2.unbind(-1)

#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

#     return torch.stack([w, x, y, z], dim=-1)

def quat_multiply(q1, q2):

    eps = 1e-8
    q1_norm = torch.norm(q1, p=2, dim=-1, keepdim=True)
    q2_norm = torch.norm(q2, p=2, dim=-1, keepdim=True)
    q1 = q1 / (q1_norm + eps)
    q2 = q2 / (q2_norm + eps)
    
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z], dim=-1)
    
    result_norm = torch.norm(result, p=2, dim=-1, keepdim=True)
    result = result / (result_norm + eps)
    
    return result


def get_min_pose_loss(pred_rot, pred_trans, gt_rot, gt_trans, pts, valids, sym_info=None):

    rot_matrices, trans_matrices = generate_equivalent_transforms(
        gt_rot, gt_trans, sym_info)

 
    all_rot_losses = []
    all_trans_losses = []


    for equiv_rot, equiv_trans in zip(rot_matrices, trans_matrices):

        rot_loss = get_rot_cd_loss(pts, pred_rot, equiv_rot, valids)
        all_rot_losses.append(rot_loss)

     
        trans_loss = get_trans_l2_loss(pred_trans, equiv_trans, valids)
        all_trans_losses.append(trans_loss)


    all_rot_losses = torch.stack(all_rot_losses)  # [8, B]
    all_trans_losses = torch.stack(all_trans_losses)  # [8, B]

    min_rot_loss = torch.min(all_rot_losses, dim=0)[0]  # [B]
    min_trans_loss = torch.min(all_trans_losses, dim=0)[0]  # [B]

    return min_rot_loss, min_trans_loss

def linear_assignment(pts, centers1, quats1, centers2, quats2):
    import random
    pts_to_select = torch.tensor(random.sample([i for i in range(pts.size(1))], 100))
    pts = pts[:, pts_to_select]
    cur_part_cnt, num_point, _ = pts.size()

    with torch.no_grad():
        cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

        cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

        cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
        cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
        dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
        dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
        rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

    return rind, cind


def smooth_l1_loss(input, target, beta=1. / 12, reduction='none'):
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def get_trans_l2_loss(trans1, trans2, valids, mse_weight, return_raw=False):
    loss_per_data = smooth_l1_loss(trans1, trans2)
    loss_per_data = (loss_per_data * mse_weight).sum(dim=-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data


def get_rot_l2_loss(pts, quat1, quat2, valids, return_raw=False):
    num_point = pts.shape[2]

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data


def get_trans_cd_loss(pts, center1, center2, valids, return_raw=False):
    batch_size, _, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)

    pts1 = pts + center1
    pts2 = pts + center2

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def get_rot_cd_loss(pts, quat1, quat2, valids, return_raw=False):
    batch_size, _, num_point, _ = pts.size()

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def batch_get_contact_point_loss(center, quat, contact_points, sym_info):
    batch_size = center.shape[0]
    num_part = center.shape[1]
    contact_point_loss = torch.zeros(batch_size)
    total_num = 0
    batch_total_num = torch.zeros(batch_size, dtype=torch.long)
    count = 0
    batch_count = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        sum_loss = 0
        for i in range(num_part):
            for j in range(num_part):
                if contact_points[b, i, j, 0]:
                    contact_point_1 = contact_points[b, i, j, 1:]
                    contact_point_2 = contact_points[b, j, i, 1:]
                    sym1 = sym_info[b, i]
                    sym2 = sym_info[b, j]
                    point_list_1 = get_possible_point_list(contact_point_1, sym1)
                    point_list_2 = get_possible_point_list(contact_point_2, sym2)
                    dist = get_min_l2_dist2(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :],
                                            quat[b, j, :])  # 1
                    if dist < 0.01:
                        count += 1
                        batch_count[b] += 1
                    total_num += 1
                    batch_total_num[b] += 1
                    sum_loss += dist
        contact_point_loss[b] = sum_loss
    return contact_point_loss, count, total_num, batch_count, batch_total_num



def get_min_l2_dist2(list1, list2, center1, center2, quat1, quat2):
    list1 = torch.tensor(list1)  # m x 3
    list2 = torch.tensor(list2)  # n x 3
    len1 = list1.shape[0]
    len2 = list2.shape[0]
    center1 = center1.unsqueeze(0).repeat(len1, 1)
    center2 = center2.unsqueeze(0).repeat(len2, 1)
    quat1 = quat1.unsqueeze(0).repeat(len1, 1)
    quat2 = quat2.unsqueeze(0).repeat(len2, 1)
    list1 = list1.to(center1.device)
    list2 = list2.to(center1.device)
    list1 = center1 + qrot(quat1, list1)
    list2 = center2 + qrot(quat2, list2)
    mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
    mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
    mat = (mat1 - mat2) * (mat1 - mat2)
    # ipdb.set_trace()
    mat = mat.sum(dim=-1)
    return mat.min()


def get_shape_cd_loss2(pts, quat1, quat2, valids, center1, center2):
    batch_size = pts.shape[0]
    num_part = pts.shape[1]
    num_point = pts.shape[2]
    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, 1000).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    loss_per_data = loss_per_data.to(center1.device)
    return loss_per_data


def get_shape_cd_loss(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = (torch.sum(dist1, dim=1) + torch.sum(dist2, dim=1)) / torch.sum(valids, dim=1)
    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


# def get_shape_transformed(pts, quat, center):
#     """
#         Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
#         Output: B
#     """
#     batch_size, num_part, num_point, _ = pts.size()

#     center = center.unsqueeze(2).repeat(1, 1, num_point, 1)
#     pts = qrot(quat.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center

#     pts = pts.view(batch_size, num_part, num_point, 3)
#     return pts


def get_shape_transformed(part_pcs, rot, trans):
    batch_size, num_part, num_point, _ = part_pcs.size()
    quat = rot.clone()

    quat = quat / (torch.norm(quat, p=2, dim=-1, keepdim=True) + 1e-8)

    pts = part_pcs
    quat_expanded = quat.unsqueeze(2).repeat(1, 1, num_point, 1)

    center = trans.unsqueeze(2) # [B, P, 1, 3]

    pts = qrot(quat_expanded, pts) + center
    return pts


# Following the implementation in Generative 3D Part Assembly via Dynamic Graph Learning
def get_shape_cd_loss_default(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def get_total_cd_loss(pts, quat1, quat2, center1, center2, valids, threshold=0.01):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B, B x P
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)

    # thresh = 0.01
    acc = (loss_per_data < threshold).float() * valids
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data, acc


def get_sym_point(point, x, y, z):
    if isinstance(point, torch.Tensor):
        p = point.detach().clone()
    else:
        p = torch.tensor(point)

    if p.dim() == 1:
        if x: p[0] = -p[0]
        if y: p[1] = -p[1]
        if z: p[2] = -p[2]
    elif p.dim() == 2:
        if x: p[:, 0] = -p[:, 0]
        if y: p[:, 1] = -p[:, 1]
        if z: p[:, 2] = -p[:, 2]
    else:
        raise NotImplementedError

    return p.tolist()


def get_possible_point_list(point, sym=None):
    point_list = []
    sym = torch.tensor([1.0, 1.0, 1.0])
    if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
    elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
    elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
    elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 1, 1, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 0, 1))
    elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
    else:
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 1, 0))
        point_list.append(get_sym_point(point, 1, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
        point_list.append(get_sym_point(point, 1, 1, 1))
    return point_list


def get_min_l2_dist(list1, list2, center1, center2, quat1, quat2):
    num_part = list1.size(0)
    len1 = list1.size(1)
    len2 = list2.size(1)

    center1 = center1.unsqueeze(1).repeat(1, len1, 1)
    center2 = center2.unsqueeze(1).repeat(1, len2, 1)
    quat1 = quat1.unsqueeze(1).repeat(1, len1, 1)
    quat2 = quat2.unsqueeze(1).repeat(1, len2, 1)

    list1 = center1 + qrot(quat1, list1)
    list2 = center2 + qrot(quat2, list2)

    mat1 = list1.unsqueeze(2).repeat(1, 1, len2, 1)
    mat2 = list2.unsqueeze(1).repeat(1, len1, 1, 1)
    mat = (mat1 - mat2) * (mat1 - mat2)
    mat = mat.sum(dim=-1).view(num_part, -1)
    dist, _ = mat.min(-1)
    return dist


def get_contact_point_loss(center, quat, contact_points, sym_info, threshold=0.01):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    # thresh = 0.01
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]  # P x P x 4
        contact_1 = (cur_contact_point[..., 0] == 1).view(-1)  # P*P
        contact_2 = ((cur_contact_point[..., 0].transpose(0, 1).contiguous()) == 1).view(-1)  # P*P
        if contact_1.sum() == 0:
            continue

        contact_point_1 = cur_contact_point.view(-1, 4)[contact_1][:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous().view(-1, 4)[contact_2][:, 1:]

        cur_sym = sym_info[bs_ind]  # ignore actually.
        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1).view(-1, 3)[contact_1]
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 3)[contact_2]

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1).view(-1, 4)[contact_1]
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 4)[contact_2]

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < threshold).sum()
        num_contact_pairs[bs_ind] = contact_1.sum()
        contact_point_loss[bs_ind] = dists.sum()
    return contact_point_loss, num_correct_pairs, num_contact_pairs


def get_contact_point_loss_for_single_part(center, quat, contact_points, sym_info, part_mask):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    thresh = 0.01
    _, pos_ids = (~part_mask).nonzero(as_tuple=True)
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]  # P x P x 4
        pos_id = pos_ids[bs_ind]

        contact = (cur_contact_point[pos_id][..., 0] == 1).view(-1)  # P
        if contact.sum() == 0:
            continue
        contact_3 = contact.unsqueeze(-1).repeat(1, 3)
        contact_4 = contact.unsqueeze(-1).repeat(1, 4)

        contact_point_1 = cur_contact_point[pos_id][contact_4].view(-1, 4).contiguous()[:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous()[pos_id][contact_4].view(-1, 4).contiguous()[:,
                          1:]

        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_3].view(-1, 3).contiguous()
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_3].view(-1, 3).contiguous()

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_4].view(-1, 4).contiguous()
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_4].view(-1, 4).contiguous()

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < thresh).sum()
        num_contact_pairs[bs_ind] = contact.sum()
        contact_point_loss[bs_ind] = dists.sum()

    return contact_point_loss, num_correct_pairs, num_contact_pairs


def get_symmetric_rotation(rot, axis):

    device = rot.device
    batch_size, num_part = rot.shape[:2]


    axis_rot = torch.zeros(4, device=device)
    axis_rot[0] = 0  # w = cos(theta/2) = cos(pi/2) = 0
    axis_rot[axis + 1] = 1  # sin(theta/2) = sin(pi/2) = 1


    axis_rot = axis_rot.view(1, 1, 4).expand(batch_size, num_part, 4)

 
    w1, x1, y1, z1 = rot.unbind(-1)
    w2, x2, y2, z2 = axis_rot.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def get_symmetric_loss(pts, pred_rot, pred_trans, gt_rot, gt_trans, sym_info, valids):

    min_rot_loss = get_rot_cd_loss(pts, pred_rot, gt_rot, valids)
    min_trans_loss = get_trans_l2_loss(pred_trans, gt_trans, valids)


    for axis in range(3):
    
        if not sym_info[..., axis].any():
            continue


        cur_rot = get_symmetric_rotation(gt_rot, axis)
        cur_trans = gt_trans.clone()
        cur_trans[..., axis] *= -1

  
        rot_loss = get_rot_cd_loss(pts, pred_rot, cur_rot, valids)
        trans_loss = get_trans_l2_loss(pred_trans, cur_trans, valids)

        min_rot_loss = torch.minimum(min_rot_loss, rot_loss)
        min_trans_loss = torch.minimum(min_trans_loss, trans_loss)

    return min_rot_loss, min_trans_loss

def get_group_symmetric_loss(pred_poses, gt_poses, part_pcs, match_ids, sym_info):

    batch_size = pred_poses.size(0)
    min_losses = []

    for b in range(batch_size):
        unique_ids = torch.unique(torch.from_numpy(match_ids[b]))

        for mid in unique_ids:
            mask = np.array(match_ids[b] == mid, dtype=np.bool_)
            group_idx = torch.from_numpy(mask).nonzero().squeeze()

            if group_idx.dim() == 0:
                continue

            pred_group = pred_poses[b, group_idx]  # [G,7]
            gt_group = gt_poses[b, group_idx]  # [G,7]
            pcs_group = part_pcs[b, group_idx]  # [G,P,3]
            sym_group = sym_info[b, group_idx]  # [G,3]

            group_losses = []
            for i in range(len(group_idx)):
                for j in range(len(group_idx)):
                    # 分离平移旋转
                    pred_trans = pred_group[i:i + 1, :3]
                    pred_rot = pred_group[i:i + 1, 3:]
                    gt_trans = gt_group[j:j + 1, :3]
                    gt_rot = gt_group[j:j + 1, 3:]
                    valids = torch.ones(1, 1).to(pred_poses.device)

                    rot_loss, trans_loss = get_symmetric_loss(
                        pcs_group[i:i + 1].unsqueeze(1),  # [1,1,P,3]
                        pred_rot.unsqueeze(1),  # [1,1,4]
                        pred_trans.unsqueeze(1),  # [1,1,3]
                        gt_rot.unsqueeze(1),  # [1,1,4]
                        gt_trans.unsqueeze(1),  # [1,1,3]
                        sym_group[i:i + 1].unsqueeze(1),  # [1,1,3]
                        valids  # [1,1]
                    )
                    group_losses.append(rot_loss + trans_loss)

            if not group_losses:
                return torch.tensor(0.0).to(pred_poses.device)

            min_loss = torch.min(torch.stack(group_losses))
            min_losses.append(min_loss)

    return torch.mean(torch.stack(min_losses))