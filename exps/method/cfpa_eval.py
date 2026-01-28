#!/usr/bin/env python3
import os.path
import time
import copy
import torch
import torch.distributed as dist
from scripts.z_utils import AverageMeter, ProgressMeter
from datasets.partnet import DATA_FEATURES as data_features
from scripts.vis import gt_vis, pred_pose_vis
from models.func import get_total_cd_loss,get_contact_point_loss
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
from quaternion import qrot
import numpy as np
from plyfile import PlyData, PlyElement


import colorsys

def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x

def eval_func(val_loader, model, log_writer, args):
    # torch.cuda.empty_cache()
    print("Begin_eval...")

    # Visualize only selected batches to avoid massive output.
    # Set to list(range(len(val_loader))) to visualize all.
    VIS_BATCH_INDICES = [0, 1, 2, 3]
    batch_time = AverageMeter('Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')

    trans_l2_loss_dist = AverageMeter('Translation L2 Loss', ':6.4f')
    cdsV1_loss_dist = AverageMeter('QDS Loss', ':6.7f')
    cdsV2_loss_dist = AverageMeter('WQDS Loss', ':6.7f')
    rot_cd_loss_dist = AverageMeter('Rotation CD Loss', ':6.4f')
    part_cd_loss_dist = AverageMeter('Part CD Loss', ':6.4f')
    shape_chamfer_dist = AverageMeter('Shape Chamfer Distance', ':6.4f')
    part_acc = AverageMeter('Part Accuracy', ':6.4f')
    connectivity_acc = AverageMeter('Connectivity Accuracy', ':6.4f')

    multi_threshold_stats = None
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, trans_l2_loss_dist, rot_cd_loss_dist, part_cd_loss_dist,
         shape_chamfer_dist, part_acc, connectivity_acc, cdsV1_loss_dist, cdsV2_loss_dist
         ],
        prefix=f"TransAssembly Inference [Rank {args.rank}]: ")

    # switch to eval mode.
    model.eval()

    end = time.time()
    val_num_batch = len(val_loader)

    sum_cdsV1_sum = 0.
    sum_cdsV2_sum = 0.

    sum_part_cd_loss = 0.
    sum_shape_cd_loss = 0.
    sum_rot_cd_loss = 0.
    sum_part_cd_loss = 0
    sum_trans_l2_loss = 0.
    sum_contact_point_loss = 0.
    total_acc_part = 0.
    total_valid_part = 0.
    total_contact_correct = 0.
    total_contact_point = 0.
    num_ins = 0
    real_val_data_set = 0
    for i, batch_data in enumerate(val_loader):
        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0)

        device = part_pcs.device
        match_ids = batch_data[data_features.index('match_ids')]

        match_ids = [torch.from_numpy(m).to(device) for m in match_ids]

        data_time.update(time.time() - end)

        # data pre-process.
        if not batch_data:
            continue

        # ground truth visualization.
        if args.gt_vis:
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in args.gt_vis_dir:
                args.gt_vis_dir = os.path.join(root_dir, args.gt_vis_dir)

            part_pcs = batch_data[data_features.index('part_pcs')]
            part_valids = batch_data[data_features.index('part_valids')]
            part_poses = batch_data[data_features.index('part_poses')]
            part_ids = batch_data[data_features.index('part_ids')]
            cur_batch_size = gt_vis(part_pcs, part_poses, part_valids, num_ins, args=args)
            num_ins += cur_batch_size
            # continue

        # process.
        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0) 
        part_valids = torch.cat(batch_data[data_features.index('part_valids')], dim=0) 
        gt_part_poses = torch.cat(batch_data[data_features.index('part_poses')], dim=0)  
        match_ids = batch_data[data_features.index('match_ids')]  
        part_ids = torch.cat(batch_data[data_features.index('part_ids')], dim=0)  
        contact_points = torch.cat(batch_data[data_features.index('contact_points')], dim=0)  
        sym_info = torch.cat(batch_data[data_features.index('sym')], dim=0)  

        # print("Processing...")

        if args.gpu is not None:
            part_pcs = part_pcs.cuda(args.gpu, non_blocking=True)
            part_valids = part_valids.cuda(args.gpu, non_blocking=True)
            gt_part_poses = gt_part_poses.cuda(args.gpu, non_blocking=True)
            part_ids = part_ids.cuda(args.gpu, non_blocking=True)
            contact_points = contact_points.cuda(args.gpu, non_blocking=True)
            sym_info = sym_info.cuda(args.gpu, non_blocking=True)

        # compute output.
        with torch.no_grad():
            trans_l2_loss, rot_cd_loss, part_cd_loss, shape_cd_loss, contact_point_loss, acc_per_batch, valid_per_batch, \
                contact_correct_per_batch, contact_point_per_batch, batch_size, output, cdsV1_sum, cdsV2_sum \
                = model(part_pcs, part_valids, gt_part_poses, match_ids, part_ids,
                        contact_points=contact_points, sym_info=sym_info)

            if args.save_vis and (i in VIS_BATCH_INDICES):
                visualize_batch(args, output["pred_poses"], gt_part_poses, part_pcs, part_valids, num_ins, i)

            if args.pred_encoder_vis:
                root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                if root_dir not in args.pred_encoder_vis_dir:
                    args.pred_encoder_vis_dir = os.path.join(root_dir, args.pred_encoder_vis_dir)

                pred_poses = output["pred_poses"]
                _ = pred_pose_vis(part_pcs, pred_poses, part_valids, num_ins, args=args)

            pred_rot = output['pred_rot']
            pred_trans = output['pred_trans']
            gt_rot = output['gt_rot']
            gt_trans = output['gt_trans']


            # cal.
            sum_cdsV1_sum += cdsV1_sum
            sum_cdsV2_sum += cdsV2_sum

            sum_part_cd_loss += part_cd_loss * batch_size
            sum_shape_cd_loss += shape_cd_loss * batch_size
            sum_rot_cd_loss += rot_cd_loss * batch_size
            sum_trans_l2_loss += trans_l2_loss * batch_size
            sum_contact_point_loss += contact_point_loss * batch_size

            total_acc_part += acc_per_batch
            total_valid_part += valid_per_batch
            total_contact_correct += contact_correct_per_batch
            total_contact_point += contact_point_per_batch
            num_ins += batch_size
            real_val_data_set += part_pcs.shape[0]

        # measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        shape_chamfer_dist.update(shape_cd_loss.item(), batch_size)
        trans_l2_loss_dist.update(trans_l2_loss.item(), batch_size)
        rot_cd_loss_dist.update(rot_cd_loss.item(), batch_size)
        part_cd_loss_dist.update(part_cd_loss.item(), batch_size)
        part_acc.update((acc_per_batch / valid_per_batch).item() * 100., batch_size)

        cdsV1_loss_dist.update(cdsV1_sum.item(), part_pcs.shape[0])
        cdsV2_loss_dist.update(cdsV2_sum.item(), part_pcs.shape[0])

        connectivity_acc.update((contact_correct_per_batch / contact_point_per_batch).item() * 100., batch_size)
        if i % args.print_freq == 0:
            infos = progress.display(i)

    # compute results.
    res_shape_cd = sum_shape_cd_loss / num_ins
    res_part_acc = total_acc_part / total_valid_part * 100.
    res_contact_acc = total_contact_correct / total_contact_point * 100.
    real_total_cdsV1 = sum_cdsV1_sum / real_val_data_set
    real_total_cdsV2 = sum_cdsV2_sum / real_val_data_set

# Now gather results from all GPUs
    if dist.is_initialized():
        all_res = {
            'shape_cd': torch.tensor([res_shape_cd.item()], device=part_pcs.device),
            'part_acc': torch.tensor([res_part_acc.item()], device=part_pcs.device),
            'contact_acc': torch.tensor([res_contact_acc.item()], device=part_pcs.device),
            'qds': torch.tensor([real_total_cdsV1.item()], device=part_pcs.device),
            'wqds': torch.tensor([real_total_cdsV2.item()], device=part_pcs.device),
            'rank': torch.tensor([args.rank], device=part_pcs.device),
        }

        # Create a placeholder for all processes
        gathered_results = {}
        for key in all_res:
            gathered_results[key] = [torch.zeros_like(all_res[key]) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_results[key], all_res[key])

        # Print results from all GPUs
        res_info = "\n==========================================================\n"
        for i in range(dist.get_world_size()):
            res_info += f"Rank {gathered_results['rank'][i].item()}: " + \
                       f"Shape Chamfer Distance: {gathered_results['shape_cd'][i].item():.9f} | " + \
                       f"Part Accuracy: {gathered_results['part_acc'][i].item():.9f} | " + \
                       f"Connectivity Accuracy: {gathered_results['contact_acc'][i].item():.9f} | " + \
                       f"QDS: {gathered_results['qds'][i].item():.9f} | " + \
                       f"WQDS: {gathered_results['wqds'][i].item():.9f}\n"
        res_info += "==========================================================\n"

        # All processes write the same collected results
        print(res_info)
        log_writer.write(res_info + "\n")
        log_writer.flush()
    else:
        # If not distributed, just print the local results
        print("==========================================================")
        print(f"Shape Chamfer Distance: {res_shape_cd.item()}")
        print(f"Part Accuracy: {res_part_acc.item()}")
        print(f"Connectivity Accuracy: {res_contact_acc.item()}")
        print(f'QDS: {real_total_cdsV1.item():.9f}')
        print(f'WQDS: {real_total_cdsV2.item():.9f}')
        print("==========================================================")

        res_info = "\n==========================================================" + "\n" + \
               f"Shape Chamfer Distance: {res_shape_cd.item()}" + "\t" + \
               f"Part Accuracy: {res_part_acc.item()}" + "\t" + \
               f"Connectivity Accuracy: {res_contact_acc.item()}" + "\t" + \
               f"QDS: {real_total_cdsV1.item():.9f}" + "\t" + \
               f"WQDS: {real_total_cdsV2.item():.9f}" + "\n" + \
               "==========================================================\n"
        log_writer.write(res_info + "\n")
        log_writer.flush()

    # return res_part_acc.item()
    return res_part_acc.item(), res_contact_acc.item()



def save_ply(file_path, points):
    vertices = [(float(x), float(y), float(z)) for x, y, z in points]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices_array = np.array(vertices, dtype=dtype)
    ply_element = PlyElement.describe(vertices_array, 'vertex')
    PlyData([ply_element]).write(file_path)

def visualize_and_save(args, pred_poses, gt_poses, part_pcs, part_valids, batch_idx):
    if args.save_vis:
        os.makedirs(args.save_dir, exist_ok=True)

        if batch_idx >= part_valids.shape[0]:
            print(f"Warning: batch_idx {batch_idx} exceeds part_valids first dimension size {part_valids.shape[0]}")
            return
            
        num_parts = part_pcs.shape[1]
        
        for p in range(num_parts):
            try:
                if part_valids[batch_idx, p] == 0:
                    continue

                gt_trans = gt_poses[batch_idx, p, :3].cpu().numpy()
                gt_rot = gt_poses[batch_idx, p, 3:7].cpu().numpy()
                pred_trans = pred_poses[batch_idx, -1, p, :3].cpu().numpy()  
                pred_rot = pred_poses[batch_idx, -1, p, 3:7].cpu().numpy()
                part_pc = part_pcs[batch_idx, p].cpu().numpy()

                gt_rot = torch.from_numpy(gt_rot).unsqueeze(0).repeat(part_pc.shape[0], 1)
                pred_rot = torch.from_numpy(pred_rot).unsqueeze(0).repeat(part_pc.shape[0], 1)
                part_pc = torch.from_numpy(part_pc)

                gt_pc = qrot(gt_rot, part_pc).numpy() + gt_trans
                pred_pc = qrot(pred_rot, part_pc).numpy() + pred_trans


                save_ply(os.path.join(args.save_dir, f"batch_{batch_idx}_part_{p}_gt.ply"), gt_pc)
                save_ply(os.path.join(args.save_dir, f"batch_{batch_idx}_part_{p}_pred.ply"), pred_pc)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx} part {p}: {str(e)}")
                print(f"Shapes - part_valids: {part_valids.shape}, part_pcs: {part_pcs.shape}, "
                      f"gt_poses: {gt_poses.shape}, pred_poses: {pred_poses.shape}")
                continue


def save_colored_ply(file_path, points, colors):
    # Ensure numpy arrays
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.detach().cpu().numpy()

    vertices = [(p[0], p[1], p[2], c[0], c[1], c[2])
                for p, c in zip(points, colors)]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_data = np.array(vertices, dtype=vertex_dtype)

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write(file_path)

def get_distinct_colors(n):
    # Predefined palette (cycled if needed).
    predefined_colors = [
        [222, 236, 246],
        [175, 200, 226],
        [226, 242, 205],
        [182, 218, 167],
        [249, 213, 213],
        [239, 152, 161],
        [251, 227, 192],
        [251, 201, 154],
        [232, 224, 239],
        [194, 177, 215],
    ]
    colors = []
    for i in range(n):
        colors.append(predefined_colors[i % len(predefined_colors)])
    return np.array(colors)

def visualize_batch(args, pred_poses, gt_poses, part_pcs, part_valids, start_global_idx, batch_idx):
    if not args.save_vis:
        return

    # Handle pred_poses possibly being a list
    if isinstance(pred_poses, list):
        pred_poses = pred_poses[-1]

    # Check if pred_poses has iteration dimension [B, Iter, P, 7]
    has_iter = (pred_poses.dim() == 4)

    batch_size = part_pcs.shape[0]
    num_parts = part_pcs.shape[1]

    part_colors = get_distinct_colors(num_parts)

    for b in range(batch_size):
        global_idx = start_global_idx + b

        dir_assembled = os.path.join(args.save_dir, 'assembled', f'shape_{global_idx}')
        os.makedirs(dir_assembled, exist_ok=True)

        dir_parts_root = os.path.join(args.save_dir, 'part', f'shape_{global_idx}')

        assembled_gt_pts = []
        assembled_pred_pts = []
        assembled_gt_colors = []
        assembled_pred_colors = []

        valid_parts_found = False

        for p in range(num_parts):
            if part_valids[b, p] == 0:
                continue

            valid_parts_found = True

            pc = part_pcs[b, p].float()
            num_points = pc.shape[0]

            gt_t = gt_poses[b, p, :3].float()
            gt_r = gt_poses[b, p, 3:].float()

            if has_iter:
                pred_t = pred_poses[b, -1, p, :3].float()
                pred_r = pred_poses[b, -1, p, 3:].float()
            else:
                pred_t = pred_poses[b, p, :3].float()
                pred_r = pred_poses[b, p, 3:].float()

            gt_r_exp = gt_r.view(1, 4).expand(num_points, -1)
            pred_r_exp = pred_r.view(1, 4).expand(num_points, -1)

            gt_pc_trans = qrot(gt_r_exp.cpu(), pc.cpu()) + gt_t.view(1, 3).cpu()
            pred_pc_trans = qrot(pred_r_exp.cpu(), pc.cpu()) + pred_t.view(1, 3).cpu()

            gt_pc_np = gt_pc_trans.numpy()
            pred_pc_np = pred_pc_trans.numpy()

            color = part_colors[p]
            color_arr = np.tile(color, (num_points, 1))

            dir_curr_part = os.path.join(dir_parts_root, f'part_{p}')
            os.makedirs(dir_curr_part, exist_ok=True)

            save_colored_ply(os.path.join(dir_curr_part, 'gt.ply'), gt_pc_np, color_arr)
            save_colored_ply(os.path.join(dir_curr_part, 'pred.ply'), pred_pc_np, color_arr)

            assembled_gt_pts.append(gt_pc_np)
            assembled_pred_pts.append(pred_pc_np)
            assembled_gt_colors.append(color_arr)
            assembled_pred_colors.append(color_arr)

        if valid_parts_found:
            assembled_gt_pts = np.concatenate(assembled_gt_pts, axis=0)
            assembled_pred_pts = np.concatenate(assembled_pred_pts, axis=0)
            assembled_gt_colors = np.concatenate(assembled_gt_colors, axis=0)
            assembled_pred_colors = np.concatenate(assembled_pred_colors, axis=0)

            save_colored_ply(os.path.join(dir_assembled, 'assembled_gt.ply'),
                             assembled_gt_pts, assembled_gt_colors)
            save_colored_ply(os.path.join(dir_assembled, 'assembled_pred.ply'),
                             assembled_pred_pts, assembled_pred_colors)

    end_global_idx = start_global_idx + batch_size - 1
    print(f"--> [Vis] Batch {batch_idx}: saved shape {start_global_idx} to {end_global_idx} in {args.save_dir}")






