#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred            

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_lvl_list


def lerf_localization(sem_map, image, clip_model, image_name, query_list, mask_thresh):
    """
    Localize objects using activation heatmaps from CLIP features.    
    Args:
        sem_map (torch.Tensor): Semantic feature maps from the model [n_head, n_prompt, h, w]
        image (torch.Tensor): Original RGB image [h, w, 3]
        clip_model: CLIP model instance for feature processing
        image_name (Path): Output path for saving visualizations
        query_list (List[str]): List of object queries to localize
        mask_thresh (float): Threshold for activation map (0.0-1.0)
    
    Returns:
        Dict: Detection results.
    """
    output_path_loca = Path(image_name) / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)
    print(f"Saving visualizations to: {output_path_loca}")
    
    valid_map = clip_model.get_max_across(sem_map)
    n_head, n_prompt, h, w = valid_map.shape
    
    detection_results = {}
    
    for k in range(len(query_list)):
        try:
            # Get activation map for current query
            select_output = valid_map[:, k] 
            
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = select_output.cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
            
            # Find best activation across attention heads
            score_lvl = np.zeros((n_head,))
            coord_lvl = []
            for i in range(n_head):
                score = avg_filtered[..., i].max()
                score_lvl[i] = score
                coord = np.nonzero(avg_filtered[..., i] == score)
                coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

            selec_head = np.argmax(score_lvl)
            coord_final = coord_lvl[selec_head]
            
            heatmap = avg_filtered[..., selec_head]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            if mask_thresh < 0.3:
                heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                _, binary_mask = cv2.threshold(heatmap_uint8, 0, 255, 
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_mask = binary_mask.astype(np.uint8) // 255
            else:
                binary_mask = (heatmap > mask_thresh).astype(np.uint8)
            
            kernel_size = max(3, int(min(heatmap.shape) * 0.02))  # Dynamic kernel size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            min_size = max(50, int(total_pixels * 0.001))  # At least 0.1% of image
            
            valid_components = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_size:
                    valid_components.append((i, area))
            
            if valid_components:
                valid_components.sort(key=lambda x: x[1], reverse=True)
                largest_comp = valid_components[0][0]
                
                margin = 0.05  # 5% margin for bounding box
                x = max(0, stats[largest_comp, cv2.CC_STAT_LEFT])
                y = max(0, stats[largest_comp, cv2.CC_STAT_TOP])
                w = stats[largest_comp, cv2.CC_STAT_WIDTH]
                h = stats[largest_comp, cv2.CC_STAT_HEIGHT]
                
                margin_x = int(w * margin)
                margin_y = int(h * margin)
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                w = min(binary_mask.shape[1] - x, w + 2 * margin_x)
                h = min(binary_mask.shape[0] - y, h + 2 * margin_y)
                
                bbox = [x, y, x + w, y + h]
                
                detection_results[query_list[k]] = {
                    'bbox': bbox,
                    'confidence': float(score_lvl[selec_head]),
                    'area': stats[largest_comp, cv2.CC_STAT_AREA]
                }

                vis_img = image.cpu().numpy()
                if vis_img.max() <= 1.0:
                    vis_img = (vis_img * 255).astype(np.uint8)
                else:
                    vis_img = vis_img.astype(np.uint8)
                
                heatmap_vis = (heatmap * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                
                alpha = 0.5
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                blended = cv2.addWeighted(vis_img, 1-alpha, heatmap_colored, alpha, 0)

                cv2.rectangle(blended, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)

                label = f"{query_list[k]} ({score_lvl[selec_head]:.2f})"
                cv2.putText(blended, label, 
                           (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                debug_path = output_path_loca / 'debug'
                debug_path.mkdir(exist_ok=True)
                save_path_cv = debug_path /f"{query_list[k]}.png"
                
                cv2.imwrite(str(save_path_cv), blended)
                cv2.imwrite(str(debug_path / f"{query_list[k]}_mask.png"), binary_mask * 255)
                cv2.imwrite(str(debug_path / f"{query_list[k]}_heatmap.png"), heatmap_vis)
            
                avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
                torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
                p_i = torch.clip(torch_relev - 0.5, 0, 1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
                mask = (torch_relev < 0.5).squeeze()
                valid_composited[mask, :] = image[mask, :] * 0.3
                
                save_path = output_path_loca / f"{query_list[k]}.png"
                show_result(valid_composited.cpu().numpy(), coord_final,
                            np.array(bbox), save_path)
        except Exception as e:
            print(f"Error processing query '{query_list[k]}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return detection_results


def evaluate(query_list: list[str], feat_dir: list[str], output_path: str, ae_ckpt_path: str, image_dir: str, mask_thresh: float, encoder_hidden_dims: list[int], decoder_hidden_dims: list[int], logger: logging.Logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    image_paths = sorted(glob.glob(os.path.join(str(image_dir), '*.jpg' or '*.png')))
    image_shape = cv2.imread(image_paths[0]).shape[:2]
    eval_index_list = [int(idx) for idx in range(len(image_paths))]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_lvl_list = []
    results_list = []
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1) 
        
        clip_model.set_positives(query_list)
        
        c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        chosen_lvl_list.extend(c_lvl)

        results = lerf_localization(restored_feat, rgb_img, clip_model, image_name, query_list, mask_thresh)
        results_list.append(results)

    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--query_list", nargs='+', type=str, default=None)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    image_dir = args.image_dir
    query_list = list(args.query_list)
    if query_list is None:
        raise ValueError("query_list is required")
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    logger.info(f"Processing {dataset_name} model with query_list: {query_list}\n")

    evaluate(query_list, feat_dir, output_path, ae_ckpt_path, image_dir, mask_thresh, args.encoder_dims, args.decoder_dims, logger)