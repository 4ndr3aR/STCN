"""
Generic evaluation script 
The segmentation mask for each object when they first appear is required
(YouTubeVOS style, but dense)

Optimized for compatibility, not speed.
We will resize the input video to 480p -- check generic_test_dataset.py if you want to change this behavior
AMP default on.

Usage: python eval_generic.py --data_path <path to data_root> --output <some output path>

Data format:
    data_root/
        JPEGImages/
            video1/
                00000.jpg
                00001.jpg
                ...
            video2/
                ...
        Annotations/
            video1/
                00000.png
            video2/
                00000.png
            ...
"""


import os
from os import path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.generic_test_dataset import GenericTestDataset
from util.tensor_util import unpad
from util.blend_img_mask import blend_img_mask
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--data_path')
parser.add_argument('--output')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp_off', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

data_path = args.data_path
out_path = args.output
args.amp = not args.amp_off

# Simple setup
os.makedirs(out_path, exist_ok=True)
torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = GenericTestDataset(data_root=data_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top
prop_model = STCN().cuda().eval()
prop_model.load_state_dict(prop_saved)

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        print(f'rgb.shape: {rgb.shape}')

        msk = data['gt'][0]
        print(f'msk.shape: {msk.shape}')

        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']
        palette = data['palette'][0]

        print('Processing', name, '...')

        '''
        imgs_dir      = test_dataset.image_dir
        rgb_jpg_str   = info['frames'][3][0]
        print(imgs_dir + '/' + name + '/' + rgb_jpg_str)
        '''

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))
        processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                    mem_every=args.mem_every, include_last=args.include_last)

        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            print(f'eval_generic START - i: {i} - frame_idx: {frame_idx} - min_idx: {min_idx}')
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            # Append the background label
            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                msk[:,frame_idx],
            ], 0).cuda()

            # We perform propagation from the current frame to the next frame with label
            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)
            print(f'eval_generic - END   - i: {i} - frame_idx: {frame_idx} - min_idx: {min_idx}')

        # Do unpad -> upsample to original size (we made it 480p)
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx

        print(info['frames'])

        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                imgs_dir      = test_dataset.image_dir
                rgb_jpg_str   = info['frames'][f][0]
                src_img_fn    = imgs_dir + '/' + name + '/' + rgb_jpg_str
                combo_jpg_str = info['frames'][f][0]
                mask_png_str  = info['frames'][f][0].replace('.jpg','.png')
                img_E = Image.fromarray(idx_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(this_out_path, mask_png_str))

                mask = img_E
                out_fn = os.path.join(this_out_path, combo_jpg_str)
                blend_img_mask(src_img_fn, mask, out_fn, 0.5)


                '''
                cv_mask = np.array(img_E) 
                print(f'{img_E.size, cv_mask.shape}')
                '''

                '''
                print(f'{rgb[:,f][0].shape}')
                print(f'{rgb[f].shape}')
                cv_rgb  = rgb[:,f][0].numpy()
                print(f'{cv_rgb.shape}')
                '''

                '''
'rgb': ...,
'gt': ...,
'info': {'name': 'video1', 'frames': ['ericsson-camera-360-00.jpg', 'ericsson-camera-360-02.jpg', 'ericsson-camera-360-03.jpg', 'ericsson-camera-360-04.jpg', 'ericsson-camera-360-05.jpg', 'ericsson-camera-360-06.jpg', 'ericsson-camera-360-07.jpg', 'ericsson-camera-360-08.jpg', 'ericsson-camera-360-09.jpg'], 'size': (1920, 3840), 'gt_obj': {0: array([ 11,  13,  14,  15,  16,  17,  46,  47,  52,  53,  59,  60,  88,
        89,  90,  94,  95,  96,  97, 102, 131, 132, 133, 138, 139],
      dtype=uint8)}, 'label_convert': {11: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 46: 7, 47: 8, 52: 9, 53: 10, 59: 11, 60: 12, 88: 13, 89: 14, 90: 15, 94: 16, 95: 17, 96: 18, 97: 19, 102: 20, 131: 21, 132: 22, 133: 23, 138: 24, 139: 25}, 'label_backward': {1: 11, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 46, 8: 47, 9: 52, 10: 53, 11: 59, 12: 60, 13: 88, 14: 89, 15: 90, 16: 94, 17: 95, 18: 96, 19: 97, 20: 102, 21: 131, 22: 132, 23: 133, 24: 138, 25: 139}, 'labels': array([ 11,  13,  14,  15,  16,  17,  46,  47,  52,  53,  59,  60,  88,
        89,  90,  94,  95,  96,  97, 102, 131, 132, 133, 138, 139],
      dtype=uint8)}, 'palette': array([...
                '''

                '''
                stuff = test_dataset[f]
                #print(len(stuff), stuff)
                #print(inputs.shape, labels.shape)
                print(stuff['rgb'].shape, stuff['gt'].shape)
                '''


                '''
                #print(os.path.join(data_path, rgb_jpg_str))
                cv_rgb          = cv2.imread(src_img_fn)
                cv_mask_rgb     = cv2.cvtColor(cv_mask, cv2.COLOR_GRAY2RGB)
                print(f'{cv_rgb.shape} - {cv_mask_rgb.shape}')
                combo_image = cv2.addWeighted(cv_rgb, 0.4, cv_mask_rgb, 0.1, 0)
                combo_image_permuted = np.transpose(combo_image , (2, 0, 1))
                #combo_image_permuted.save(os.path.join(this_out_path, combo_jpg_str))
                out_fn = os.path.join(this_out_path, combo_jpg_str)
                print(f'Saving combo image: {out_fn}')
                cv2.imwrite(out_fn, combo_image)
                '''

        del rgb
        del msk
        del processor
