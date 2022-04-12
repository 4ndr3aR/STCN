import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class GenericTestDataset(Dataset):
    def __init__(self, data_root, res=240):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        if res != -1:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(res, interpolation=InterpolationMode.BICUBIC),
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])

            self.mask_transform = transforms.Compose([
            ])

    def process_one_image(self, i, f, info, vid_im_path, vid_gt_path, images, masks, debug=False):
        if debug:
            print(f'GenericTestDataset.__getitem__() - LOOP START - i: {i} - f: {f}')
        img = Image.open(path.join(vid_im_path, f)).convert('RGB')
        #images.append(self.im_transform(img))
        images[i] = self.im_transform(img)
        
        mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
        if path.exists(mask_file):
            mask = Image.open(mask_file).convert('P')
            palette = mask.getpalette()
            #masks.append(np.array(mask, dtype=np.uint8))
            masks[i] = np.array(mask, dtype=np.uint8)
            this_labels = np.unique(masks[-1])
            this_labels = this_labels[this_labels!=0]
            info['gt_obj'][i]  = this_labels
            print(f'info["gt_obj"][i]: {info["gt_obj"][i]}')
            info['palette'][i] = palette
            if debug:
                print(f'Setting palette in dict with key: {i}')
        else:
            # Mask not exists -> nothing in it
            #masks.append(np.zeros(self.shape[video]))
            video = info['name']
            masks[i] = np.zeros(self.shape[video])
        if debug:
            print(f'GenericTestDataset.__getitem__() - LOOP END   - i: {i} - f: {f}')

    def __getitem__(self, idx, debug=True):
        print(f'GenericTestDataset.__getitem__() - START - idx: {idx}')
        video = self.videos[idx]
        info = {}
        info['name']    = video
        info['frames']  = self.frames[video] 
        info['size']    = self.shape[video] # Real sizes
        info['gt_obj']  = {} # Frames with labelled objects
        info['palette'] = {} # Palettes of labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video)

        frames = self.frames[video]


        #pool = multiprocessing.Pool(4)
        #out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))


        images = [None] * len(frames)
        masks  = [None] * len(frames)
        #images = []
        #masks  = []
        for i, f in enumerate(frames):
            self.process_one_image(i, f, info, vid_im_path, vid_gt_path, images, masks, debug=debug)
            '''
            print(f'GenericTestDataset.__getitem__() - LOOP START - i: {i} - f: {f}')
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))
            
            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            if path.exists(mask_file):
                mask = Image.open(mask_file).convert('P')
                palette = mask.getpalette()
                masks.append(np.array(mask, dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
                print(f'info["gt_obj"][i]: {info["gt_obj"][i]}')
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))
            print(f'GenericTestDataset.__getitem__() - LOOP END   - i: {i} - f: {f}')
            '''

        for i, f in enumerate(frames):
            if images[i] is None:
                print(f'images[{i}] is None!')
            if masks[i] is None:
                print(f'masks[{i}] is None!')
            if i in info['palette']:
                if debug:
                    print(f'key: {i} exists into dict: {info["palette"]} and has value: {info["palette"][i]}')
                #palette = info['palette'][i]
            print(type(images[i]), type(masks[i]), images[i].shape, masks[i].shape)
        
        images = torch.stack(images, 0)
        masks  = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
            'palette': np.array(palette),
        }

        print(f'GenericTestDataset.__getitem__() - END   - idx: {idx}')

        #attrs = vars(self)
        #print(', '.join("%s: %s" % item for item in attrs.items()))

        print(data, len(data), type(data))

        return data

    def __len__(self):
        return len(self.videos)
