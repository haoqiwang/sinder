#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import cv2

import numpy as np
from PIL import Image
from tqdm import tqdm

from sinder import (
    get_tokens,
    load_model,
    load_visual_data,
    pca_array,
)

os.environ['XFORMERS_DISABLED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument(
        'imgs', nargs='+', type=str, help='path to image/images'
    )
    parser.add_argument(
        '--model', type=str, default='dinov2_vitg14', help='model name'
    )
    parser.add_argument('--workdir', type=str, default='visualize')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint. Default is None, which loads the official pretrained weights',
    )
    parser.add_argument(
        '--visual_size',
        type=int,
        default=518,
        help='short side size of input image',
    )

    args = parser.parse_args()
    return args


def visualize(args, model, visual_dataset):
    model.eval()

    for d in tqdm(range(len(visual_dataset))):
        visual_image = visual_dataset[d]
        visual_tokens_all = get_tokens(model, visual_image)
        visual_tokens, visual_tokens_cls = zip(*visual_tokens_all)
        filename = Path(visual_dataset.files[d]).stem

        t = visual_tokens[-1].detach().cpu()
        h, w, c = t.shape
        norm = ((t.norm(dim=-1) / t.norm(dim=-1).max()) * 255).byte().numpy()
        norm_img = Image.fromarray(norm).resize((w * 14, h * 14), 0)
        norm = cv2.applyColorMap(np.array(norm_img), cv2.COLORMAP_JET)
        cv2.imwrite(args.folder / f'{filename}_norm.png', norm)

        pca_img = pca_array(visual_tokens[-1])
        pca_img.save(args.folder / f'{filename}_pca.png')


def main():
    args = parse_args()

    args.folder = Path(args.workdir).expanduser()
    os.makedirs(args.folder, exist_ok=True)
    print(args)
    print(' '.join(sys.argv))

    model = load_model(args.model, args.checkpoint)
    visual_dataset = load_visual_data(args, model)
    visualize(args, model, visual_dataset)


if __name__ == '__main__':
    main()
