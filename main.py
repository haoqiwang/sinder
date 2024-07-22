#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sinder import (
    get_neighbor_loss,
    get_tokens,
    load_data,
    load_model,
    load_visual_data,
    pca_array,
    replace_back,
    replace_linear_addition_noqk,
)

os.environ['XFORMERS_DISABLED'] = '1'
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description='Beautify network')
    parser.add_argument(
        '--model', type=str, default='dinov2_vitg14', help='config file'
    )
    parser.add_argument('--work_dir', type=str, default='results')
    parser.add_argument('--resolution', type=int, default=518)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_iter', type=int, default=30000)
    parser.add_argument('--num_train_max', type=int, default=30000)
    parser.add_argument('--mask_thr', type=float, default=4)
    parser.add_argument('--skip_less_than', type=int, default=3)
    parser.add_argument('--visual_size', type=int, default=448 * 2)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--save_at_skip', type=int, nargs='+', default=[75])
    parser.add_argument('--limit_layers', type=int, default=10)

    args = parser.parse_args()
    return args


def prepare_train(args, model):
    model.train()

    all_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False

    replace_linear_addition_noqk(model, 'model')
    for name, param in model.named_parameters():
        if '.epsilon' in name and param.requires_grad is True:
            all_params.append(param)

    grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params.append(name)

    assert len(grad_params) == len(all_params)
    print(len(grad_params), grad_params)
    print(len(all_params), all_params)
    optimizer = torch.optim.SGD(
        all_params,
        lr=args.lr,
        momentum=0.9,
    )

    return optimizer


def save_model(args, model):
    print('save model')
    model.eval()

    replace_back(model, 'model')

    torch.save(model.state_dict(), args.folder / 'model.pt')


def train(args, model, dataset, optimizer, visual_dataset):
    print('training')
    skip_history = [False] * 1000
    model.train()

    for global_iter in tqdm(range(args.max_iter)):
        img = dataset[global_iter % len(dataset)]
        H = img.shape[1] // model.patch_size
        W = img.shape[2] // model.patch_size
        density = np.array(skip_history[-1000:]).astype(float).mean()
        print(f'{global_iter=} {W=} {H=} {density=:.2f}')

        for percent in args.save_at_skip:
            if percent / 100 <= density:
                print(f'save checkpoint at {density=}')
                args.save_at_skip.remove(percent)
                torch.save(model, args.folder / f'checkpoint_p{percent}.pth')
        if len(args.save_at_skip) == 0:
            break

        model.zero_grad()

        model.train()
        with torch.enable_grad():
            image_batch = img.unsqueeze(0).cuda()
            result = get_neighbor_loss(
                model,
                image_batch,
                skip_less_than=args.skip_less_than,
                mask_thr=args.mask_thr,
                kernel=args.kernel,
            )

        if result is None:
            skip_history.append(True)
            print('no loss, skip')
        else:
            skip_history.append(False)
            (
                layer,
                loss,
                I,
                J,
                T,
                alpha,
                mask_angle,
                x_token,
            ) = result

            print(
                f'{global_iter=}, {layer=}, {density=}, {alpha=:.2f}, {len(I)=}, '
                f'{loss.item()=:.2f}'
            )

            if torch.isnan(loss).any():
                print('nan loss, skip')
                continue
            loss.backward()

            # set some grad to 0
            if args.limit_layers:
                with torch.no_grad():
                    for t in range(layer - args.limit_layers + 1):
                        for p in model.blocks[t].parameters():
                            p.grad = None

            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f'nan grad at {name}, skip')
                    has_nan = True
            if has_nan:
                continue
            optimizer.step()

            # visualize
            if global_iter % 100 == 0:
                try:
                    print(f'visualization at {global_iter=}')
                    pca_img = pca_array(x_token)
                    pca_img.save(args.folder / 'pca.png')
                    mask_img = Image.fromarray(
                        (mask_angle * 255)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    ).resize((W * 7, H * 7), resample=Image.NEAREST)
                    mask_img.save(args.folder / 'mask.png')
                    Image.fromarray(
                        (
                            (
                                img.permute((1, 2, 0)).cpu().numpy() * 0.22
                                + 0.45
                            )
                            * 255
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    ).save(args.folder / 'img.png')
                    if global_iter % 1000 == 0:
                        pca_img.save(args.folder / f'{global_iter:05}_pca.png')
                        mask_img.save(
                            args.folder / f'{global_iter:05}_mask.png'
                        )
                        Image.fromarray(
                            (
                                (
                                    img.permute((1, 2, 0)).cpu().numpy() * 0.22
                                    + 0.45
                                )
                                * 255
                            )
                            .clip(0, 255)
                            .astype(np.uint8)
                        ).save(args.folder / f'{global_iter:05}_img.png')
                except Exception as e:
                    print(e)

                for d in range(len(visual_dataset)):
                    visual_image = visual_dataset[d]
                    visual_tokens_all = get_tokens(model, visual_image)
                    visual_tokens, visual_tokens_cls = zip(*visual_tokens_all)
                    pca_img = pca_array(visual_tokens[-1])
                    pca_img.save(args.folder / f'{d}_pca.png')
                    if global_iter % 500 == 0:
                        pca_img.save(
                            args.folder / f'{global_iter:05}_{d}_pca.png'
                        )

    torch.save(model, args.folder / 'checkpoint.pth')


def main():
    print('Start beautify')
    args = parse_args()

    name = f'res{args.resolution}_lr{args.lr}_{args.num_train_max}_skipless{args.skip_less_than}_maskthr{args.mask_thr}_limit{args.limit_layers}_ker{args.kernel}'
    args.folder = Path(args.work_dir) / name
    os.makedirs(args.folder, exist_ok=True)
    print(args)
    print(' '.join(sys.argv))
    print(f'work dir {args.folder}')
    model = load_model(args.model)
    dataset = load_data(args, model)
    visual_dataset = load_visual_data(args, model)
    optimizer = prepare_train(args, model)
    train(args, model, dataset, optimizer, visual_dataset)
    save_model(args, model)


if __name__ == '__main__':
    main()
