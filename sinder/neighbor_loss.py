import torch
from torch.nn.functional import normalize


def check_anomaly_theoretical(
    x,
    H,
    W,
    anomaly_dir=None,
    temperature=0.1,
    mask_thr=0.001,
    kernel=3,
):
    x_token = x[:, 1:]
    B = x.shape[0]
    assert B == 1
    x_token = x_token.reshape(H, W, -1).contiguous()

    with torch.no_grad():
        feature = normalize(x_token, dim=-1)
        direction = normalize(anomaly_dir, dim=-1)

        logits = -(feature * direction).sum(dim=-1).abs()
        prob = torch.exp(logits / temperature)

        assert kernel in (3, 5)
        pad = kernel // 2

        w = prob.unfold(0, kernel, 1).unfold(1, kernel, 1)
        w = w / w.sum(dim=(-1, -2), keepdims=True)

        if kernel == 3:
            gaussian = (
                torch.FloatTensor(
                    [
                        1 / 16,
                        1 / 8,
                        1 / 16,
                        1 / 8,
                        1 / 4,
                        1 / 8,
                        1 / 16,
                        1 / 8,
                        1 / 16,
                    ]
                )
                .to(w.device)
                .reshape(1, 1, 3, 3)
            )
        elif kernel == 5:
            gaussian = (
                torch.tensor(
                    [
                        [1, 4, 7, 4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4, 7, 4, 1],
                    ]
                )
                .float()
                .to(w.device)
                / 273
            )

        w2 = w * gaussian

        w2 = w2 / w2.sum(dim=(-1, -2), keepdims=True)

        T = x_token.unfold(0, kernel, 1).unfold(1, kernel, 1)
        T = (T * w2[:, :, None].to(T.device)).sum(dim=(-1, -2))

        mask_full = logits < logits.mean() - mask_thr * logits.std()
        mask_full[:pad, :] = False
        mask_full[:, :pad] = False
        mask_full[-pad:, :] = False
        mask_full[:, -pad:] = False
        index_tensor = torch.nonzero(mask_full.flatten()).flatten()
        if len(index_tensor) == 0:
            return None
        rows = index_tensor // W
        cols = index_tensor % W

        alpha = x_token[pad:-pad, pad:-pad].norm(dim=-1).mean()

    loss_neighbor = (
        (x_token[rows, cols] - T[rows - pad, cols - pad]).norm(dim=-1)
    ).mean() / alpha

    return loss_neighbor, rows, cols, T, alpha, mask_full, x_token


def get_neighbor_loss(
    model,
    x,
    skip_less_than=1,
    mask_thr=0.001,
    kernel=3,
):
    H = x.shape[2]
    W = x.shape[3]
    x = model.prepare_tokens_with_masks(x)

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        assert len(model.singular_defects) > 0
        result = check_anomaly_theoretical(
            x,
            H // model.patch_size,
            W // model.patch_size,
            model.singular_defects[i],
            mask_thr=mask_thr,
            kernel=kernel,
        )
        if result is not None:
            (
                loss_neighbor,
                rows,
                cols,
                T,
                alpha,
                mask_angle,
                x_token,
            ) = result
            if len(rows) >= skip_less_than:
                assert not torch.isnan(loss_neighbor).any()
                return (
                    i,
                    loss_neighbor,
                    rows,
                    cols,
                    T,
                    alpha,
                    mask_angle,
                    x_token,
                )
    return None
