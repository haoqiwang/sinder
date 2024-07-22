# compute singular defect directions
import torch
import torch.nn.functional as F


def anomaly_dir_attn(
    blk,
    identity=False,
    bias=False,
    centered=False,
    homogeneous=False,
):
    with torch.no_grad():
        N = blk.ls1.gamma.shape[0]
        dev = blk.ls1.gamma.device

        A4 = torch.diag(blk.ls1.gamma)
        A3 = blk.attn.proj.weight
        B3 = blk.attn.proj.bias
        A2 = blk.attn.qkv.weight.chunk(3, dim=0)[-1]
        B2 = blk.attn.qkv.bias.chunk(3, dim=0)[-1]
        A1 = torch.diag(blk.norm1.weight)
        B1 = blk.norm1.bias
        A0 = (torch.eye(N) - 1 / N * torch.ones(N, N)).to(dev)
        A = A4 @ A3 @ A2 @ A1

        if centered:
            A = A @ A0
        B = A4 @ (A3 @ (A2 @ B1)) + A4 @ (A3 @ B2) + A4 @ B3

        if bias:
            A = torch.cat((A, B[:, None]), dim=1)
            if homogeneous:
                onehot = torch.cat(
                    (torch.zeros_like(B), torch.ones(1).to(dev))
                )
                A = torch.cat((A, onehot[None]), dim=0)

        if identity:
            iden = torch.eye(N).to(dev)
            A[:N, :N] += iden
        u, _, _ = torch.linalg.svd(A)

    return u[:N, 0], A, B


def w12(blk, x):
    with torch.no_grad():
        x1, x2 = blk.mlp.w12(x).chunk(2, dim=-1)
    return F.silu(x1) * x2


def anomaly_dir_mlp_ls(
    blk,
    identity=False,
    bias=False,
    centered=False,
    homogeneous=False,
    bias_ls=False,
):
    with torch.no_grad():
        N = blk.ls2.gamma.shape[0]
        M = blk.mlp.w3.weight.shape[1]
        dev = blk.ls2.gamma.device

        A4 = torch.diag(blk.ls2.gamma)
        A3 = blk.mlp.w3.weight
        B3 = blk.mlp.w3.bias

        X = torch.randn(100000, N, device=dev)
        Y = w12(blk, X)
        if bias_ls:
            X_one = torch.cat((X, torch.ones(100000, 1).to(dev)), dim=1)
        else:
            X_one = X
        sol = torch.linalg.lstsq(X_one, Y)
        if bias_ls:
            A2 = sol.solution.T[:, :-1]
            B2 = sol.solution.T[:, -1]
        else:
            A2 = sol.solution.T
            B2 = torch.zeros(M).to(dev)

        A1 = torch.diag(blk.norm2.weight)
        B1 = blk.norm2.bias
        A0 = (torch.eye(N) - 1 / N * torch.ones(N, N)).to(dev)
        A = A4 @ A3 @ A2 @ A1

        if centered:
            A = A @ A0
        B = A4 @ (A3 @ (A2 @ B1)) + A4 @ (A3 @ B2) + A4 @ B3

        if bias:
            A = torch.cat((A, B[:, None]), dim=1)
            if homogeneous:
                onehot = torch.cat(
                    (torch.zeros_like(B), torch.ones(1).to(dev))
                )
                A = torch.cat((A, onehot[None]), dim=0)

        if identity:
            iden = torch.eye(N).to(dev)
            A[:N, :N] += iden
        u, s, vt = torch.linalg.svd(A)

    return u[:N, 0], A, B


def anomaly_dir(blk, homogeneous=False):
    _, A, b = anomaly_dir_attn(
        blk,
        identity=True,
        bias=homogeneous,
        centered=True,
        homogeneous=homogeneous,
    )
    _, C, d = anomaly_dir_mlp_ls(
        blk,
        identity=True,
        bias=homogeneous,
        bias_ls=False,
        centered=True,
        homogeneous=homogeneous,
    )

    with torch.no_grad():
        N = b.shape[0]
        AA = C @ A
        if homogeneous:
            BB = 0
        else:
            BB = C @ b + d
        u, _, _ = torch.linalg.svd(AA)

    return u[:N, 0], AA, BB


def singular_defect_directions(model):
    accumulative_anomalies = []
    anomaly_dab = [anomaly_dir(blk) for blk in model.blocks]
    anomaly_as = [dab[1] for dab in anomaly_dab]

    with torch.no_grad():
        aaa = torch.eye(anomaly_as[0].shape[0]).to(anomaly_as[0])
        for a in anomaly_as:
            aaa = a @ aaa
            u, _, _ = torch.linalg.svd(aaa)
            accumulative_anomalies.append(u[:, 0])
    return accumulative_anomalies
