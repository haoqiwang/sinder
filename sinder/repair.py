import torch


class SVDLinearAddition(torch.nn.Module):
    def __init__(self, linear, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        w, b = linear.weight, linear.bias

        self.bias = b

        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(w, full_matrices=False)
            self.U = torch.nn.Parameter(U)
            self.S = torch.nn.Parameter(S)
            self.Vt = torch.nn.Parameter(Vt)
            self.epsilon = torch.nn.Parameter(torch.zeros_like(S))

    def forward(self, x):
        w = self.U @ torch.diag(self.S + self.epsilon) @ self.Vt
        x = torch.nn.functional.linear(x, w, self.bias)
        return x


class SVDLinearAdditionQKV(torch.nn.Module):
    def __init__(self, linear, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        w, b = linear.weight, linear.bias

        self.bias = b

        with torch.no_grad():
            c = w.shape[0] // 3
            q = w[:c]
            k = w[c : 2 * c]
            v = w[2 * c :]
            U_q, S_q, Vt_q = torch.linalg.svd(q, full_matrices=False)
            self.U_q = torch.nn.Parameter(U_q)
            self.S_q = torch.nn.Parameter(S_q)
            self.Vt_q = torch.nn.Parameter(Vt_q)
            self.epsilon_q = torch.nn.Parameter(torch.zeros_like(S_q))
            U_k, S_k, Vt_k = torch.linalg.svd(k, full_matrices=False)
            self.U_k = torch.nn.Parameter(U_k)
            self.S_k = torch.nn.Parameter(S_k)
            self.Vt_k = torch.nn.Parameter(Vt_k)
            self.epsilon_k = torch.nn.Parameter(torch.zeros_like(S_k))
            U_v, S_v, Vt_v = torch.linalg.svd(v, full_matrices=False)
            self.U_v = torch.nn.Parameter(U_v)
            self.S_v = torch.nn.Parameter(S_v)
            self.Vt_v = torch.nn.Parameter(Vt_v)
            self.epsilon_v = torch.nn.Parameter(torch.zeros_like(S_v))

    def forward(self, x):
        w = torch.concatenate(
            (
                self.U_q @ torch.diag(self.S_q + self.epsilon_q) @ self.Vt_q,
                self.U_k @ torch.diag(self.S_k + self.epsilon_k) @ self.Vt_k,
                self.U_v @ torch.diag(self.S_v + self.epsilon_v) @ self.Vt_v,
            )
        )
        x = torch.nn.functional.linear(x, w, self.bias)
        return x


def replace_linear_addition_noqk(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            if 'qkv' in attr_str:
                print('replaced: ', name, attr_str)
                svd_linear_qkv = SVDLinearAdditionQKV(target_attr)
                svd_linear_qkv.U_q.requires_grad = False
                svd_linear_qkv.U_k.requires_grad = False
                svd_linear_qkv.U_v.requires_grad = False
                svd_linear_qkv.S_q.requires_grad = False
                svd_linear_qkv.S_k.requires_grad = False
                svd_linear_qkv.S_v.requires_grad = False
                svd_linear_qkv.Vt_q.requires_grad = False
                svd_linear_qkv.Vt_k.requires_grad = False
                svd_linear_qkv.Vt_v.requires_grad = False
                svd_linear_qkv.epsilon_q.requires_grad = False
                svd_linear_qkv.epsilon_k.requires_grad = False
                svd_linear_qkv.epsilon_v.requires_grad = True
                svd_linear_qkv.bias.requires_grad = False
                setattr(module, attr_str, svd_linear_qkv)
            else:
                print('replaced: ', name, attr_str)
                svd_linear = SVDLinearAddition(target_attr)
                svd_linear.U.requires_grad = False
                svd_linear.S.requires_grad = False
                svd_linear.Vt.requires_grad = False
                svd_linear.bias.requires_grad = False
                svd_linear.epsilon.requires_grad = True
                setattr(module, attr_str, svd_linear)

    for name, immediate_child_module in module.named_children():
        replace_linear_addition_noqk(immediate_child_module, name)


def replace_back(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == SVDLinearAddition:
            print('replaced back: ', name, attr_str)
            with torch.no_grad():
                linear = torch.nn.Linear(
                    target_attr.Vt.shape[1],
                    target_attr.U.shape[0],
                    device=target_attr.U.device,
                )
                linear.weight.add_(
                    target_attr.U
                    @ torch.diag(target_attr.S + target_attr.epsilon)
                    @ target_attr.Vt
                    - linear.weight
                )
                linear.bias.add_(target_attr.bias - linear.bias)

            setattr(module, attr_str, linear)

        elif type(target_attr) == SVDLinearAdditionQKV:
            print('replaced back: ', name, attr_str)
            with torch.no_grad():
                w = torch.concatenate(
                    (
                        target_attr.U_q
                        @ torch.diag(target_attr.S_q + target_attr.epsilon_q)
                        @ target_attr.Vt_q,
                        target_attr.U_k
                        @ torch.diag(target_attr.S_k + target_attr.epsilon_k)
                        @ target_attr.Vt_k,
                        target_attr.U_v
                        @ torch.diag(target_attr.S_v + target_attr.epsilon_v)
                        @ target_attr.Vt_v,
                    )
                )
                linear = torch.nn.Linear(
                    w.shape[1], w.shape[0], device=target_attr.U_q.device
                )
                linear.weight.add_(w - linear.weight)
                linear.bias.add_(target_attr.bias - linear.bias)

            setattr(module, attr_str, linear)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_back(immediate_child_module, name)
