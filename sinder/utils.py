from pathlib import Path

import torch
from PIL import Image
from sklearn.decomposition import PCA

import sinder
from .singular_defect import singular_defect_directions


def pca_array(tokens, whiten=False):
    h, w, c = tokens.shape
    tokens = tokens.detach().cpu()

    pca = PCA(n_components=3, whiten=whiten)
    pca.fit(tokens.reshape(-1, c))
    projected_tokens = pca.transform(tokens.reshape(-1, c))

    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    array = (normalized_t * 255).byte().numpy()
    array = array.reshape(h, w, 3)

    return Image.fromarray(array).resize((w * 7, h * 7), 0)


def get_tokens(model, image, blocks=1):
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).cuda()
        image_batch = image_batch.cuda()
        H = image_batch.shape[2]
        W = image_batch.shape[3]
        print(f'{W=} {H=}')
        tokens = model.get_intermediate_layers(
            image_batch, blocks, return_class_token=True, norm=False
        )
        tokens = [
            (
                t.reshape(
                    (H // model.patch_size, W // model.patch_size, t.size(-1))
                ),
                tc,
            )
            for t, tc in tokens
        ]

    return tokens


def load_model(model_name, checkpoint=None):
    print(f'using {model_name} model')
    model = torch.hub.load(
        repo_or_dir=Path(sinder.__file__).parent.parent,
        source='local',
        model=model_name,
    )
    if checkpoint is not None:
        states = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(states, strict=False)
    model = model.cuda()
    model.eval()
    model.interpolate_antialias = True
    model.singular_defects = singular_defect_directions(model)
    print(f'model loaded. patch size: {model.patch_size}')

    return model
