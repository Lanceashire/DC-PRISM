"""external-v15: frozen, label-independent deep representations for eight standard multiclass benchmarks.

Vision: CIFAR-10 (10), CIFAR-100 coarse (20), Fashion-MNIST (10), SVHN cropped digits (10).
Text:   AG News (4), DBpedia-14 (14), IMDB (2), Yelp Polarity (2).
Encoders (pretrained, frozen, no fine-tuning): vision DINOv2-small (primary) and MobileNetV3-Small;
text BGE-small-en-v1.5 (primary, CLS pooling) and all-MiniLM-L6-v2 (mean pooling).
Per dataset a fixed random subsample (seed 99001) of up to 24,000 official-train and 4,000 official-test records;
texts are exact-deduplicated (normalized whitespace/case) across both splits before sampling.
Labels are stored but never used by the encoders.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT/'hf_cache'))
import hashlib, json, sys, time
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi

torch.set_num_threads(6); torch.backends.cudnn.benchmark = False
OUT = ROOT/'features'; OUT.mkdir(exist_ok=True)
MAN = ROOT/'assets_manifest.json'
man = json.loads(MAN.read_text()) if MAN.exists() else {'assets': {}, 'features': {}}
api = HfApi()
TRAIN_CAP, TEST_CAP = 24000, 4000

DATASETS = {
    'cifar10':   dict(repo='uoft-cs/cifar10', config=None, kind='image', x='img', y='label', classes=10),
    'cifar100c': dict(repo='uoft-cs/cifar100', config=None, kind='image', x='img', y='coarse_label', classes=20),
    'fmnist':    dict(repo='zalando-datasets/fashion_mnist', config=None, kind='image', x='image', y='label', classes=10),
    'svhn':      dict(repo='ufldl-stanford/svhn', config='cropped_digits', kind='image', x='image', y='label', classes=10),
    'agnews':    dict(repo='fancyzhx/ag_news', config=None, kind='text', x='text', y='label', classes=4),
    'dbpedia':   dict(repo='fancyzhx/dbpedia_14', config=None, kind='text', x=('title', 'content'), y='label', classes=14),
    'imdb':      dict(repo='stanfordnlp/imdb', config=None, kind='text', x='text', y='label', classes=2),
    'yelp':      dict(repo='fancyzhx/yelp_polarity', config=None, kind='text', x='text', y='label', classes=2),
}
ENCODERS = {'image': ['dinov2s', 'mnv3s'], 'text': ['bge', 'minilm']}
MODEL_REPO = {'dinov2s': 'facebook/dinov2-small', 'mnv3s': 'timm/mobilenetv3_small_100.lamb_in1k',
              'bge': 'BAAI/bge-small-en-v1.5', 'minilm': 'sentence-transformers/all-MiniLM-L6-v2'}


def rev(repo, kind):
    if repo not in man['assets']:
        man['assets'][repo] = {'revision': api.repo_info(repo, repo_type=kind).sha, 'kind': kind}
        MAN.write_text(json.dumps(man, indent=2))
    return man['assets'][repo]['revision']


def sample_rows(name, spec):
    ds = load_dataset(spec['repo'], spec['config'], revision=rev(spec['repo'], 'dataset')) if spec['config'] else \
        load_dataset(spec['repo'], revision=rev(spec['repo'], 'dataset'))
    rng = np.random.default_rng(99001); rows = []; seen = set(); dup = 0
    for split, cap in [('train', TRAIN_CAP), ('test', TEST_CAP)]:
        d = ds[split]; perm = rng.permutation(len(d)); got = 0
        for i in perm:
            r = d[int(i)]
            if spec['kind'] == 'text':
                t = (r[spec['x'][0]] + '. ' + r[spec['x'][1]]) if isinstance(spec['x'], tuple) else r[spec['x']]
                key = ' '.join(t.lower().split())
                if key in seen: dup += 1; continue
                seen.add(key); x = t
            else:
                x = r[spec['x']]
            rows.append((x, int(r[spec['y']]), f'{split}:{int(i)}', split)); got += 1
            if got >= cap: break
    return rows, dup


def encode_images(enc, xs):
    if enc == 'dinov2s':
        from transformers import AutoImageProcessor, AutoModel
        repo = MODEL_REPO[enc]; r = rev(repo, 'model')
        proc = AutoImageProcessor.from_pretrained(repo, revision=r); model = AutoModel.from_pretrained(repo, revision=r).eval().cuda()
        def f(batch):
            inp = proc(images=[b.convert('RGB') for b in batch], return_tensors='pt').to('cuda')
            return model(**inp).pooler_output
    else:
        import timm
        repo = MODEL_REPO[enc]; r = rev(repo, 'model')
        model = timm.create_model(f'hf-hub:{repo}@{r}', pretrained=True, num_classes=0).eval().cuda()
        tr = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
        def f(batch):
            return model(torch.stack([tr(b.convert('RGB')) for b in batch]).cuda())
    out = []
    for s in range(0, len(xs), 128):
        with torch.inference_mode(): out.append(f(xs[s:s+128]).float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


def encode_texts(enc, xs):
    from transformers import AutoTokenizer, AutoModel
    repo = MODEL_REPO[enc]; r = rev(repo, 'model')
    tok = AutoTokenizer.from_pretrained(repo, revision=r); model = AutoModel.from_pretrained(repo, revision=r).eval().cuda()
    out = []
    for s in range(0, len(xs), 64):
        inp = tok(xs[s:s+64], padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            h = model(**inp).last_hidden_state.float()
        if enc == 'bge':
            x = h[:, 0]
        else:
            m = inp.attention_mask.unsqueeze(-1).float(); x = (h*m).sum(1)/m.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(x, dim=1).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


def main(names):
    for name in names:
        spec = DATASETS[name]; t0 = time.time()
        todo = [e for e in ENCODERS[spec['kind']] if not (OUT/f'{name}__{e}.npz').exists()]
        if not todo: continue
        rows, dup = sample_rows(name, spec)
        xs = [r[0] for r in rows]; y = np.array([r[1] for r in rows]); ids = np.array([r[2] for r in rows]); sp = np.array([r[3] for r in rows])
        for e in todo:
            X = encode_images(e, xs) if spec['kind'] == 'image' else encode_texts(e, xs)
            p = OUT/f'{name}__{e}.npz'
            np.savez_compressed(p, X=X.astype('float32'), y=y, ids=ids, official_split=sp)
            man['features'][f'{name}__{e}'] = dict(dataset=spec['repo'], config=spec['config'], encoder=MODEL_REPO[e], rows=len(y),
                                                    dim=int(X.shape[1]), classes=spec['classes'], train=int((sp == 'train').sum()),
                                                    test=int((sp == 'test').sum()), text_exact_duplicates_skipped=dup,
                                                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(), seconds=round(time.time()-t0, 1))
            MAN.write_text(json.dumps(man, indent=2)); print(name, e, man['features'][f'{name}__{e}'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(DATASETS))
