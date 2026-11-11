"""Download pinned HF assets and extract frozen, label-independent deep representations."""
import os
os.environ.setdefault('HF_HOME', str(__import__('pathlib').Path(__file__).resolve().parent/'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(6)
torch.manual_seed(99001)
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available()
api = HfApi()
manifest = {'started_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'assets': {},
            'scope': 'Frozen pretrained encoders; no end-to-end fine-tuning. Binary subsets, not full multiclass CDALBench.'}
if (ROOT/'assets_manifest.json').exists():
    manifest.update(json.loads((ROOT/'assets_manifest.json').read_text()))

def revision(repo, kind):
    if repo in manifest['assets']:
        return manifest['assets'][repo]['revision']
    info = api.repo_info(repo, repo_type=kind)
    manifest['assets'][repo] = {'revision': info.sha, 'url': f'https://huggingface.co/{"datasets/" if kind=="dataset" else ""}{repo}'}
    (ROOT/'assets_manifest.json').write_text(json.dumps(manifest, indent=2))
    return info.sha

def save(name, xs, ys, ids, split, model, extra):
    p = ROOT/(name+'.npz')
    np.savez_compressed(p, X=np.concatenate(xs).astype('float32'), y=np.asarray(ys,dtype='int64'),
                        ids=np.asarray(ids), official_split=np.asarray(split))
    manifest[name] = {'rows': len(ys), 'features': int(np.concatenate(xs).shape[1]), 'model': model,
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), **extra}
    (ROOT/'assets_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(name, manifest[name], flush=True)

if not (ROOT/'cifar10_cat_dog.npz').exists():
    import timm
    repo='timm/mobilenetv3_small_100.lamb_in1k'; rev=revision(repo,'model')
    model=timm.create_model(f'hf-hub:{repo}@{rev}', pretrained=True, num_classes=0).eval().cuda()
    trans=timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dsrepo='uoft-cs/cifar10'; dsrev=revision(dsrepo,'dataset')
    ds=load_dataset(dsrepo, revision=dsrev)
    xs=[]; ys=[]; ids=[]; ss=[]; batch=[]
    for split in ['train','test']:
        for i,row in enumerate(ds[split]):
            if row['label'] not in [3,5]: continue
            batch.append(trans(row['img'].convert('RGB')))
            ys.append(int(row['label']==5)); ids.append(f'{split}:{i}'); ss.append(split)
            if len(batch)==64:
                with torch.inference_mode(): xs.append(model(torch.stack(batch).cuda()).cpu().numpy())
                batch=[]
                if len(ys)%1024==0: print('vision features',len(ys),flush=True)
    if batch:
        with torch.inference_mode(): xs.append(model(torch.stack(batch).cuda()).cpu().numpy())
    save('cifar10_cat_dog',xs,ys,ids,ss,repo,{'classes':{'0':'cat','1':'dog'},'parameters':sum(p.numel() for p in model.parameters()),'image_size':224})
    del model
    torch.cuda.empty_cache()

if not (ROOT/'agnews_business_scitech.npz').exists():
    from transformers import AutoTokenizer, AutoModel
    repo='sentence-transformers/all-MiniLM-L6-v2'; rev=revision(repo,'model')
    tok=AutoTokenizer.from_pretrained(repo,revision=rev)
    model=AutoModel.from_pretrained(repo,revision=rev).eval().cuda()
    dsrepo='fancyzhx/ag_news'; dsrev=revision(dsrepo,'dataset')
    ds=load_dataset(dsrepo,revision=dsrev)
    xs=[]; ys=[]; ids=[]; ss=[]; texts=[]; seen=set(); duplicates=0
    # Exact normalized text deduplication across official splits. Keep first in train then test.
    for split in ['train','test']:
        count={2:0,3:0}; cap=5000 if split=='train' else 1000
        perm=np.random.default_rng(99001).permutation(len(ds[split]))
        for i in perm:
            row=ds[split][int(i)]; label=row['label']
            if label not in count or count[label]>=cap: continue
            key=' '.join(row['text'].lower().split())
            if key in seen: duplicates+=1; continue
            seen.add(key); count[label]+=1
            texts.append(row['text']); ys.append(int(label==3)); ids.append(f'{split}:{i}'); ss.append(split)
            if all(v==cap for v in count.values()): break
    for start in range(0,len(texts),64):
        inp=tok(texts[start:start+64],padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            h=model(**inp).last_hidden_state; mask=inp.attention_mask.unsqueeze(-1)
            x=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            xs.append(torch.nn.functional.normalize(x,p=2,dim=1).cpu().numpy())
        if start%1024==0: print('text features',start,len(texts),flush=True)
    save('agnews_business_scitech',xs,ys,ids,ss,repo,{'classes':{'0':'Business','1':'Sci/Tech'},'parameters':sum(p.numel() for p in model.parameters()),'max_length':128,'exact_duplicates_removed':duplicates})

manifest['elapsed_complete_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
manifest['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
(ROOT/'assets_manifest.json').write_text(json.dumps(manifest,indent=2))
