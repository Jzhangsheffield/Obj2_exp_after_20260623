#!/usr/bin/env python3
"""High-dimensional held-out-person diagnostics plus temporal perturbation tests."""
from __future__ import annotations
import argparse, json, sys
from functools import partial
from pathlib import Path
import numpy as np
import torch

def normalize(x): return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
def effective_rank(x):
    x=x-x.mean(0,keepdims=True); s=np.linalg.svd(x,compute_uv=False); v=s*s; p=v/np.clip(v.sum(),1e-12,None)
    return float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())), float(p[:5].sum())
def geometry(x,y,persons=None):
    from sklearn.metrics import silhouette_score
    z=normalize(x); classes=np.unique(y); cent=np.stack([normalize(z[y==c].mean(0,keepdims=True))[0] for c in classes])
    within=float(np.mean([np.mean(1-z[y==c]@cent[i]) for i,c in enumerate(classes)])); pair=1-cent@cent.T; between=float(pair[np.triu_indices_from(pair,k=1)].mean())
    er,top5=effective_rank(z)
    result={"silhouette_cosine":float(silhouette_score(z,y,metric="cosine")),"effective_rank":er,"top5_variance_fraction":top5,"within_cosine_distance":within,"between_centroid_cosine_distance":between,"between_within_ratio":between/max(within,1e-12)}
    if persons is not None and len(np.unique(persons)) > 1:
        result["person_silhouette_cosine"] = float(silhouette_score(z,persons,metric="cosine"))
    return result
def perturb(x,mode,seed):
    if mode=="normal": return x
    if mode=="reverse": return x.flip(2)
    if mode=="repeat_center": return x[:,:,x.shape[2]//2:x.shape[2]//2+1].expand_as(x)
    if mode=="shuffle":
        g=torch.Generator(device="cpu"); g.manual_seed(seed); idx=torch.randperm(x.shape[2],generator=g).to(x.device); return x.index_select(2,idx)
    raise ValueError(mode)
def build_loader(args,manifest,label_map):
    from utils_.mapstype_dataloader_with_index import PackedMultiModalConfig,build_packed_mapstyle_dataset,build_packed_mapstyle_loader_from_dataset
    cfg=PackedMultiModalConfig(n_frames=args.n_frames,rgb_two_views=False,rgb_camera_id=args.rgb_camera_id,rgb_preserve_aspect_pad=args.rgb_preserve_aspect_pad,
        use_modalities=("rgb",),load_labels=True,label_map_path=str(args.label_map),tier_mode="tier1",is_train=False,rgb_out_hw=(224,224),rgb_mean=tuple(args.rgb_mean),rgb_std=tuple(args.rgb_std))
    ds=build_packed_mapstyle_dataset(args.dataset_root,str(manifest),cfg,label_map=label_map,verify_paths_on_init=True)
    return build_packed_mapstyle_loader_from_dataset(ds,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,drop_last=False,pin_memory=True,prefetch_factor=2 if args.num_workers else None)
def load_model(args,device):
    from backbone.video_backbone import generate_video_model
    if args.checkpoint is None:
        return generate_video_model(args.backbone,num_classes=args.proj_dim,model_depth=18).to(device).eval()
    from backbone.MoCo_VAR_supcon_wds import MoCo3D
    model=MoCo3D(partial(generate_video_model,backbone_name=args.backbone,model_depth=18),dim=args.proj_dim,K=args.queue_size,T=.07,mlp=True,exclude_invalid_queue=True)
    obj=torch.load(args.checkpoint,map_location="cpu",weights_only=False); state=obj.get("state_dict",obj); state={(k[7:] if k.startswith("module.") else k):v for k,v in state.items()}
    msg=model.load_state_dict(state,strict=False); bad=[x for x in msg.unexpected_keys if not (x.startswith("round2_aux_classifier") or x.startswith("repair_xmodal_adapter"))]
    missing=[x for x in msg.missing_keys if not x.startswith("queue")]
    if bad or missing: raise RuntimeError(f"Checkpoint mismatch missing={missing[:20]} unexpected={bad[:20]}")
    return model.encoder_q.to(device).eval()
@torch.inference_mode()
def extract(encoder,loader,device,mode,seed):
    b=[];p=[];y=[]
    for batch_idx,batch in enumerate(loader):
        x=batch["rgb"].permute(0,2,1,3,4).contiguous().to(device); x=perturb(x,mode,seed+batch_idx)
        f=encoder.forward_features(x); z=encoder.fc(f); b.append(f.float().cpu());p.append(z.float().cpu());y.append(batch["tier_ids"]["tier1"].long().cpu())
    return torch.cat(b).numpy(),torch.cat(p).numpy(),torch.cat(y).numpy()
def predictive(train_x,train_y,val_x,val_y,seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score
    a=normalize(train_x);b=normalize(val_x);clf=LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed).fit(a,train_y);pred=clf.predict(b)
    sim=b@a.T; idx=sim.argmax(1); knn=train_y[idx]
    return {"linear_balanced_accuracy":float(balanced_accuracy_score(val_y,pred)),"linear_macro_f1":float(f1_score(val_y,pred,average="macro",zero_division=0)),"knn1_balanced_accuracy":float(balanced_accuracy_score(val_y,knn)),"linear_confusion_matrix":confusion_matrix(val_y,pred,labels=np.unique(train_y)).tolist()}
def manifest_persons(path):
    values=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): values.append(json.loads(line)["person"])
    return np.asarray(values)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--src-root",type=Path,required=True);p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--val-manifest",type=Path,required=True);p.add_argument("--label-map",type=Path,required=True)
    p.add_argument("--backbone",required=True);p.add_argument("--representation",required=True);p.add_argument("--temporal-mode",required=True);p.add_argument("--rgb-camera-id",required=True);p.add_argument("--rgb-mean",nargs=3,type=float,required=True);p.add_argument("--rgb-std",nargs=3,type=float,required=True);p.add_argument("--rgb-preserve-aspect-pad",action="store_true")
    p.add_argument("--n-frames",type=int,required=True);p.add_argument("--checkpoint",type=Path);p.add_argument("--backbone-init",choices=["random","kinetics400"],default="random");p.add_argument("--freeze-patch-embed",action="store_true");p.add_argument("--queue-size",type=int,default=1088);p.add_argument("--proj-dim",type=int,default=128);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--num-workers",type=int,default=8);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=1)
    args=p.parse_args(); package=Path(__file__).resolve().parents[1];sys.path[:0]=[str(package),str(args.src_root.resolve())]
    from common.runtime_patch import install
    install(args.src_root,args.representation,args.temporal_mode,args.backbone_init,args.freeze_patch_embed)
    from utils_.mapstype_dataloader_with_index import load_label_map_json
    label_map=load_label_map_json(str(args.label_map));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");encoder=load_model(args,device)
    train=extract(encoder,build_loader(args,args.train_manifest,label_map),device,"normal",args.seed); vals={m:extract(encoder,build_loader(args,args.val_manifest,label_map),device,m,args.seed) for m in ("normal","reverse","shuffle","repeat_center")}
    train_persons=manifest_persons(args.train_manifest);val_persons=manifest_persons(args.val_manifest)
    if len(train_persons)!=len(train[2]) or len(val_persons)!=len(vals["normal"][2]): raise RuntimeError("Manifest/person order does not match extracted feature count")
    result={"checkpoint":str(args.checkpoint) if args.checkpoint else None,"backbone_init":args.backbone_init,"patch_embed_frozen":bool(args.freeze_patch_embed),"backbone":args.backbone,"representation":args.representation,"n_frames":args.n_frames,"temporal_mode":args.temporal_mode,"features":{}}
    for fi,name in enumerate(("backbone","projection")):
        normal=vals["normal"]; entry={"train_geometry":geometry(train[fi],train[2],train_persons),"val_geometry":geometry(normal[fi],normal[2],val_persons),"normal_predictive":predictive(train[fi],train[2],normal[fi],normal[2],args.seed),"perturbations":{}}
        for mode in ("reverse","shuffle","repeat_center"):
            current=vals[mode]; cos=float(np.mean(np.sum(normalize(normal[fi])*normalize(current[fi]),axis=1)))
            entry["perturbations"][mode]={"original_perturbed_mean_cosine":cos,**predictive(train[fi],train[2],current[fi],current[2],args.seed)}
        result["features"][name]=entry
    args.output.mkdir(parents=True,exist_ok=True);(args.output/"diagnostics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    np.save(args.output/"train_backbone.npy",train[0]);np.save(args.output/"val_backbone.npy",vals["normal"][0]);np.save(args.output/"train_labels.npy",train[2]);np.save(args.output/"val_labels.npy",vals["normal"][2]);np.save(args.output/"train_persons.npy",train_persons);np.save(args.output/"val_persons.npy",val_persons)
    from sklearn.decomposition import PCA
    joined=np.concatenate((train[0],vals["normal"][0]));coords=PCA(n_components=2,random_state=args.seed).fit_transform(joined);np.save(args.output/"pca2d_backbone.npy",coords)
    try:
        import umap
        np.save(args.output/"umap2d_backbone.npy",umap.UMAP(n_components=2,metric="cosine",random_state=args.seed).fit_transform(joined))
    except Exception as exc: result["umap_note"]=f"UMAP unavailable ({type(exc).__name__}); PCA coordinates were saved instead"
    (args.output/"diagnostics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
