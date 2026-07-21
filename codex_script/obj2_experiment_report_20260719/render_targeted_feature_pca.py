"""Fast, dependency-light targeted feature comparison.

Uses a shared random projection, orthogonal Procrustes alignment, and a joint
2-D PCA.  It is the deterministic fallback when UMAP/numba startup is too slow.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "obj2_experiment_report_20260719"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
FEATURES = OUT / "targeted_features" / "01_features"
ROLES = ["scratch", "supcon", "module"]
ROLE_LABEL = {"scratch": "Scratch", "supcon": "SupCon", "module": "Best module"}
COLORS = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#DB2777"]


def local_path(value):
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def align(source, reference):
    s = normalize(source.astype(float)); r = normalize(reference.astype(float))
    s -= s.mean(0, keepdims=True); r -= r.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(s.T @ r, full_matrices=False)
    return s @ (u @ vt)


def metrics(x, labels):
    x = normalize(x.astype(float)); n = len(x)
    dist = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=min(5, n-1)-1, axis=1)[:, :min(5, n-1)]
    purity = float(np.mean(labels[nn] == labels[:, None]))
    unique = np.unique(labels); sil = []
    cents = {}
    for c in unique:
        cents[c] = normalize(x[labels == c].mean(0, keepdims=True))[0]
    within = []
    for i,c in enumerate(labels):
        same = np.where(labels == c)[0]; same = same[same != i]
        a = float(np.mean(dist[i, same])) if len(same) else 0.0
        if len(unique) > 1:
            b = min(float(np.mean(dist[i, labels == other])) for other in unique if other != c)
            sil.append((b-a)/max(a,b,1e-12))
        within.append(1 - x[i] @ cents[c])
    nearest = []
    for c in unique:
        if len(unique) > 1:
            nearest.append(min(1 - cents[c] @ cents[o] for o in unique if o != c))
    w = float(np.mean(within)); nd = float(np.mean(nearest)) if nearest else math.nan
    return dict(silhouette_cosine=float(np.mean(sil)) if sil else math.nan, knn5_purity=purity,
                within_class_cosine_distance=w, nearest_other_centroid_distance=nd,
                centroid_margin=nd-w if not math.isnan(nd) else math.nan)


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def draw_plot(modality, coords, meta, harmed, selected_classes):
    W,H = 1800,650; image = Image.new("RGB", (W,H), "white"); d = ImageDraw.Draw(image)
    d.text((W/2,25), f"{modality.upper()}: harmed classes in a shared aligned PCA", fill="#111827", font=font(28,True), anchor="ma")
    color = {c: COLORS[i % len(COLORS)] for i,c in enumerate(selected_classes)}
    panel_w=520; lefts=[70,640,1210]; top=95; bottom=555
    xmin,xmax=float(coords[:,0].min()),float(coords[:,0].max()); ymin,ymax=float(coords[:,1].min()),float(coords[:,1].max())
    for role,left in zip(ROLES,lefts):
        d.rectangle((left,top,left+panel_w,bottom), outline="#94A3B8", width=2)
        d.text((left+panel_w/2,70), ROLE_LABEL[role], fill="#111827", font=font(22,True), anchor="ma")
        role_mask = meta.role.eq(role).to_numpy()
        for cls in selected_classes:
            idx=np.where(role_mask & meta.class_name.eq(cls).to_numpy())[0]
            for i in idx:
                x=left+20+(coords[i,0]-xmin)/max(xmax-xmin,1e-9)*(panel_w-40)
                y=bottom-20-(coords[i,1]-ymin)/max(ymax-ymin,1e-9)*(bottom-top-40)
                if cls in harmed:
                    d.ellipse((x-4,y-4,x+4,y+4), fill=color[cls], outline="white")
                else:
                    d.line((x-5,y-5,x+5,y+5), fill=color[cls], width=2); d.line((x-5,y+5,x+5,y-5), fill=color[cls], width=2)
        d.text((left+panel_w/2,bottom+12), "Shared PCA 1", fill="#475569", font=font(15), anchor="ma")
    x=80
    for cls in selected_classes:
        marker="●" if cls in harmed else "×"
        d.text((x,615),f"{marker} {cls}",fill=color[cls],font=font(17)); x += 35 + d.textlength(cls,font=font(17)) + 55
    image.save(FIGURES / f"harmed_class_feature_space_{modality}.png", dpi=(240,240))


def main():
    models=pd.read_csv(TABLES/"selected_module_models.csv")
    per_class=pd.read_csv(TABLES/"selected_module_per_class.csv")
    diagnostics=[]; harmed_rows=[]
    for mi,modality in enumerate(["rgb","emg","imu"]):
        group=models[models.modality == modality]
        raw={}; metas={}
        for _,r in group.iterrows():
            role=r.model_role; folder=FEATURES/r.feature_model_id/"test"
            raw[role]=np.load(folder/"features_512.npy")
            metas[role]=pd.read_csv(folder/"samples.csv")
        names=metas["supcon"].sample_name.astype(str).tolist()
        for role in ROLES:
            index={str(n):i for i,n in enumerate(metas[role].sample_name)}
            take=np.array([index[n] for n in names]); raw[role]=raw[role][take]; metas[role]=metas[role].iloc[take].reset_index(drop=True)
        q=per_class[per_class.modality == modality].sort_values("delta_vs_supcon")
        harmed=q[q.delta_vs_supcon < 0].class_name.head(5).tolist() or q.class_name.head(1).tolist()
        module_row=group[group.model_role == "module"].iloc[0]
        pred=pd.read_csv(local_path(module_row.per_sample_path))
        present_true=set(pred.true_label_name.astype(str))
        confused=[]
        for cls in harmed:
            e=pred[(pred.true_label_name == cls) & (pred.correct == 0)]
            if len(e):
                candidates=[str(x) for x in e.pred_label_name.value_counts().index if str(x) in present_true]
                if candidates: confused.append(candidates[0])
        selected_classes=list(dict.fromkeys(harmed+confused))
        for cls in harmed:
            harmed_rows.append(dict(modality=modality,harmed_class=cls,delta_vs_supcon=float(q.set_index("class_name").loc[cls,"delta_vs_supcon"]),included_confusion_classes="; ".join(selected_classes)))
        for role in ROLES:
            meta=metas[role]; mask=meta.true_action.astype(str).isin(selected_classes).to_numpy(); lab=meta.loc[mask,"true_action"].astype(str).to_numpy()
            m=metrics(raw[role][mask],lab)
            diagnostics.append(dict(modality=modality,role=role,scope="selected_harmed_and_confusion_classes",classes="; ".join(selected_classes),n_samples=int(mask.sum()),**m,classification_recall=math.nan))
            for cls in harmed:
                cm=meta.true_action.astype(str).eq(cls).to_numpy(); x=normalize(raw[role][cm].astype(float)); cent=normalize(x.mean(0,keepdims=True))[0]
                diagnostics.append(dict(modality=modality,role=role,scope="harmed_class",classes=cls,n_samples=int(cm.sum()),silhouette_cosine=math.nan,knn5_purity=math.nan,within_class_cosine_distance=float(np.mean(1-x@cent)),nearest_other_centroid_distance=math.nan,centroid_margin=math.nan,classification_recall=float(meta.loc[cm,"correct"].mean())))
        rng=np.random.default_rng(42+mi); proj=rng.normal(size=(raw["supcon"].shape[1],30))/math.sqrt(30)
        reduced={r:raw[r]@proj for r in ROLES}; ref=reduced["supcon"]
        aligned={"supcon":align(ref,ref),"scratch":align(reduced["scratch"],ref),"module":align(reduced["module"],ref)}
        chunks=[]; plot_meta=[]
        for role in ROLES:
            mask=metas[role].true_action.astype(str).isin(selected_classes).to_numpy(); chunks.append(aligned[role][mask]); plot_meta.append(pd.DataFrame({"role":role,"class_name":metas[role].loc[mask,"true_action"].astype(str).to_numpy()}))
        allx=np.vstack(chunks); allx-=allx.mean(0,keepdims=True); _,_,vt=np.linalg.svd(allx,full_matrices=False); coords=allx@vt[:2].T
        draw_plot(modality,coords,pd.concat(plot_meta,ignore_index=True),harmed,selected_classes)
    pd.DataFrame(diagnostics).to_csv(TABLES/"targeted_feature_diagnostics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(harmed_rows).to_csv(TABLES/"harmed_class_selection.csv",index=False,encoding="utf-8-sig")
    guide=pd.read_csv(TABLES/"figure_reading_guide.csv")
    mask=guide.figure_type.eq("Shared UMAP feature plots") | guide.figure_type.eq("Shared aligned PCA feature plots")
    guide.loc[mask,"figure_type"]="Shared aligned PCA feature plots"
    guide.loc[mask,"encoding"]="The same samples are projected to a common 30-D basis, orthogonally aligned to SupCon, and jointly reduced to 2-D PCA. Circles are harmed classes; x marks are common confusion targets."
    guide.loc[mask,"how_to_read"]="Compare mixing and compactness, not absolute coordinate meaning; confirm with high-dimensional silhouette, purity, and distance metrics."
    guide.to_csv(TABLES/"figure_reading_guide.csv",index=False,encoding="utf-8-sig")
    print("targeted PCA feature figures and diagnostics refreshed")


if __name__ == "__main__": main()
