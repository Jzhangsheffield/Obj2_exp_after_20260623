#!/usr/bin/env python3
"""Single Windows/HPC launcher for all stages in this experiment package."""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
from common.config import PACKAGE_ROOT, PLAN_PATH, SELECTION_PATH, append, load_plan, read_json, resolve_rgb, roots, select
STAGE6_SELECTION_PATH = PACKAGE_ROOT / "config" / "stage6_selection.json"

class Runner:
    def __init__(self, args):
        self.args=args; self.plan=load_plan(); self.project,self.dataset=roots(self.plan,args.platform,args.project_root,args.dataset_root)
        self.python=args.python_bin or sys.executable; self.output=self.project/self.plan["output_rel"]
        self.base=self.project/self.plan["base_source_package_rel"]; self.src=self.base/"src"
        self.splits=self.output/"runtime"/"splits"; self.crop_stats=self.output/"runtime"/"motion_crop_train_stats.json"
        profiles={"screen":("train_MJ.jsonl","val_MR.jsonl"),"fold_M":("fold_M_train_JMR.jsonl","fold_M_val_M.jsonl"),"fold_J":("fold_J_train_MMR.jsonl","fold_J_val_J.jsonl"),"fold_MR":("fold_MR_train_MJ.jsonl","fold_MR_val_MR.jsonl")}
        self.train_manifest=self.splits/profiles[args.split_profile][0]; self.val_manifest=self.splits/profiles[args.split_profile][1]
        self.teacher_cache=self.output/self.plan["imu_teacher"]["cache_rel"] if args.split_profile=="screen" else self.output/"runtime"/"imu_teacher"/f"{args.split_profile}_features.pth"
        self.env=dict(os.environ); self.env["PYTHONPATH"]=str(self.src)+(os.pathsep+self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else "")
    def run(self, cmd):
        print("[Command]", shlex.join([str(x) for x in cmd]))
        if not self.args.dry_run: subprocess.run([str(x) for x in cmd], cwd=str(self.project), env=self.env, check=True)
    def stage_exp(self): return select(self.plan["stages"][self.args.stage], self.args.index, self.args.experiment)
    def rgb(self, exp):
        if self.args.dry_run and exp["rgb_source"]=="motion_crop" and not self.crop_stats.is_file():
            t=self.plan["task"]; return {"camera":t["motion_crop_rgb_key"],"mean":t["original_mean"],"std":t["original_std"],"preserve":True}
        value=resolve_rgb(self.plan,exp,self.crop_stats)
        if "rgb_mean" in exp: value["mean"]=exp["rgb_mean"]
        if "rgb_std" in exp: value["std"]=exp["rgb_std"]
        return value
    def prepare(self):
        src=self.plan["source_manifests"]
        self.run([self.python, PACKAGE_ROOT/"tools"/"prepare_protocol.py", "--dataset-root",self.dataset,"--train",src["train"],"--val",src["val"],"--output",self.splits])
        self.run([self.python,self.base/"tools"/"compute_rgb_stats.py","--dataset-root",self.dataset,"--manifest",self.splits/"train_MJ.jsonl","--rgb-key","rgb_cam_00143_motion_crop_m32","--output",self.crop_stats])
    def pretrain_path(self, stage, exp):
        root=self.output/"pretrain" if self.args.split_profile=="screen" else self.output/"pretrain_crossval"/self.args.split_profile
        return root/stage/exp["id"]
    def pretrain(self, exp):
        common=self.plan["pretrain_common"]; aug=self.plan["augmentation"]; task=self.plan["task"]; rgb=self.rgb(exp)
        out=self.pretrain_path(self.args.stage,exp); source=self.src/"train"/"pretrain_supcon.py"
        cmd=[self.python,"-u",PACKAGE_ROOT/"common"/"pretrain_entry.py","--repair-source",source,"--repair-src-root",self.src,
             "--repair-representation",exp["representation"],"--repair-temporal-mode",exp["temporal_mode"],"--repair-lr-warmup-epochs",exp.get("lr_warmup_epochs",0),
             "--repair-backbone-init",exp.get("backbone_init","random"),
             "--repair-aux-ce-weight",exp.get("aux_ce_weight",0),"--repair-xmodal-weight",exp.get("xmodal_weight",0),"--repair-xrel-weight",exp.get("xrel_weight",0),
             "--repair-xrel-temperature",self.plan["imu_teacher"]["relation_temperature"],"--repair-auto-resume"]
        if exp.get("xmodal_weight",0)>0 or exp.get("xrel_weight",0)>0: cmd += ["--repair-teacher-cache",self.teacher_cache]
        values={"--dataset_root":self.dataset,"--train_manifest_name":self.train_manifest,"--label_map_json":self.dataset/self.plan["source_manifests"]["label_map"],
            "--weight_save_path":out,"--tier_mode":task["tier_mode"],"--n_frames":exp["n_frames"],"--backbone":exp["backbone"],"--model_depth":18,
            "--rgb_camera_id":rgb["camera"],"--temporal_view_mode":aug["temporal_view_mode"],"--batch_size":exp.get("batch_size",common["batch_size"]),
            "--num_workers":common["num_workers"],"--proj_dim":common["proj_dim"],"--K_queue":common["queue_size"],"--temperature":common["temperature"],
            "--contrastive_loss":"suploss","--num_positive":common["num_positive"],"--ablation_mode":"contrastive_only","--epochs":exp.get("epochs",common["epochs"]),
            "--learning_rate":exp.get("learning_rate",common["learning_rate"]),"--weight_decay":common["weight_decay"],"--optimizer":common["optimizer"],"--seed":exp["seed"],
            "--save_interval":exp.get("save_interval",common["save_interval"]),"--print_freq":common["print_freq"],"--sampler_type":"none",
            "--rgb_hflip_p":aug["hflip_p"],"--rgb_vflip_p":aug["vflip_p"],"--rgb_jitter_p":aug["jitter_p"],
            "--rgb_jitter_brightness":aug["jitter_strength"][0],"--rgb_jitter_contrast":aug["jitter_strength"][1],"--rgb_jitter_saturation":aug["jitter_strength"][2],
            "--rgb_jitter_hue":aug["jitter_strength"][3],"--rgb_gray_p":aug["gray_p"],"--rgb_blur_p":aug["blur_p"],"--rgb_blur_kernel":aug["blur_kernel"]}
        for k,v in values.items(): append(cmd,k,v)
        for k,v in {"--rgb_mean":rgb["mean"],"--rgb_std":rgb["std"],"--rgb_out_hw":[task["image_size"]]*2,"--rrc_scale":aug["rrc_scale"],"--rrc_ratio":aug["rrc_ratio"],"--rgb_blur_sigma":aug["blur_sigma"],"--schedule":common["step_schedule"]}.items(): append(cmd,k,v)
        cmd += ["--rgb_apply_spatial_aug","--mlp","--no_ddp","--no-use_syncbn","--verify_paths_on_init","--exclude_invalid_queue"]
        if rgb["preserve"]: cmd.append("--rgb_preserve_aspect_pad")
        if exp.get("freeze_patch_embed"): cmd.append("--repair-freeze-patch-embed")
        if exp["lr_schedule"]=="cosine": cmd.append("--cos")
        if self.args.validate_command: cmd.insert(cmd.index("--repair-auto-resume"),"--repair-parse-only")
        self.run(cmd)
    def checkpoint(self, stage, exp): return self.pretrain_path(stage,exp)/f"checkpoint_{exp.get('epochs',self.plan['pretrain_common']['epochs']):04d}.pth"
    def diagnose(self, exp):
        rgb=self.rgb(exp); ckpt=self.checkpoint(self.args.stage,exp)
        cmd=[self.python,"-u",PACKAGE_ROOT/"tools"/"diagnose_features.py","--src-root",self.src,"--dataset-root",self.dataset,
             "--train-manifest",self.train_manifest,"--val-manifest",self.val_manifest,"--label-map",self.dataset/self.plan["source_manifests"]["label_map"],
             "--backbone",exp["backbone"],"--representation",exp["representation"],"--temporal-mode",exp["temporal_mode"],"--rgb-camera-id",rgb["camera"],
             "--rgb-mean",*rgb["mean"],"--rgb-std",*rgb["std"],"--n-frames",exp["n_frames"],"--checkpoint",ckpt,"--backbone-init",exp.get("backbone_init","random"),
             "--queue-size",self.plan["pretrain_common"]["queue_size"],"--proj-dim",self.plan["pretrain_common"]["proj_dim"],
             "--batch-size",min(32,exp.get("batch_size",64)),"--num-workers",self.plan["pretrain_common"]["num_workers"],"--output",(self.output/"diagnostics"/self.args.stage/exp["id"] if self.args.split_profile=="screen" else self.output/"diagnostics_crossval"/self.args.split_profile/self.args.stage/exp["id"]),"--seed",exp["seed"]]
        if rgb["preserve"]: cmd.append("--rgb-preserve-aspect-pad")
        if exp.get("freeze_patch_embed"): cmd.append("--freeze-patch-embed")
        self.run(cmd)
    def stage6a_rows(self): return self.plan["stage6"]["stage6a"]
    def stage6_selection(self): return read_json(STAGE6_SELECTION_PATH)
    def stage6_config(self, exp_id):
        return select(self.stage6a_rows(),None,exp_id)
    def require_stage6(self,key):
        selection=self.stage6_selection()
        if not selection.get(key) and not self.args.dry_run: raise RuntimeError(f"Stage 6 gate {key}=false; update config/stage6_selection.json after reviewing the preceding results")
        return selection
    def stage6a(self,exp):
        rgb=self.rgb(exp); out=self.output/"diagnostics"/"stage6a"/exp["id"]
        cmd=[self.python,"-u",PACKAGE_ROOT/"tools"/"diagnose_features.py","--src-root",self.src,"--dataset-root",self.dataset,
             "--train-manifest",self.train_manifest,"--val-manifest",self.val_manifest,"--label-map",self.dataset/self.plan["source_manifests"]["label_map"],
             "--backbone",exp["backbone"],"--backbone-init",exp["backbone_init"],"--representation","rgb","--temporal-mode","current",
             "--rgb-camera-id",rgb["camera"],"--rgb-mean",*rgb["mean"],"--rgb-std",*rgb["std"],"--n-frames",exp["n_frames"],
             "--proj-dim",self.plan["pretrain_common"]["proj_dim"],"--batch-size",exp["batch_size"],"--num-workers",self.plan["pretrain_common"]["num_workers"],"--output",out,"--seed",exp["seed"]]
        self.run(cmd)
    def stage6b_rows(self,confirm=False):
        sel=self.require_stage6("stage6a_ready"); cfg=self.plan["stage6"]["stage6b"]; rows=[]
        if confirm:
            self.require_stage6("stage6b_ready"); candidates=[sel["stage6b_winner"]]; policies=[sel["stage6b_policy"]]; seeds=cfg["confirm_seeds"]
        else:
            candidates=sel["stage6b_candidates"]; policies=cfg["policies"]; seeds=[cfg["screen_seed"]]
        for candidate in candidates:
            base=self.stage6_config(candidate)
            for policy in policies:
                for seed in seeds:
                    rows.append({**base,"index":len(rows),"id":f"s6b_{candidate[4:]}_{policy}_s{seed}","policy":policy,"seed":seed})
        return rows
    def stage6c_rows(self,confirm=False):
        sel=self.require_stage6("stage6b_ready"); base=self.stage6_config(sel["stage6b_winner"]); cfg=self.plan["stage6"]["stage6c"]
        names=list(cfg["experiments"])
        if base["backbone"] not in {"mvit_v2_s","swin3d_t"}: names.remove("k400_supcon_patch_frozen")
        if confirm:
            self.require_stage6("stage6c_ready"); names=[sel["stage6c_winner"]]; seeds=cfg["confirm_seeds"]
        else: seeds=[1]
        rows=[]
        for name in names:
            if name=="direct_k400": continue
            for seed in seeds:
                init="random" if name=="random_supcon" else "kinetics400"
                transformer=base["backbone"] in {"mvit_v2_s","swin3d_t"}
                rows.append({**base,"index":len(rows),"id":f"s6c_{name}_s{seed}","seed":seed,"backbone_init":init,
                    "freeze_patch_embed":name.endswith("patch_frozen"),"epochs":cfg["epochs"],"save_interval":cfg["save_interval"],
                    "batch_size":base["batch_size"] if self.plan["pretrain_common"]["queue_size"] % base["batch_size"] == 0 else 16,
                    "learning_rate":cfg["transformer_lr"] if transformer else cfg["cnn_lr"],"lr_schedule":"cosine","lr_warmup_epochs":cfg["warmup_epochs"],"aux_ce_weight":0.0})
        return rows
    def stage6_classifier(self,row,phase):
        cfg=self.plan["stage6"]["stage6b"]; rgb=self.rgb(row); policy=row["policy"]; epochs=cfg["epochs"][policy]
        transformer=row["backbone"] in {"mvit_v2_s","swin3d_t"}; backbone_lr=cfg["transformer_backbone_lr"] if transformer else cfg["cnn_backbone_lr"]
        out=self.output/"classifier"/phase/row["id"]; source=self.src/"ft_and_test"/"train_classifier.py"
        if out.is_dir() and list(out.rglob("last.pth")) and not self.args.dry_run: print(f"[Skip] completed: {out}"); return
        cmd=[self.python,"-u",PACKAGE_ROOT/"common"/"classifier_entry.py","--repair-source",source,"--repair-src-root",self.src,"--repair-representation","rgb","--repair-temporal-mode","current",
             "--repair-backbone-init",row.get("backbone_init","kinetics400"),"--repair-finetune-policy",policy]
        values={"--run_mode":"train","--save_path":out,"--datamap_csv_path":out/"datamaps","--dataset_root":self.dataset,"--label_map_json":self.dataset/self.plan["source_manifests"]["label_map"],
            "--train_manifest":self.train_manifest,"--val_manifest":self.val_manifest,"--tier_mode":"tier1","--n_frames":row["n_frames"],"--use_modality":"rgb","--num_classes":15,
            "--backbone":row["backbone"],"--model_depth":18,"--rgb_camera_id":rgb["camera"],"--rgb_size":224,"--rrc_scale_min":.85,"--rrc_scale_max":1.0,"--rrc_ratio_min":.9,"--rrc_ratio_max":1.1,
            "--rgb_hflip_p":.5,"--rgb_vflip_p":0,"--rgb_jitter_p":0,"--rgb_gray_p":0,"--rgb_blur_p":0,"--epochs":epochs,"--batch_size":row.get("batch_size",16),
            "--num_workers_train":self.plan["finetune_common"]["num_workers_train"],"--num_workers_val":self.plan["finetune_common"]["num_workers_val"],"--optimizer":"adamw",
            "--learning_rate":cfg["head_lr"],"--weight_decay":cfg["weight_decay"],"--seed":row["seed"],"--finetune_mode":"head_only" if policy=="head" else "full","--save_period":cfg["save_period"],"--best_after_epoch":0}
        for k,v in values.items(): append(cmd,k,v)
        append(cmd,"--rgb_mean",rgb["mean"]);append(cmd,"--rgb_std",rgb["std"]);append(cmd,"--schedules",[25,40] if epochs==50 else ([50,75] if epochs==100 else [15,20]))
        cmd += ["--rgb_apply_spatial_aug","--enable_amp"]
        if policy!="head": cmd += ["--use_discriminative_lr","--backbone_learning_rate",str(backbone_lr),"--head_learning_rate",str(cfg["head_lr"])]
        if row.get("pretrained_checkpoint"): append(cmd,"--pretrained_weight_paths",row["pretrained_checkpoint"])
        if self.args.validate_command: cmd.append("--repair-parse-only")
        self.run(cmd)
    def stage6c_finetune_rows(self,confirm=False):
        sel=self.stage6_selection(); rows=self.stage6c_rows(confirm); policy=sel["stage6b_policy"]
        for row in rows:
            row["policy"]=policy; row["pretrained_checkpoint"]=self.checkpoint("stage6c",row)
        return rows
    def cache_teacher(self):
        t=self.plan["imu_teacher"]
        self.run([self.python,"-u",PACKAGE_ROOT/"tools"/"cache_imu_teacher.py","--project-root",self.project,"--dataset-root",self.dataset,
            "--manifest",self.train_manifest,"--checkpoint",self.project/t["checkpoint_rel"],"--args-json",self.project/t["args_rel"],"--output",self.teacher_cache])
    def final_rows(self):
        selection=read_json(SELECTION_PATH)
        if not selection.get("selection_ready") and not self.args.dry_run: raise RuntimeError("Stage 5 is locked: edit config/final_selection.json and set selection_ready=true after Stage 1-4 selection")
        rows=[]
        for seed in selection["seeds"]:
            for mode in ("scratch_full","sup_head","sup_full"):
                rows.append({"index":len(rows),"id":f"{mode}_s{seed}","seed":seed,"mode":mode,"selection":selection})
        return rows
    def classifier(self, row):
        sel=row["selection"]; exp={**sel,"id":row["id"]}; rgb=self.rgb(exp); common=self.plan["finetune_common"]
        head=row["mode"]=="sup_head"; pretrained=row["mode"]!="scratch_full"; epochs=common["epochs_head"] if head else common["epochs_full"]
        out=self.output/"classifier"/"stage5"/row["id"]; source=self.src/"ft_and_test"/"train_classifier.py"
        if out.is_dir() and list(out.rglob("last.pth")) and not self.args.dry_run: print(f"[Skip] completed: {out}"); return
        cmd=[self.python,"-u",PACKAGE_ROOT/"common"/"classifier_entry.py","--repair-source",source,"--repair-src-root",self.src,"--repair-representation",sel["representation"],"--repair-temporal-mode",sel["temporal_mode"]]
        values={"--run_mode":"train","--save_path":out,"--datamap_csv_path":out/"datamaps","--dataset_root":self.dataset,
            "--label_map_json":self.dataset/self.plan["source_manifests"]["label_map"],"--train_manifest":self.splits/"train_MJ.jsonl","--val_manifest":self.splits/"val_MR.jsonl",
            "--tier_mode":"tier1","--n_frames":sel["n_frames"],"--use_modality":"rgb","--num_classes":15,"--backbone":sel["backbone"],"--model_depth":18,
            "--rgb_camera_id":rgb["camera"],"--rgb_size":224,"--rrc_scale_min":.85,"--rrc_scale_max":1.0,"--rrc_ratio_min":.9,"--rrc_ratio_max":1.1,
            "--rgb_hflip_p":.5,"--rgb_vflip_p":0,"--rgb_jitter_p":0,"--rgb_gray_p":0,"--rgb_blur_p":0,"--epochs":epochs,
            "--batch_size":common["batch_size"],"--num_workers_train":common["num_workers_train"],"--num_workers_val":common["num_workers_val"],
            "--optimizer":"adamw","--learning_rate":common["head_lr"],"--weight_decay":common["weight_decay"],"--seed":row["seed"],
            "--finetune_mode":"head_only" if head else "full","--save_period":common["save_period"],"--best_after_epoch":0}
        for k,v in values.items(): append(cmd,k,v)
        for k,v in {"--rgb_mean":rgb["mean"],"--rgb_std":rgb["std"],"--schedules":common["schedule"]}.items(): append(cmd,k,v)
        cmd += ["--rgb_apply_spatial_aug","--enable_amp"]
        if rgb["preserve"]: cmd.append("--rgb_preserve_aspect_pad")
        if pretrained:
            append(cmd,"--pretrained_weight_paths",self.project/sel["selected_pretrain_rel"])
            if not head: cmd += ["--use_discriminative_lr","--backbone_learning_rate",str(common["backbone_lr"]),"--head_learning_rate",str(common["head_lr"])]
        self.run(cmd)
    def best_classifier(self,row):
        root=self.output/"classifier"/"stage5"/row["id"]; found=list(root.rglob("best_val_balanced.pth"))
        if len(found)!=1: raise FileNotFoundError(f"Expected one best_val_balanced.pth under {root}, found {len(found)}")
        return found[0]
    def test(self,row):
        lock=self.plan["test_lock"]
        if (not self.args.dry_run) and os.environ.get(lock["environment_variable"])!=lock["required_value"]: raise RuntimeError(f"Locked N test: set {lock['environment_variable']}={lock['required_value']} only after Stage 5 validation is complete")
        sel=row["selection"]; exp={**sel,"id":row["id"]}; rgb=self.rgb(exp); out=self.output/"test"/"stage5"/row["id"]
        cmd=[self.python,"-u",PACKAGE_ROOT/"common"/"classifier_entry.py","--repair-source",self.src/"ft_and_test"/"train_classifier.py","--repair-src-root",self.src,"--repair-representation",sel["representation"],"--repair-temporal-mode",sel["temporal_mode"]]
        values={"--run_mode":"test","--save_path":out,"--datamap_csv_path":out/"datamaps","--test_results_csv":out/"test_results.csv","--dataset_root":self.dataset,
            "--label_map_json":self.dataset/self.plan["source_manifests"]["label_map"],"--test_manifest":self.dataset/self.plan["source_manifests"]["locked_test"],
            "--test_weight_paths":(self.output/"classifier"/"stage5"/row["id"]/'<run_dir>'/"best_val_balanced.pth" if self.args.dry_run else self.best_classifier(row)),"--tier_mode":"tier1","--n_frames":sel["n_frames"],"--use_modality":"rgb","--num_classes":15,
            "--backbone":sel["backbone"],"--model_depth":18,"--rgb_camera_id":rgb["camera"],"--rgb_size":224,"--batch_size":64,"--num_workers_test":self.plan["finetune_common"]["num_workers_test"]}
        for k,v in values.items(): append(cmd,k,v)
        append(cmd,"--rgb_mean",rgb["mean"]); append(cmd,"--rgb_std",rgb["std"]); cmd.append("--enable_amp")
        if rgb["preserve"]: cmd.append("--rgb_preserve_aspect_pad")
        self.run(cmd)
    def summarize(self): self.run([self.python,PACKAGE_ROOT/"tools"/"summarize.py","--results-root",self.output])
    def validate(self): self.run([self.python,PACKAGE_ROOT/"tools"/"validate_package.py","--project-root",self.project,"--dataset-root",self.dataset])

def main():
    stage6_actions=["stage6a","stage6b","stage6b_confirm","stage6c_pretrain","stage6c_diagnose","stage6c_finetune","stage6c_confirm_pretrain","stage6c_confirm_finetune"]
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["validate","prepare","pretrain","diagnose","cache_teacher","classifier","test","summarize",*stage6_actions])
    p.add_argument("--stage",choices=["stage1","stage2","stage3","stage4"]); g=p.add_mutually_exclusive_group(); g.add_argument("--index",type=int); g.add_argument("--experiment")
    p.add_argument("--platform",choices=["auto","windows","hpc"],default="auto"); p.add_argument("--split-profile",choices=["screen","fold_M","fold_J","fold_MR"],default="screen"); p.add_argument("--project-root"); p.add_argument("--dataset-root"); p.add_argument("--python-bin"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--validate-command",action="store_true")
    args=p.parse_args(); r=Runner(args)
    if args.action in {"pretrain","diagnose"}:
        if not args.stage or (args.index is None and args.experiment is None): p.error("pretrain/diagnose require --stage and --index or --experiment")
        exp=r.stage_exp(); r.pretrain(exp) if args.action=="pretrain" else r.diagnose(exp)
    elif args.action in {"classifier","test"}:
        if args.index is None and args.experiment is None: p.error("classifier/test require --index or --experiment")
        row=select(r.final_rows(),args.index,args.experiment); r.classifier(row) if args.action=="classifier" else r.test(row)
    elif args.action=="stage6a":
        if args.index is None and args.experiment is None: p.error("stage6a requires --index or --experiment")
        r.stage6a(select(r.stage6a_rows(),args.index,args.experiment))
    elif args.action in {"stage6b","stage6b_confirm"}:
        if args.index is None and args.experiment is None: p.error(f"{args.action} requires --index or --experiment")
        confirm=args.action.endswith("confirm"); row=select(r.stage6b_rows(confirm),args.index,args.experiment)
        r.stage6_classifier(row,"stage6b_confirm" if confirm else "stage6b")
    elif args.action.startswith("stage6c_"):
        if args.index is None and args.experiment is None: p.error(f"{args.action} requires --index or --experiment")
        confirm="confirm" in args.action; rows=r.stage6c_finetune_rows(confirm) if args.action.endswith("finetune") else r.stage6c_rows(confirm)
        row=select(rows,args.index,args.experiment); r.args.stage="stage6c"
        if args.action.endswith("pretrain"): r.pretrain(row)
        elif args.action=="stage6c_diagnose": r.diagnose(row)
        else: r.stage6_classifier(row,"stage6c_confirm" if confirm else "stage6c")
    else: getattr(r,args.action)()
if __name__=="__main__": main()
