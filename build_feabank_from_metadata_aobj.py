import os
import sys
import math
import glob
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import librosa
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

# ============================================================
# 路径和参数
# ============================================================
REPO_ROOT = "/home/dy/DDESeg-main"
METADATA_CSV = "/mnt/sdc/dy/data/Re_AVS/metadata_w_silent.csv"
DATA_ROOT = "/mnt/sdc/dy/data/Re_AVS"
AUDIO_CKPT = "/mnt/sdc/dy/HTSAT_AudioSet_Saved_1_clean.ckpt"
LABEL2IDX_PATH = "/mnt/sdc/dy/data/Re_AVS/label2idx.json"
OUTPUT_NPY = "/home/dy/DDESeg-main/feabank2.npy" # 1是center

# KMeans 原型生成方式: "center" or "nearest"
MODE = "nearest"
K = 5

# 只用 train，避免信息泄漏。需要时可改成 {"train", "val"} 或 None(全部)
SPLIT_FILTER = {"train", "val"}

# 使用哪些subset {"v1s", "v1m", "v2"}。None 表示 metadata 里出现的都用。
SUBSET_FILTER = None

# 只保留 semantic mask 去掉背景后“恰好一个非背景类”的秒级样本
REQUIRE_SINGLE_NON_BG_CLASS = True

# 可选：该唯一前景类在 mask 中占比至少达到这个阈值才保留；设 0.0 表示不限制
MIN_FOREGROUND_RATIO = 0.0

# 音频设置
SAMPLE_RATE = 22050
AUD_DUR = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ignore label（若 mask 有 255 之类无效值，可在这里加）
IGNORE_LABELS = {255}

# 如果 mask 中出现 0，则默认视作 bg=0, fg=1..70，对应 bank idx 直接等于 mask id
MASK_ENCODING = "bg0_fg1to70"

# 如果某类样本不足 K 个，是否允许重复补齐
REPEAT_IF_FEW = True
# ============================================================


def add_repo_to_path(repo_root: str):
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


"""
    构建音频模型
    build_audio_model() 会从你的 DDESeg 仓库里导入 HTSAT_Swin_Transformer，加载 checkpoint，然后切到 eval()。后面所有音频特征都是从这个模型的 latent_output 里拿的。
"""
def build_audio_model(repo_root: str, audio_ckpt: str):
    add_repo_to_path(repo_root)
    try:
        from models.audio_branch.htsat import HTSAT_Swin_Transformer
    except Exception as e:
        raise ImportError(
            f"Failed to import HTSAT_Swin_Transformer from repo_root={repo_root}. "
            f"Please check your repo path and module layout. Original error: {e}"
        )

    model = HTSAT_Swin_Transformer(
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        in_chans=1,
        num_classes=71,
        num_heads=[4, 8, 16, 32],
        window_size=8,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        config={
            "window_size": 1024,
            "mel_bins": 64,
            "hop_size": 256,
            "sample_rate": 22050,
            "fmin": 50,
            "fmax": 14000,
            "htsat_attn_heatmap": False,
            "htsat_hier_output": False,
            "htsat_use_max": False,
            "enable_tscam": True,
            "token_label_range": [0.2, 0.6],
            "enable_time_shift": False,
            "enable_label_enhance": False,
            "enable_repeat_mode": False,
            "loss_type": "clip_bce",
        }
    )

    ckpt = torch.load(audio_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(DEVICE)
    return model


def load_label_map(path: str) -> Dict[str, int]:
    """Load label2idx.json and normalize to 0-based bank indices. class1-71对应的是mask里面的0-70"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    values = list(obj.values())
    if min(values) == 1:
        return {k: int(v) - 1 for k, v in obj.items()}
    return {k: int(v) for k, v in obj.items()}


def load_audio_1s(path: str, sample_rate: int = 22050, aud_dur: float = 1.0) -> np.ndarray:
    wav, sr = librosa.load(path, sr=None, mono=True)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)

    target_len = int(sample_rate * aud_dur)
    if len(wav) < target_len:
        repeat_times = math.ceil(target_len / len(wav))
        wav = np.tile(wav, repeat_times)
    wav = wav[:target_len]
    return wav.astype(np.float32)


@torch.no_grad()
def extract_latent_output(model, wav_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(wav_np).unsqueeze(0).to(DEVICE)
    out = model(x)
    feat = out["latent_output"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return feat


def split_uid_second_level(uid: str) -> Tuple[str, str]:
    """
    metadata_w_silent.csv 里的 uid 形如:
        __GOGlHL23s_4000_9000_0
    返回:
        clip_uid = __GOGlHL23s_4000_9000
        sec_idx  = 0
    """
    if "_" not in uid:
        raise ValueError(f"Unexpected uid format: {uid}")
    clip_uid, sec_idx = uid.rsplit("_", 1)
    return clip_uid, sec_idx


def resolve_audio_path(data_root: str, subset: str, uid: str) -> str:
    clip_uid, sec_idx = split_uid_second_level(uid)
    candidates = [
        os.path.join(data_root, subset, clip_uid, "audios", f"{sec_idx}.wav"),
        os.path.join(data_root, subset, clip_uid, "audios", f"{int(sec_idx)}.wav") if sec_idx.isdigit() else "",
        os.path.join(data_root, subset, uid, "audios", f"{sec_idx}.wav"),
        os.path.join(data_root, subset, uid, "audios", f"{int(sec_idx)}.wav") if sec_idx.isdigit() else "",
    ]
    candidates = [p for p in candidates if p]

    for p in candidates:
        if os.path.exists(p):
            return p

    # 再做一次模糊匹配兜底
    fuzzy_dirs = [
        os.path.join(data_root, subset, clip_uid, "audios"),
        os.path.join(data_root, subset, uid, "audios"),
    ]
    for d in fuzzy_dirs:
        if os.path.isdir(d):
            wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
            if sec_idx.isdigit():
                target = os.path.join(d, f"{int(sec_idx)}.wav")
                if os.path.exists(target):
                    return target
            if len(wavs) == 1:
                return wavs[0]

    raise FileNotFoundError(
        "No 1-second wav found for row:\n"
        f"  subset={subset}, uid={uid}\n"
        f"  clip_uid={clip_uid}, sec_idx={sec_idx}\n"
        f"Tried:\n  " + "\n  ".join(candidates)
    )


def resolve_mask_path(data_root: str, subset: str, uid: str) -> str:
    clip_uid, sec_idx = split_uid_second_level(uid)
    candidates = [
        os.path.join(data_root, subset, clip_uid, "labels_semantic", f"{sec_idx}.png"),
        os.path.join(data_root, subset, clip_uid, "labels_semantic", f"{int(sec_idx)}.png") if sec_idx.isdigit() else "",
        os.path.join(data_root, subset, uid, "labels_semantic", f"{sec_idx}.png"),
        os.path.join(data_root, subset, uid, "labels_semantic", f"{int(sec_idx)}.png") if sec_idx.isdigit() else "",
    ]
    candidates = [p for p in candidates if p]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "No semantic mask found for row:\n"
        f"  subset={subset}, uid={uid}\n"
        f"  clip_uid={clip_uid}, sec_idx={sec_idx}\n"
        f"Tried:\n  " + "\n  ".join(candidates)
    )


def load_mask(mask_path: str) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def iter_filtered_rows(metadata_csv: str) -> List[dict]:
    df = pd.read_csv(metadata_csv)
    rows = []
    for _, row in df.iterrows():
        split = str(row["split"])
        subset = str(row["subset"])
        if SPLIT_FILTER is not None and split not in SPLIT_FILTER:
            continue
        if SUBSET_FILTER is not None and subset not in SUBSET_FILTER:
            continue
        rows.append({
            "vid": str(row["vid"]),
            "uid": str(row["uid"]),
            "split": split,
            "subset": subset,
        })
    return rows


def mask_to_single_class(mask: np.ndarray, encoding: str) -> Optional[Tuple[int, float, List[int]]]:
    vals = np.unique(mask)
    vals = [int(v) for v in vals if int(v) not in IGNORE_LABELS]

    if encoding == "bg0_fg1to70":
        non_bg = [v for v in vals if v != 0]
        if REQUIRE_SINGLE_NON_BG_CLASS and len(non_bg) != 1:
            return None
        if len(non_bg) == 0:
            return None
        raw_cls = non_bg[0]
        bank_cls = raw_cls  # bank idx 直接等于 mask id
    elif encoding == "bg1_fg2to71":
        non_bg = [v for v in vals if v != 1]
        if REQUIRE_SINGLE_NON_BG_CLASS and len(non_bg) != 1:
            return None
        if len(non_bg) == 0:
            return None
        raw_cls = non_bg[0]
        bank_cls = raw_cls - 1  # 1-based -> 0-based
    else:
        raise ValueError(f"Unknown mask encoding: {encoding}")

    if bank_cls < 0 or bank_cls > 70:
        return None

    fg_ratio = float((mask == raw_cls).sum()) / float(mask.size)
    if fg_ratio < MIN_FOREGROUND_RATIO:
        return None

    return bank_cls, fg_ratio, non_bg


def collect_features_from_semantic_mask(
    metadata_csv: str,
    data_root: str,
    model,
    num_classes: int = 71,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[str, int]]:
    rows = iter_filtered_rows(metadata_csv)
    print(f"[Info] Filtered rows: {len(rows)}")

    encoding = "bg0_fg1to70"
    print(f"[Info] Inferred MASK_ENCODING = {encoding}")

    class_to_feats: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_classes)}
    stats = {
        "total_rows": 0,
        "used_rows": 0,
        "missing_audio": 0,
        "missing_mask": 0,
        "invalid_mask": 0,
        "multi_class_mask": 0,
        "empty_mask": 0,
    }

    for row in tqdm(rows, desc="Extracting features"):
        stats["total_rows"] += 1
        subset = row["subset"]
        uid = row["uid"]

        try:
            mask_path = resolve_mask_path(data_root, subset, uid)
        except FileNotFoundError:
            stats["missing_mask"] += 1
            continue

        mask = load_mask(mask_path)
        result = mask_to_single_class(mask, encoding)
        if result is None:
            vals = [int(v) for v in np.unique(mask).tolist() if int(v) not in IGNORE_LABELS]
            if encoding == "bg0_fg1to70":
                non_bg = [v for v in vals if v != 0]
            else:
                non_bg = [v for v in vals if v != 1]
            if len(non_bg) == 0:
                stats["empty_mask"] += 1
            elif len(non_bg) > 1:
                stats["multi_class_mask"] += 1
            else:
                stats["invalid_mask"] += 1
            continue

        bank_cls, _, _ = result

        try:
            audio_path = resolve_audio_path(data_root, subset, uid)
        except FileNotFoundError:
            stats["missing_audio"] += 1
            continue

        wav = load_audio_1s(audio_path, sample_rate=SAMPLE_RATE, aud_dur=AUD_DUR)
        feat = extract_latent_output(model, wav)
        class_to_feats[bank_cls].append(feat)
        stats["used_rows"] += 1

    return class_to_feats, stats


# 构建前景语义类音频原型库，不是“前景+背景都对等”的原型库
def build_bank(class_to_feats: Dict[int, List[np.ndarray]], k: int = 5, mode: str = "center") -> np.ndarray:
    num_classes = len(class_to_feats)
    embed_dim = None
    for feats in class_to_feats.values():
        if feats:
            embed_dim = feats[0].shape[0]
            break
    if embed_dim is None:
        raise RuntimeError("No features collected. Please check mask/audio paths and filtering rules.")

    bank = np.zeros((num_classes, k, embed_dim), dtype=np.float32)

    for cls_idx in tqdm(range(num_classes), desc="Building bank"):
        feats = class_to_feats[cls_idx]
        if len(feats) == 0:
            print(f"[Warn] class {cls_idx} has no samples; leaving zeros.")
            continue

        arr = np.stack(feats, axis=0)

        if len(arr) < k:
            if REPEAT_IF_FEW:
                repeat_idx = np.arange(k) % len(arr)
                bank[cls_idx] = arr[repeat_idx]
                print(f"[Info] class {cls_idx} only has {len(arr)} samples; repeated to fill k={k}.")
                continue
            raise RuntimeError(f"class {cls_idx} only has {len(arr)} samples, fewer than k={k}")

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(arr)
        centers = kmeans.cluster_centers_.astype(np.float32)

        if mode == "center":
            bank[cls_idx] = centers
        elif mode == "nearest":
            reps = []
            for j in range(k):
                idxs = np.where(labels == j)[0]
                cluster_feats = arr[idxs]
                dists = np.linalg.norm(cluster_feats - centers[j], axis=1)
                reps.append(cluster_feats[np.argmin(dists)])
            bank[cls_idx] = np.stack(reps, axis=0).astype(np.float32)
        else:
            raise ValueError("mode must be 'center' or 'nearest'")

    return bank


def print_class_histogram(class_to_feats: Dict[int, List[np.ndarray]], label2idx: Dict[str, int]):
    idx2label = {v: k for k, v in label2idx.items()}
    print("\n" + "=" * 80)
    print("Per-class sample counts")
    print("=" * 80)
    total_nonempty = 0
    for cls_idx in range(len(class_to_feats)):
        cnt = len(class_to_feats[cls_idx])
        name = idx2label.get(cls_idx, f"class_{cls_idx}")
        if cnt > 0:
            total_nonempty += 1
        print(f"[{cls_idx:02d}] {name:20s} -> {cnt}")
    print("=" * 80)
    print(f"Non-empty classes: {total_nonempty}/{len(class_to_feats)}")
    print("=" * 80 + "\n")


def main():
    label2idx = load_label_map(LABEL2IDX_PATH)

    print("=" * 80)
    print("Build feabank from semantic masks only (no a_obj)")
    print(f"REPO_ROOT            = {REPO_ROOT}")
    print(f"METADATA_CSV         = {METADATA_CSV}")
    print(f"DATA_ROOT            = {DATA_ROOT}")
    print(f"AUDIO_CKPT           = {AUDIO_CKPT}")
    print(f"LABEL2IDX_PATH       = {LABEL2IDX_PATH}")
    print(f"OUTPUT_NPY           = {OUTPUT_NPY}")
    print(f"MODE                 = {MODE}")
    print(f"K                    = {K}")
    print(f"SPLIT_FILTER         = {SPLIT_FILTER}")
    print(f"SUBSET_FILTER        = {SUBSET_FILTER}")
    print(f"REQUIRE_SINGLE_NON_BG_CLASS = {REQUIRE_SINGLE_NON_BG_CLASS}")
    print(f"MIN_FOREGROUND_RATIO = {MIN_FOREGROUND_RATIO}")
    print(f"MASK_ENCODING        = {MASK_ENCODING}")
    print(f"DEVICE               = {DEVICE}")
    print("=" * 80)

    model = build_audio_model(REPO_ROOT, AUDIO_CKPT)
    class_to_feats, stats = collect_features_from_semantic_mask(
        metadata_csv=METADATA_CSV,
        data_root=DATA_ROOT,
        model=model,
        num_classes=len(label2idx),
    )

    print("\n" + "=" * 80)
    print("Collection stats")
    print("=" * 80)
    for k_, v_ in stats.items():
        print(f"{k_:20s}: {v_}")
    print("=" * 80)

    print_class_histogram(class_to_feats, label2idx)

    bank = build_bank(class_to_feats, k=K, mode=MODE)

    os.makedirs(os.path.dirname(OUTPUT_NPY) or ".", exist_ok=True)
    np.save(OUTPUT_NPY, bank.astype(np.float32))
    print(f"Saved to {OUTPUT_NPY}")
    print(f"Shape: {bank.shape}, dtype: {bank.dtype}")


if __name__ == "__main__":
    main()
