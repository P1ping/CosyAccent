import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_FILENAME = "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="User-friendly CosyAccent wav-to-wav inference."
    )

    parser.add_argument(
        "--source_wav", required=True, help="Path to source speech wav."
    )
    parser.add_argument(
        "--reference_wav",
        default=None,
        help="Path to reference speaker wav. If omitted, source_wav is used.",
    )
    parser.add_argument(
        "--output_wav", required=True, help="Path to save generated wav."
    )

    parser.add_argument(
        "--model_config",
        default=None,
        help="Local config yaml path. If omitted and model checkpoint is downloaded, use bundled config.",
    )

    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="Local path to CosyAccent checkpoint (.pt). If omitted, download CosyAccent bundle by --model_tag.",
    )
    parser.add_argument(
        "--hift_checkpoint",
        default=None,
        help="Local path to HiFT checkpoint (.pt). If omitted, will download from Hugging Face.",
    )

    parser.add_argument(
        "--hf_repo_id",
        default="Piping/CosyAccent",
        help="Hugging Face model repo id used when checkpoint path is not provided.",
    )
    parser.add_argument(
        "--hf_subfolder", default="checkpoints", help="Checkpoint subfolder in HF repo."
    )
    parser.add_argument(
        "--model_tag",
        default="emilia_pretrained",
        help="CosyAccent model folder tag under hf_subfolder (e.g., checkpoints/<model_tag>/...).",
    )
    parser.add_argument(
        "--model_filename",
        default="cosyaccent.pt",
        help="CosyAccent checkpoint filename inside model_tag folder.",
    )
    parser.add_argument(
        "--config_filename",
        default=DEFAULT_CONFIG_FILENAME,
        help="CosyAccent config filename inside model_tag folder (default: config.yaml).",
    )
    parser.add_argument(
        "--hift_filename", default="hift.pt", help="HiFT checkpoint filename."
    )
    parser.add_argument(
        "--hf_revision", default=None, help="Optional HF revision (branch/tag/commit)."
    )
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token.")

    parser.add_argument(
        "--cache_dir",
        default="checkpoints",
        help="Local cache directory for checkpoints.",
    )

    parser.add_argument(
        "--n_timesteps", type=int, default=32, help="Flow matching inference steps."
    )
    parser.add_argument(
        "--full_cfg",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for null (condition+speaker) branch.",
    )
    parser.add_argument(
        "--cond_cfg",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for condition branch.",
    )
    parser.add_argument(
        "--spk_cfg",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale for speaker branch.",
    )
    parser.add_argument(
        "--preserve_total_duration",
        action="store_true",
        help="If set, preserve total duration of source speech.",
    )
    parser.add_argument(
        "--speech_len_ratio",
        type=float,
        default=1.0,
        help="Used only with --preserve_total_duration to scale output duration.",
    )
    parser.add_argument(
        "--mel_path",
        default=None,
        help="Optional path to save predicted mel tensor (.pt).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )

    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_if_needed(
    local_path: Optional[str],
    *,
    repo_id: str,
    subfolder: str,
    filename: str,
    cache_dir: str,
    revision: Optional[str],
    token: Optional[str],
) -> str:
    if local_path is not None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Checkpoint file not found: {local_path}")
        return local_path

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=cache_dir,
        revision=revision,
        token=token,
    )
    return downloaded


def _download_config(
    *,
    repo_id: str,
    subfolder: str,
    model_tag: str,
    cache_dir: str,
    revision: Optional[str],
    token: Optional[str],
    config_filename: str,
) -> str:
    model_subfolder = f"{subfolder}/{model_tag}" if subfolder else model_tag
    return hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        subfolder=model_subfolder,
        local_dir=cache_dir,
        revision=revision,
        token=token,
    )


def _load_audio_as_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0)


def _extract_spk_embedding(reference_wav: str, device: torch.device) -> torch.Tensor:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except Exception as exc:
        raise RuntimeError(
            "Failed to import resemblyzer. Please install requirements_inference.txt."
        ) from exc

    encoder_device = "cuda" if device.type == "cuda" else "cpu"
    encoder = VoiceEncoder(device=encoder_device)
    preprocessed = preprocess_wav(reference_wav)
    emb = encoder.embed_utterance(preprocessed)
    emb_t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).unsqueeze(0)
    emb_t = F.normalize(emb_t, p=2, dim=-1)
    return emb_t.to(device)


def _load_model_and_vocoder(
    config_path: str,
    model_ckpt_path: str,
    hift_ckpt_path: str,
    device: torch.device,
    config_overrides: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides=config_overrides)

    model = configs["model"].to(device).eval()
    model_ckpt = torch.load(model_ckpt_path, map_location="cpu")
    for key in ["epoch", "step"]:
        model_ckpt.pop(key, None)
    model.load_state_dict(model_ckpt, strict=True)
    for p in model.parameters():
        p.requires_grad = False

    vocoder = configs["hift"].to(device).eval()
    vocoder.load_state_dict(torch.load(hift_ckpt_path, map_location="cpu"), strict=True)
    for p in vocoder.parameters():
        p.requires_grad = False

    sample_rate = int(configs["sample_rate"])
    return model, vocoder, sample_rate


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    print(f"[CosyAccent] Using device: {device}")

    if args.model_checkpoint is None:
        model_bundle_subfolder = (
            f"{args.hf_subfolder}/{args.model_tag}"
            if args.hf_subfolder
            else args.model_tag
        )

        model_ckpt = hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.model_filename,
            subfolder=model_bundle_subfolder,
            local_dir=args.cache_dir,
            revision=args.hf_revision,
            token=args.hf_token,
        )
        if args.model_config:
            if not os.path.isfile(args.model_config):
                raise FileNotFoundError(f"Config file not found: {args.model_config}")
            model_config_path = args.model_config
        else:
            model_config_path = _download_config(
                repo_id=args.hf_repo_id,
                subfolder=args.hf_subfolder,
                model_tag=args.model_tag,
                cache_dir=args.cache_dir,
                revision=args.hf_revision,
                token=args.hf_token,
                config_filename=args.config_filename,
            )
        print(f"[CosyAccent] Downloaded bundle tag: {args.model_tag}")
        print(f"[CosyAccent] Bundled config: {model_config_path}")
    else:
        if not os.path.isfile(args.model_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint file not found: {args.model_checkpoint}"
            )
        model_ckpt = args.model_checkpoint
        if args.model_config:
            if not os.path.isfile(args.model_config):
                raise FileNotFoundError(f"Config file not found: {args.model_config}")
            model_config_path = args.model_config
        else:
            model_config_path = _download_config(
                repo_id=args.hf_repo_id,
                subfolder=args.hf_subfolder,
                model_tag=args.model_tag,
                cache_dir=args.cache_dir,
                revision=args.hf_revision,
                token=args.hf_token,
                config_filename=args.config_filename,
            )

    hift_ckpt = _download_if_needed(
        args.hift_checkpoint,
        repo_id=args.hf_repo_id,
        subfolder=args.hf_subfolder,
        filename=args.hift_filename,
        cache_dir=args.cache_dir,
        revision=args.hf_revision,
        token=args.hf_token,
    )
    print(f"[CosyAccent] Model checkpoint: {model_ckpt}")
    print(f"[CosyAccent] HiFT checkpoint: {hift_ckpt}")
    print(f"[CosyAccent] Model config: {model_config_path}")

    model, vocoder, sample_rate = _load_model_and_vocoder(
        model_config_path,
        model_ckpt,
        hift_ckpt,
        device,
    )

    src_wav = _load_audio_as_mono(args.source_wav, target_sr=16000)
    src_wav = src_wav.unsqueeze(0).to(device)
    src_wav_len = torch.tensor([src_wav.shape[1]], dtype=torch.long, device=device)
    reference_wav = args.reference_wav if args.reference_wav else args.source_wav
    spk_embed = _extract_spk_embedding(reference_wav, device=device)

    if args.preserve_total_duration:
        use_predicted_duration = False
        speech_len_ratio = float(args.speech_len_ratio)
    else:
        use_predicted_duration = True
        speech_len_ratio = None

    speech_feat, _ = model.inference(
        src_feat_or_wav=src_wav,
        src_feat_or_wav_len=src_wav_len,
        spk_embed=spk_embed,
        n_timesteps=int(args.n_timesteps),
        full_cfg=float(args.full_cfg),
        cond_cfg=float(args.cond_cfg),
        spk_cfg=float(args.spk_cfg),
        use_predicted_duration=use_predicted_duration,
        speech_len_ratio=speech_len_ratio,
    )

    speech_feat_for_vocoder = speech_feat.transpose(1, 2)
    hift_cache_source = torch.zeros(1, 1, 0, device=device)
    speech, _ = vocoder.inference(
        speech_feat=speech_feat_for_vocoder, cache_source=hift_cache_source
    )

    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), speech.squeeze(0).cpu().numpy(), sample_rate)
    print(f"[CosyAccent] Wrote audio: {output_path}")

    if args.mel_path:
        mel_path = Path(args.mel_path)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(speech_feat.cpu(), mel_path)
        print(f"[CosyAccent] Wrote mel tensor: {mel_path}")


if __name__ == "__main__":
    run_inference(parse_args())
