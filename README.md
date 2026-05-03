# CosyAccent

Official implementation of the paper:
"CosyAccent: Duration-Controllable Accent Normalization Using Source-Synthesis Training Data"
(Accepted to ICASSP 2026)

Paper: https://arxiv.org/abs/2602.19166v1

## Dataset

The training data (L2-LibriTTSR) is hosted on Hugging Face:
https://huggingface.co/datasets/Piping/L2-LibriTTSR

## Inference Quick Start

Install inference dependencies:

```bash
pip install -r requirements.txt
```

Run wav-to-wav inference:

```bash
python infer_wav.py \
  --source_wav /path/to/source.wav \
  --output_wav outputs/result.wav
```

Model selection is done by `--model_tag`:

- default: `emilia_pretrained`
- paper setting: `paper`

Example with paper setting:

```bash
python infer_wav.py \
  --source_wav /path/to/source.wav \
  --output_wav outputs/result.wav \
  --model_tag paper
```

- `--reference_wav` (default: None -> `source_wav`; used for timbre conditioning)
- `--n_timesteps` (default: `32`; flow-matching steps)
- `--full_cfg` (default: `1.0`)
- `--cond_cfg` (default: `1.0`)
- `--spk_cfg` (default: `0.0`)
- `--preserve_total_duration` (flag; disabled by default)
- `--speech_len_ratio` (default: `1.0`; used only when `--preserve_total_duration` is set)
- `--device` (`auto`/`cpu`/`cuda`, default: `auto`)
- `--mel_path` (optional output path to save predicted mel tensor)
- `--hf_repo_id`, `--hf_subfolder`, `--model_filename`, `--config_filename`, `--hift_filename`, `--hf_revision`, `--hf_token` (optional advanced Hugging Face download settings)
- `--model_checkpoint`, `--model_config`, `--hift_checkpoint` (optional local paths; when `--model_checkpoint` is set, `--model_config` is required)

`infer_wav.py` auto-downloads required checkpoints from Hugging Face if local checkpoint paths are not provided.

## Acknowledgements

This repository builds on several excellent open-source projects:

- **CosyVoice**: the HiFT checkpoint is leveraged from CosyVoice2.
- **ESPnet**: the Conformer/Transformer encoder components are adapted from ESPnet (via CosyVoice and direct modifications in this repo).
- **OpenAI Whisper**: Whisper encoder frontend is used for speech feature extraction.

## Citation

```bibtex
@inproceedings{bai2026cosyaccent,
  title={CosyAccent: Duration-Controllable Accent Normalization Using Source-Synthesis Training Data},
  author={Bai, Qibing and Shi, Shuhao and Wang, Shuai and Ju, Yukai and Wang, Yannan and Li, Haizhou},
  booktitle={ICASSP 2026},
  year={2026}
}
```