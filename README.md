# Facial Motion Generator

[日本語](#概要) | [English](#overview)

音声からリアルタイムで表情動作を生成するシステム / Audio-driven real-time facial expression generation system

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 概要

**Facial Motion Generator** は、音声入力のみから52次元のARKit BlendShape係数をリアルタイムに推定し、3Dアバターの表情を駆動するシステムです。

### 特徴
- 軽量モデル（約0.72M パラメータ）
- 高速推論（0.62ms/frame）
- OSC出力対応（VRChat, Unity, Blender等と連携可能）
- 単一話者データで学習済み

### 性能

| 項目 | Audio2Face-3D | Facial Motion Generator |
|------|--------------|-------------------------|
| MAE | 0.147 | 0.111 |
| MSE | 0.074 | 0.035 |
| パラメータ数 | 181M | 0.72M |
| 推論時間 | 21.5 ms/frame | 0.62 ms/frame |

---

## Overview

**Facial Motion Generator** is a system that estimates 52-dimensional ARKit BlendShape coefficients in real-time from audio input only, driving 3D avatar facial expressions.

### Features
- Lightweight model (~0.72M parameters)
- Fast inference (0.62ms/frame)
- OSC output support (compatible with VRChat, Unity, Blender, etc.)
- Trained on single-speaker data

---

## クイックスタート / Quick Start

### 環境構築 / Setup

```bash
git clone https://github.com/atsuki-ichikawa/facial-motion-generator.git
cd facial-motion-generator
pip install -r requirements.txt
```

### 推論の実行 / Run Inference

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth
```

OSCでBlendShape値が `127.0.0.1:9000` に送信されます。

---

## システム構成 / Architecture

![Architecture](docs/static/images/architecture_image.svg)

### 入力 / Input
- 音声: 16kHz, モノラル
- Log-mel特徴量: 80次元 × 7フレーム
- eGeMAPS特徴量: 88次元

### 出力 / Output
- ARKit BlendShape: 52次元
- 更新レート: 30Hz（設定可能）
- 出力形式: OSC (UDP)

---

## 学習 / Training

### データセット準備 / Dataset Preparation

本システムは [NVIDIA Audio2Face-3D Dataset (Claire)](https://huggingface.co/datasets/nvidia/Audio2Face-3D-Dataset-v1.0.0-claire) を使用して学習されました。

```bash
# データセット準備
python scripts/prepare_dataset.py --source /path/to/data --output data/processed

# キャッシュ作成（初回のみ）
python scripts/train.py --prepare-cache
```

### 学習の実行 / Run Training

```bash
python scripts/train.py --config configs/default.yaml
```

学習の再開:
```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

---

## 設定 / Configuration

`configs/default.yaml` で各種パラメータを調整できます。

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `inference.update_rate` | 推論更新レート | 30 Hz |
| `inference.osc_port` | OSC出力ポート | 9000 |
| `model.n_blendshapes` | BlendShape次元数 | 52 |
| `training.batch_size` | バッチサイズ | 32 |
| `training.learning_rate` | 学習率 | 0.0001 |

---

## 外部連携 / Integration

### OSC受信設定

BlendShape値は以下のOSCアドレスで送信されます：
- `/blendshapes` - 52次元配列
- `/blendshape/{i}` - 個別の値（i = 0-51）

### Unity
OSC受信プラグイン（例: [extOSC](https://assetstore.unity.com/packages/tools/input-management/extosc-open-sound-control-72005)）を使用してBlendShape値を受信できます。

### VRChat
OSCを有効にし、表情パラメータにマッピングしてください。

### Blender
[NodeOSC](https://github.com/maybites/blender.NodeOSC) アドオンを使用してBlendShape値を受信できます。

---

## 後処理オプション / Post-processing (Optional)

キャリブレーション機能を使用すると、出力値を調整できます：

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
    --calibration configs/calibration.json
```

---

## 引用 / Citation

```bibtex
@inproceedings{ichikawa2026facial,
  title={3Dアバターに適用可能な音声駆動型リアルタイム表情生成システム},
  author={市川淳貴 and 徳久良子},
  booktitle={言語処理学会 第32回年次大会},
  year={2026}
}
```

---

## ライセンス / License

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。

---

## 謝辞 / Acknowledgments

- 本研究はNVIDIAが公開する [Audio2Face-3D Dataset](https://huggingface.co/datasets/nvidia/Audio2Face-3D-Dataset-v1.0.0-claire) を使用しています。
- 愛知工業大学・理化学研究所

---

## 既知の制限事項 / Known Limitations

- 単一話者（Claire）のデータで学習されているため、多話者への一般化は未検証です
- 眉・頬・眼瞼など顔の上半分の動きは限定的です
- eGeMAPS特徴量の蓄積に最初の5秒が必要です
