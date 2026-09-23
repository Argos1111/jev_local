# Sarashina対応ランタイムをソースからビルドする

[配布済みランタイム](SARASHINA_RELEASE.md)が環境に合わない場合（別のGPU世代、AVX2非対応のCPU、別のCUDA / ROCm版など）に、同じランタイムをソースからビルドする手順です。mmprojは配布済みのものをそのまま使えます。

## 必要なもの

- Linux / WSL2、Python 3.12以上、GCC / g++（C++17）、CMake 3.24以上、Ninja、GNU patch
- CUDA版: NVIDIAドライバーとCUDA Toolkit（nvcc）。PyTorchのCUDA wheelだけではビルドできません
- HIP版: ROCm開発環境（hipcc、hipBLAS / rocBLASの開発ファイル）

`CC` / `CXX` / `CMAKE` / `NINJA` / `PATCH`で実行ファイルを指定できます。OSのパッケージやドライバーは自動インストールしません。

## ビルド

ソース（llama.cpp b11042）は自動で取得・検証し、`native/`のパッチを適用して`.cache/sarashina-official-runtime/`にビルドします。通常のランタイム・保存済みのモデル選択は変わりません。

```bash
# NVIDIA GPU。--cuda-architectures はGPUに合わせる（下表）
python3 scripts/build_sarashina.py --backend cuda --cuda-architectures 89 --jobs 8

# AMD GPU。省略時はCMake / HIPの自動検出
CMAKE_PREFIX_PATH=/opt/rocm python3 scripts/build_sarashina.py --backend hip --jobs 8
# GPUを明示する例: --hip-architectures gfx1100  /  --hip-architectures 'gfx1100;gfx1201'

# CPUのみ
python3 scripts/build_sarashina.py --backend cpu --jobs 8
```

| NVIDIA GPU | `--cuda-architectures` |
|---|---|
| RTX 30シリーズ | `86` |
| RTX 40シリーズ | `89` |
| RTX 50シリーズ | `120`（CUDA Toolkit 12.8以降） |

複数世代には`'89;120'`のように指定します。CUDAコンパイラは`CUDACXX`、HIPコンパイラは`--cmake-arg=-DCMAKE_HIP_COMPILER=...`で指定できます。

GPU用のFlash Attentionカーネルの数値検査をビルド後に行うには`--test-fa`を追加します（対象GPUは`--test-device CUDA0`などで指定）。検査は任意で、未実施や失敗でもビルド成果物は使えます。

## 起動

```bash
unset LFM_MODEL LFM_MMPROJ
python3 scripts/run_sarashina.py --backend cuda --device CUDA0
# AMD GPU: --backend hip --device ROCm0
# CPU:     --backend cpu
# Q8_0:    --model sarashina-q8
```

`--build`を省略すると、上記でビルドした`.cache/sarashina-official-runtime/build-<backend>/`を使います。mmprojは`models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf`を既定とし、[配布済みのもの](SARASHINA_RELEASE.md#1-モデルファイルをmodelsへ置く)をそのまま使えます。そのほかのオプションは配布版と同じです。

## GPUの動作確認とフィードバック

配布版で確認済みのGPUはRadeon AI PRO R9700とRX 7900 XTXで、NVIDIA GPUでは動作を確認できていません。他のGPUで動かした結果は歓迎します。

Flash Attentionカーネルの数値検査は、モデルをダウンロードする前でも実行できます。

```bash
python3 -m tools.verify_sarashina_fa --backend cuda --device CUDA0 --output results/my-gpu-kernels
```

Issueには、GPU名、OS、ドライバーとCUDA / ROCmのバージョン、ビルドコマンド、`.cache/sarashina-official-runtime/build-<backend>/jev-build.json`、上記の検査結果を添えてください。ログにはローカルのパスや入力内容が含まれることがあるので、公開前に確認してください。

## mmprojを自分で変換する

配布済みのmmprojで足りるため通常は不要です。公式checkpointから変換し直す場合は、[公式のSarashinaページ](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)でアクセス承認を受け、HF CLI（`.venv-hf`）でログインしてから実行します。約7.6 GBのcheckpointを取得します。

```bash
python3 -m venv .venv-hf && .venv-hf/bin/python -m pip install 'huggingface_hub==1.32.0'
.venv-hf/bin/hf auth login
.venv-hf/bin/python scripts/fetch_sarashina_reference.py

python3 -m venv .venv-sarashina
.venv-sarashina/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-sarashina/bin/python -m pip install -r scripts/requirements-sarashina.txt
.venv-sarashina/bin/python scripts/convert_sarashina_mmproj.py \
  --config .cache/sarashina-official/config.json \
  --checkpoint .cache/sarashina-official/model.safetensors \
  --output models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf
```

変換はcheckpoint付属のPythonコードを実行せず、既存の出力を上書きしません。変換結果の`.gguf.json`には入力・出力のSHA256を記録し、起動時に照合します。
