# Sarashinaで画像を使う

Sarashinaの画像入力には、修正版のmmproj（画像エンコーダー）と、それに対応するllama.cppランタイムが必要です。どちらも配布済みで、ビルドやHugging Faceへのログインは不要です。

- 対応環境: Linux x86_64 / WSL2。CPU、AMD GPU（ROCm）、NVIDIA GPU（CUDA）
- Jev Localのルートで作業します（Python 3.12以上。追加のPythonパッケージは不要）

## 1. モデルファイルを`models/`へ置く

言語GGUF（Q4_K_M、約2.07 GB）を取得します。取得先は[mradermacher/sarashina2.2-vision-3b-GGUF](https://huggingface.co/mradermacher/sarashina2.2-vision-3b-GGUF)で、SHA256を検証します。通常のランタイム選択は変わりません。

```bash
./setup.sh --model sarashina --model-only
```

Q8_0（約3.57 GB）を使う場合は`--model sarashina-q8`を指定します。

mmprojは[argos1111/sarashina2.2-vision-3b-mmproj-jev-f16](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16)から次の2ファイルを取得します（合計約893 MB）。

```bash
HF=https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16/resolve/main
curl -L -o models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf      "$HF/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf"
curl -L -o models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json "$HF/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json"
```

同じ場所の`SHA256SUMS`で照合できます。`.gguf.json`は起動時の整合性確認に使うので、必ずGGUFと一緒に置いてください。

## 2. ランタイムを展開する

[リリースページ sarashina-llama-b11042-pre1](https://github.com/Argos1111/jev_local/releases/tag/sarashina-llama-b11042-pre1)から環境に合うアーカイブをダウンロードします。

| 環境 | アーカイブ | 別途必要なもの |
|---|---|---|
| CPUのみ | `llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu.tar.gz` | AVX2 / FMA / F16C対応CPU |
| AMD GPU（RX 7900シリーズ、Radeon AI PRO R9700などgfx1100 / gfx1201） | `llama-b11042-jev-sarashina-pre1-linux-x86_64-hip-gfx1100-gfx1201.tar.gz` | ROCm 7系のHIP / hipBLAS / rocBLAS共有ライブラリとドライバー |
| NVIDIA GPU（RTX 40シリーズ、sm_89） | `llama-b11042-jev-sarashina-pre1-linux-x86_64-cuda13-sm89.tar.gz` | CUDA 13のRuntime / cuBLASと対応ドライバー |

いずれもUbuntu 24.04でビルドしており、GLIBC 2.38以上（Ubuntu 24.04以降など）が必要です。Ubuntu 22.04では動きません。他のGPU世代やCPUでは[ソースからビルド](SARASHINA_BUILD.md)してください。

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu.tar.gz
RUNTIME="$PWD/llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu"
"$RUNTIME/bin/llama-server" --list-devices
```

`--list-devices`にGPUが表示されない場合は、GPUライブラリのパスを`LD_LIBRARY_PATH`に設定してください。ROCmの分割インストールでは`ROCBLAS_TENSILE_LIBPATH` / `HIPBLASLT_TENSILE_LIBPATH`も必要な場合があります。

**CUDA版はNVIDIA GPUでの動作を確認できていません。** 問題があれば[Issue](https://github.com/Argos1111/jev_local/issues)で知らせてください。

## 3. 起動して画像で質問する

```bash
# Jev Localのルートで実行。--backendはアーカイブに合わせる（cpu / hip / cuda）
python3 scripts/run_sarashina.py --backend cpu --build "$RUNTIME"

# 別ターミナル
python3 systemone_client.py --input examples/vision.json --image /path/to/photo.jpg
```

- Q8_0を使う場合は`--model sarashina-q8`を追加します。
- APIは`http://127.0.0.1:8080`で待ち受けます。ポートを変える場合は`--port`と`--backend-port`を指定します。
- 複数GPUで使うGPUを選ぶには`--device ROCm1`のように指定します。
- 長い入力・大きな画像には`CTX_SIZE=32768`を前置します。
- 通常の`run.sh`の設定・保存済みのモデル選択は変わりません。ログは`.cache/sarashina-official-runtime/`以下に分けて保存します。

画像はPNG / JPEG、1枚4 MiBまで、最大4枚です。文書画像は縮小せず元の解像度で送る方が読み取りが安定します。質問の書き方は[API](API.md#画像入力ローカル拡張)を参照してください。

## ランタイムの内容

llama.cpp b11042に、Sarashinaの画像入力に必要な修正（projectorの最終LayerNorm、公式の画像前処理、画像境界トークン）と、同じ画像を複数質問で使う際のエンコード結果の再利用、GPU向けFlash Attention（Sarashinaのヘッド次元160への対応）を加えたものです。未修正のllama.cppではこのmmprojは正しく動きません。

アーカイブにはLICENSE / NOTICE、ビルド設定、検査記録、SHA256SUMSを同梱しています。モデル、GPU / OSの共有ライブラリ、ドライバーは含みません。

## 既知の制約

- 単色画像の色や、2枚以上の画像の順番に関する質問で誤答が残っています。
- 候補の順番で判定が変わることがあります。
- 文書画像を小さく縮小すると読み取り精度が落ちます。
- 通常の`run.sh`と同じく、Choiceは26候補までです。
