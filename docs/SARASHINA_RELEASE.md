# Sarashina llama.cpp Pre-release

Sarashinaの修正版mmprojに対応する、Jev Local独自のllama.cpp b11042ビルドです。Linux x86_64向けのCPU / HIP / CUDA版を用意しています。

配布ページ: **[sarashina-llama-b11042-pre1](https://github.com/Argos1111/jev_local/releases/tag/sarashina-llama-b11042-pre1)**（Pre-release、Latestにはしない）。GitHubと、公式由来mmprojを置いたHFリポジトリはどちらもPublicです。

## 配布ファイル

| アーカイブ名の末尾 | 内容・外部依存 |
|---|---|
| `linux-x86_64-cpu.tar.gz` | CPU。AVX2 / FMA / F16C |
| `linux-x86_64-hip-gfx1100-gfx1201.tar.gz` | HIP、ROCm SDK 10.0でビルド。HIP 7 / hipBLAS 3 / rocBLAS 5と推移的依存・Tensileデータが別途必要 |
| `linux-x86_64-cuda13-sm89.tar.gz` | CUDA 13.2でビルド、`sm_89`と`compute_89`。CUDA Runtime 13 / cuBLAS 13 / 対応NVIDIAドライバーが別途必要。NVIDIA実機未検証 |

共通のファイル名prefixは`llama-b11042-jev-sarashina-pre1-`です。Linux x86_64 / WSL2向けで、Windows native・macOS版ではありません。Ubuntu 24.04の隔離環境でビルドしています。ELFが要求する最大symbol versionはGLIBC 2.38、GLIBCXX 3.4.32、CXXABI 1.3.13です。Ubuntu 22.04の標準環境では動きません。

同梱: `llama-server`、必要なllama.cpp共有ライブラリ（`libllama-common`も含む）、診断ツール、パッチ・ビルド入力、ビルドmanifest、検査記録、LICENSE / NOTICE、SHA256SUMS。モデル、GPU/OS共有ライブラリ、GPUドライバー、Python環境、Web UIは含みません。

GPUのコンパイル対象はこのアーカイブの内容であって、実行時の型番ホワイトリストではありません。別archはソースからビルドできます。検査未実施・失敗を起動禁止の条件にしません。

## 利用方法

配布ページから該当アーカイブと`SHA256SUMS`をダウンロードし、検査してから任意の場所へ展開します。以下の`RUNTIME`は展開ディレクトリへ置き換えてください。

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu.tar.gz
RUNTIME="$PWD/llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS)
"$RUNTIME/bin/llama-server" --version
"$RUNTIME/bin/llama-server" --list-devices
```

Jev Localをこのリリースに対応するソースへ更新し、言語GGUFと公式由来mmproj・同名`.gguf.json`を`models/`へ置いて起動します。

```bash
# Jev Localのルートで実行
python3 scripts/run_sarashina.py --backend cpu --build "$RUNTIME"
```

HIP / CUDAはbackendとアーカイブを合わせます。ライブラリが標準パスにない場合は、その利用者のインストール先を`LD_LIBRARY_PATH`へ指定してください。ROCmの分割SDKでは`ROCBLAS_TENSILE_LIBPATH` / `HIPBLASLT_TENSILE_LIBPATH`も必要な場合があります。**ビルドしたマシンのライブラリパスは配布manifestへ埋め込んでいません。**

`--build`指定の起動はログ・slot cacheをmanifestのhashごとに隔離し、既存のビルド・通常の保存済みモデル/ランタイム選択を変えません。バイナリ・projectorの整合性と前処理の互換性は検査します。

公式由来mmprojは[HFの専用リポジトリ](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16)で**Public公開済み**です。認証なしで取得できます。次の2ファイルをJev Localの`models/`へ配置してください。

- [`sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf`](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16/resolve/main/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf)（約893 MB）
- [`sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json`](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16/resolve/main/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json)

言語GGUFは`./setup.sh --model sarashina --model-only`（Q8_0は`--model sarashina-q8`）で`models/`へ取得します。ログイン不要で、SHA256を検証し、ランタイム選択は変更しません。配布済みmmprojを使う場合、公式checkpointの取得・変換は不要です。ランタイムにはモデル重みを含みません。

## 今回の配布ビルドの検査

- 配布準備時のPython単体テスト91件。展開したCPU / HIP版を`--build`指定で起動し、API契約・赤/青の画像判定・保存済み選択の維持を確認。
- Ubuntu 24.04 userspaceで、元のビルドディレクトリを隠して別の場所へ展開。CPU / HIPで`--version`・依存解決・画像生成を確認。青・赤・青の四角形3回に対応する回答と画像cache hitを確認。
- 公式AutoProcessorとの画素照合: 15/15、最大絶対差7.16e-7以下（基準1e-6）。
- encoder照合: CPU / HIP各5/5。cosineの最小値はCPU 0.999805、HIP 0.999747、RMSEの最大値はCPU 0.01986、HIP 0.02259。
- HIP FA160: R9700 / RX 7900 XTXで各252/252。F16 KV、NMSE基準5e-4。追加vector-slice検査は無効。
- CUDA: ビルドと移動後のCPU側起動まで。NVIDIA GPU/ドライバーなしで、GPU数値検査は0/0（合格ではない）。CUDA版による画像推論・FA160実行・速度は未検証。

これは再配置・数値・小規模smoke検査で、モデル精度や全GPUでの動作保証ではありません。単色・複数画像・候補順・言語GGUFのtokenizer差などの制約は[詳細](SARASHINA.md)を参照してください。配布ビルドの速度比較はしていません。WSL2そのものの検査も未実施です。

## 再ビルド・梱包

Ubuntu 24.04とGCC 13.3、CMake 3.28.3、Ninja 1.11.1を使用。CUDAは13.2.51 / cuBLAS 13.3.0.5 / CCCL 3.2.0、HIPはROCm SDK 10.0.0 / AMD clang 23.0.0gitを使用しました。ホストOS・既存venvは変更せず、Ubuntu Baseを`.cache/sarashina-prerelease/`へ展開してbwrapで隔離しました。

```bash
CC=gcc-13 CXX=g++-13 python3 scripts/build_sarashina.py \
  --backend cpu --base .cache/sarashina-prerelease/runtime \
  --cmake-arg='-DCMAKE_BUILD_RPATH=$ORIGIN' \
  --cmake-arg=-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  --cmake-arg='-DCMAKE_INSTALL_RPATH=$ORIGIN' \
  --cmake-arg=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

HIP / CUDAは[ビルド手順](SARASHINA.md#実験版のビルドと起動)のtoolchain指定と、それぞれ`--hip-architectures 'gfx1100;gfx1201'` / `--cuda-architectures 89`を追加します。CUDA配布版は`--cmake-arg=-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`も指定しています。正確なCMake引数はアーカイブ内のmanifestへ記録しています。

```bash
python3 scripts/prepare_sarashina_release.py \
  --build .cache/sarashina-prerelease/runtime/build-cpu \
  --name llama-b11042-jev-sarashina-pre1-linux-x86_64-cpu
```

梱包スクリプトはネットワーク・認証・Git操作を行いません。既存出力を上書きせず、source/patch/binary hash、共有ライブラリの閉包、`$ORIGIN`、外部依存、LICENSE原文を検査し、再展開してhashを確認します。GPU検査結果は任意添付で、合否を配布可否の制限にはしません。LICENSEは[`licenses/runtime/`](../licenses/runtime/)と固定b11042ソースから取得します。独自コードへの新しいライセンス選択はしていません。

## 公開手順

1. GitHub CLIへログイン。HFトークンではなくGitHub認証を使い、トークンをソース・チャットへ書かない。
2. この変更をJev Localへcommit/pushし、公開するソースcommitを確定。通常モデルの既定は変更しない。
3. 確定commitへタグを付けてpushし、`gh release create ... --verify-tag --prerelease --latest=false`で作成する。まずDraftでアーカイブを送信・検査してから公開する。
4. GitHub APIのasset size/digestと再ダウンロードhashを照合する。HFの可視性変更は別操作として扱い、配布担当者の明示的な承認を得る（今回のmmprojは別途承認を受けPublic公開済み）。

GitHub認証用CLIはこの環境では`.cache/release-tools/gh_2.101.0_linux_amd64/bin/gh`へ隔離して用意しています。Release公開のためにggml-org/llama.cppへPR・commit・pushする必要はありません。
