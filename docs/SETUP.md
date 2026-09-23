# セットアップ

## 標準構成

`./setup.sh --model text`は文章用のLFM2.5-1.2B Instruct Q8_0、`./setup.sh --model vision`は画像対応のLFM2.5-VL-1.6B Q8_0とmmprojを取得します。`--model sarashina`はSarashina2.2 Vision 3B Q4_K_M、`--model sarashina-q8`は同Q8_0の言語部分だけを取得します（[Sarashinaバックエンド](SARASHINA.md)）。GPUを検出して対応するllama.cpp公式バイナリも導入します。Python 3.12以上とBashを使用し、pip installやコンパイルは不要です。

| 環境 | 自動選択 |
|---|---|
| Linux x86_64 + NVIDIA | CUDA 12.8版（CUDAユーザー空間ライブラリも取得） |
| Linux x86_64 + AMD | ROCm 10.0版 |
| macOS Apple Silicon | Metal版 |
| Linux x86_64・arm64 / macOS Intel、またはGPU利用不可 | CPU版 |

NVIDIAとAMDの両方があればCUDA、ROCm、CPUの順に試します。Intel GPUとLinux arm64 GPUの自動導入は未対応で、CPU版を選びます。WindowsではWSL2を使用します。

**MLXはllama.cppのバックエンドではなく別の推論基盤です。** このプロジェクトはGGUFとllama.cpp APIを使うため、Apple SiliconにはMetal版を導入します。

取得した実行ファイルの`--list-devices`で、対象バックエンドのGPUが実際に認識されることまで確認します。ドライバー・共有ライブラリ・アクセス権などにより認識できない場合は理由を表示し、次の候補を試します。ダウンロード失敗やSHA256不一致はCPUへの切り替えで隠さず、セットアップを停止します。

GPUドライバーはインストール済みである必要があります。ROCm版は対応するシステム側ROCmランタイムも必要です。セットアップはOSのドライバーやパッケージを変更しません。GPU検出は推論成功の保証ではなく、VRAM不足や対応外のGPUアーキテクチャなどは起動時に判明する場合があります。

```bash
./setup.sh                    # 自動検出・導入
./setup.sh --dry-run          # 候補の表示のみ。ダウンロードしない
./setup.sh --backend cpu      # CPUを明示
./setup.sh --backend cuda     # CUDAを明示。利用不可ならエラー
./setup.sh --backend rocm     # ROCmを明示。利用不可ならエラー
./setup.sh --backend metal    # Apple SiliconのMetalを明示
```

- llama.cpp: **b11042**（[公式リリース](https://github.com/ggml-org/llama.cpp/releases/tag/b11042)）
- 文章モデル: **LFM2.5-1.2B-Instruct-Q8_0.gguf**
- 画像対応モデル: **LFM2.5-VL-1.6B-Q8_0.gguf**（約1.25 GB）
- 画像エンコーダー: **mmproj-LFM2.5-VL-1.6b-F16.gguf**（約0.85 GB）
- Sarashina比較用: **sarashina2.2-vision-3b.Q4_K_M.gguf**（約2.07 GB）または**Q8_0**（約3.57 GB）。mmprojなしのテキスト入力のみ
- 全配布物のURL・モデルリビジョン・SHA256: [`scripts/runtime.json`](../scripts/runtime.json)
- 保存先: `.cache/runtime/b11042/<環境>/`、`models/`
- ランタイム選択結果: `.cache/runtime/selected.json`。`run.sh`と`run_server.sh`が読み込み、GPU版では自動で全レイヤーをオフロードします。

ダウンロード済みアーカイブとモデルはSHA256を検証して再利用します。展開済みランタイムは起動確認して再利用します。SHA256不一致の場合は自動上書きせず停止します。該当ファイルを別の場所に移して再実行してください。セットアップの同時実行は避けてください。

VL構成ではモデル本体と画像エンコーダーの計約2.1 GBに加え、選択するランタイム（CUDA版は付属ライブラリ込みで約760 MBのダウンロード）、展開用ディスク、推論用メモリが必要です。初回はGitHubとHugging FaceへのHTTPSアクセスが必要で、取得完了後のデモ推論はローカルで動作します。

モデルの選択は`.cache/runtime/model.json`に保存します。`--model`を省略すると`LFM_PROFILE`、保存した選択、`text`の順に解決します。各モデルを取得済みなら、サーバーを停止し`./run.sh --model text`・`--model vision`・`--model sarashina`・`--model sarashina-q8`で起動し直して切り替えます。起動時の指定は保存された選択を変更しません。`run_server.sh`でも同じオプションを使えます。

## 起動設定

`./run.sh`が推論サーバーの準備完了を待ってからAPIを起動します。CPUスレッド数は8 threads、4 slots、合計context 8192（各slot 2048）です。

```bash
THREADS=4 CTX_SIZE=16384 ./run.sh
./run.sh --port 8081 --backend-port 8098
python3 systemone_client.py --url http://127.0.0.1:8081
```

| 設定 | 既定値 | 用途 |
|---|---|---|
| `LLAMA_SERVER` | 自動選択した実行ファイル | 別の推論バイナリ |
| `LFM_PROFILE` | 保存した選択、未設定なら`text` | `text` / `vision` / `sarashina` / `sarashina-q8`。CLIの`--model`が優先 |
| `LFM_MODEL` | 選択したモデルの絶対パス | 既存GGUF。標準モデル選択を上書き（変数名は歴史的なもの。Sarashinaにも適用） |
| `LFM_MMPROJ` | `vision`のみ標準mmproj、他はなし | 画像エンコーダー。空文字で無効化 |
| `THREADS` | `8` | CPUスレッド数 |
| `CTX_SIZE` | `8192` | 全slotの合計コンテキスト |
| `GPU_LAYERS` | GPU版は`all`、CPU版は`0` | 自動設定を上書き |
| `GPU_DEVICE` | 未指定 | GPU版で明示する場合のデバイス名 |
| `SLOT_CACHE_DIR` | `.cache/slots`の絶対パス | 一時状態の保存先 |
| `JEV_BACKEND_LOG` | `.cache/llama-server.log` | `run.sh`のbackendログ。実験ランチャーは専用パスを指定 |
| `JEV_API_KEY` | `local-dev` | サーバー・クライアント共通のローカルキー |

環境変数はシェルで`export`するか、コマンドの前に指定します。`.env`の自動読み込みはありません。パスに空白がある場合は引用符で囲んでください。

サービスを個別に起動することもできます。

```bash
./run_server.sh       # ターミナル1、localhost:8097
./run_api_server.sh   # ターミナル2、localhost:8080
```

個別起動でslotディレクトリを変更する場合は、推論側の`SLOT_CACHE_DIR`とAPI側の`--slot-cache-dir`を合わせます。`run.sh`では両側を自動で揃えます。

## 既存モデル・GPU・他のOS

自動導入とは別のバージョンやビルドを使う場合は、公式[ビルド手順](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)に従って用意し、`LLAMA_SERVER`を指定してください。手動指定時は`GPU_LAYERS=all`も指定します。

```bash
# 既存のバイナリとGGUFを使う場合、setup.shは不要
export LLAMA_SERVER="/absolute/path/to/llama-server"
export LFM_MODEL="/absolute/path/to/LFM2.5-VL-1.6B-Q8_0.gguf"
export LFM_MMPROJ="/absolute/path/to/mmproj-LFM2.5-VL-1.6b-F16.gguf"
GPU_LAYERS=all ./run.sh
```

`LFM_MODEL`を明示すると既定mmprojの自動指定を止めます。画像を使う場合は、そのモデルに対応した`LFM_MMPROJ`も指定してください。CPU設定（`GPU_LAYERS=0`）では画像エンコーダーもCPUで実行します。

モデル本体とmmprojだけ必要なら`./setup.sh --model vision --model-only`（文章モデルは`--model text`）で取得できます。LM Studio同梱バイナリを使う場合も`LLAMA_SERVER`を明示します。必要な共有ライブラリは配布元の手順で設定してください。

Windowsは`fcntl`によるファイルロックとBashを使用するため、ネイティブ実行ではなくWSL2を使用してください。動作確認はLinux x86_64（AMD GPU / ROCm、CPU）で行っています。macOS（Metal）とNVIDIA（CUDA）の自動選択は実装済みですが実機では未確認です。ROCmの共有ライブラリが見つからない場合はCPUへ自動で切り替わるので、`LD_LIBRARY_PATH`にROCmライブラリのパスを設定してください。

バックエンドは`/props`、`/apply-template`、`/tokenize`、`/completion`（`post_sampling_probs=false`のlogprob応答）に対応する必要があります。共通State再利用にはslot保存・復元も必要です。古いllama.cppとの互換性は保証しません。

## トラブルシューティング

- **Connection refused**: 送信先APIのポートでサーバーが動いていません。`run.sh`の`Ready:`表示を待ち、そのターミナルを開いたまま送信してください。`--url`はAPIポート（既定8080）で、推論バックエンドの8097ではありません。
- **起動できない**: `.cache/llama-server.log`を確認します。バイナリが見つからなければ`./setup.sh`を実行してください。
- **GPUで動かない**: `./setup.sh --backend cuda`（または`rocm` / `metal`）でエラーを確認します。ドライバー・ROCmランタイム・デバイスアクセス権を確認してください。CPUに固定するには`./setup.sh --backend cpu`を実行します。
- **共有ライブラリ／GLIBCのエラー**: 配布バイナリとOSの組み合わせが非対応です。対応するLinux環境を使うか、その環境向けにllama.cppをビルドし`LLAMA_SERVER`を指定します。
- **ポートが使用中**: 既存プロセスを終了するか`--port`と`--backend-port`で別ポートを指定します。
- **長い入力で422**: `CTX_SIZE=32768 ./run.sh`などで増やします。必要メモリも増加します。
- **slot保存・復元のエラー**: `./run.sh --state-cache off`で切り分けます。[共通Stateの説明](STATE_CACHE.md)も確認してください。
- **401**: APIとクライアントに同じ`JEV_API_KEY`を設定します。

APIは既定で`127.0.0.1`に待ち受けます。`/health`はAPIプロセスの生存確認で、推論の確認には`python3 -m tools.verify_api`を使います。

画像の実行には`/props`のvision対応とmedia_marker、および`/completion`のmultimodal_data対応が必要です。画像もコンテキストを消費するため、大きな画像・複数画像で422になる場合は画像を縮小するか`CTX_SIZE`を増やしてください。
