# Jev Local

文章・JSON・画像に対する質問を、**Choice（選択肢）・Score（段階評価）・Noul（真偽）の確率**として返すローカルAPIです。同じ`/v1/systemone`を、LLMの先頭トークン判定とModernBERTの候補ペア採点で提供します。

| | LFM（既定） | Sarashina（追加） | ModernBERT |
|---|---|---|---|
| モデル | LFM2.5-1.2B Instruct / LFM2.5-VL-1.6B（Q8_0） | Sarashina2.2 Vision 3B（Q4_K_M / Q8_0） | ModernBERT-Ja 310Mをcross-encoderとしてfine-tune |
| 判定方式 | 回答先頭トークンのlogprob（zero-shot） | 同左（llama.cpp） | (質問＋State, 候補)ペアの採点（学習済み重みを配布） |
| 入力 | 文章・JSON・画像 | 文章・JSON。**画像は別ビルドの実験機能** | 文章・JSON |
| 強いところ | 知識・未知の分類体系・自由な指示文 | 今回の比較では常識QA・知識・ニュース分類がLFMより高い | 日本語の意図・関係判定と速度。候補順に依存しない |
| 追加依存 | なし（バイナリを自動取得） | 標準文章構成はなし。実験版は追加ビルド（画像はmmproj変換も必要） | torch＋transformers（別venv） |
| JGLUE test | JNLI 17% / JComQA 69% | Q4: JNLI 34% / JComQA 86% | JNLI 93% / JComQA 92%（trainで学習） |

クライアント・評価ツール・API仕様は共通です。まずLFMで動かしてから、[Sarashina比較バックエンド](docs/SARASHINA.md)や[ModernBERTバックエンド](docs/MODERNBERT.md)を追加できます。学習条件と量子化が異なるため、上表は同一APIでの実用比較であり、モデル能力の公平な順位付けではありません。

**Jev本体のモデル・学習・精度を再現するものではありません。** TypeSafeの`/v1/systemone`形式に合わせた非公式アダプターです。画像入力はローカル拡張で、公式SDKの画像互換を意味しません。[APIの互換範囲](docs/API.md)を参照してください。

## セットアップと起動（LFM）

Python **3.12以上**とBashが必要です。自動セットアップはLinux x86_64・arm64、macOS（Apple Silicon・Intel）に対応し、WindowsではWSL2を使用します。通常実行に追加Pythonパッケージやコンパイルは不要です。

```bash
git clone https://github.com/Argos1111/jev_local.git
cd jev_local
./setup.sh --model text
./run.sh --model text
```

画像も入力する場合は、次の構成を選びます。

```bash
./setup.sh --model vision
./run.sh --model vision
```

`text`は文章用モデルだけ、`vision`はVLモデルと画像エンコーダーを取得します。選択は保存され、以後の`./run.sh`はそのモデルを使います。両方を取得済みなら、`./run.sh --model text`／`--model vision`だけで切り替えられます。切り替える際は稼働中のサーバーをCtrl+Cで停止してから起動してください。未設定時の既定は`text`です。

`setup.sh`はGPUを検出してCUDA・ROCm・Metal版のllama.cppを選択し、選択したモデルを取得・SHA256検証します。VL構成は画像エンコーダー（mmproj F16）込みで約2.1 GBです。GPUを利用できない場合は理由を表示してCPU版に切り替えます。初回のみGitHub・Hugging Faceへのネット接続が必要です。

GPUドライバーやROCmの共有ライブラリは別途必要です。OSのパッケージは自動インストールしません。CPUに固定する場合は`./setup.sh --backend cpu`を使います。[対応環境・手動設定・トラブルシューティング](docs/SETUP.md)を参照してください。

**`Ready: http://127.0.0.1:8080`が表示されたら、サーバーを動かしたまま別ターミナルから入力を送ります。** Ctrl+Cで推論サーバーとAPIの両方が停止します。`Connection refused`の場合は起動状態と送信先ポートを確認してください。

ランタイムを変更せずモデルだけ追加する場合は`./setup.sh --model vision --model-only`（文章用は`--model text`）を使います。`LFM_MODEL`・`LFM_MMPROJ`を設定済みの場合はそれらが優先されるため、標準モデルの選択を使う前に`unset LFM_MODEL LFM_MMPROJ`で解除してください。

## テキスト・JSONで試す

```bash
python3 systemone_client.py
python3 systemone_client.py --input examples/systemone.json --format json
python3 systemone_client.py --input examples/systemone.json --output results/my_result.json
```

既定のサンプルは顧客の問い合わせに対する12問です。`--format json`で生JSONを表示します。`--output`の保存内容は表示形式にかかわらずJSONです。自分の入力には[`examples/systemone.json`](examples/systemone.json)の`state`と`questions`を書き換えてください。

## 画像で試す

`./run.sh --model vision`で起動します。画像を自分で用意し、[ローカルの画像フォルダ](sample_pics/README.md)などに置きます。画像はリポジトリに同梱していません。

```bash
python3 systemone_client.py \
  --input examples/vision.json \
  --image sample_pics/photo.jpg
```

GSS資料向けの[12問](examples/vision_gss.json)と[4問](examples/vision_gss_4.json)もあります。対応する画像を用意した場合は次のように実行できます。

```bash
python3 systemone_client.py \
  --input examples/vision_gss_4.json \
  --image sample_pics/20260911_image_resized.png \
  --format json \
  --output results/my_vision_gss_4.json
```

**LFMで速度を優先する場合は、画像の縦横をともに512px以内に収め（縦横比を維持）、質問を4問に絞る構成を推奨します。** 今回の検証では、512×286px・4問・GPU・画像エンコード再利用の組み合わせが最も速く、同じ画像の再送は中央値約98msでした。4問は4並列の1回分に収まります。画像サイズや質問数を網羅的に比較した結果ではなく、1〜3問より4問が速いことを意味しません。縮小と質問の選定は入力前に行います（自動変換・4問制限はありません）。

- PNG/JPEG、1枚4 MiB、最大4枚。複数枚は`--image`を繰り返します。
- HTTPではトップレベルの`images`配列にbase64 data URLを渡します。[画像API仕様](docs/API.md#画像入力ローカル拡張)を参照してください。
- 大きな画像・複数画像には`CTX_SIZE=32768 ./run.sh --model vision`などでコンテキストを増やします。既定は4 slots・合計8192トークンです。
- 各質問は独立に推論し、最大4問を並列処理します。画像入力ではdecoder側の共通Stateキャッシュを使いません。

## 画像エンコードの再利用（実験機能）

同じ画像を複数質問に使う場合、画像エンコーダーの出力だけをメモリ内で再利用するオプションがあります。**Linux x86_64 / AMD ROCm向けの追加ビルド**が必要です。通常の`setup.sh`・`run.sh`だけでは有効になりません。[ビルドと起動手順](docs/IMAGE_CACHE.md)に従って準備してください。

```bash
# ビルド済みの場合。通常サーバーとは別ポートを使用する例
./run_image_cache_gpu.sh --port 18080 --backend-port 18097
```

このサーバーに送るクライアントでは`--url http://127.0.0.1:18080`を追加します。画像＋stateを読み込むdecoderの処理と各質問の判定は、引き続き質問ごとに実行します。

Radeon AI PRO R9700・512×286 px画像・4問での測定例:

| 条件 | 応答時間 |
|---|---:|
| 再利用なし、10回の中央値 | 358 ms |
| 初回の画像、1回計算＋3回再利用（2回測定） | 192〜212 ms |
| 同じ画像を再送、10回の中央値 | 98 ms |

モデルロードを除くHTTP応答時間です。1画像での実験値で、速度や正答率を保証するものではありません。[測定条件と精度上の制約](docs/VISION_EVALUATION.md)も参照してください。

## ModernBERTバックエンド（文章のみ）

LLMの先頭トークン判定ではなく、[sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)を「(質問＋State, 候補)ペアの採点器」としてfine-tuneし、同じAPIを提供する構成です。torch＋transformersを別のvenvに導入します。候補順への依存がなく、1リクエストの全質問を1回のバッチ推論で処理します。

```bash
./setup_modernbert.sh        # .venv-modernbert（torch/transformers）。GPUを自動検出
./run_modernbert.sh          # 学習済み重み argos1111/modernbert-ja-310m-jev を取得して起動（初回約1.3 GB）
python3 systemone_client.py  # クライアントは共通
```

学習済み重みは[Hugging Face Hub](https://huggingface.co/argos1111/modernbert-ja-310m-jev)で公開しています（CC BY-SA 4.0）。推論はVRAM約2 GB、CPUでも動作します（12問で約2秒）。

自分で学習する場合は`./train_modernbert.sh`を実行します。公開日本語データ（JGLUE train・JCoLA・JCommonsenseMorality・MASSIVE）を自動取得し、R9700で約40分、bf16でVRAM約20 GB（`--pair-budget 64`で約10 GB）です。`models/modernbert-ja-310m-jev/`ができると`./run_modernbert.sh`はそちらを優先します。

画像入力は非対応で、`images`を含むリクエストは422を返します。JGLUEの高い数値は同じデータのtrainで学習した結果です。学習に使っていないタスクでは特性が分かれます：知識を問うタスク（ニュース分類・JMMLU）はLFMの方が高く、短い日本語の意図判定（顧客対応の手作り16例）はModernBERTの方が高い結果でした。数値と条件、学習データとライセンス、OS別の対応状況は[ModernBERTバックエンド](docs/MODERNBERT.md)を参照してください。

## Sarashina比較バックエンド

[Sarashina2.2 Vision 3B](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)の言語部分を、LFMと同じ先頭トークンlogprob方式で利用します。追加のPython依存はありません。

```bash
./setup.sh --model sarashina --model-only  # 既存ランタイムを維持。Q4_K_M、約2.07 GB
./run.sh --model sarashina
# 別ターミナル
python3 systemone_client.py
```

新規環境では`--model-only`を外してランタイムも取得します。量子化の対照用には`--model sarashina-q8`（Q8_0、約3.57 GB）を使えます。`setup.sh`のモデル選択は保存されます。LFMへ戻すには`./run.sh --model text`または`--model vision`で起動してください。

**通常のプロファイルは文章専用で、公開mmprojを自動取得・有効化しません。** SarashinaではChoiceは26候補までです（27以上は422）。Q4でも今回の常識QA・知識・ニュース分類はLFMより高い一方、JNLIと候補順への依存は課題です。

画像には、projectorの欠落LayerNorm・前処理・境界tokenを修正した**別の実験ビルド**を追加しました。CUDA/HIP向けの160次元GPU Flash Attentionと画像エンコード再利用も利用できます。GPU型番での利用制限はなく、未検証環境からのフィードバックも受け付けます。変換・ビルド済みなら:

```bash
python3 scripts/run_sarashina.py --port 18080 --backend-port 18097
# CUDA版をビルドした場合: --backend cuda --device CUDA0
# Q8: --model sarashina-q8
```

RTX 4090向けビルドは`python3 scripts/build_sarashina.py --backend cuda --cuda-architectures 89`（Linux／WSL2＋CUDA Toolkit）。RTX 5090など他世代の指定・mmproj変換・任意検査は[再現手順](docs/SARASHINA.md#実験版のビルドと起動)、報告するログは[フィードバック案内](docs/SARASHINA.md#gpu数値検査とフィードバック)を参照してください。**CUDAはsm_89向けビルド確認済み、実機推論は未検証**です。未検証や検査失敗を理由に起動を禁止せず、警告と記録に留めます。ビルド済みCPU / HIP / CUDA版と、[HFでPublic公開した公式由来mmproj](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16)の取得・起動は[Pre-releaseの利用案内](docs/SARASHINA_RELEASE.md)を参照してください。

公式checkpointからmmprojを再生成し、公式AutoProcessorに合わせて前処理を追加修正しました。R9700での元解像度文書＋4問の再送はQ4で約2.15秒→0.69秒、Q8で約2.05秒→0.60秒（同一ビルド内の画像キャッシュOFF/ON比較）。文書12問は両者12/12ですが、単色・複数画像の失敗と、言語GGUFの公式tokenizerとの分割差が残ります。旧成果物は保存し、**通常設定は変更せず、制約付きの実験機能**として提供します。[検証範囲・更新した変換／ビルド手順](docs/SARASHINA.md)を参照してください。

## API

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-dev' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

| 質問のtype | 得られるもの |
|---|---|
| `choice` | 選んだ候補、候補ごとの確率、分布の集中度 |
| `score` | 段階番号の期待値、各段階の確率、分布の集中度 |
| `noul` | 真である確率（0〜1） |

確率は指定した候補の中で正規化した値です。`confidence`は分布の集中度で、正解率として校正された値ではありません。[仕組みと制約](docs/DESIGN.md)を参照してください。

## フォルダ構成

```text
api_server.py / systemone.py       HTTP APIと型付き判断
jev_local.py / state_cache.py      候補確率の計算・テキストStateの再利用
image_input.py                    画像の検証・data URLへの変換
systemone_client.py / display.py   クライアント・結果表示
modernbert/                       ModernBERT cross-encoderの学習・推論・API
scripts/                          セットアップ・起動・実験ビルド
native/                           画像エンコードキャッシュとC++単体テスト
examples/                         リクエストJSON
sample_pics/                      ローカル入力画像（画像はGit対象外）
tests/                            Python単体テスト
tools/                            動作確認・評価・速度測定
docs/                             詳細な設定・仕様・測定条件
models/ .cache/ results/ .venv*/   ローカル生成物（Git対象外）
```

モデル・画像・生ログ・計測結果・ビルド成果物はGit管理しません。**修正版mmprojはHF、対応ランタイムはGitHub Releasesで別配布**し、取得元/SHA256・パッチ・変換／ビルドスクリプトをソースツリーに含めます。公開する測定要約は`docs/`にまとめています。`results/`内のファイルを編集しても配布コードにはならないため、再利用するスクリプトは`tools/`で管理します。

## 開発・検証

```bash
python3 -m unittest discover -s tests -v  # モデル・ダウンロード不要
python3 -m tools.verify_api              # 起動済みAPIの実通信確認
python3 -m tools.verify_vision           # 合成画像で画像入力を確認
python3 -m tools.benchmark_jglue --output results/jglue      # JGLUE test（各バックエンド共通）
python3 -m tools.evaluate_heldout_tasks --output results/ho  # 学習に使っていないタスクでの比較
```

CIではPythonテスト・シェル構文と、モデル不要のC++キャッシュテストを実行します。ModernBERTのテストも疑似エンコーダーで動くため、CIにtorchは不要です。GPU推論の確認はローカルで行います。

- [セットアップ詳細](docs/SETUP.md)
- [API・公式SDK接続](docs/API.md)
- [画像エンコードキャッシュ](docs/IMAGE_CACHE.md)
- [画像評価の記録](docs/VISION_EVALUATION.md)
- [評価ツール・速度測定](docs/EVALUATION.md)
- [共通Stateの再利用](docs/STATE_CACHE.md)
- [JGLUE評価](docs/JGLUE.md)
- [ModernBERTバックエンド](docs/MODERNBERT.md)
- [Sarashinaバックエンド・全方式の比較](docs/SARASHINA.md)

依存する[llama.cpp](https://github.com/ggml-org/llama.cpp)と[LFM2.5モデル](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF)、[Sarashina2.2 Vision](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)・[ModernBERT-Ja](https://huggingface.co/sbintuitions/modernbert-ja-310m)（MIT）は、それぞれの配布元の利用条件に従います。ModernBERTの学習に使う公開データセット（JGLUE・JCoLA・JMMLU: CC BY-SA 4.0、JCommonsenseMorality: MIT、MASSIVE: CC BY 4.0、livedoor: CC BY-ND 2.1 JP）は実行時に取得し、リポジトリには含めません。学習済みモデルを再配布する場合はCC BY-SAの継承条件に留意してください。Gitのソースツリーにバイナリ・モデル重みは同梱しません。
