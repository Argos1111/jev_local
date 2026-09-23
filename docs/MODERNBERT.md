# ModernBERTバックエンド

LFM（llama.cpp）の代わりに、[sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)（MIT License）を**cross-encoder**としてfine-tuneし、同じ`/v1/systemone` APIを提供する構成です。テキスト・JSONのStateのみ対応し、画像入力はありません。

## LFM方式との違い

| | LFM 1.2B（既定） | ModernBERT-Ja 310M |
|---|---|---|
| 判定方法 | 回答先頭トークン（A/B/C…）のlogprob | (質問＋State, 候補) ペアごとに1つのlogit |
| 学習 | zero-shot、プロンプトのみ | 公開日本語データでfine-tune（学習済み重みを配布） |
| 候補順への依存 | あり（位置バイアス） | なし（各候補を独立に採点） |
| 候補数の上限 | 26（英字）／255（数字トークン） | 制限なし（255はAPI側の上限） |
| 1リクエストの推論 | 質問ごとに1回、最大4並列 | 全質問・全候補を1バッチで1回 |
| 画像 | VLモデルで可 | 不可 |
| 実行環境 | llama.cppバイナリ、追加Python不要 | torch＋transformers（別venv） |
| 未知のタスク | 指示文をそのまま解釈 | 学習分布に依存。指示文の意味理解は限定的 |

確率は候補間のsoftmaxで正規化した値、`confidence`は同じ`1 − 正規化エントロピー`です。校正された正解率ではありません。

## セットアップ

Python 3.12以上。GPUはNVIDIA（CUDA）またはAMD（ROCm 10.0、Linux）を自動検出し、なければCPU版を選びます。

```bash
./setup_modernbert.sh            # .venv-modernbert を作成
./setup_modernbert.sh --backend cpu
./setup_modernbert.sh --backend cuda --cuda cu130   # 新しいNVIDIAドライバー向け。既定はcu126
./setup_modernbert.sh --dry-run  # 検出結果とpipコマンドのみ表示
```

| バックエンド | wheelの取得元 | ダウンロード量の目安 | 検証状況 |
|---|---|---|---|
| ROCm 10.0（AMD, Linux） | stable.repo.amd.com | 約3〜5 GB（ROCmランタイム同梱） | R9700・RX 7900 XTXで学習・推論を確認 |
| CUDA cu126 / cu130（NVIDIA） | download.pytorch.org | 約2.5〜3 GB（cuDNN等同梱） | `pip install --dry-run`で依存解決を確認。実機未実行 |
| CPU | download.pytorch.org | 約200 MB | 同上。推論速度は未測定 |

ROCm版はPython側のwheelにROCmランタイムが含まれるため、システムにROCmを別途入れる必要はありません。GPUドライバー（amdgpuカーネルモジュール、`/dev/kfd`へのアクセス権）は必要です。NVIDIAはCUDA対応ドライバーが必要で、cu126はドライバー525以降、cu130は580以降を想定します。

### OS別の状況

| OS | 推論 | 学習 | 備考 |
|---|---|---|---|
| Linux x86_64 | ROCm / CUDA / CPU | 同左 | ROCmで検証済み |
| Windows（WSL2） | Linuxと同じ | 同左 | 既存のLFM構成と同じ前提。GPUはWSL2経由で利用可（未検証） |
| Windowsネイティブ | 未対応 | 未対応 | torch自体のWindows wheel（CPU / cu126 / cu130 / ROCm）は存在しますが、付属シェルスクリプトと`state_cache.py`の`fcntl`がPOSIX前提です。`python -m modernbert.server`を直接起動すれば動く可能性はありますが未検証です |
| macOS Apple Silicon | MPS（Apple GPU）または CPU | CPU / MPS（低速、推奨しない） | `setup_modernbert.sh`はPyPIのmacOS wheelを入れます。`--device auto`はMPSがあれば選びます。MPS・macOSは実機未検証です |
| macOS Intel | CPU | CPU（非推奨） | torch 2.13.0にIntel Mac wheelはありません。動作保証外 |

CPU推論の目安（24スレッド、float32）: README 12問サンプル約1.9秒、noul 1問約110 ms。GPUより大幅に遅いですが、少数の質問なら実用範囲です。CPU・MPSではbf16を使わずfloat32で動作します。

学習には1.2 GBのモデルダウンロード（Hugging Face Hub、初回のみ）と、bf16で約20 GBのVRAMが必要です（`--pair-budget 64`なら約10 GB）。推論だけなら約2 GBで動きます。

AMDでは`/sys/class/kfd`からgfxターゲット（例: Radeon AI PRO R9700 = gfx1201、RX 7900 XTX = gfx1100）を検出し、対応するデバイスパッケージを入れます。`--gfx`で明示できます。AMDのwheelが対応するgfxは`https://stable.repo.amd.com/rocm/whl-next/`の`amd-torch-device-gfx*`一覧で確認できます（2026-09時点でgfx908/90a/942/950とgfx10xx/11xx/12xxを含みます）。

**異世代のAMD GPUを2枚以上搭載している場合**（検証環境: R9700 gfx1201 + RX 7900 XTX gfx1100）、2枚目以降を`--device cuda:1`のように指定するとROCmランタイムがSegmentation faultで落ちます（torch 2.13.0+rocm10.0.0で確認。デバイスパッケージを両方入れても同じ）。使いたいGPUを環境変数で単独に見せてしまうと動きます。

```bash
HIP_VISIBLE_DEVICES=1 ./run_modernbert.sh        # 2枚目だけを使う（プロセス内ではcuda:0になる）
```

AMD版torchは一部演算をTriton JITに回すためCコンパイラを要求します。付属スクリプトは`TORCH_DISABLE_NATIVE_JIT=1`を設定して通常カーネルを使います。

## 学習

```bash
./train_modernbert.sh                       # 既定: 全タスク、2 epoch、lr 3e-5、128ペア/step
./train_modernbert.sh --tasks jnli jcommonsenseqa --epochs 1 --output models/mb-nli-only
./train_modernbert.sh --limit 400 --valid-limit 200 --epochs 1 --output /tmp/smoke  # 動作確認
```

学習は`ModernBertForSequenceClassification(num_labels=1)`で、1質問のK候補を`<s>質問+State</s><s>候補</s>`のKペアとしてエンコードし、K個のlogitに対するcross-entropy（listwise）で更新します。推論時のsoftmaxと同じ形です。State側のみ`max_length=512`で切り詰め、質問文が切れないように「質問→状況」の順で描画します。描画コードは学習・推論で`modernbert/prompting.py`を共有し、`FORMAT_VERSION`をチェックポイントに記録して不一致なら起動を拒否します。

出力は`models/modernbert-ja-310m-jev/`（約1.26 GB、Git対象外）。`jev_modernbert.json`に学習設定と検証結果を保存します。検証セット（valid）のマクロ平均正解率が最良のステップを保存します。

Radeon AI PRO R9700（bf16、128ペア/step）で2 epochが約41分、VRAM使用は約20 GBです。

学習中はvalidの正解率とNLLをタスクごとに表示し、最終的な検証結果を`jev_modernbert.json`に保存します。配布している重みでは検証3,000問のマクロ平均正解率が約90%です。

### 学習データ

すべて公開データで、リビジョンとSHA256を`modernbert/data.py`に固定しています。System Oneの形（State・質問・候補・説明）に変換し、testスプリットは学習・検証に使いません。

| 元データ | ライセンス | 変換後の質問 | 型 |
|---|---|---|---|
| [JGLUE](https://github.com/yahoojapan/JGLUE) JNLI | CC BY-SA 4.0 | 前提と仮説の関係（3択）、および「仮説も必ず正しいか？」（真偽） | choice / noul |
| JGLUE JCommonsenseQA | CC BY-SA 4.0 | 常識質問（5択） | choice |
| JGLUE JSTS | CC BY-SA 4.0 | 2文の意味の近さ（0〜5の6段階） | score |
| [JCoLA](https://github.com/osekilab/JCoLA) in-domain | CC BY-SA 4.0 | 文法的に自然か（真偽） | noul |
| [JCommonsenseMorality](https://github.com/Language-Media-Lab/commonsense-moral-ja) | MIT | 道徳的に問題があるか（真偽） | noul |
| [MASSIVE 1.1](https://github.com/alexa/massive) ja-JP | CC BY 4.0 | 発話の分野（18分野、および2〜6択のランダム部分集合） | choice |

学習94,384問（約345,000ペア）、検証12,811問（学習中の評価は先颂3,000問）。JCommonsenseQAとMASSIVEには「答えは〇〇か？」形式の真偽問も生成し、全体の約41%が真偽（true率約46%）です。一部の問題では候補の説明文をランダムに落とし、説明なしの候補にも対応させています。JNLIの説明文は[JGLUE評価](JGLUE.md)で使うものと同一です。

**顧客対応など、README例のような業務ドメインのデータは含みません。** そのドメインでの正解率は測定していません。

## 起動

```bash
./run_modernbert.sh                     # http://127.0.0.1:8080
./run_modernbert.sh --port 8081 --device cuda:0 --dtype bfloat16
./run_modernbert.sh --checkpoint models/mb-nli-only              # ローカルの別チェックポイント
./run_modernbert.sh --checkpoint argos1111/modernbert-ja-310m-jev@main   # Hub id（@でリビジョン指定可）
```

`--checkpoint`省略時は、`models/modernbert-ja-310m-jev/`があればそれを、なければHubの[argos1111/modernbert-ja-310m-jev](https://huggingface.co/argos1111/modernbert-ja-310m-jev)を使います（初回に約1.3 GBを`.cache/hf`へ取得）。Hub側にも`jev_modernbert.json`が含まれ、`format_version`の照合は同じです。

llama.cppは起動しません。既存の`./run.sh`（LFM）と同時に動かす場合はポートを分けてください。クライアントは同じです。

```bash
python3 systemone_client.py --url http://127.0.0.1:8080
python3 -m tools.verify_api --url http://127.0.0.1:8080
python3 -m tools.benchmark_jglue --url http://127.0.0.1:8080 --output results/jglue-test-modernbert
```

応答ヘッダー`X-Jev-Local-State-Cache`は`encoder-batch`固定、`X-Jev-Local-Processed-Tokens`はバッチ全体の入力トークン数です。`images`を含むリクエストは422を返します。

未学習のベースモデルで起動する`--allow-untrained`は、ヘッドが乱数初期化のため出力に意味がありません。配線確認用です。

## 公開済みの重み

学習済みチェックポイントは[Hugging Face Hub](https://huggingface.co/argos1111/modernbert-ja-310m-jev)に**CC BY-SA 4.0**で公開しています。ベースモデル（MIT）と学習データ（CC BY-SA 4.0 / CC BY 4.0 / MIT）の条件を継承した保守的な選択です。モデルカード（[`modernbert/MODEL_CARD.md`](../modernbert/MODEL_CARD.md)）に学習設定・データ出典・評価値・transformersからの直接利用例を記載しています。

自分の学習結果を公開するには、writeトークンを`HF_TOKEN`に設定して次を実行します。トークンは保存しません。

```bash
HF_TOKEN=hf_xxx .venv-modernbert/bin/python scripts/publish_modernbert.py --repo <user>/<name> --checkpoint models/<dir>
```

## 評価

JGLUE test（学習・検証に未使用）を、LFMと同じ[`tools/benchmark_jglue.py`](JGLUE.md)で同じAPI経由・同じプロンプト文言で測れます。ModernBERTはJGLUE trainで学習しているため、`--method`にその旨を記録してください。

```bash
python3 -m tools.benchmark_jglue --url http://127.0.0.1:8080 --output results/jglue-test-modernbert \
  --method "fine-tuned on JGLUE train (ModernBERT-Ja 310M cross-encoder)"
```

配布している重みの結果はJNLI 92.6% / JCommonsenseQA 92.4%（[READMEの比較表](../README.md)）で、LFMのzero-shotとは条件が異なります。公式モデルカードの値（JNLI 92.93 / JComQA 93.53）はタスク別に専用ヘッドを学習した値で、本リポジトリは全タスクを一つのペア採点器で学習しています。

学習に使っていないタスクでの汎化は、同じ入力・同じ採点でLFMと比較できます。

```bash
python3 -m tools.evaluate_heldout_tasks --url http://127.0.0.1:8080 --output results/heldout-modernbert
```

配布している重みでは、知識・語彙の広さが要るタスク（ニュース分類・JMMLU）はLFMより低く、短い日本語の意図判定（顧客対応の例）はLFMより高い結果でした。**学習したタスク族（NLI・意図分類・段階評価・常識QA）の近傍では汎化するが、汎用的な知識や未知の分類体系はLFMより弱い**という特性です。用途が固定的な業務判定なら、少量のドメインデータを`modernbert/data.py`に追加して再学習するのが最も効きます。

fine-tuneせずMasked LMの穴埋めでzero-shot判定する方式も検討しましたが、テンプレートによって正解率が大きく振れ、汎用APIとして安定しないため採用していません。

## 制約

- 指示文（`instructions`）と候補説明（`criteria`）はテキストとして読みますが、学習で見た質問の型から外れると精度は保証されません。新しい判定タスクには、同じ形式のデータを`modernbert/data.py`に追加して再学習する設計です。
- 1ペアあたり`max_length=512`トークン。長いStateは末尾を切り詰め、`diagnostics.truncated`で通知します（HTTPヘッダーには出ません）。
- CPU推論はfloat32で動作します。
- Python 3.14での動作はROCm 10.0 wheel（cp314）で確認済み。CUDA・CPU wheelは`pip install --dry-run`で2.13.0の依存解決が通ることまで確認していますが、実機では未実行です。
- 学習済みチェックポイント（1.26 GB）はGitリポジトリに含めず、Hugging Face Hubから取得します。`./train_modernbert.sh`で再学習した結果は、乱数シード固定でもGPU・ドライバーの差で完全一致はしません。
