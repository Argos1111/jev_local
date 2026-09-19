# ModernBERTバックエンド

LFM（llama.cpp）の代わりに、[sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)（MIT License）を**cross-encoder**としてfine-tuneし、同じ`/v1/systemone` APIを提供する構成です。テキスト・JSONのStateのみ対応し、画像入力はありません。

## LFM方式との違い

| | LFM 1.2B（既定） | ModernBERT-Ja 310M |
|---|---|---|
| 判定方法 | 回答先頭トークン（A/B/C…）のlogprob | (質問＋State, 候補) ペアごとに1つのlogit |
| 学習 | zero-shot、プロンプトのみ | 公開日本語データでfine-tune必須 |
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

CPU推論の目安（Core Ultra 7 270K Plus、24スレッド、float32）: README 12問サンプル（33ペア）約1.9秒、noul 1問約110 ms。GPUの約90倍遅いですが、少数の質問なら実用範囲です。CPU・MPSではbf16を使わずfloat32で動作します。

学習には1.2 GBのモデルダウンロード（Hugging Face Hub、初回のみ）と、bf16で約20 GBのVRAMが必要です（`--pair-budget 64`なら約10 GB）。推論だけなら約2 GBで動きます。

AMDでは`/sys/class/kfd`からgfxターゲット（例: Radeon AI PRO R9700 = gfx1201、RX 7900 XTX = gfx1100）を検出し、対応するデバイスパッケージを入れます。`--gfx`で明示できます。AMDのwheelが対応するgfxは`https://stable.repo.amd.com/rocm/whl-next/`の`amd-torch-device-gfx*`一覧で確認できます（2026-09時点でgfx908/90a/942/950とgfx10xx/11xx/12xxを含みます）。

**異世代のAMD GPUを2枚以上搭載している場合**（検証環境: R9700 gfx1201 + RX 7900 XTX gfx1100）、2枚目以降を`--device cuda:1`のように指定するとROCmランタイムがSegmentation faultで落ちます（torch 2.13.0+rocm10.0.0で確認。デバイスパッケージを両方入れても同じ）。使いたいGPUを環境変数で単独に見せてしまうと動きます。

```bash
HIP_VISIBLE_DEVICES=1 ./run_modernbert.sh        # 2枚目だけを使う（プロセス内ではcuda:0になる）
```

RX 7900 XTX単独では12問サンプルが約23 msで、R9700と同等の速度です。

AMD版torchは一部演算をTriton JITに回すためCコンパイラを要求します。付属スクリプトは`TORCH_DISABLE_NATIVE_JIT=1`を設定して通常カーネルを使います。

## 学習

```bash
./train_modernbert.sh                       # 既定: 全タスク、2 epoch、lr 3e-5、128ペア/step
./train_modernbert.sh --tasks jnli jcommonsenseqa --epochs 1 --output models/mb-nli-only
./train_modernbert.sh --limit 400 --valid-limit 200 --epochs 1 --output /tmp/smoke  # 動作確認
```

学習は`ModernBertForSequenceClassification(num_labels=1)`で、1質問のK候補を`<s>質問+State</s><s>候補</s>`のKペアとしてエンコードし、K個のlogitに対するcross-entropy（listwise）で更新します。推論時のsoftmaxと同じ形です。State側のみ`max_length=512`で切り詰め、質問文が切れないように「質問→状況」の順で描画します。描画コードは学習・推論で`modernbert/prompting.py`を共有し、`FORMAT_VERSION`をチェックポイントに記録して不一致なら起動を拒否します。

出力は`models/modernbert-ja-310m-jev/`（約1.26 GB、Git対象外）。`jev_modernbert.json`に学習設定と検証結果を保存します。検証セット（valid）のマクロ平均正解率が最良のステップを保存します。

Radeon AI PRO R9700（bf16 autocast、128ペア/step）で約0.45 s/step、2 epoch（5,386 step）で約41分でした。VRAM使用は約20 GBです。256ペア/stepは32 GBでOOMになりました。

検証3,000問での最終ステップの正解率（トレーナー内部の集計）：

| タスク | n | 正解率 | NLL |
|---|---:|---:|---:|
| jnli（3択） | 546 | 94.3% | 0.22 |
| jnli-noul | 292 | 95.2% | 0.13 |
| jcommonsenseqa（5択） | 271 | 95.9% | 0.30 |
| jcommonsenseqa-noul | 106 | 92.5% | 0.25 |
| jsts（6段階を完全一致で採点） | 338 | 60.7% | 0.90 |
| jcola | 212 | 89.6% | 0.28 |
| moral | 462 | 87.7% | 0.31 |
| massive-scenario（18択） | 101 | 86.1% | 0.53 |
| massive-scenario-subset（2、6択） | 468 | 98.5% | 0.06 |
| massive-noul | 204 | 98.0% | 0.11 |
| マクロ平均 | 3,000 | 89.9% | 0.29 |

JSTSは連続値ど0.5単位の丸めを使っているため完全一致は厳しく、Scoreの期待値での評価はしていません。

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
./run_modernbert.sh                     # http://127.0.0.1:8080、models/modernbert-ja-310m-jev
./run_modernbert.sh --port 8081 --device cuda:0 --dtype bfloat16
./run_modernbert.sh --checkpoint models/mb-nli-only
```

llama.cppは起動しません。既存の`./run.sh`（LFM）と同時に動かす場合はポートを分けてください。クライアントは同じです。

```bash
python3 systemone_client.py --url http://127.0.0.1:8080
python3 -m tools.verify_api --url http://127.0.0.1:8080
python3 -m tools.benchmark_jglue --url http://127.0.0.1:8080 --output results/jglue-test-modernbert
```

応答ヘッダー`X-Jev-Local-State-Cache`は`encoder-batch`固定、`X-Jev-Local-Processed-Tokens`はバッチ全体の入力トークン数です。`images`を含むリクエストは422を返します。

未学習のベースモデルで起動する`--allow-untrained`は、ヘッドが乱数初期化のため出力に意味がありません。配線確認用です。

## 評価

JGLUE test（学習・検証に未使用）を、LFM 1.2Bと同じ[`tools/benchmark_jglue.py`](JGLUE.md)で同じAPI経由・同じプロンプト文言・同じ4並列で測りました。

```bash
python3 -m tools.benchmark_jglue --url http://127.0.0.1:8080 --output results/jglue-test-modernbert \
  --method "fine-tuned on JGLUE train (ModernBERT-Ja 310M cross-encoder)"
```

| バックエンド | 条件 | JNLI | JCommonsenseQA | p50応答（4並列） |
|---|---|---:|---:|---:|
| LFM2.5-1.2B Instruct Q8_0 / llama.cpp ROCm | zero-shot | 17.15% | 68.87% | 56 / 49 ms |
| ModernBERT-Ja 310M cross-encoder / torch ROCm bf16 | JGLUE trainで学習 | **92.62%** | **92.40%** | 33 / 32 ms |

いずれもRadeon AI PRO R9700です。ModernBERTはJNLI/JCommonsenseQAの**trainスプリットで学習しているため**、LFMのzero-shotとは条件が異なります。「同じAPI形式・同じハードウェアでどの程度の精度・速度になるか」の比較であり、モデルの能力を同条件で比べるものではありません。公式モデルカードの値（JNLI 92.93 / JComQA 93.53）はタスク別に専用ヘッドを学習した値で、本リポジトリは全タスクを一つのペア採点器で学習しています。

その他の確認（すべてR9700、HTTP往復含む）：

- READMEの12問サンプル（33ペアを1バッチ）：中央値約22 ms。1問（noul）は約8 ms。
- `tools/evaluate.py`の手作り8例（日英の返金要求判定）：8/8。学習データにこのドメインは含まれていませんが、8例では一般化の証拠にはなりません。
- 候補順を反転しても確率は完全に一致します（ペアごとに独立に採点するため）。

結果は[`results/jglue-test-modernbert/`](../results)に保存しています（Git対象外）。

### 学習に使っていないタスクでの汎化

JGLUEの数値は学習分布内の性能です。学習に使っていないタスクを同じAPI経由で測るツールを用意しました。LFM（zero-shot）とModernBERT（fine-tuned）を**同じ入力・同じ採点**で比較します。

```bash
python3 -m tools.evaluate_heldout_tasks --url http://127.0.0.1:8080 --output results/heldout-modernbert
```

| タスク | 内容 | n | 多数派 | LFM 1.2B zero-shot | ModernBERT fine-tuned |
|---|---|---:|---:|---:|---:|
| livedoor | ニュース記事の9カテゴリ分類（説明付き候補）。学習にニュース分類なし | 500 | 14.2% | **46.8%** | 35.6% |
| JMMLU | 4択の知識問題（解剖学・天文・日本史など10科目）。学習にない科目 | 500 | 28.8% | **43.6%** | 34.8% |
| synthetic | 手作りの顧客対応16例（README例と同種のnoul/choice/score）。学習に顧客対応ドメインなし | 16 | 25.0% | 56.2% | **93.8%** |

Radeon AI PRO R9700、seed 0、livedoor/JMMLUはランダム抽出500件（[CC BY-ND 2.1 JP](https://www.rondhuit.com/download.html) / [CC BY-SA 4.0](https://github.com/nlp-waseda/JMMLU)）。synthetic 16例は本リポジトリで作成した指標的なもので、統計的な結論には足りません。

読み方：

- **知識・語彙の広さが要るタスク（livedoor, JMMLU）ではLFMの方が高く**、ModernBERTは多数派より上・チャンス（11% / 25%）より上ですが、学習分布から外れると精度が落ちます。livedoorでは記事の約4割を「トピックニュース」「独女通信」に寄せており、未知のカテゴリ説明を読んで分類する力が弱いことが分かります。JMMLUの正解確率は正答時0.77・誤答時0.66で、間違いにも自信を持ちます。
- **短い日本語の意図判定（synthetic）ではModernBERTの方が高く**、LFMは「返金を要求しているか」のような真偽問で肯定側へ強く寄る傾向（7件中5件がTrue誤答、P=0.89〜1.00）を示しました。ModernBERTのこの結果は、JNLI（含意判定）・MASSIVE（発話の意図分類）・JCommonsenseMoralityで学んだ「文の意図を読む」能力が、顧客対応という未学習ドメインにある程度転移していると解釈できます。ただし「パスワードリセットのメール→billing」のような、ドメイン知識が要る誤りは残ります。
- 総合すると、ModernBERT版は**学習したタスク族（NLI・意図分類・段階評価・常識QA）の近傍では汎化するが、汎用的な知識や未知の分類体系はLFMより弱い**、という特性です。用途が固定的な業務判定なら少量のドメインデータを追加学習する（`modernbert/data.py`に追加）のが最も効きます。

## 検討した別案：MLMの穴埋めによるzero-shot

fine-tuneせず、事前学習済みのMasked LMヘッドで「答えの数字は`<mask>`だ」を埋めさせる方式も試しました（`sbintuitions/modernbert-ja-310m`そのまま、JGLUE test先頭1,000件、候補トークンだけでsoftmax）。

| タスク | テンプレート | 正解率 | 予測の偏り |
|---|---|---:|---|
| JNLI（多数派 neutral 55.6%） | 選択肢「1：含意」「2：矛盾」「3：中立」、答えの数字は`<mask>` | 25.1% | 1に71%、2は0件 |
| JNLI | 全角①②③ | 55.6% | 全件③（多数派と一致しただけ） |
| JNLI | 「したがって、『仮説』は`<mask>`」→ 正しい/誤り/不明 | 34.9% | 不明が0件 |
| JCommonsenseQA | 選択肢列挙 → 答えの数字は`<mask>` | 36.3% | 5に72% |
| JCommonsenseQA | 「選択肢：1=…、5=…\n質問『…』の正解は`<mask>`番です」 | 79.5% | 均等 |
| JCommonsenseQA（5候補すべて1トークンの300件） | 「質問 答えは`<mask>`です」に候補語を直接入れる | 81.3% | — |

観察：

- 数字ラベルで「選択肢問題を解く」というタスク自体をMLMは学習していないため、数字の出現頻度・直前の文脈への引きずられ方でテンプレートごとに結果が大きく振れます（JCQAで36%⇄80%）。正解率の高いテンプレートも、候補順を反転すると17%の設問で予測が変わりました。
- JNLIのような「関係の判定」は語彙の穴埋めに変換しづらく、どのテンプレートでも多数派ベースライン以下でした。
- 業務ドメイン例（返金要求8例）では「答え：`<mask>`」→はい/いいえで8/8でしたが、数字ラベル版は5/8でP(true)が0.41〜0.50に張り付き、判別になっていません。
- APIは任意の`instructions`と`criteria`を受けるため、テンプレートを設問ごとに手で調整する前提は取れません。

質問の型が固定された用途（例：候補が単語で、質問文をテンプレートに埋め込める）なら、学習なしで70〜80%程度は出る場面があります。汎用APIとして安定させるにはfine-tuneが必要、という判断で現在の方式にしています。数値はテンプレート探索を含む1,000件のプローブで、モデルの能力の一般的な推定ではありません。

## 制約

- 指示文（`instructions`）と候補説明（`criteria`）はテキストとして読みますが、学習で見た質問の型から外れると精度は保証されません。新しい判定タスクには、同じ形式のデータを`modernbert/data.py`に追加して再学習する設計です。
- 1ペアあたり`max_length=512`トークン。長いStateは末尾を切り詰め、`diagnostics.truncated`で通知します（HTTPヘッダーには出ません）。
- CPU推論はfloat32で動作しますが速度は未測定です。
- Python 3.14での動作はROCm 10.0 wheel（cp314）で確認済み。CUDA・CPU wheelは`pip install --dry-run`で2.13.0の依存解決が通ることまで確認していますが、実機では未実行です。
- 学習済みチェックポイント（1.26 GB）はリポジトリに含めません。利用者は`./train_modernbert.sh`で再学習します（データは自動取得、R9700で約41分）。学習結果は乱数シード固定でも、GPU・ドライバーの差で完全一致はしません。
