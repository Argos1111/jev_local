---
language:
- ja
license: cc-by-sa-4.0
base_model: sbintuitions/modernbert-ja-310m
pipeline_tag: text-classification
library_name: transformers
tags:
- modernbert
- cross-encoder
- japanese
- jev-local
datasets:
- yahoojapan/JGLUE
- osekilab/JCoLA
- Language-Media-Lab/commonsense-moral-ja
- AmazonScience/massive
---

# modernbert-ja-310m-jev

[sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m) を、
[Jev Local](https://github.com/Argos1111/jev_local) の ModernBERT バックエンド用に fine-tune した cross-encoder です。
「質問＋状況（State）」と「候補」のペアを 1 本の系列として読み、1 つのスコアを出します。
候補ごとのスコアを softmax すると Choice（選択肢）・Score（段階評価）・Noul（真偽）の確率になります。

**TypeSafe の Jev 本体とは無関係の非公式モデルです。** Jev の学習・精度を再現するものではありません。

## 使い方

Jev Local から使う場合（`/v1/systemone` 互換 API として起動）:

```bash
git clone https://github.com/Argos1111/jev_local.git && cd jev_local
./setup_modernbert.sh
./run_modernbert.sh --checkpoint argos1111/modernbert-ja-310m-jev
python3 systemone_client.py   # 別ターミナル
```

ローカルに学習済みモデルがなければ `./run_modernbert.sh` は既定でこのリポジトリを取得します。

transformers から直接使う場合:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "argos1111/modernbert-ja-310m-jev"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

state = "二重請求です。返金してください。解決しないなら解約します。"
question = "担当部署は？"
candidates = ["billing — 請求・返金", "technical — 技術的な障害", "sales"]

context = f"質問: {question}\n状況: {state}"      # 質問 → 状況 の順（学習時と同じ描画）
enc = tok([context] * len(candidates), candidates, padding=True,
          truncation="only_first", max_length=512, return_tensors="pt")
with torch.inference_mode():
    logits = model(**enc).logits[:, 0]
print(dict(zip(candidates, logits.softmax(0).tolist())))
# {'billing — 請求・返金': 0.9996, 'technical — 技術的な障害': 0.0, 'sales': 0.0004}
```

描画形式（`質問: …\n状況: …` / `ラベル — 説明`）は学習時と一致させる必要があります。
Jev Local 側は `modernbert/prompting.py` でこれを共有し、`jev_modernbert.json` の `format_version`（`modernbert-jev/1`）で照合します。
State が JSON の場合は `json.dumps(..., ensure_ascii=False)` した文字列を入れます。真偽（Noul）は候補 `["true", "false"]` として扱います。

## 学習

| 項目 | 値 |
|---|---|
| ベース | sbintuitions/modernbert-ja-310m @ `77675fc9` |
| ヘッド | `ModernBertForSequenceClassification(num_labels=1)`、CLS pooling |
| 損失 | 各質問の K 候補に対する listwise cross-entropy |
| 学習データ | 94,384 問（約 345,000 ペア）、下記公開データを System One 形式に変換 |
| 設定 | 2 epoch（5,386 step）、lr 3e-5 linear warmup 6% + decay、AdamW wd 0.01、128 ペア/step、bf16 autocast、seed 0 |
| 系列長 | 512（State 側のみ切り詰め） |
| 環境 | AMD Radeon AI PRO R9700、torch 2.13.0+rocm10.0.0、transformers 5.17.0、約 41 分 |

### 学習データ

test スプリットは学習・検証に使っていません。

| 元データ | ライセンス | 変換後の質問 |
|---|---|---|
| [JGLUE](https://github.com/yahoojapan/JGLUE) JNLI train | CC BY-SA 4.0 | 前提と仮説の関係（3 択）、「仮説も必ず正しいか？」（真偽） |
| JGLUE JCommonsenseQA train | CC BY-SA 4.0 | 常識質問（5 択）、「答えは○○か？」（真偽） |
| JGLUE JSTS train | CC BY-SA 4.0 | 2 文の意味の近さ（0〜5 の 6 段階） |
| [JCoLA](https://github.com/osekilab/JCoLA) in-domain train | CC BY-SA 4.0 | 文法的に自然か（真偽） |
| [JCommonsenseMorality](https://github.com/Language-Media-Lab/commonsense-moral-ja) train | MIT | 道徳的に問題があるか（真偽） |
| [MASSIVE 1.1](https://github.com/alexa/massive) ja-JP train | CC BY 4.0 | 発話の分野（18 分野、およびランダムな 2〜6 択）、「○○に関するものか？」（真偽） |

約 41% が真偽問（true 率約 46%）。一部の問題では候補の説明文をランダムに落としています。
変換コードは [`modernbert/data.py`](https://github.com/Argos1111/jev_local/blob/main/modernbert/data.py)。

## 評価

### JGLUE test（学習に未使用）

Jev Local の HTTP API 経由、1 問 1 リクエスト、4 並列、R9700。

| タスク | 正解率 | LFM2.5-1.2B zero-shot（同 API） |
|---|---:|---:|
| JNLI | 92.62% (2323/2508) | 17.15% |
| JCommonsenseQA | 92.40% (1033/1118) | 68.87% |

同じデータの train で学習しているため、zero-shot の LFM とは条件が異なります。
公式モデルカードの値（JNLI 92.93 / JComQA 93.53）はタスク別に専用ヘッドを学習した値で、本モデルは全タスクを 1 つの採点器で学習しています。

### 検証スプリット（学習中の評価、3,000 問）

| タスク | 正解率 | | タスク | 正解率 |
|---|---:|---|---|---:|
| jnli | 94.3% | | jcola | 89.6% |
| jnli-noul | 95.2% | | moral | 87.7% |
| jcommonsenseqa | 95.9% | | massive-scenario (18 択) | 86.1% |
| jcommonsenseqa-noul | 92.5% | | massive-scenario-subset | 98.5% |
| jsts (6 段階完全一致) | 60.7% | | massive-noul | 98.0% |

### 学習に使っていないタスク

| タスク | n | 多数派 | 本モデル | LFM2.5-1.2B zero-shot |
|---|---:|---:|---:|---:|
| livedoor ニュース 9 カテゴリ分類 | 500 | 14.2% | 35.6% | **46.8%** |
| JMMLU 4 択（10 科目） | 500 | 28.8% | 34.8% | **43.6%** |
| 手作りの顧客対応 16 例（noul / choice / score） | 16 | 25.0% | **93.8%** | 56.2% |

知識・未知の分類体系が要るタスクは LFM より弱く、短い日本語の意図・関係判定は学習分布外のドメインでも高めです。
16 例は指標的なもので統計的な結論には足りません。

## 制約

- 学習した質問の型（NLI・意図分類・段階評価・常識 QA・真偽）から外れると精度は保証されません。指示文と候補説明はテキストとして読みますが、LLM のような指示追従はしません。
- 業務ドメイン（顧客対応など）のデータは学習に含まれていません。
- 誤答時も高い確率を出すことがあります（JMMLU 誤答時の平均最大確率 0.66）。`confidence` は分布の集中度で、校正された正解率ではありません。
- 画像入力は非対応。日本語と英語以外は未評価です。
- 系列長 512 を超える State は末尾が切り詰められます。

## ライセンス

重みは **CC BY-SA 4.0** で公開します。ベースモデル（MIT）と学習データ（CC BY-SA 4.0 / CC BY 4.0 / MIT）の条件を継承した、保守的な選択です。
利用時は上記データセットの出典表記を保持してください。

## 引用

ベースモデル・各データセットの引用は、それぞれのリポジトリ／モデルカードを参照してください。
