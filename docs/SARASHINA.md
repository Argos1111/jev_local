# Sarashinaバックエンド

[Sarashina2.2 Vision 3B](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)を、LFMと同じllama.cpp・同じ`/v1/systemone` APIで使います。日本語の常識QA・知識問題・ニュース分類ではLFMより高い正解率が出ており、画像入力にも対応しています。

| | 文章・JSONのみ | 画像も使う |
|---|---|---|
| 必要なもの | 言語GGUF | 言語GGUF＋mmproj＋対応ランタイム |
| 起動 | `./run.sh --model sarashina` | `python3 scripts/run_sarashina.py --build ...` |
| 対応環境 | LFMと同じ（Linux / macOS / WSL2） | Linux x86_64 / WSL2（CPU、AMD GPU、NVIDIA GPU） |
| 手順 | このページ | [Sarashinaで画像を使う](SARASHINA_RELEASE.md) |

## モデルの選択

| プロファイル | ファイル | サイズ | 用途 |
|---|---|---:|---|
| `sarashina` | `sarashina2.2-vision-3b.Q4_K_M.gguf` | 約2.07 GB | 標準。容量を抑えて試す |
| `sarashina-q8` | `sarashina2.2-vision-3b.Q8_0.gguf` | 約3.57 GB | 量子化の影響を確かめる |

どちらも[mradermacher/sarashina2.2-vision-3b-GGUF](https://huggingface.co/mradermacher/sarashina2.2-vision-3b-GGUF)から取得し、URL・revision・SHA256は[`scripts/runtime.json`](../scripts/runtime.json)に固定しています。Q8の方が常識QA・知識問題でわずかに高く、短い判定タスクでは少し速い一方、JNLI（含意判定）はQ4の方が高いなど、量子化の影響は一様ではありません。まずQ4で試し、必要に応じてQ8と比較してください。

## 文章・JSONで使う

```bash
# 既にLFMのランタイムがある場合
unset LFM_MODEL LFM_MMPROJ
./setup.sh --model sarashina --model-only
./run.sh --model sarashina

# 別ターミナル（クライアントは共通）
python3 systemone_client.py
```

新規環境では`./setup.sh --model sarashina`でランタイムも導入します。Q8は`./setup.sh --model sarashina-q8 --model-only`で取得し、サーバーを停止してから`./run.sh --model sarashina-q8`で起動します。

- `setup.sh`は選択したプロファイルを保存しますが、`run.sh --model ...`だけでは保存済みの選択を変えません。LFMへ戻すには`./run.sh --model text`または`--model vision`で起動します。
- `--model-only`は既存のCPU/GPUランタイム選択を変えません。GPUで動かす場合は[セットアップ](SETUP.md)を参照してください。
- `LFM_MODEL`・`LFM_MMPROJ`はSarashinaにも適用されるため、通常は両方をunsetします。
- 標準起動は4 slots・合計context 8192（各2048）です。長い入力には`CTX_SIZE=32768 ./run.sh --model sarashina`で増やします。

## 画像で使う

通常の`run.sh --model sarashina`は文章・JSON専用で、画像を送ると502を返します。画像には、修正版mmprojと対応ランタイム（配布済み）を使う専用の起動方法があります。手順は[Sarashinaで画像を使う](SARASHINA_RELEASE.md)を参照してください。

配布版が環境に合わない場合（別のGPU世代、AVX2非対応のCPUなど）は、[ソースからのビルド](SARASHINA_BUILD.md)ができます。

## 特性と制約

- **Choiceは26候補まで。** 27候補以上は422を返します。LFM / ModernBERTの255候補対応は変わりません。
- **含意判定（JNLI）は弱く**、多数派を常に選ぶベースラインを下回ります。「AならばBか」のような関係判定にはModernBERTを検討してください。
- **候補の順番で結果が変わることがあります。** 候補順を反転すると判定が変わる例を確認しています。重要な判定では候補順を入れ替えて確認してください。
- **確率は校正された値ではありません。** `confidence`が高くても正答の保証にはなりません。
- 応答時間はLFM（1.2B）の約2.5〜4倍です。GPUではFlash Attentionを使える[対応ランタイム](SARASHINA_RELEASE.md)の方が速くなります。
- 画像入力では、単色画像の色や、2枚以上の画像の順番に関する質問で誤答が残っています。文書画像の読み取りは元の解像度で送る方が安定します。

同じAPI・同じハードウェアでLFM・ModernBERTと比較した正解率と応答時間は[READMEの比較表](../README.md)を参照してください。

## 評価ツール

Sarashinaでも[JGLUE評価](JGLUE.md)・[評価ツール](EVALUATION.md)をそのまま使えます。APIを起動した状態で、モデルごとに別の出力ディレクトリを指定してください。

```bash
python3 -m tools.verify_api --url http://127.0.0.1:8080
python3 -m tools.benchmark_jglue --url http://127.0.0.1:8080 --output results/sarashina-q4-jglue
python3 -m tools.evaluate_heldout_tasks --url http://127.0.0.1:8080 --limit 500 --seed 0 --output results/sarashina-q4-heldout
```

画像入力の確認には、専用ランタイムを起動した状態で`python3 -m tools.verify_sarashina_vision --url http://127.0.0.1:8080`を実行します。既知の制約（単色・複数画像）も検査するため、現状は不合格を含む結果を返します。
