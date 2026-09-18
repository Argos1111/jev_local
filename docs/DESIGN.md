# 仕組みと再現範囲

入力のStateと各質問をモデルのチャットテンプレートで組み立て、回答の最初の位置における候補トークンのlogprobを取得します。選択肢はA/B/C等のラベルに対応付け、候補集合の中で正規化します。

```text
P(i | candidates) = exp(logprob_i) / sum_j exp(logprob_j)
```

`/tokenize`でラベルが単一トークンであることを確認します。`/completion`は`n_predict=1`、`post_sampling_probs=false`で呼び出します。候補が取得した上位トークンに含まれなければ取得数を増やし、それでも不足すればエラーにします。欠落を確率0として扱いません。

複数質問は独立した推論です。[共通prefixの保存・復元](STATE_CACHE.md)で入力評価の一部を共有しますが、1回の出力で全質問を判断するモデルではありません。

## 型付きの出力

APIは元の候補から応答を組み立てます。Choiceは最多候補、Scoreは段階番号の期待値、Noulは真の確率を返します。型を維持できても、判断が正しいとは限りません。

`confidence`は本実装で定義した分布の集中度です。

```text
H = -sum(p_i * ln(p_i))
confidence = 1 - H / ln(K)
```

Kは候補数です。均等分布で0、1候補に集中すると1となり、単一Choiceは1とします。正答確率としての校正や、公式Jevと同じ算出式であることは保証しません。候補集合外に大きな確率があっても集中度は高くなり得ます。

実験CLIでは全語彙中の候補の合計確率`candidate_mass`も返します。APIの応答形式と実験CLIの形式は異なります。

## 実験CLI

APIを経由せず、推論サーバーに直接接続できます。APIによるslot管理と競合させないため、APIを停止して`./run_server.sh`だけを起動してください。

```bash
python3 jev_local.py demo
python3 jev_local.py decide --input examples/decide.json --output results/decision.json
python3 jev_local.py view --input results/decision.json
```

既定のLFM2.5 Instructでは空のthinkブロックを追加しません。`--empty-think`は推論型モデルでの過去の実験向けのオプションです。別モデルに変更しても同じ精度やラベルのトークン化になる保証はありません。
