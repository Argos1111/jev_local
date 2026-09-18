# 評価ツール

以下は任意の開発・評価用ツールです。プロジェクトルートで`python3 -m tools.<名前>`として実行してください。測定結果は`results/`に保存し、Gitには含めません。

## APIの実通信確認

`./run.sh`が起動済みの状態で実行します。モデルの正答率ではなく、認証・入力検証・3種の質問・確率の正規化を確認します。

```bash
python3 -m tools.verify_api
```

公式SDKとの接続も確認する場合のみ、追加依存をインストールします。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install typesafe-sdk==0.7.0
.venv/bin/python -m tools.verify_api --sdk
```

## 直接推論の速度・小規模診断

APIを停止し、`./run_server.sh`だけを起動してから実行します。

```bash
python3 jev_local.py benchmark --rounds 3 --output results/benchmark.json
python3 -m tools.evaluate --output results/sanity.json
python3 -m tools.evaluate --semantic-tokens --output results/semantic.json
python3 -m tools.evaluate --json-baseline --output results/with_json.json
```

速度比較の既定入力は4問（`examples/original.json`）です。`--input examples/decide.json`で7問に変更できます。モデルロードを除外し、JSON Schema制約付きの値のみの生成と比較します。prompt cacheは既定で無効です。logprob方式にはtemplate適用のHTTP通信時間が含まれ、JSON生成側はtemplate適用後から計時するため、完全に対称な計測ではありません。

精度診断は手作りの返金要求8例で、候補順の反転による変化も調べます。一般的なモデル性能を推定するベンチマークではありません。

## その他

- [JGLUE評価](JGLUE.md): JNLI / JCommonsenseQAの固定splitをAPI経由で評価。
- [共通Stateの計測](STATE_CACHE.md): 専用バックエンドで保存・復元の効果を測定。
- `python3 -m tools.probe_prefix`: 過去の回答prefix探索用ツール。

過去の個別PC上の測定ログは配布対象に含めません。実験条件と新しい測定結果をセットで保存してください。
