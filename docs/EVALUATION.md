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

- `python3 -m tools.evaluate_heldout_tasks --url ... --output results/heldout`: ModernBERTの学習に使っていないタスク（livedoorニュース分類・JMMLU・手作り顧客対応16例）をAPI経由で採点。LFMとModernBERTの汎化を同条件で比較します（[結果](MODERNBERT.md#学習に使っていないタスクでの汎化)）。
- [JGLUE評価](JGLUE.md): JNLI / JCommonsenseQAの固定splitをAPI経由で評価。`--url`を変えれば[ModernBERTバックエンド](MODERNBERT.md)にも同じ手順で使えます。
- [Sarashinaと全方式の比較](SARASHINA.md): LFM text / VL・Sarashina Q4 / Q8・ModernBERTを同じテキスト入力で測定。画像修正、数値照合、GPU FA160、画像エンコード再利用の同一ビルド比較も記録。
- [共通Stateの計測](STATE_CACHE.md): 専用バックエンドで保存・復元の効果を測定。
- `python3 -m tools.probe_prefix`: 過去の回答prefix探索用ツール。

過去の個別PC上の測定ログは配布対象に含めません。実験条件と新しい測定結果をセットで保存してください。

## 画像入力と速度測定

### モデルの画像入力を確認

```bash
python3 -m tools.verify_vision --url http://127.0.0.1:8080
```

赤・青のPNGをコードで生成して送り、画像に応じた回答と、テキストStateキャッシュとの同時実行を確認します。外部の画像ファイルは不要です。APIは`--state-cache auto`または`shared`で起動してください。

Sarashinaはこの単色テストを公式checkpoint＋AutoProcessorでも誤答しました。専用の`tools.verify_sarashina_vision`は図形・画像順・再送・テキストとの同時実行を検証し、既知の失敗も記録して非ゼロ終了します。前処理/embeddingの数値照合は`tools.verify_sarashina_embeddings`、公式参照生成との比較は`tools.verify_sarashina_reference`。後者は言語GGUFのtokenizerとの差も検出します。どちらの公式参照ツールもコードのレビューと`--allow-reviewed-code`が必要です。[Sarashinaの再検証手順](SARASHINA.md#再検証コマンド)を参照してください。

### 手元の画像で応答時間を測る

```bash
python3 -m tools.benchmark_vision \
  --url http://127.0.0.1:18080 \
  --input examples/vision_gss_4.json \
  --image sample_pics/20260911_image_resized.png \
  --rounds 5 --output results/my_vision_benchmark.json
```

起動済みAPIに対してウォームアップ1回＋指定回数を計測し、中央値・平均・最小・最大、全回答とヘッダーを保存します。`--url`は実際のAPIのポートに合わせてください。4問に限らず、別の入力JSONも指定できます。画像を使い回す測定なので、画像エンコードキャッシュを有効にしたサーバーでは再送時の速度を測ることになります。

### GSS資料の12問を採点

```bash
python3 -m tools.evaluate_vision_gss \
  --url http://127.0.0.1:18080 --backend-url http://127.0.0.1:18097 \
  --image sample_pics/20260911_image_resized.png

python3 -m tools.evaluate_vision_gss \
  --url http://127.0.0.1:18080 --backend-url http://127.0.0.1:18097 \
  --image sample_pics/20260911_image_resized.png \
  --reverse-choices --output results/vision_gss/reversed_choices.json
```

画像あり・なしを比較します。`--reverse-choices`はChoiceの順序だけを反転し、入力ファイルは変更しません。資料に対応するローカル画像が必要です。[測定記録と採点上の制約](VISION_EVALUATION.md)を参照してください。

### 画像エンコード再利用の有無を比較

[実験ビルドの準備](IMAGE_CACHE.md)後、以下を実行します。

```bash
python3 -m tools.benchmark_image_cache \
  --input examples/vision_gss_4.json \
  --image sample_pics/20260911_image_resized.png \
  --device ROCm0 --rounds 5
```

必要なROCm共有ライブラリがシステムにない場合は、`LD_LIBRARY_PATH`を設定してから実行します。同じ実験バイナリを無効→有効→有効→無効の順に起動し、初回と再送時を分けて記録します。この比較ツールはChoice質問用です。`--workers 1`で逐次実行、`--port`・`--backend-port`で使用ポート、`--output`で保存先を指定できます。初期値の19080/19097は測定用で、通常の8080/8097や手動起動例の18080/18097とは別です。

Sarashinaの修正版では`--model sarashina`（または`--model sarashina-q8`）、専用`--server`・`--mmproj`を明示します。FA単独の比較には`tools.benchmark_sarashina_fa`を使用してください。[測定範囲の違いとコマンド](SARASHINA.md#再検証コマンド)を参照してください。
