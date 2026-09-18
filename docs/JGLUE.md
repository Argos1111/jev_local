# JGLUE API評価

[Yahoo Japan / Kawahara LabのJGLUE](https://github.com/yahoojapan/JGLUE) v1.3のJNLIとJCommonsenseQAを、ローカル`/v1/systemone` API経由で評価します。公式リポジトリのリビジョン`6f071c09316baae89c3d083a90985b4b1cb9968c`に固定しています。データはCC BY-SA 4.0で公開されています。出典・ライセンスは[公式README](https://github.com/yahoojapan/JGLUE/tree/6f071c09316baae89c3d083a90985b4b1cb9968c)を参照してください。保存する設問・選択肢も同データに由来します。

## 実行

推論サーバーとAPIを起動した状態で実行します。追加Pythonパッケージは不要です。

```bash
python3 -m tools.benchmark_jglue
```

既定はtest全件（JNLI 2,508件、JCommonsenseQA 1,118件）、4並列、zero-shotです。データを`.cache/jglue`に取得し、結果を`results/jglue-test`へ保存します。既存結果の上書きは拒否するので、再実行では出力先を変えます。

```bash
python3 -m tools.benchmark_jglue --output results/jglue-test-rerun
python3 -m tools.benchmark_jglue --split valid --output results/jglue-valid
python3 -m tools.benchmark_jglue --limit 10 --output results/jglue-smoke-10
python3 -m tools.benchmark_jglue --tasks jnli --workers 1 --output results/jnli-serial
```

接続先は`--url http://127.0.0.1:8080`、モデル指定は`--model jev-latest`、APIキーは環境変数`JEV_API_KEY`（既定`local-dev`）で設定します。実際の応答モデル名も記録します。

## 評価方法

- JNLI：sentence1を前提、sentence2を仮説にしてStateへ渡し、entailment / contradiction / neutralの3択。日本語で定義を付け、前提から仮説への方向を固定します。
- JCommonsenseQA：質問をStateへ渡し、choice0〜choice4を元の順序で5択として渡します。
- 正解ラベルとデータセットIDは推論入力に含めません。few-shot例、学習、test結果を使うプロンプト調整は行っていません。
- 1データ＝1 APIリクエスト＝1 Choiceです。設問間でStateが異なるため、同一Stateのprefix共有を測る実験ではありません。4並列のHTTPリクエストでバックエンドの4 slotsを利用します。
- APIの予測choiceと正解ラベルの完全一致によるAccuracyを主指標にします。失敗リクエストは正解に数えず、分母には含め、別途エラー件数も出します。自動リトライはしません。
- 確率・確信度はAPIが返す値を保存します。確信度は正答率として校正された値ではありません。平均正解確率、NLL、混同行列も集計します。
- 固定の候補順を使用するため、候補位置バイアスの平均化はしていません。一般的な選択肢文の全文尤度を比較する評価とは異なり、現在のAPIの最初の回答ラベルトークンによる分類を測っています。

公式掲載のfine-tuning済みモデルや、異なるfew-shot・プロンプト・splitを使う評価結果と同一条件ではありません。ここでの数値はこのAPI・モデル・プロンプトの組み合わせの評価です。モデルの学習データに対するtest汚染の有無は検証していません。

## 保存結果

- `report.md`：正解率と速度の一覧。
- `summary.json`：設定、データ取得URLとSHA256、コードSHA256、集計値。
- `jnli.jsonl` / `jcommonsenseqa.jsonl`：全件の入力state・質問・候補、正解、予測、確率、確信度、生の応答JSON、診断ヘッダー、応答時間。行ごとに即時保存します。
- 必要に応じてバックエンドの設定を別ファイルに記録してください。

時間はモデルをロード済みとし、合成入力1件のウォームアップ後に測定します。HTTP往復・API処理・バックエンド待ちを含み、データダウンロードは除外します。p50/p95は4並列時の各リクエストの応答時間であり、単独リクエストのレイテンシとは異なります。全体時間にはクライアントでの結果保存も含みます。
