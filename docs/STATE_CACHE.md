# 共通Stateの再利用

公式互換APIの同一リクエスト内で、各質問に共通する入力prefixを一度だけ計算します。各質問の完全なチャット入力を先にトークン化し、実際に一致するトークン列を共有するため、文字列連結によるトークン境界の変化はありません。State以外の共通システム指示も対象です。

LFM2.5は再帰状態を持つため、`cache_prompt=true`だけに依存せず、llama-serverのslot保存・復元でattentionと再帰状態の両方を引き継ぎます。質問ごとに独立したslotを借り、prefixを復元してから質問固有の後半だけ評価します。候補logprobの再取得時にも必ず復元します。各質問の`cache_n`を検査し、再利用されなければエラーにします。

画像を含むリクエストはテキストtoken列の保存・復元を行わず、各質問でdecoderの画像入力を評価します。`--state-cache shared`指定時も同様で、診断ヘッダーは`off-images`です。[画像エンコードキャッシュ](IMAGE_CACHE.md)を有効にした実験ビルドではencoderの出力だけを別途再利用できます。画像推論も同じslot貸し出しを使うため、テキストリクエストの復元処理とslotを取り合いません。

## 起動

既存の推論サーバーとAPIサーバーをそれぞれCtrl-Cで止め、別ターミナルで再起動します。

```bash
./run_server.sh
```

```bash
./run_api_server.sh
```

```bash
python3 systemone_client.py
```

既定は`--state-cache auto --cache-min-tokens 256`です。2問以上の共通prefixが256トークン以上なら共有し、短ければ保存・復元のコストを避けます。256は今回の測定による目安で、GPU・質問数に応じた自動学習はしません。短いStateでも試すには`./run_api_server.sh --state-cache shared`、無効化するには`--state-cache off`を指定します。

推論サーバーは`.cache/slots`を`--slot-save-path`として使います。変更する場合は、推論側の`SLOT_CACHE_DIR=/absolute/path`とAPI側の`--slot-cache-dir /absolute/path`を合わせてください。両プロセスから同じファイルが見える必要があります。保存機能のない既存バックエンドには`--state-cache off`を使うか、起動し直してください。

slot管理のため、推論サーバーはこのAPI専用にしてください。別クライアントから直接推論させる構成は対応しません。同一URL・保存ディレクトリのAPI重複起動はロックで防ぎますが、別URL表記・別ディレクトリや外部クライアントまでは検出できません。

snapshotはリクエストごとに一意のファイルに保存し、全質問終了後にエラー時も削除します。リクエストをまたぐ永続キャッシュではありません。強制終了時は残る可能性があります。保存先には入力由来のモデル状態が含まれるため、他ユーザーと共有しないディレクトリを使ってください。

## 確認

```bash
curl -sS -D - http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-dev' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

公式JSON形式は変更せず、HTTPヘッダーに診断値を返します。

| ヘッダー末尾（`X-Jev-Local-`に続く） | 内容 |
|---|---|
| `State-Cache` | `shared` / `off-short` / `off` |
| `Prefix-Tokens` | 共有prefixの長さ |
| `Cached-Tokens` | 各質問・再試行で再利用したトークン数の合計 |
| `Processed-Tokens` | 準備を含め実際に評価した入力トークン数 |

既定の12問デモはprefixが32トークンなので`off-short`になります。長いStateまたは`shared`指定で再利用を確認できます。準備ではバックエンドの都合で無視する1トークンを生成します。そのトークンは共有状態には入りませんが、`usage.output_tokens`には含めます。`usage.input_tokens`はバックエンドの論理入力長の合計であり、計算削減率そのものを表しません。

## 効果の目安

共通prefixが長いほど効果が大きく、既定の`auto`では約365トークンで1.3倍、約1,000トークンで2.4倍程度の応答時間短縮を確認しています（LFM 1.2B、GPU、12問）。短いprefixでは効果がないため、既定では256トークン未満は共有しません。

**計算の区切りが変わるため、非共有時と確率・確信度が完全に一致する保証はありません。** 0.5付近のNoulの判断が反転し得ます。プロンプト・候補・確率の計算式は変えていません。

自分の環境で測るには、他のAPIが接続していない専用バックエンドに対して次を実行します。結果は`results/state_cache_benchmark.json`に保存されます。

```bash
python3 -m tools.benchmark_state_cache --backend-url http://127.0.0.1:8097 --rounds 5
```
