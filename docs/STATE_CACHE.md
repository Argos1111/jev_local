# 共通Stateの再利用

公式互換APIの同一リクエスト内で、各質問に共通する入力prefixを一度だけ計算します。各質問の完全なチャット入力を先にトークン化し、実際に一致するトークン列を共有するため、文字列連結によるトークン境界の変化はありません。State以外の共通システム指示も対象です。

LFM2.5は再帰状態を持つため、`cache_prompt=true`だけに依存せず、llama-serverのslot保存・復元でattentionと再帰状態の両方を引き継ぎます。質問ごとに独立したslotを借り、prefixを復元してから質問固有の後半だけ評価します。候補logprobの再取得時にも必ず復元します。各質問の`cache_n`を検査し、再利用されなければエラーにします。

画像を含むリクエストはテキストtoken列の保存・復元を行わず、各質問でdecoderの画像入力を評価します。`--state-cache shared`指定時も同様で、診断ヘッダーは`off-images`です。[画像エンコードキャッシュ](IMAGE_CACHE.md)を有効にした実験ビルドではencoderの出力だけを別途再利用できます。画像推論も同じslot貸し出しを使うため、テキストリクエストの復元処理とslotを取り合いません。以下の速度測定は旧テキストモデルでの結果です。

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

## 実測

Radeon AI PRO R9700、ROCm、LFM2.5-1.2B-Instruct Q8_0、4 slots、同じ12問。各条件5回の中央値、モデルロードを除外し、各リクエストのtokenize・prefix準備・保存復元を含みます。中・長は同じ顧客情報の履歴を追加した合成入力です。

| 共通prefix | 無効 | 強制共有 | 既定auto |
|---|---:|---:|---:|
| 32 tokens | 114.9 ms | 162.5 ms | 125.0 ms |
| 365 tokens | 241.8 ms | 183.6 ms | 181.0 ms |
| 1025 tokens | 552.7 ms | 231.7 ms | 230.7 ms |

長い例はautoで約2.40倍、中程度は約1.34倍高速です。短い例ではautoも共有判定用tokenizeのコストが残ります。長い例の実評価トークン数は13,105から1,830へ減りました。すべての入力・負荷での性能保証ではありません。

**計算の区切りが変わるため、非共有時と確率・確信度が完全一致する保証はありません。** 実測でもNoulが0.508から0.479へ動いた例があり、0.5付近の判断は反転し得ます。プロンプト・候補・確率の計算式は変えていません。

再測定は、他のAPIが接続していない専用バックエンドに対して行います。

```bash
python3 -m tools.benchmark_state_cache --backend-url http://127.0.0.1:8097 --rounds 5
```

応答・診断・全測定値は`results/state_cache_benchmark.json`へ保存します。従来の`jev_local.py`直接実行には今回の共有機能は適用していません。
