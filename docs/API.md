# TypeSafe / Jev互換のローカルAPI

2026-09-19時点の[公式HTTP API](https://docs.typesafe.ai/api)、[Choice](https://docs.typesafe.ai/primitives/choice)、[Score](https://docs.typesafe.ai/primitives/score)、[Noul](https://docs.typesafe.ai/primitives/noul)、[公式Python SDK](https://docs.typesafe.ai/sdk/python/api/clients/sync)を参照した入出力互換アダプターです。**Jev本体ではなくLFM2.5を実行します。**

## 起動

[セットアップ](SETUP.md)完了後、`./run.sh`でまとめて起動できます。個別に起動する場合はプロジェクトディレクトリで次の順に実行します。

```bash
# ターミナル1：推論サーバー（起動済みなら不要）
./run_server.sh

# ターミナル2：互換API（標準ライブラリのみ）
./run_api_server.sh

# ターミナル3：公式形式の入力を送信し、stateと判定を表で表示
python3 systemone_client.py
```

推論サーバーは8097、互換APIは8080です。`run_api_server.sh --backend-url http://127.0.0.1:8097 --port 8080`で変更できます。`/health`はアダプターの生存確認で、モデルの稼働確認ではありません。

既定APIキーはローカル開発用の`local-dev`。変更する場合はサーバーとクライアントの両方に`JEV_API_KEY`を設定してください。公式TypeSafeの実APIキーは不要です。

## HTTP

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-dev' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

入力例は`examples/systemone.json`。`state`には文字列・object・array、`questions`には次の3種を混在させられます。

| type | 入力 | 応答 |
|---|---|---|
| choice | instructions、criteria: 選択肢名→説明 | choice、probabilities、confidence |
| score | instructions、criteria: 順序付き段階の配列 | score、legend、probabilities、confidence |
| noul | instructions、任意のcriteria: true/falseの説明 | noulのみ（typeを除く） |

トップレベルの応答は`model`, `answers`, `usage`だけです。追加のローカル診断情報で公式SDKの応答形を変更しません。`state`や質問文は応答には加えず、整形クライアントが元リクエストから表示します。

- `score = Σ(段階番号 × その確率)`。段階番号は0始まりです。
- Scoreの`legend`と`probabilities`のキーはHTTP上では文字列。Python SDKは整数キーへ変換します。
- Noulの`noul`はP(true)。boolや最大候補の確率ではありません。独立したconfidenceは付けません。
- Choiceは1〜255候補、Scoreは2〜10段階です。単一Choiceは推論せず唯一の候補を返します。
- Choiceのキーと説明を両方プロンプトに含めます。質問IDは推論には渡しません。
- instructions・criteriaの説明は文字列・object・arrayに対応し、Choiceの説明はnullも受け付けます。SDKに合わせinstructionsの省略/nullも許容し、Noulはinstructionsまたはcriteriaを必要とします。
- 選択肢が27個以上の場合は0〜254の数字ラベルに割り当て、単一トークンであることをバックエンドで確認します。

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-dev'
```

`jev-latest`と`jev-preview`はこのローカルサービス内での別名として受け付けます。実際の返却モデル名は起動時にバックエンドのGGUF名から取得します。例: `lfm2.5-1.2b-instruct-q8_0`。公式のモデルバージョン名を実行したようには表示しません。モデル一覧のrelease_dateはローカルアダプターの提供日です。

## 公式SDK

`typesafe-sdk==0.7.0`の同期・非同期クライアントで実通信を確認しました。混在3種・構造化入力・27択・255択を検証しています。サーバー自身にはSDKのインストールは不要です。

```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul

with TypeSafeClient(
    api_key="local-dev",
    base_url="http://127.0.0.1:8080",  # /v1はSDKが付ける
) as client:
    result = client.system_one(
        state={"message": "二重請求です。返金してください。"},
        questions={
            "refund": Noul(instructions="返金を要求しているか？"),
            "department": Choice(
                instructions="担当部署は？",
                criteria={"billing": "請求・返金", "technical": "技術的な問題"},
            ),
            "urgency": Score(
                instructions="対応の緊急度は？",
                criteria=["通常", "早め", "即時"],
            ),
        },
    )
    print(result.answers["refund"].noul)
    print(result.answers["department"].choice)
    print(result.answers["urgency"].score)
```

SDKを試す場合は次のように別途インストールします。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install typesafe-sdk==0.7.0
.venv/bin/python -m tools.verify_api --sdk  # 起動済み互換APIが必要
```

## 表示・保存

```bash
python3 systemone_client.py --input examples/systemone.json
python3 systemone_client.py --input examples/systemone.json --output results/my_systemone.json
python3 systemone_client.py --format json
```

保存・JSON出力は公式形式のままです。Scoreの返答は期待値なので、その値自体への「確率」は表示しません。Noulの確信度列は「仕様なし」と表示します。従来の`jev_local.py`は直接llama-serverを呼ぶ実験用CLIとして残しています。

## エラーとローカル制限

- 401: Bearerキー不一致／欠落。
- 422: 不正なJSON・質問・モデル名。通常の入力検証は`detail`配列に対象フィールドを含みます。
- 529 + Retry-After: 同時処理中のAPIリクエストが8件に達した場合。
- 502 / 504: 推論バックエンドの障害／タイムアウト。
- 413: 2MiB超のリクエスト。HTTP chunked uploadは非対応です。

エラーJSONの全フィールドまで公式サービスと同一であるとは保証しません。公式側の429課金レート制限や64k/32kコンテキスト契約も再現していません。ローカルの既定コンテキストは4 slotsに合計8192（各2048）です。多い候補や長い入力には、推論サーバーを例えば`CTX_SIZE=32768 ./run_server.sh`で起動してください。切り捨てられた入力から正常な回答を返さずエラーにします。

## 意味上の非互換

- **モデル品質・校正・内部アーキテクチャはJevと異なります。** 複数質問は個別の入力として処理するため、質問を増やすと処理量が増えます。
- **confidenceの厳密な数値互換は保証しません。** [公式Confidenceページ](https://docs.typesafe.ai/confidence)では分布から計算すると説明されていますが、確認した資料には計算式の定義がありません。本アダプターは従来の`1 − H(p)/ln(K)`を使います。`X-Jev-Local-Confidence`ヘッダーにも方式を記載します。
- `usage`はllama-serverが報告した入力・生成トークンの合計です。再利用済みprefixを含む各質問の論理的な入力長、共通prefixの準備、候補確率の再取得も含み、公式の課金カウントとは異なります。実際の評価量は`X-Jev-Local-Processed-Tokens`で確認できます。
- 構造化したJSONは文字列化してLFMに渡します。公式モデルの構造化入力エンコードを再現するものではありません。

参考資料は冒頭の公式ドキュメントを参照してください。文書の説明とSDK型定義で差がある箇所（省略可能instructions、構造化criteria/legend）はSDK型も参照して実装しました。

### 12問のデモ

`systemone_client.py`の既定入力`examples/systemone.json`は、Choice・Score・Noulを各4問、計12問含みます。返金・担当部署・緊急度に加え、解約予告、解約済みか、返金後の継続、口調、返金範囲、返答方法、不満、解約リスク、調査に必要な情報の充足を尋ねます。返答方法などは「記載なし」の候補も含めています。

`python3 systemone_client.py`でそのまま実行できます。`--output results/systemone.json`で実行結果を保存できます。

## 共通Stateの再利用

既定の`--state-cache auto`では、2問以上の共通prefixが256トークン以上なら一度だけ評価して全質問へ復元します。短い入力は従来どおり処理します。JSONの入出力形は変更していません。既存環境では推論サーバーとAPIサーバーの両方を再起動してください。[設定・計測・制約](STATE_CACHE.md)を参照してください。
