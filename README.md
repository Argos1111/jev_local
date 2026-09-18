# Jev Local

Jevの型付き判断APIを、ローカルの小型LLMで試すための非公式実装です。文章やJSONを渡すと、**選択肢・スコア・真偽の確率**を返します。推論には **LFM2.5-1.2B-Instruct Q8_0 + llama.cpp** を使います。

**Jev本体のモデル・学習・精度を再現するものではありません。** TypeSafeの`/v1/systemone`形式に合わせたローカルアダプターです。APIの互換範囲は[APIガイド](docs/API.md)を参照してください。

## クイックスタート

自動セットアップの対象は **Linux x86_64・arm64 / macOS（Apple Silicon・Intel）**。Python 3.12以上とBashが必要です。GPUを検出し、NVIDIAはCUDA版、AMDはROCm版、Apple SiliconはMetal版を選びます。GPUを利用できない場合は理由を表示してCPU版に切り替えます。LM Studio・APIキーの取得・追加Pythonパッケージは不要です。WindowsではWSL2のLinux環境を使ってください。モデルのダウンロードは約1.25 GBで、初回のみネット接続が必要です。

このリポジトリをクローンし、そのディレクトリで実行します。

```bash
git clone https://github.com/Argos1111/jev_local.git
cd jev_local
./setup.sh  # GPUを自動検出し、対応する推論サーバーとモデルを取得・SHA256検証
./run.sh    # 推論サーバーと互換APIを起動。Ctrl-Cで両方停止
```

別のターミナルを開き、同じディレクトリで実行します。

```bash
python3 systemone_client.py
```

顧客の問い合わせを入力に、返金要求・担当部署・緊急度など12問の結果が表で表示されます。推論結果はモデルの判断なので、回答や確率の一致を保証するデモではありません。

GPU版には対応ドライバーが必要で、ROCm版には対応するROCmランタイムも必要です。OS側へのインストールは自動化しません。CPU版を指定する場合は`./setup.sh --backend cpu`を使います。

起動に失敗した場合や、既存のモデルを利用する場合は[セットアップ詳細](docs/SETUP.md)を参照してください。

## 自分の入力で試す

[入力例](examples/systemone.json)の`state`と`questions`を書き換えて渡します。

```bash
python3 systemone_client.py --input examples/systemone.json
python3 systemone_client.py --input examples/systemone.json --format json
python3 systemone_client.py --output results/my_result.json
```

HTTPからも利用できます。

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-dev' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

| 質問のtype | 得られるもの |
|---|---|
| `choice` | 選んだ候補、候補ごとの確率、分布の集中度 |
| `score` | 段階番号の期待値、各段階の確率、分布の集中度 |
| `noul` | 真である確率（0〜1） |

確率は指定した候補の中で正規化した値です。`confidence`は分布の集中度で、正解率として校正された値ではありません。[仕組みと制約](docs/DESIGN.md)を参照してください。

## 構成

```text
api_server.py / systemone.py   HTTP APIと型付き判断
jev_local.py / state_cache.py  推論・共通入力の再利用
display.py / systemone_client.py  表示・サンプルクライアント
scripts/                      セットアップと起動管理、取得物の固定情報
examples/                     入力サンプル
tests/                        モデル不要のユニットテスト
tools/                        任意の評価・ベンチマーク
docs/                         設定、API仕様、仕組み、評価方法
```

モデル、ダウンロードキャッシュ、実行結果、仮想環境はGit管理対象外です。クローンには実行コードと入力例だけが含まれ、モデル等は`setup.sh`が取得します。

## 開発・検証

```bash
python3 -m unittest discover -s tests -v  # ダウンロード・推論不要
python3 -m tools.verify_api              # 起動済みAPIへの実通信テスト
```

- [API・公式SDK接続](docs/API.md)
- [共通Stateの再利用](docs/STATE_CACHE.md)
- [評価ツール](docs/EVALUATION.md)
- [JGLUE評価](docs/JGLUE.md)

依存する[llama.cpp](https://github.com/ggml-org/llama.cpp)と[LFM2.5モデル](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)は、それぞれの配布元の利用条件に従います。本プロジェクトにバイナリ・重みは同梱しません。
