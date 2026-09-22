# 修正版Sarashina mmprojをHugging Faceへアップロードする

**約893 MBのprojectorだけを専用のモデルリポジトリへアップロード**します。言語GGUF・llama.cppバイナリ・ログ・参照shard・トークンは含めません。Git LFSの操作は不要で、`hf upload`が大きいファイルの転送を処理します。

このページの準備スクリプトはローカル処理のみです。ログイン・アップロード・Publicへの変更は利用者が明示的に実行します。2026-09-21にCLI `huggingface_hub==1.32.0`のヘルプを確認しました。

**公式checkpoint由来のmmprojをPublic公開済み**です。公開後、認証・既存HFキャッシュを使わずに全ファイルを再取得し、約893 MBのGGUF全体を含めてSHA256一致を確認しました。利用だけならHFのREADMEもしくは[対応ランタイムとモデルの取得手順](SARASHINA_RELEASE.md#利用方法)を参照してください。

- 配布先: [argos1111/sarashina2.2-vision-3b-mmproj-jev-f16](https://huggingface.co/argos1111/sarashina2.2-vision-3b-mmproj-jev-f16)
- 現在のrevision: `09ce275e3473c2127c5d1211a1a644fd93e5b03b`（公開時は`aabea115c03f21dbd5b0018c24770f632cb8c93d`）
- 公開状態: **Public、ゲートなし**。利用者の明示的な承認を受け、このリポジトリだけPrivateから切り替えました。
- 配布ファイルは`mmproj-jev-official-f16.gguf`（SHA256 `7c170758...`）と同名`.gguf.json`。重み・変換manifest・LICENSEは初回の公式由来版revision `0261fe2a2e9974fe533710d075196eade265d4a8`から変更していません。
- READMEは利用者向けの手順（clone・モデル取得・ランタイム取得・起動・画像での質問）に書き直し、NOTICEのソースを`5edbee00dcdd01690cbbd9cd1dd4d8f300862cd0`へ固定しました。
- 旧クローン由来の`mmproj-jev-f16.gguf`とmanifestは利用者の指示でHFから削除しました（ローカルの`models/`には残しています）。履歴は保持しており、旧版のrevision `22f873e5e3f13d997e75fa5d8f41ed1128288b30`で確認できます。
- 公開前の全履歴を点検し、想定外ファイル・認証情報パターンがないことを確認しました。`.gitattributes`、ModernBERTなど他リポジトリ、公式モデルのアクセス設定は変更していません。

初回の送信・再取得記録は`.cache/hf-projector-publication-official/`、公開時の履歴点検・可視性変更・未認証ダウンロードの記録は`.cache/hf-projector-publication-public/`に保存しています。

## 1. 配布条件を確認してPrivateリポジトリを作る

[配布時のライセンス確認](SARASHINA_DISTRIBUTION.md)を確認し、[公式Sarashinaページ](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)へご自身でログインして、同意画面に追加条件がないか確認してください。ゲート設定をこちらで勝手に引き継いだり、解除したりはしません。不明な条件がある間はPublicへの切り替えを保留します。PrivateへのアップロードもHFへデータを送る操作で、利用条件の代わりにはなりません。

[HFのNew model](https://huggingface.co/new)で作成します。

- Owner: 自分のHFユーザー名。GitHubのユーザー名と同じとは限りません。
- Model nameの例: `sarashina2.2-vision-3b-mmproj-jev-f16`
- Visibility: **Private**を推奨。まずファイルと説明を確認し、後からPublicにします。
- Licenseを選択する場合: Sarashinaの主ライセンスのMIT。SigLIP部分のApache-2.0はNOTICEに記載し、両方のLICENSE全文を同梱します。

既存モデルのファイルを上書きしないよう、**専用の新しいリポジトリ**を使います。

## 2. アップロード専用トークンを発行する

[Access Tokens](https://huggingface.co/settings/tokens)でfine-grained tokenを作り、作成したリポジトリへの**Read / Write権限だけ**を与えます。ブラウザで先にリポジトリを作ることで、アカウント全体の作成・書き込み権限を渡さずに済みます。UIの権限名はHFの更新により変わる場合があります。

**トークンはチャット・ソースコード・README・Git remote URLへ貼らないでください。** また、シェル履歴へ残る`HF_TOKEN=hf_...`の直書きも避けます。HF CLIのログイン用入力欄にだけ貼り付けます。

## 3. HF CLIを隔離して用意し、ログインする

既存の`.venv-modernbert` / `.venv-sarashina`を更新しないでください。HF Hub 1.xと参照診断用Transformersの依存条件は異なります。以下はLinux / WSL2での例です。

```bash
# リポジトリのルートで実行。既に .venv-hf があれば作り直さない
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install 'huggingface_hub==1.32.0'

.venv-hf/bin/hf auth login --format human
.venv-hf/bin/hf auth whoami
```

ログイン方法の選択が出たら`Paste an access token`を選び、トークンを入力します。Git credentialへの登録を尋ねられたら今回は`No`で構いません。HF CLIはGitHubとは別の認証です。

- ログイン情報は通常`~/.cache/huggingface/`（`HF_HOME`指定時はその場所）に保存されます。リポジトリへ保存されませんが、暗号化保管ではないため共有PCやバックアップの取り扱いに注意します。
- `HF_TOKEN`環境変数が既にあると、保存したトークンより優先される場合があります。意図したアカウントか`whoami`で確認してください。トークンそのものを表示するコマンドは不要です。
- `venv`で`ensurepip is not available`になる場合、OSに対応する`python3-venv`パッケージが必要です。この調査マシンではOSを変更せず、既存pipの`--python`オプションで**インストール先を`.venv-hf`だけ**に指定して準備済みです。

## 4. アップロード用の8ファイルを準備する

既存の公式由来F16 mmprojが`models/`にある場合（[取得・変換手順](SARASHINA.md#実験版のビルドと起動)）:

```bash
python3 scripts/prepare_sarashina_upload.py
(cd .cache/hf-sarashina-mmproj-official && sha256sum -c SHA256SUMS)
```

出力は新しい`.cache/hf-sarashina-mmproj-official/`です。旧`.cache/hf-sarashina-mmproj/`とモデルは上書きしません。既存ディレクトリは上書きしないので、既に準備済みならそのまま検査し、作り直す場合は別の`--output`を指定します。約893 MBのコピー用空き容量が必要です。

```text
.cache/hf-sarashina-mmproj-official/
  sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf
  sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json
  sarashina-reference.json
  README.md
  NOTICE.md
  LICENSE.sarashina2.2-vision-3b
  LICENSE.siglip
  SHA256SUMS
```

スクリプトは、文書化したF16成果物のsize / SHA256・変換manifest・LICENSE原文を検査してから、指定のファイルだけをコピーします。元のmmprojを書き換えず、ダウンロード・トークン読み取り・アップロードもしません。F32診断版や将来の別版を、そのまま現在の説明文で公開する用途には使えません。

`README.md`は概要・必要ファイル・使い方に絞ったHFモデルカードです。検証の詳細はJev Localのドキュメント、帰属・利用条件は同梱のNOTICE / LICENSEに分けています。**GitHubに`native/sarashina-official-preprocess-b11042.patch`等の新版コードが公開されていることを確認し、そのcommitへリンクを固定してから公開**してください。旧`cf973239...`版だけでは新前処理を実行できません。README等を編集した場合は`SHA256SUMS`も更新します。

## 5. 専用フォルダだけアップロードする

`YOUR_HF_USERNAME`を実際のHFユーザー名に置き換えます。リポジトリ名も手順1で作ったものと一致させてください。

```bash
HF_REPO='YOUR_HF_USERNAME/sarashina2.2-vision-3b-mmproj-jev-f16'

.venv-hf/bin/hf upload "$HF_REPO" .cache/hf-sarashina-mmproj-official . \
  --repo-type model --private \
  --include 'sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf' \
  --include 'sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json' \
  --include 'sarashina-reference.json' \
  --include 'README.md' --include 'NOTICE.md' \
  --include 'LICENSE.sarashina2.2-vision-3b' --include 'LICENSE.siglip' \
  --include 'SHA256SUMS' \
  --commit-message 'Upload official-source Sarashina F16 projector and notices'
```

- **Jev Localのルートや`models/`全体をアップロードしないでください。** 専用フォルダ＋明示した8ファイルだけが対象です。
- `--private`は**新規作成時のみ**有効で、既存PublicリポジトリをPrivateに戻す指定ではありません。送信先と意図した可視性をブラウザで確認してください。非公開で事前確認したい場合は別のPrivateリポジトリを使います。
- `--include`を繰り返す書式は上記CLI版で確認しています。`--delete`は使わず、サーバー側の他ファイルを削除しません。既存リポジトリへ新版を送ると**README・NOTICE・source lock・SHA256SUMSは新版へ更新され、旧GGUFは残ります**。この変更を意図する場合だけ実行してください。完全に分けたい場合は別の専用Privateリポジトリを作り、`HF_REPO`を変更します。
- 通信が中断した場合は同じコマンドを再実行できます。403の場合はリポジトリ名、`whoami`、そのリポジトリへのトークンのWrite権限を確認します。トークンをログやチャットへ貼る必要はありません。

## 6. アップロード結果を確認してからPublicへ

HFの`Files and versions`で8ファイルとモデルカードの表示を確認します。より確実に転送結果を検査するには、別ディレクトリへダウンロードし直します（約893 MBの通信と追加空き容量が必要）。

```bash
.venv-hf/bin/hf download "$HF_REPO" --local-dir .cache/hf-sarashina-mmproj-official-check
(cd .cache/hf-sarashina-mmproj-official-check && sha256sum -c SHA256SUMS)
```

公式同意条件と内容の確認が済み、公開する意思がある場合だけ、HFリポジトリのSettingsでVisibilityをPublicへ変更します。**アップロードコマンドだけで既存Privateリポジトリを公開にはしません。** 公開後のURLとcommit hashを控えておけば、ダウンロード元をrevision固定で案内できます。

アップロード後も、利用者には別途対応ランタイムとSarashina言語GGUFが必要です。mmproj公開だけで一般の未修正llama.cppへ対応するわけではありません。

必要なら、専用トークンをHFの設定画面で無効化し、ローカルの認証情報も`hf auth logout --token-name <発行したトークン名>`で削除できます。ログアウトだけではHF側のトークン失効にはなりません。
