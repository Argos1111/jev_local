# Sarashina成果物の配布時の表記・ライセンス確認

確認日: **2026-09-21**。対象は修正版F16 mmprojと、llama.cpp b11042に本リポジトリのパッチを適用したCPU / HIP / CUDAランタイムです。公開文書とローカルのソース・ビルドを確認した記録であり、新しい構成やモデルのPublic配布まで一律に監査済みとするものではありません。

## 1. 修正版mmproj

### Sarashina本体: MIT

[公式LICENSE（revision固定）](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b/blob/46d9cc3929a54f7d2b91ce5668d7a9c5833991ed/LICENSE)を認証なしで取得できました。著作権表示は次のとおりです。

```text
Copyright (c) 2025 SB Intuitions
```

MIT本文は改変・再配布・商用利用を許諾し、コピーまたは実質的部分に著作権表示と許諾文を含めることを求めています。配布物に**免責条項を含むLICENSE全文**を同梱してください。「MIT」と書くだけ、または外部リンクだけで済ませない構成にします。

原文を[`licenses/LICENSE.sarashina2.2-vision-3b`](../licenses/LICENSE.sarashina2.2-vision-3b)に保存しました。取得した原文のSHA256は`c2142adf07ff4607749b33314f969ffca01ce0bd629eee341323bdf5ca825b3f`です。調査初期はprocessor等がHTTP 401でしたが、**同日、利用者によるアクセス承認・HFログイン後に公式checkpoint / AutoProcessorの取得にも成功**しました。認証後に取得したLICENSEも同じhashです。

### 画像エンコーダーの由来: SigLIP / Apache-2.0

[公式Sarashinaモデルカード](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)は、画像エンコーダーの元モデルを[Google SigLIP SO400M patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)と明記しています。SigLIPのカードはApache-2.0で、元の[big_visionの説明](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/README.md#license)もモデルを含むApache-2.0の適用を明記しています。

そのため、画像エンコーダーを含むmmprojの配布は、**SarashinaのMIT表記だけで済ませず、SigLIP由来部分のApache-2.0条件も引き継ぐ構成**にします。

- Apache-2.0全文を同梱する。
- 変更したファイルに変更の事実を明示する。GGUFの説明メタデータに加え、README / NOTICEにも変換・変更内容を記載する。
- 適用される元の著作権・特許・商標・帰属表示を保持する。
- 元の配布物にNOTICEがある場合、その適用部分も保持する。今回確認したSigLIPのHFファイル一覧とbig_visionの上記revisionには、独立したNOTICEファイルは見つかりませんでした。将来別の配布元を使う場合は再確認する。
- Apache-2.0は、由来の説明などを除いて商標の利用許諾を与えるものではない。公式版・公式推奨品と誤認させない。

原文を[`licenses/LICENSE.siglip`](../licenses/LICENSE.siglip)に保存しました。big_visionの上記revisionのLICENSEとバイト単位で同一で、SHA256は`43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`です。Sarashina全体をこちらの判断でApache-2.0へ変更する趣旨ではなく、部品ごとの条件を保持します。

### 実際の取得元と変更内容の表示

以下は出自・変更内容・検証範囲を正しく伝えるための表示です。すべてがMIT本文上の独立した義務という意味ではありません。

| 項目 | 記載する内容 |
|---|---|
| 原モデル | SB Intuitions / `sbintuitions/sarashina2.2-vision-3b` |
| 実際の重み取得元 | **公式`sbintuitions/sarashina2.2-vision-3b`**。利用者の承認・HF認証後に取得 |
| 取得revision | `46d9cc3929a54f7d2b91ce5668d7a9c5833991ed` |
| 変換 | Jev Localの`convert_sarashina_mmproj.py`によるF16 GGUF化。追加学習なし |
| 変更 | 元の公開mmprojに欠けていたpost-merger LayerNormの2 tensor、FFN幅4304、独自projector種別`sarashina2vl`、公式出典・前処理版のmetadata |
| 実行条件 | 対応するSarashina言語GGUFと、`sarashina-official-preprocess-b11042.patch`を含む更新ランタイムが必要 |
| 検証の限界 | 公式前処理15例・embedding 5例を照合。単色・複数画像・候補順の失敗と、既存言語GGUFのtokenizer差は残る |
| 独立した配布 | SB Intuitions / Google / llama.cpp上流による公式リリース・承認ではない |

旧修正版は公開クローン`AnalyticPudding/sarashina2.2-vision-3b-clone`（`05710ee40ae41ff322da991298ab893cf54ce110`）由来でした。旧GGUFとそのmanifestはローカル・HFとも変更せず、新版を同じPrivateリポジトリへ追加しました。公式と旧クローンの全624 tensorの同一性は今回確認できましたが、**新版はクローンではなく公式の取得ファイルから再変換**し、公式の前処理を追加修正しています。過去の取得元を遡って公式由来と呼び替えることはしません。

`mradermacher`は変更していない言語Q4/Q8 GGUFの量子化配布元です。公式由来の新版mmprojとは区別します。言語GGUFも一緒に配布する場合は、その配布元・revision・変更の有無を追加します。

### 公式HFのアクセス条件は別に確認が必要

公式API / 公開ページで次を確認しました。

- `gated: auto`。アクセスには連絡先共有への同意を求める表示がある。
- カードの`extra_gated_eu_disallowed: true`。[HFの説明](https://huggingface.co/docs/hub/models-gated#restricting-access-for-eu-users)では、ゲート対象モデルのEUからのアクセスを制限する設定。
- 当初は未ログインで追加条件を確認できませんでした。その後**利用者自身が同意・承認を済ませたと伝え、保存済み認証で公式ファイルを取得できた**ことを確認しました。エージェントは同意画面を操作せず、その文面を独立に監査したわけではありません。

**HF上のアクセス設定と、MITの著作権許諾は区別します。** このフラグだけから「MIT本文にEUでの利用・再配布禁止がある」とは断定しません。一方で、ログイン後の同意画面や追加条件を未確認のまま「世界中へ無条件にミラー配布できる」とも断定しません。重みの公開前に配布担当者が公式の同意画面を確認し、MITとの関係が不明な条件があればSB Intuitionsへ確認してください。これはランタイムに新しい地域・GPU・検査結果の実行制限を追加する話ではありません。

### 配布物の構成案

```text
sarashina-projector/
  sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf
  sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf.json
  README.md                         # 出自・変更・必要ランタイム・既知の制約
  NOTICE.md                         # 部品ごとの帰属と変更記録
  LICENSE.sarashina2.2-vision-3b      # MIT全文
  LICENSE.siglip                     # Apache-2.0全文
  SHA256SUMS
```

LICENSEはGGUFと一緒に取得できる場所へ置き、アーカイブ配布ならアーカイブにも含めます。NOTICEはLICENSE全文の代わりにはなりません。既存の重み・hashを書き換える必要はなく、必要な文書を別添できます。

ローカルの配布フォルダを作る`prepare_sarashina_upload.py`と、Privateリポジトリへ送信して確認する[HFアップロード手順](SARASHINA_UPLOAD.md)を用意しました。準備スクリプト自体は認証・アップロードを行いません。

## 2. 修正版llama.cppバイナリ

### llama.cppのMITだけでは第三者コードすべてをカバーしない

b11042の著作権表示は`Copyright (c) 2023-2026 The ggml authors`です。原文は既に[`native/LICENSE.llama.cpp`](../native/LICENSE.llama.cpp)にあります。**配布するバイナリアーカイブにも同梱**します。

また、今回のserver / mtmd / CPUバックエンドには第三者コードが組み込まれています。独立した`.so`が存在しない静的リンク・ヘッダ実装も対象です。b11042のソースから確認した主な表記元は以下です。完成した配布物の網羅的な一覧ではありません。

| 部品 | 確認したライセンス / 原文の場所（llama.cpp内） |
|---|---|
| cpp-httplib | MIT、`vendor/cpp-httplib/LICENSE`（yhirose） |
| nlohmann/json | MIT、`licenses/LICENSE-jsonhpp`。`vendor/nlohmann/json.hpp`内のHedley・Grisu・UTF-8処理等の帰属表示も保持する |
| xxHash | BSD-2-Clause、`vendor/hash/xxhash/LICENSE`。バイナリ配布にも著作権・条件・免責の再掲が必要 |
| rotate-bits | MIT、`vendor/hash/rotate-bits/LICENSE.md`（William Casarin） |
| stb_image | MITまたはUnlicense、`vendor/stb/stb_image.h`末尾。MITを選ぶ場合はSean Barrettの著作権表示と全文を保持 |
| miniaudio | MIT-0またはUnlicense、`vendor/miniaudio/miniaudio.h`末尾。通常のMITと異なりMIT-0は帰属表示義務なし。選択した原文と出自の同梱を推奨 |
| SHA-1 / SHA-256 | Public Domain、`vendor/hash/sha1/sha1.c`冒頭 / `vendor/hash/sha256/LICENSE` |
| subprocess / base64 | Unlicense、`vendor/sheredom/subprocess.h` / `common/base64.hpp`冒頭 |
| CPU tinyBLAS / llamafile | MIT、`ggml/src/ggml-cpu/llamafile/sgemm.cpp`冒頭（Mozilla Foundation）。今回`GGML_LLAMAFILE=ON` |
| YaRN由来の実装 | MIT、`ggml/src/ggml-cpu/ops.cpp`と`ggml/src/ggml-cuda/rope.cu`にJeffrey Quesnelle / Bowen Pengの表示 |

これらを配布用の`THIRD_PARTY_NOTICES` / `licenses/`へ原文付きで集約します。単なる部品名の一覧では不十分です。ソース一式も配るなら、ビルドで未使用の部分も含めて、そのソース内のライセンス表示を削除しないでください。

今回はWeb UI・prebuilt UI・OpenSSL・OpenMP・LLGuidanceを無効にした構成です。有効化する場合は依存関係と表記を追加します。b11042のライセンス埋め込みは`llama-app`向けで、今回`LLAMA_BUILD_APP=OFF`です。serverに必要な全表記が自動同梱される前提にはしません。

### GPU・OSランタイムを同梱する場合

- **CUDA**: 実ビルドは`libcudart.so.13` / `libcublas.so.13` / NVIDIAドライバーの`libcuda.so.1`等へ動的リンクしています。CUDA Toolkitのファイルを配る場合はMITではなく[NVIDIA EULA](https://docs.nvidia.com/cuda/eula/index.html)の対象です。手元のCUDA Runtime 13.2.51のLICENSE（2026-01-26更新）では、§1.1.2に配布条件、§2.6 Attachment Aに`libcudart` / `libcublas` / `libcublasLt`等の再配布対象があります。対象・版・条件を確認し、該当EULA・第三者表記を同梱する構成にします。Toolkitやドライバー一式を無条件でコピーできるわけではありません。
- **HIP / ROCm**: `libamdhip64` / `libhipblas` / `librocblas`等と、その推移的依存・Tensileデータを配る場合は実際の各パッケージのLICENSE / NOTICESを保持します。手元のHIPはMITですが、hipBLAS / rocBLASのLICENSEにはMITに加えて大学・研究機関由来のBSD条件も含まれます。「ROCmは全部AMDのMIT」と一括しないでください。
- **C/C++ランタイム**: `libstdc++` / `libgcc_s` / glibc等まで同梱するなら、GPL＋Runtime Library Exception / LGPL等を各版・リンク方法に応じて別途確認します。
- **外部依存にする構成**: GPU・OSランタイムを利用者側で導入してもらえば、その共有ライブラリ自体の同梱は不要です。ただし、ビルド時にヘッダ・テンプレート・静的コードから取り込まれる部分の確認まで不要になるわけではありません。

実際に配るファイルを確定してから、直接依存だけでなく推移的依存と静的に取り込まれたコードも棚卸しします。今回の[Pre-release](SARASHINA_RELEASE.md)ではGPU/OS共有ライブラリは外部依存とし、CPU / HIP / CUDA別のアーカイブへb11042の第三者表記と[`licenses/runtime/`](../licenses/runtime/)を同梱しました。CUDA device runtime・HIP device libraries・ヘッダ由来部分も表記対象に含めています。これはGPU SDK一式を同梱する構成の監査ではありません。

## 3. 公開前に残る確認

1. **モデルの公式同意条件**: 利用者の同意・アクセス成功は確認済み。追加条件の文面や再配布への適用をエージェントが独立に確認済みとはしません。公開前の最終確認は配布担当者が行います。
2. **本プロジェクト独自部分のライセンス選択**: 現在、プロジェクト全体・独自追加コードの利用条件を定めるルートLICENSEは未設定です。上流LICENSEを同梱しても、自動的に独自コード全体へ同じ条件が適用されるわけではありません。利用者に与える再利用・再配布の許諾は権利者が決めます。ここでは新しいライセンスを選定していません。
3. **ランタイムの公開時チェック**: Pre-releaseのローカルアーカイブに第三者LICENSE / NOTICE、依存一覧、変更点、ビルド設定、SHA256を同梱済み。公開するソースcommitとタグを確定し、アップロード後の再取得hashを検査する。GPU SDK・追加機能を同梱する場合は棚卸しし直す。
4. **検証範囲の表示**: CUDA sm_89はビルド／リンクのみ確認、NVIDIA実機での動作は未検証。HIPのR9700 / 7900 XTXの確認とは分けて記載する。公式版や全GPU動作保証と表示しない。

調査に用いたHTTP応答・取得URL・SHA256は`.cache/license-review-2026-09-21/`、公式取得と再変換は`.cache/sarashina-official-review/`・`results/sarashina-official/`に保存しています（Git対象外）。新版mmprojは利用者の指示でHFの専用Privateリポジトリへアップロードし、再取得によるhash一致を確認しました（[送信記録](SARASHINA_UPLOAD.md)）。**HFの可視性は変更していません。** ランタイムの配布物・公開先・検査範囲は[Pre-releaseの案内](SARASHINA_RELEASE.md)を参照してください。
