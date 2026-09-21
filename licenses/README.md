# 第三者ライセンス原文

このディレクトリは、Sarashinaの成果物を配布する際に必要な第三者ライセンスを保管します。**Jev Local本体や独自追加コードのライセンスを設定するものではありません。** ランタイムのtoolchain表記は[`runtime/`](runtime/)にあり、llama.cpp内の第三者表記は固定ソースアーカイブから梱包時に取得します。

| ファイル | 対象・原文の取得元 |
|---|---|
| [LICENSE.sarashina2.2-vision-3b](LICENSE.sarashina2.2-vision-3b) | Sarashina2.2 Vision 3B、MIT、Copyright (c) 2025 SB Intuitions。[公式revision `46d9cc3929a54f7d2b91ce5668d7a9c5833991ed`](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b/blob/46d9cc3929a54f7d2b91ce5668d7a9c5833991ed/LICENSE)から取得 |
| [LICENSE.siglip](LICENSE.siglip) | 画像エンコーダーの元モデルSigLIP、Apache-2.0。[big_vision revision `0127fb6b337ee2a27bf4e54dea79cff176527356`](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/LICENSE)から取得。Sarashinaの学習に使われたbig_visionのrevisionを特定したものではない |

2026-09-21取得。どちらも取得した原文を変更せず保存しています。llama.cpp自身のMIT原文は[`native/LICENSE.llama.cpp`](../native/LICENSE.llama.cpp)にあります。

[配布時の必要表記・未確認事項](../docs/SARASHINA_DISTRIBUTION.md)を参照してください。実際に配布する重み・バイナリにも該当するLICENSE / NOTICEを同梱する必要があり、このリポジトリにファイルを追加しただけでは配布物への同梱になりません。
