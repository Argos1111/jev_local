# ネイティブ実験コード

- `source.json`: パッチ対象の公式llama.cppソースアーカイブ（b11042）のURLとSHA256。
- `vision_embedding_cache.h`: モデルコンテキストごとの容量制限付きLRU。画像encoder/projectorの出力だけを保持し、decoderのStateは保持しません。
- `test_vision_embedding_cache.cpp`: モデル不要の並列処理・キー分離・エラー・容量制限の単体テスト。

パッチ適用は[`scripts/patch_vision_cache.py`](../scripts/patch_vision_cache.py)、ビルドは[`scripts/build_image_cache.py`](../scripts/build_image_cache.py)で行います。ソース展開先とビルド成果物は`.cache/`内でGit管理対象外です。公式配布バイナリは書き換えません。

```bash
g++ -std=c++17 -Wall -Wextra -Werror -pthread native/test_vision_embedding_cache.cpp -o /tmp/test-vision-cache
/tmp/test-vision-cache
```

[導入・設計・制限](../docs/IMAGE_CACHE.md)を参照してください。GPUプラグインとリンクする実験ビルドはGCC/libstdc++を使います。

## Sarashina（独立した実験パッチ）

- `sarashina-reference.json`: 公式の固定revision・単一checkpoint・AutoProcessor/tokenizer等のサイズとSHA256。公開クローンへのfallbackなし。
- `sarashina-vision-b11042.patch`: 私有projector種別`sarashina2vl`、post-merger LayerNorm、境界token・前処理修正、Sarashina限定encoderキャッシュ。既存Qwen/LFMの挙動は維持。
- `sarashina-official-preprocess-b11042.patch`: 上記に重ねて適用。公式の正規化後float bicubicと面積上限1,016,064へ対応。新mmprojのmetadataで選択し、旧版は旧前処理＋警告で維持。
- `sarashina-fa160-b11042.patch`: CUDA/HIP共通の160次元GPU FAと252ケースの任意演算検査。GPU型番のホワイトリストなし。テンプレート生成を含めて適用。
- `test_sarashina_embedding.cpp`: 正規化画素・画像embeddingを出力し、参照実装と比較する診断プログラム。`pixels-only`ならencoder推論を省略。

ビルドは[`scripts/build_sarashina.py`](../scripts/build_sarashina.py)（`--backend cuda` / `hip` / `cpu` / `stock-rocm`、`--base`で別のビルド先）、変換は[`scripts/convert_sarashina_mmproj.py`](../scripts/convert_sarashina_mmproj.py)、起動は[`scripts/run_sarashina.py`](../scripts/run_sarashina.py)（`--build`で展開した配布版も指定可能）。任意のGPU数値検査は[`tools/verify_sarashina_fa.py`](../tools/verify_sarashina_fa.py)。未検証・検査失敗・別GPUを理由に実行を禁止しません。`.cache/sarashina-official-runtime/`に隔離し、通常ランタイム・LFM用キャッシュビルドを書き換えません。[ビルド手順](../docs/SARASHINA_BUILD.md)を参照してください。

配布用アーカイブは[`scripts/prepare_sarashina_release.py`](../scripts/prepare_sarashina_release.py)でローカルに梱包します。`sarashina-runtime/`は配布アーカイブ、`sarashina-mmproj/`はHFのmmproj配布に同梱するREADME / NOTICEのテンプレートです。配布物の利用手順は[Sarashinaで画像を使う](../docs/SARASHINA_RELEASE.md)です。

これは本リポジトリ独自の実験パッチです。llama.cpp上流への投稿はしていません。パッチに含まれる上流由来のコードの著作権・MITライセンス表記は[`LICENSE.llama.cpp`](LICENSE.llama.cpp)に収録しています。バイナリ配布時にはこれに加えて第三者コード・同梱GPUライブラリ等の条件確認が必要で、配布アーカイブには[`licenses/`](../licenses/)とb11042ソース由来の表記を同梱しています。上流への変更提案には人間によるレビュー・理解・継続保守と、対象リポジトリのcontribution規約の確認が必要です。
