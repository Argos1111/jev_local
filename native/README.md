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

- `sarashina-reference.json`: 公開クローンのrevision・全8 shard等のSHA256。公式checkpointとの同等性は未検証。
- `sarashina-vision-b11042.patch`: 私有projector種別`sarashina2vl`、post-merger LayerNorm、境界token・前処理修正、Sarashina限定encoderキャッシュ。既存Qwen/LFMの挙動は維持。
- `sarashina-fa160-b11042.patch`: CUDA/HIP共通の160次元GPU FAと252ケースの任意演算検査。GPU型番のホワイトリストなし。テンプレート生成を含めて適用。
- `test_sarashina_embedding.cpp`: 正規化画素・画像embeddingを出力し、参照実装と比較する診断プログラム。`pixels-only`ならencoder推論を省略。

ビルドは[`scripts/build_sarashina.py`](../scripts/build_sarashina.py)（`--backend cuda` / `hip` / `cpu` / `stock-rocm`）、変換は[`scripts/convert_sarashina_mmproj.py`](../scripts/convert_sarashina_mmproj.py)、起動は[`scripts/run_sarashina.py`](../scripts/run_sarashina.py)。任意のGPU数値検査は[`tools/verify_sarashina_fa.py`](../tools/verify_sarashina_fa.py)。未検証・検査失敗・別GPUを理由に実行を禁止しません。`.cache/sarashina-runtime/`に隔離し、通常ランタイムやLFM用キャッシュビルドを書き換えません。[条件・照合結果・再現コマンド](../docs/SARASHINA.md)を参照してください。

これは本リポジトリ独自の実験パッチです。llama.cpp上流への投稿はしていません。パッチに含まれる上流由来のコードの著作権・MITライセンス表記は[`LICENSE.llama.cpp`](LICENSE.llama.cpp)に収録しています。上流への変更提案には人間によるレビュー・理解・継続保守と、対象リポジトリのcontribution規約の確認が必要です。
