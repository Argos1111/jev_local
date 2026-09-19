# 画像エンコードキャッシュの実験コード

- `source.json`: パッチ対象の公式llama.cppソースアーカイブ（b11042）のURLとSHA256。
- `vision_embedding_cache.h`: モデルコンテキストごとの容量制限付きLRU。画像encoder/projectorの出力だけを保持し、decoderのStateは保持しません。
- `test_vision_embedding_cache.cpp`: モデル不要の並列処理・キー分離・エラー・容量制限の単体テスト。

パッチ適用は[`scripts/patch_vision_cache.py`](../scripts/patch_vision_cache.py)、ビルドは[`scripts/build_image_cache.py`](../scripts/build_image_cache.py)で行います。ソース展開先とビルド成果物は`.cache/`内でGit管理対象外です。公式配布バイナリは書き換えません。

```bash
g++ -std=c++17 -Wall -Wextra -Werror -pthread native/test_vision_embedding_cache.cpp -o /tmp/test-vision-cache
/tmp/test-vision-cache
```

[導入・設計・制限](../docs/IMAGE_CACHE.md)を参照してください。GPUプラグインとリンクする実験ビルドはGCC/libstdc++を使います。
