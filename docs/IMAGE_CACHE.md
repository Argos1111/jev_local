# 画像エンコード結果の再利用（実験機能）

LFM2.5-VLの画像エンコーダー＋projectorが出力したembeddingを、llama.cpp内部で再利用します。モデルへの画像・state・質問の入力処理（decoder prefill）は質問ごとに実行し、KVや再帰状態は共有しません。PNG/JPEGの読み込み・リサイズ等の前処理も質問ごとに行います。

公式b11042のソースに小さなパッチを適用した、別の実験用バイナリです。通常の`./run.sh`や配布済みバイナリには変更を加えていません。このビルド手順はLinux/ROCm向けです。

**このページのランチャーはLFM用です。** Sarashinaは同じLRU部品を使いますが、projectorと前処理の修正が必要なため[専用の配布ランタイムと起動方法](SARASHINA_RELEASE.md)へ分離しています。

## 起動

初回は下記「ビルドと再検証」の手順でビルドしてください。バイナリはリポジトリに含めません。同じポートのサーバーが動いている場合は止めてから起動します。

```bash
./run_image_cache_gpu.sh --port 18080 --backend-port 18097
```

別ターミナルから既存のクライアントをそのまま使えます。下の例の画像は各自で`sample_pics/`に配置します（画像は同梱していません）。

```bash
python3 systemone_client.py \
  --url http://127.0.0.1:18080 \
  --input examples/vision.json \
  --image sample_pics/your-image.png
```

同じ実験バイナリで無効にする場合:

```bash
JEV_IMAGE_CACHE_MIB=0 ./run_image_cache_gpu.sh --port 18080 --backend-port 18097
```

`GPU_DEVICE`の既定値は`ROCm0`です。ROCmライブラリが標準パスにない場合は`ROCM_LIBRARY_DIR`で指定できます。

## 再利用範囲と寿命

- LFM2のvisionのみが対象で、音声や他モデルには適用しません。
- 正規化後の画素の完全なバイト列・形状・タイルの付加情報をキーにします。同じ画像IDを持つ別タイルの取り違えを防ぎます。
- モデルコンテキストごとに独立したメモリ上のLRUです。画像を変更すると別のキーになり、モデル終了時に全削除します。ディスクには保存しません。
- 既定の上限はキーとembeddingの合計128 MiB。超過時は古いエントリーから破棄します（管理オブジェクトや一時作業メモリは上限外）。`JEV_IMAGE_CACHE_MIB=0..4096`で変更できます。
- 同時に来た同一画像のエンコードを重複させないよう、初回計算を含めてロックします。失敗した計算結果は保存しません。
- 4問の初回送信では1回計算＋3回再利用、同じ画像の再送信では4回とも再利用します。複数タイル画像では各タイル・サムネイルごとに同様に処理します。
- HTTPの`X-Jev-Local-State-Cache: off-images`は引き続き**decoder側のState共有が無効**であることを示します。画像embeddingのhit/missはバックエンドログの`JEV_IMAGE_CACHE`で確認します。

## 効果の目安

同じ画像を複数の質問で使う場合に効きます。512×286 pxの画像＋4問（R9700）では、初回リクエストで約1.9倍、同じ画像を再送した場合で約3.7倍の応答時間短縮を確認しています。選択結果は有無で変わりません。

画像のサイズや質問数で効果は変わります。候補順による偏りや誤答を改善する機能ではありません。自分の環境で測るには[評価ツール](EVALUATION.md#画像エンコード再利用の有無を比較)の`benchmark_image_cache`を使います。

## ビルドと再検証

Linux x86_64、GCC（g++、C++17）、CMake 3.14以上、Ninja、利用可能なAMD ROCm環境が必要です。通常の推論と異なり、コンパイルを行います。ROCmプラグインは公式b11042ランタイムの`libggml-hip.so`を利用します。C++ ABIの一致のためGCC/libstdc++を使ってください。`CC`・`CXX`・`CMAKE`・`NINJA`で各実行ファイルを指定できます。依存ツールやOSパッケージは自動インストールしません。

```bash
./setup.sh --model vision --backend rocm
python3 scripts/build_image_cache.py --jobs 8

# ROCmライブラリが標準パスにない場合はLD_LIBRARY_PATHで指定します
python3 -m tools.benchmark_image_cache --image sample_pics/your-image.png
```

`setup.sh --model vision --backend rocm`も同じROCm共有ライブラリが必要です。標準パス以外にある場合は、セットアップ前から`LD_LIBRARY_PATH`を設定してください。

比較ツールは同一の実験ビルドでキャッシュを無効・有効に切り替えます。入力は`--input`（Choiceのみ）、画像は`--image`、GPUは`--device`で指定できます。GPUの既定は`ROCm0`で、機種名は固定しません。比較ツールは実験用ポート19080/19097を使い、終了・失敗時に自分が起動したプロセスを停止します。ポートが使用中なら停止し、`--port`・`--backend-port`で変更できます。既存の測定JSONとログは上書きします。別保存先は`--output`、数値比較の逐次実行は`--workers 1`で指定します。

ソースのURLとSHA256は`native/source.json`に固定し、変更箇所は`scripts/patch_vision_cache.py`と`native/vision_embedding_cache.h`で管理しています。ビルド成果物のハッシュは`.cache/vision-build-gcc/jev-build.json`に保存します。
