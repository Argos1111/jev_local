# ローカルの入力画像

画像ファイルはここに置いてください。このフォルダの画像は`.gitignore`で除外し、GitHubには含めません。

GSS資料でのローカル検証では次のファイルを使いました。

- `20260911_image.png`: 元画像（1190×665 px）
- `20260911_image_resized.png`: 縮小版（512×286 px）

対応する質問は[`examples/vision_gss.json`](../examples/vision_gss.json)（12問）と[`examples/vision_gss_4.json`](../examples/vision_gss_4.json)（4問）です。クローンした環境には画像は付属しません。自分で用意した画像を`--image`で指定してください。一般の画像には[`examples/vision.json`](../examples/vision.json)の質問を使えます。

```bash
python3 systemone_client.py --input examples/vision.json --image sample_pics/photo.jpg
```

PNG/JPEG、1枚4 MiB、最大4枚。複数枚は`--image`を繰り返します。画像の自動縮小はクライアントでは行いません。
