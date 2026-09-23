# Sarashina2.2 Vision 3Bによる比較

[SB IntuitionsのSarashina2.2 Vision 3B](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b)を、[mradermacher配布のGGUF](https://huggingface.co/mradermacher/sarashina2.2-vision-3b-GGUF)で実行します。LFMと同じllama.cpp・先頭回答トークンのlogprob・`/v1/systemone`を使い、追加学習はしません。

- **通常の`run.sh --model sarashina` / `sarashina-q8`は文章・JSON専用**のままです。LFMの既定設定や保存済みの選択は変更しません。
- **公式checkpoint由来の修正版mmprojと画像ランタイム**を別に用意しました。元解像度の文書画像で12問中12問が一致し、画像エンコード再利用とGPU Flash Attentionの速度改善も実測しています。ただし単色・複数画像・候補順の問題が残り、一般的な画像対応の合格とはしていません。
- 2026-09-21、利用者によるHFのアクセス承認・ログイン後に**公式checkpointとAutoProcessorを取得し、別ファイルへ再変換**しました。旧クローンと全624 tensorは同一でしたが、前処理の差を追加修正しています。既存の言語GGUFには公式tokenizerとの分割差があり、推論全体の完全な同等性は主張しません。

[画像修正と検証](#画像経路の修正実験機能) / [GPU FAの実測](#実験版gpu-flash-attention) / [ビルドと起動](#実験版のビルドと起動) / [配布用Pre-releaseの利用](SARASHINA_RELEASE.md)

## 量子化の選択

**容量を抑えて試すならQ4_K_Mは妥当です。ただし「Q4の方が速い」「Q8なら判定の偏りが直る」とは限りません。** R9700での今回の短い判定タスクではQ8_0の方が少し速く、JNLIの正解率は逆にQ4の方が高くなりました。

| ファイル | 実サイズ（10進GB） | 用途 |
|---|---:|---|
| `sarashina2.2-vision-3b.Q4_K_M.gguf` | 2.066 | 標準の比較用。`--model sarashina` |
| `sarashina2.2-vision-3b.Q8_0.gguf` | 3.568 | 量子化の対照用。`--model sarashina-q8` |
| `sarashina2.2-vision-3b.mmproj-f16.gguf` | 0.893 | 元の公開projector。調査用に保持、自動取得・有効化しない |
| `sarashina2.2-vision-3b.mmproj-jev-f16.gguf` | 0.893 | 旧クローン由来の修正版。ローカルに履歴・比較用として保持（HFの配布からは削除） |
| `sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf` | 0.893 | 公式由来の新版。明示的に変換し、新しい実験ランタイムで使用 |

Q4はQ8よりファイル容量が約42%小さくなります。これらは重みのサイズで、VRAMの実測値ではありません。推論にはKVキャッシュ・計算用バッファも必要です。実験画像構成はQ4本体＋修正版mmprojで約2.96 GBです。

モデルカードによると、言語部分はSarashina2.2-3B-Instruct、画像側を含む全体は約3.8Bパラメーターです。日本語画像・文書QAの比較対象として興味深い一方、公式の生成ベンチマーク値がGGUFやこの1トークン判定方式で再現するとは限りません。

## 標準ランタイムでのセットアップと起動

```bash
# 既にLFMのランタイムがある場合
unset LFM_MODEL LFM_MMPROJ
./setup.sh --model sarashina --model-only
./run.sh --model sarashina

# 別ターミナル（クライアントは共通）
python3 systemone_client.py
```

新規環境では`./setup.sh --model sarashina`でランタイムも導入します。Q8を比較する場合はサーバーを停止してから次を実行します。

```bash
./setup.sh --model sarashina-q8 --model-only
./run.sh --model sarashina-q8
```

- モデルのURL・revision・SHA256は[`scripts/runtime.json`](../scripts/runtime.json)に固定しています。revisionは`18b014396fa28c005d0551146558240189fc9ce8`です。
- `setup.sh`は選択したプロファイルを保存しますが、`run.sh --model ...`だけでは保存済みの選択を変えません。未設定時の既定は従来どおりLFMの`text`です。
- `--model-only`は既存のCPU/GPUランタイム選択を変えません。GPUで測る場合は、[GPUセットアップ](SETUP.md)または`LLAMA_SERVER`・`GPU_LAYERS=all`を明示してください。
- `LFM_MODEL`・`LFM_MMPROJ`は歴史的な変数名ですがSarashinaにも適用されます。通常は両方をunsetします。SarashinaプロファイルにLFMのmmprojが自動で付くことはありません。
- GGUFのコンテキスト長は8192。標準起動は4 slots・合計8192（各2048）です。必要なら`CTX_SIZE=32768`で各slotを8192にできます。モデルの学習長を延ばす設定ではありません。

今回のR9700測定は、保存済みランタイムを変えず、次の環境指定で行いました。ROCm共有ライブラリのパスはこのマシン固有で、他環境では対応するインストール先を指定します。

```bash
LD_LIBRARY_PATH="$HOME/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v4${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LLAMA_SERVER="$PWD/.cache/runtime/b11042/linux-x86_64-rocm/llama-b11042/llama-server" \
GPU_LAYERS=all GPU_DEVICE=ROCm0 CTX_SIZE=8192 \
  ./run.sh --model sarashina --port 28080 --backend-port 28097
```

## 全方式のテキスト比較

2026-09-20、各モデルを順に起動して再測定しました。以前のLFMの記録とはGPUバッチの違いなどによる小さな差があります。ここでは今回の再測定値に揃えています。

- GPU: AMD Radeon AI PRO R9700。llama.cpp公式b11042 ROCm、全レイヤーをROCm0に配置、CPU 8 threads、4 slots、合計context 8192。
- ModernBERT: ローカルのfine-tuned checkpoint、torch ROCm、`cuda:0`、bf16。
- LFM text / LFM VL / Sarashina Q4 / Sarashina Q8は同じプロンプト構築・A/B/Cラベルの先頭トークン判定。モデル固有のチャットテンプレートを使用し、テストに合わせたプロンプト調整なし。
- **すべて画像なし。** LFM VLもSarashinaもmmprojなしで言語部分を実行。画像エンコードキャッシュなし。
- JNLI 2,508件・JCommonsenseQA 1,118件の全test、livedoor 500件・JMMLU 500件（10科目、seed 0）、顧客対応の手作り16件。既存の評価ツールを変更せず使用。
- APIへの同時リクエスト数4、1リクエスト1問、state-cache auto（この構成ではprefix共有なし）。全方式・全タスクでHTTPエラー0件。
- LLMはこのリポジトリではzero-shot、ModernBERTはJGLUE train等でfine-tune済み。**同じAPIでの実用比較であり、学習条件・パラメーター数・量子化を揃えた能力比較ではありません。** 「学習に使っていないタスク」はこのリポジトリのModernBERT fine-tuningに対する表現で、LLMの事前学習への混入を否定するものではありません。

### 正解率

| バックエンド | JNLI | JComQA | livedoor | JMMLU | 顧客対応16例 |
|---|---:|---:|---:|---:|---:|
| LFM2.5-1.2B Instruct Q8_0 | 16.99% | 68.69% | 46.4% | 43.4% | 9/16 |
| LFM2.5-VL-1.6B Q8_0（画像なし） | 58.69% | 73.35% | 49.2% | 47.2% | 11/16 |
| Sarashina2.2 Vision 3B Q4_K_M（画像なし） | 33.61% | 85.96% | 69.4% | 57.0% | 16/16 |
| Sarashina2.2 Vision 3B Q8_0（画像なし） | 18.26% | 87.21% | 69.4% | 58.0% | 16/16 |
| ModernBERT-Ja 310M（fine-tuned） | 92.62% | 92.40% | 35.6% | 34.8% | 15/16 |

### HTTP応答時間の中央値（ms）

| バックエンド | JNLI | JComQA | livedoor | JMMLU | 顧客対応16例 |
|---|---:|---:|---:|---:|---:|
| LFM text Q8_0 | 54.3 | 47.5 | 104.1 | 48.3 | 41.2 |
| LFM VL Q8_0（画像なし） | 54.4 | 47.6 | 103.2 | 48.6 | 41.1 |
| Sarashina Q4_K_M | 147.7 | 139.0 | 408.7 | 176.2 | 150.0 |
| Sarashina Q8_0 | 137.4 | 130.7 | 371.9 | 166.5 | 144.8 |
| ModernBERT（fine-tuned） | 32.4 | 31.3 | 96.4 | 32.7 | 32.2 |

モデルロードを除外。JGLUEは合成入力によるウォームアップ後、他タスクも起動・契約テスト・JGLUE実行後に計測しています。各データセット1回の走査で、複数回の実験の中央値ではありません。SarashinaではFlash Attention非対応による無効化、ローカルのhipBLASLtライブラリ探索警告があり、正常完走していても最適化された速度上限を示す値ではありません。

今回の用途は長文生成ではなく、prefill＋1トークン判定です。Q4の省メモリがそのまま速度向上になるわけではなく、実行カーネル・入力長・バッチ条件に依存します。

### 公式b11042ではFlash Attentionが使えない理由

**この節は未修正の公式llama.cpp b11042の話です。** 上表の測定でSarashinaのFAが無効だった理由と、公式ビルドでFAを強制しても速くならないことを説明します。Jev Localの修正版ランタイムはこの制約を解消するために160次元対応のGPUカーネルを追加しており、そちらではFAを有効にすると速くなります（[実験版GPU Flash Attention](#実験版gpu-flash-attention)）。

SarashinaのK/Vヘッド次元は**160**なのに対し、b11042の[CUDA/ROCm共用FAカーネル選択](https://github.com/ggml-org/llama.cpp/blob/b11042/ggml/src/ggml-cuda/fattn.cu)は160に対応していません。GPU全体がFA非対応という意味ではなく、このモデルの形状と実装の組み合わせの問題です。公式ビルドでの挙動は次のとおりです。

- 既定の`-fa auto`は未対応のFAを無効化し、通常のAttentionをGPUで実行します。上表の測定はこの状態で、AttentionをCPU実行していたわけではありません。
- `-fa on`で強制すると、FAだけがCPU側に配置され、CPU/GPU間のグラフ分割が2から66に増えて逆に遅くなります。**公式ビルドでSarashinaを使う場合、`-fa on`は指定しないでください。**

R9700、同じ4 slots・合計context 8192で、常識QAとlivedoorから各1入力を固定し、1件ずつ測定しました。ウォームアップ2回を除く5回の中央値、prompt cacheなし、1トークン判定、同じslotを使用。トークン化・template適用を事前に済ませたバックエンド直接呼び出しなので、前掲の4並列SystemOne API時間と直接比較しないでください。

| 公式b11042でのSarashina Q4の設定 | 常識QA 109 tokens | livedoor 423 tokens |
|---|---:|---:|
| `-fa auto`（自動で無効化。既定） | 25.4 ms | 62.0 ms |
| `-fa off` | 24.8 ms | 62.2 ms |
| `-fa on`（FAがCPU側に配置される） | 59.3 ms | 241.6 ms |

対照として、FA対応のLFM text（ヘッド次元64）に同じ入力を送ると、短い入力121 tokensはOFF 13.5 → ON 13.2 ms、長めの523 tokensはOFF 37.1 → ON 27.4 msでした。FAに速度改善の余地はありますが、この結果からSarashinaのGPU版FAの改善率は推定できません。

GGUFとロードログで確認した言語モデルの構造:

| | LFM text | Sarashina Visionの言語部分 |
|---|---:|---:|
| パラメーター数 | 約1.17B | 約3.36B |
| 層数 | 16 | 32 |
| Attention層 | 6（残り10はshort convolution） | 32 |
| ヘッド次元 | 64 | 160 |

**速度差はパラメーター数だけでなく、層数・Attentionの構成・GPUカーネル・並列条件にも依存します。** FAはAttentionのメモリアクセスと中間テンソルを効率化するもので、FFNや投影などの計算は減らしません。入力が長いほど効く余地が増えますが、今回の短いprefill＋1トークン判定で、FAだけによりLFM並みになるとは言えません。処理時間の寄与率を特定するGPUプロファイリングは未実施です。

追加測定の全応答・ログは`results/sarashina-flash-attn/`に保存しています。通常の起動設定は変更していません。

修正版ランタイムではこの制約はなく、同一ビルド内のFA OFF→ONで常識QA 25.2→23.2 ms、livedoor 62.8→52.0 ms（Q4、1並列）と短縮しています。詳細は[実験版GPU Flash Attention](#実験版gpu-flash-attention)を参照してください。

### 解釈と確率上の注意

- **常識QA・知識・ニュース分類ではSarashinaが今回のLFMより高い**結果でした。Q4でも試す価値があります。顧客対応16/16は小規模な手作りセットであり、実業務の完全正解を意味しません。
- **JNLIは依然弱いです。** 多数派のneutralを常に選ぶだけで54.43%なのに対し、SarashinaはQ4/Q8とも下回りました。2,508件中contradictionの予測がどちらも0件で、Q4はentailment 1,634 / neutral 874、Q8はentailment 2,321 / neutral 187に偏っています。
- Q4とQ8のJNLI差は15.35ポイントもあります。Q4が一般に高品質なのではなく、現行のプロンプト・候補ラベル・logitによる判定が量子化にも敏感なことを示します。Q8を基準に「量子化差」と「判定方式の問題」を分けて見るべきです。
- `tools.evaluate`の返金要求8例は元の候補順でQ4/Q8とも8/8でしたが、**候補順を反転するとどちらも4/8で予測が変化**しました。P(true)の平均絶対差はQ4が0.345、Q8が0.398。Q8でも順序依存は解消しません。
- 通常のデモ7問ではQ4の候補確率質量`candidate_mass`は約0.999〜1でした。先頭に説明文や空白を生成してしまう問題はそのデモでは見られませんでしたが、質量が高くても正答・校正済み確率の保証ではありません。

したがって、知識を使う分類器の追加候補として有用ですが、「Jevらしい安定した確率判定器」としては、候補順を変えた検証・対象業務の検証用データによるNLL/Brier等の評価を続ける必要があります。校正やプロンプト変更を行うならtestではなく別の検証splitを使います。

## API上の制約

- Choice / Score / Noul・文章/JSONのStateは共通APIで動きます。A〜Zが各1トークンであること、26択、長いStateのslot保存・復元、両量子化の`run.sh`起動を実機確認しました。
- **Choiceは26候補まで。** 27以上で使う数字ラベルについて、Sarashinaは`26`を`2`と`6`に分割します。1トークン判定を守るため422の入力エラーとし、候補の切り捨てや多トークンの先頭だけによる近似はしません。LFM / ModernBERTの255択対応は変わりません。
- 画像なしのプロファイルへの画像リクエストは、既存のテキスト専用llama.cppバックエンドと同様、502とVisionバックエンドの設定案内を返します。

## 画像経路の修正（実験機能）

### 原因と変更内容

元の公開mmproj（SHA256 `fc3cd632b72c76e7586652dd77463a50ab8b3a567af752cf4351e50435b02728`）はロードできますが、b11042のQwen2VL互換経路には次の相違がありました。ロード成功や`/props`の`vision: true`だけでは正しさを確認できません。

| 箇所 | 修正版 |
|---|---|
| merger出力後の正規化 | 欠けていた2560次元の`norm.weight` / `norm.bias`を収録し、LayerNorm（epsilon `1e-5`）を適用 |
| 画像境界 | Qwenの文字列ではなく`<\|prefix\|>` / `<\|suffix\|>`。token ID 102397 / 102398を検査 |
| 前処理 | RGBをrescale・mean/std各0.5で正規化**してから**float bicubic（係数−0.75、`align_corners=False`、antialiasなし）。padding・uint8への再丸め・値のclampなし |
| 画像面積 | 公式設定の3136〜1,016,064 pixels（4〜1296画像トークン相当）。14px patch・2×2 merge |
| リサイズ境界 | Pythonと同じties-to-evenの丸め。28の倍数の中間値・極小画像・面積上限も照合 |
| 活性化 | encoderはtanh GELU、mergerはerf GELU |
| メタデータ | vision FFN幅を2560ではなく4304として記録 |

私有projector種別`sarashina2vl`に分離し、Qwen / LFMのグラフを変更しません。語彙なしのメモリ見積りでも安全に動くよう境界token検査を条件付きにしています。converterは全333入力tensorの名前・形状を検査して442出力tensorを生成し、チェックポイント付属Pythonを実行しません。

新しい`native/sarashina-official-preprocess-b11042.patch`は公式の`F.interpolate(mode="bicubic")`に合わせます。旧版の「Pillowでuint8をリサイズしてから正規化」とは異なります。公式JSONにある`resample: 2`は公式Pythonでは参照されず、実際はfloat bicubicです。旧再構成版の上限1280も公式設定の1296へ変更しました。

公式版は892,579,936 bytes、SHA256 **`7c170758eabf18eaf60d6c584d27b9955c3fdd595975d21fc4e7b9c93ca1b321`**。別ファイルへの再変換でもバイト単位の一致を確認しました。`jev.sarashina.preprocess_version=1`と出典をGGUFへ記録します。旧ランタイムでは新版の前処理を扱えないため**再ビルドが必要**です。新ランタイムは旧mmprojも旧前処理＋警告で実行できます。

旧クローン（`AnalyticPudding/sarashina2.2-vision-3b-clone`、revision `05710ee40ae41ff322da991298ab893cf54ce110`）との比較では、**公式checkpointの全624 tensorが名前・型・形状・データのSHA256まで一致**しました。モデル/configuration Pythonも同一で、configの差はdtypeの表記とTransformers版の記録です。変換入力は以後すべて公式版です。

新旧修正版mmprojの**442 tensorもすべて同一**で、ファイルhashの違いは出典・前処理メタデータです。旧版は892,579,328 bytes、SHA256 `094e86c9c7d54361a5c19384c60f379064346396f64fcd77c52e000942eb7843`のまま保存しました。mradermacher元mmprojとの共通440 tensorも一致し、追加分はpost-merger LayerNormの2 tensorです。言語Q4/Q8 GGUFはmradermacher配布のまま変更していません。

### 公式参照との数値照合

参照は**`sbintuitions/sarashina2.2-vision-3b`、revision `46d9cc3929a54f7d2b91ce5668d7a9c5833991ed`**。全重みは8 shardではなく7,603,021,272 bytesの`model.safetensors`一つです。config・processor・tokenizer・Python・LICENSEの15ファイルを取得し、revision・サイズ・SHA256を[`native/sarashina-reference.json`](../native/sarashina-reference.json)に固定しました。取得・変換ではcheckpoint付属コードを実行しません。診断時だけ、コードをレビューして明示許可後にローカルで実行します。

- **前処理15/15**: 単色・図形・勾配・縮小文書・丸め境界・極小・縦横比200・面積上限・1190×665相当・checkerboard・乱数画像を公式AutoProcessorと比較。形状一致、正規化画素の最大絶対差 **7.16e-7以下**（基準1e-6）。1008×1008は1296 tokens。旧版の「12ケース画素差0」は再構成したQwen処理との結果であり、公式処理との一致ではありませんでした。
- **encoder 5/5、CPUとHIP**: 公式重みのCPU float32式を参照。F16 projector＋R9700 / FA ONでcosine **0.999747以上**、RMSE **0.00145〜0.02259**。CPU nativeもcosine 0.999805以上、RMSE 0.000603〜0.01986。合格基準は従来と同じcosine ≥0.999 / RMSE ≤0.04 / 全値finiteです。
- **bitwise一致ではありません**: HIPの最大要素差は約0.3575、CPUは約0.3907。GGUF精度・ggmlのConv2D/im2col入力のF16丸めなどの差が残ります。`--reference-half-pixels`は切り分け専用で、本来のfloat32参照とは区別します。
- **生成文**: 公式bf16 / Transformers 4.57.1＋公式AutoProcessorに対し、native **Q4は11/12、Q8は12/12**一致（EOSを除く）。図形・単色・非整倍画像・2画像・元解像度文書を含む12例、最大80生成tokenです。Q4の文書回答は表現が異なりましたが、タイトル・検知日は読めています。正答率や全出力の一致保証ではありません。
- **tokenizerの未解決差**: 公式tokenizerとnativeから独立に得た入力ID列が一致したのは**1/12**、入力token数は10/12でした（文書は一致、図形2例はnativeが1 token多い）。例: 公式の「答えて／ください」に対しnativeは「答え／てください」。slow/fast公式tokenizerの両方で再現します。既存言語GGUFの`tokenizer.ggml.model=llama`と公式Unigram経路の違いを確認しましたが、原因確定・言語GGUFの修正は未実施です。mmprojとは別の検討事項です。参照生成ツールはこの不一致を隠さず**非ゼロ終了**します。旧版はnativeのIDを共用していたため、この差を検査できていませんでした。

### 画像の読解結果と残る制約

R9700、公式由来mmproj、新版HIPビルド、FA auto（有効）、画像キャッシュ128 MiB、**4 slots・合計context 32768**、既存`/v1/systemone`で再確認しました。

| 入力・条件 | Q4 | Q8 |
|---|---:|---:|
| 白背景の赤/青/緑の四角形、候補反転・画像の再送/交互入力 | 10/10 | 10/10 |
| 別画像3件の同時送信 | 3/3 | 3/3 |
| GSS元画像1190×665、12問 | 12/12 | 12/12 |
| 同上、画像なし対照 | 8/12 | 8/12 |
| 元画像、Choice順反転 | 10/12 | 11/12 |
| 反転、画像なし対照 | 3/12 | 3/12 |
| GSS縮小画像512×286、元順12問 | 10/12 | 10/12 |
| 2枚目の四角形の色、画像順2通り | **1/2** | **1/2** |

GSSは既存の12問を各条件1回ずつ採点した1資料の診断です。元のChoiceの正解は先頭に寄っており、12/12だけでは一般的な読解能力を示せません。旧前処理では順反転10/12・縮小9/12でしたが、新旧1回ずつの差を一般的な精度改善とはしません。**LFM用の512px推奨をSarashinaの文書読解へ無条件に転用しないでください。** Q4/Q8と公式参照の自由生成では文書タイトル・検知日「令和8年6月25日」を読めました。

単色5色をすべて「白」と答える問題と、blue/red順の2枚目を「青」と答える問題は、今回は**公式checkpoint＋公式AutoProcessorでも再現**しました。クローン固有の問題とは考えられませんが、少数入力・指定プロンプトでの観測です。`tools.verify_sarashina_vision`はこれらを記録し、全体としては**非ゼロ終了（不合格）**にします。単画像図形の成功だけで失敗を隠しません。画像と長いテキストStateの同時送信・slot保存/復元もQ4/Q8で再確認しています。

### 画像エンコード再利用の速度

画像encoder＋projector出力だけを、モデルごとのLRU（128 MiB）で再利用します。キーは正規化画素全体・形状・付加情報。decoderのKV / Stateは共有せず、画像前処理と各質問のprefillは毎回行います。`X-Jev-Local-State-Cache: off-images`は変わりません。実装は既存の[`vision_embedding_cache.h`](../native/vision_embedding_cache.h)を利用しますが、LFM用ビルドとは別です。

公式前処理版での同一バイナリOFF→ON→ON→OFF比較。GSS **1190×665px**（前処理後1176×672、1008画像tokens）、4 Choice、4 workers、context 32768、FA ON。各ブロック初回1回＋5回、下表の再送値は各mode 10回のHTTP中央値です。モデルロード・ファイル読み込み・base64変換は除外します。

| 量子化 | 再利用なし | 同じ画像を再送 | 倍率 | ON初回（各1 miss＋3 hits） |
|---|---:|---:|---:|---:|
| Q4_K_M | 2146 ms | **688 ms** | **3.12倍** | 1443〜1444 ms |
| Q8_0 | 2053 ms | **600 ms** | **3.42倍** | 1313〜1314 ms |

4問の選択は全測定で一致。全応答を先頭の応答と比較した候補確率差の最大はQ4 0.10416、Q8 0.02801で、キャッシュOFFでも実行バッチによる揺れがあります。**Q4を1 workerで比較すると全候補確率差0**、2436→968 ms（2.52倍、各mode 4回）でした。キャッシュの意味的な同一性確認と4並列の速度測定を分けています。

これは1資料の再利用による改善で、新しい画像1枚のエンコードを3倍速くしたという意味ではありません。12問の精度診断時間と4問の再送時間も区別してください。旧前処理版の記録はQ4 2117→657 ms、Q8 2004→545 msです。前処理が変わったため新版の再測定値を示しており、厳密な新旧性能比較ではありません。

## 実験版GPU Flash Attention

`native/sarashina-fa160-b11042.patch`はCUDA/HIP共通のFAカーネルへ**head size 160**を追加します。GPU型番のホワイトリストはありません。上流のGPU能力判定に従ってtile/MMAを選択し、K/Vの型も上流の対応・F16変換経路を利用します。テンプレート追加だけでなく、Vのvectorized copyに必要な内部paddingを256へ修正しています。初期版は単token decodeで数値検査に失敗したため破棄し、修正版で再検査しました。vision encoderのhead sizeは72で、160対応は言語側の話です。

**実行を試せることと、検証済みであることは別です。** CUDAや他のAMD GPUもビルド・起動を試せます。既存の精度・速度表はR9700・F16 KVの結果で、他GPUでの確認範囲は次節に分けています。量子化KVについても一律に拒否しませんが、252ケースの検査対象はF16 KVであり、全GPU・全形式の動作保証ではありません。

- 既存`test-backend-ops`にdecode/prefill、KV view・permute、1/4 sequence、maskあり/なしの252ケースを追加。batch 1〜512、KV 113〜8192。**R9700対CPU比較252/252通過**（上流のNMSE上限`5e-4`）。検査時は追加のFA vector sliceテストを省略し、160次元で使うtile/MMA経路を検証します。
- **ビルド時のGPU数値検査は任意**（`--test-fa`）。未実行・失敗・別デバイスを理由にビルド成果物の利用を禁止しません。結果はmanifestやログへ記録し、ランチャーは注意を表示して起動を続けます。ファイル欠落・hash不一致・モデル/projector取り違えの検査は維持します。
- R9700のnativeログでFA有効、graph splits **2**を確認。公式b11042で無理に`-fa on`にした時のCPU fallback（66 splits）ではありません。他GPUのフィードバックでも、速度だけでなく数値検査とbackendログを添付してください。

### CUDA / 他AMD GPUへの公開検証経路（2026-09-21）

GPU型番・検査合格を起動条件にしない版で、次を確認しました。

| 対象 | 確認範囲 |
|---|---|
| CUDA、sm_89（RTX 4090向け） | CUDA Toolkit 13.2.51、GCC 14.3、CMake 3.31.6でserver・GPUライブラリ・演算検査プログラムをビルド／リンク完了。160次元tile/MMAも収録。**このマシンにはNVIDIA GPUがなく、GPU実行・数値・速度は未検証** |
| R9700、gfx1201 | 型番制限撤去後もF16 KVの演算検査252/252通過 |
| RX 7900 XTX、gfx1100 | 同じHIPビルドでF16 KVの演算検査252/252通過。Q4のテキスト推論・画像推論・API契約も実行確認 |

7900 XTXのQ4で同じ2入力をOFF→ON→ON→OFF、各2 warmups＋3回、F16 KV、4 slots・context 8192、キャッシュなしで確認しました。バックエンド直接呼び出しの中央値は、常識QA109 tokensが1並列27.3→26.1 ms、4並列77.3→71.8 ms。ニュース423 tokensが1並列70.5→63.7 ms、4並列261.2→230.4 ms。選択と短い自由生成2件は一致、候補確率は完全一致ではありません（先頭応答との差の最大0.13286）。FA ONでgraph splits 2。これは限定的な動作確認で、CUDAでの性能や全タスクの精度を示しません。

7900 XTXの画像診断も図形10/10・並列別画像3/3・長文State共有は成功しましたが、2画像順序は1/2、単色2問は1/2で、診断全体は不合格のままです。以前の制約を解消したと解釈しないでください。

記録は`results/sarashina-portability/`（Git対象外）。CUDA用の任意検査をGPUなしで実行した場合、0/0を合格扱いにせず記録することも確認しました。これによってビルド済み成果物や起動経路がロックされることはありません。

### R9700の同じビルド内のOFF/ON比較

モデルだけ常駐、mmprojなし、4 slots・context 8192、CPU 8 threads、prompt cacheなし、1回答token。tokenize/templateは計時前に済ませ、バックエンド`/completion`を直接呼びます。OFF→ON→ON→OFF、各2 warmups＋5回。**SystemOne全体のHTTP時間ではありません。**

| モデル・入力 | 1並列 OFF→ON | 4並列 OFF→ON |
|---|---:|---:|
| Q4・常識QA 109 tokens | 25.2→23.2 ms（1.08倍） | 69.9→63.6 ms（1.10倍） |
| Q4・ニュース423 tokens | 62.8→52.0 ms（1.21倍） | 263.8→206.5 ms（1.28倍） |
| Q8・常識QA 109 tokens | 22.3→20.7 ms（1.08倍） | 65.4→54.1 ms（1.21倍） |
| Q8・ニュース423 tokens | 54.5→43.6 ms（1.25倍） | 230.2→167.2 ms（1.38倍） |

離散的な選択と2つの短い自由生成は全modeで一致。ただし確率は同一ではなく、先頭応答に対する差の最大は1並列で0.04486、Q4長文の4並列で0.12461でした。OFFの1ブロック内でも先頭応答から最大0.09168の揺れを観測したため、並列全体の差をFAだけの誤差とはしていません。kernel合格はモデルの全logit・全文章での同一性を保証しません。

### R9700でのSystemOne API再評価

前掲の全test / heldoutと同じツール・件数・4同時リクエストで実験HIPビルドを再評価しました。**この表は公式checkpoint取得前の記録です。** 言語GGUF・FA160パッチは新版でも変更していませんが、全testの再走査は行っていません。mmprojはロード済みですが、全入力は画像なし。FA autoで有効、エラーは全タスク0件。各cellは**正解率 / HTTP p50 ms**です。

| モデル | JNLI | JComQA | livedoor | JMMLU | 顧客対応16例 |
|---|---:|---:|---:|---:|---:|
| Q4 | 33.13% / 107.3 | 85.78% / 95.1 | 69.2% / 278.0 | 56.8% / 104.5 | 16/16 / 82.6 |
| Q8 | 18.02% / 105.2 | 87.12% / 96.8 | 69.0% / 245.5 | 57.8% / 103.6 | 15/16 / 84.4 |

公式ランタイム比ではQ4のJComQAが139.0→95.1 ms、ニュースが408.7→278.0 ms。ただし**ROCmライブラリ・コンパイラ・ビルドも変わっているため、この差をFA単独の効果としません**。FAの寄与は上の同一ビルド比較で見ます。

スコアの大勢は維持しましたが、出力同一ではありません。特にJNLIは公式実行からQ4 284/2508件、Q8 65/2508件の選択が変化。Q8の顧客対応も1件悪化しました。LFM / ModernBERTを全面置換するものではなく、JNLI・確率安定性・選択肢順の弱点は残ります。

## 実験版のビルドと起動

**配布済み版を使う場合は、[Publicのmmprojと対応Pre-releaseの取得・起動手順](SARASHINA_RELEASE.md#利用方法)を参照してください。** mmprojのダウンロードにHFログインは不要です。以下の公式checkpoint取得・再変換は省略でき、必要なのは言語GGUF・mmproj・同名`.gguf.json`と対応ランタイムです。

### 1. 参照ファイルとGGUFを明示的に取得

モデルの利用条件を公式HFページで確認してアクセス承認を受け、HF CLIでログインしてください（[隔離したCLIの導入](SARASHINA_UPLOAD.md#3-hf-cliを隔離して用意しログインする)）。スクリプトが代わりに同意・申請することはありません。

```bash
.venv-hf/bin/python scripts/fetch_sarashina_reference.py --model sarashina
# Q8も必要なら --model sarashina-q8 で再実行
# 公式AutoProcessor / 参照生成まで比較する場合は --full を追加
```

`.cache/sarashina-official/`へ公式のconfig・前処理設定・**単一の全checkpoint（約7.60 GB）**を取得します。画像重みだけのshardは公式版にはありません。`--full`はtokenizer・Python・LICENSE等の小ファイルも取得します。CLIの保存済み認証を使い、URLやコードへtokenを埋め込みません。アクセス失敗時に公開クローンへfallbackせず、既存入力のhash不一致も上書きしません。

`.cache/runtime/selected.json`・`model.json`は変更しません。言語GGUFは引き続きmradermacherの固定revisionから取得します。

### 2. 別venvでprojectorを変換

既存の`.venv-modernbert`を変更しないでください。**CUDA/HIPの推論ランタイムと、変換用torchは独立**です。CUDA機でもCPU版torchでmmprojを変換できます。通常APIの実行だけならこのvenvは不要です。

```bash
python3 -m venv .venv-sarashina
.venv-sarashina/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-sarashina/bin/python -m pip install -r scripts/requirements-sarashina.txt

.venv-sarashina/bin/python scripts/convert_sarashina_mmproj.py \
  --config .cache/sarashina-official/config.json \
  --checkpoint .cache/sarashina-official/model.safetensors \
  --output models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf
```

参照モデルの生成までGPUで比較する場合は、CPU版の代わりにそのGPUに対応するtorchを導入してください。R9700測定ではAMD配布の`torch[device-gfx1201]==2.13.0+rocm10.0.0`を使用しました。CUDAのllama.cppを動かすためにROCm版torchを入れる必要はありません。

既存の出力・manifest・partialファイルがある場合は上書きせず停止します。configの隣の`preprocessor_config.json`も読み、公式の入力サイズ・hashを検証します。生成された`.gguf.json`には公式repo/revision・入力/出力hash・前処理版・tensor数を保存します。元の公開mmproj・旧クローン由来の修正版は置換しません。F32診断用には別ファイルへ`--outtype f32`で変換します。

### 3. 隔離したランタイムをビルド

Linux / WSL2、Python 3.12以上、GCC/g++（C++17・libstdc++）、CMake（3.24以上推奨）、Ninja、GNU patchが必要です。CUDAにはNVIDIAドライバーとCUDA Toolkit（nvcc・開発用ライブラリ）、HIPにはROCm開発環境が必要です。**PyTorchのCUDA wheelだけではビルド環境になりません。** CUDA Toolkitが対応するGCCを使ってください。OSパッケージやドライバーは自動インストールしません。Windowsネイティブ向けのDLLビルド手順ではなく、WindowsではWSL2を使用します。

`CMAKE` / `NINJA` / `CC` / `CXX` / `PATCH`で実行ファイルを指定可能。CUDAコンパイラは`CUDACXX`または`--cmake-arg=-DCMAKE_CUDA_COMPILER=...`、HIPコンパイラは`--cmake-arg=-DCMAKE_HIP_COMPILER=...`で渡します。

```bash
# RTX 4090（Ada、sm_89）。ほかのGPUではCUDA architectureを変更または省略
python3 scripts/build_sarashina.py --backend cuda --cuda-architectures 89 --jobs 8

# ROCm。architecture省略時はCMake/HIPの自動検出
CMAKE_PREFIX_PATH=/opt/rocm \
  python3 scripts/build_sarashina.py --backend hip --jobs 8
# 7900 XTXを明示: --hip-architectures gfx1100
# 両AMD GPU用: --hip-architectures 'gfx1100;gfx1201'

# GPUなしの動作確認用
python3 scripts/build_sarashina.py --backend cpu --jobs 8
```

CUDAのビルド対象はGPUに合わせて指定できます（型番による起動制限ではありません）。

| GPUの例 | `--cuda-architectures` |
|---|---|
| RTX 3090 | `86` |
| RTX 4090 | `89` |
| RTX 5090 | `120`（CUDA Toolkit 12.8以降。b11042が内部で`120a`へ変換） |

複数世代には`--cuda-architectures '89;120'`、省略時は上流CMakeの既定選択を使います。**こちらでコンパイル確認済みなのは`89`で、`120`のビルドやCUDA実機での推論は未検証**です。

GPUの数値検査をビルド後に行う場合だけ`--test-fa`を追加します（CUDA既定`CUDA0`、HIP既定`ROCm0`。`--test-device`で変更）。検査を省略したり、GPUのないビルド機でコンパイルしたりしても利用できます。`--test-fa`の失敗・タイムアウトは警告と記録になり、ビルド自体は成功としてmanifestを残します。**検査専用コマンドは不合格時に非ゼロ終了するので、CIでの判定も可能**です。

Python ROCm SDKを開発環境にも使うR9700向けの例（system ROCmの代わりに選ぶ場合。他GPUではTensileパスの`gfx1201`もそのGPUへ変更）:

```bash
.venv-sarashina/bin/python -m pip install \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --extra-index-url https://pypi.org/simple rocm-sdk-devel==10.0.0
.venv-sarashina/bin/python -m rocm_sdk init
export ROCM_PATH="$(.venv-sarashina/bin/python -m rocm_sdk path --root)"
export CMAKE_PREFIX_PATH="$ROCM_PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ROCM_LIBS="$(.venv-sarashina/bin/python -c 'import pathlib, _rocm_sdk_libraries; print(pathlib.Path(next(iter(_rocm_sdk_libraries.__path__))).resolve())')"
export ROCBLAS_TENSILE_LIBPATH="$ROCM_LIBS/lib/rocblas/library"
export HIPBLASLT_TENSILE_LIBPATH="$ROCM_LIBS/lib/hipblaslt/library/gfx1201"
python3 scripts/build_sarashina.py --backend hip --jobs 8 \
  --cmake-arg="-DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++" \
  --cmake-arg="-DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH" \
  --cmake-arg="-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM_PATH -I$ROCM_PATH/include"
```

今回の環境ではGCC 15.2 / HIP Clang 23 / ROCm SDK 10.0を使用し、OSにgccがないため展開済みGCCのパスを`CC` / `CXX`に指定、HIP flagsに`--gcc-toolchain=/path/to/gcc-root/usr`を追加しました。Python 3.14.4、torch 2.13.0+rocm10.0.0、Transformers 4.57.1です。既存torchの再利用でModernBERT側のパッケージを更新していません。

`stock-rocm`も選べます。これは修正したCPU/server/mtmdと**公式b11042の`libggml-hip.so`を再利用する構成で、GPU FA160には対応しません**。公式プラグインがない場合、保存選択を変更せず取得するには:

```bash
python3 - <<'PY'
import json
from scripts.setup_runtime import ROOT, install_runtime
spec = json.loads((ROOT/'scripts/runtime.json').read_text())['llama']
install_runtime(spec, 'linux-x86_64-rocm')
PY
python3 scripts/build_sarashina.py --backend stock-rocm --jobs 8
```

成果物は新しい**`.cache/sarashina-official-runtime/`**以下に隔離し、以前の`.cache/sarashina-runtime/`や`.cache/sarashina-portability/`は残します。ソースarchiveのhash、適用patchのhash、変更/生成ソースのhash、CMake設定、任意カーネル検査の状態、実行ファイル/共有ライブラリのhash、指定した実行環境をmanifestへ保存します。ソースやpatchを変更した場合はローカル編集を上書きせず停止するので、その実験のsource/buildを別の場所へ保管して再ビルドしてください。

公式前処理追加後にCPU・HIP（gfx1100/gfx1201）・CUDA（sm_89）を再ビルドしました。CPU/HIPのembedding照合、R9700 / 7900 XTXのFA160各252/252を確認。CUDAビルドのCPU前処理検査も15/15通過しましたが、**NVIDIA GPU実行は依然未検証**です。stock-rocmは旧版で画像推論を確認した構成で、今回の再測定対象ではありません。

### 4. 明示的に起動

```bash
unset LFM_MODEL LFM_MMPROJ
# CUDA版をビルドした場合
python3 scripts/run_sarashina.py --backend cuda --device CUDA0 --port 18080 --backend-port 18097
# HIP版: --backend hip --device ROCm0（7900 XTXがROCm1なら --device ROCm1）
# Q8: --model sarashina-q8
# CPU: --backend cpu --flash-attn off
# 公式GPUプラグイン: --backend stock-rocm --flash-attn off
# 画像なし: --text-only
# 切り分け: --flash-attn off --image-cache-mib 0
```

backend省略時はhip、Q4、**公式由来mmproj**、FA auto、画像キャッシュ128 MiB、4 slots・context 8192。manifestのhashと前処理の対応版を検証し、明示した環境変数がなければビルド時のlibrary/Tensileパスを使います。`--device`、`GPU_DEVICE`、backend別の既定（CUDA0 / ROCm0）の順でGPUを選びます。**検査時と異なるGPUでも起動を禁止しません**。stock-rocmでFA ONを指定した場合も、CPU fallbackの警告を表示して続行します。context拡大は`CTX_SIZE=32768`で指定可能です。

```bash
# 別ターミナル。元画像を使う例
python3 systemone_client.py --url http://127.0.0.1:18080 \
  --input examples/vision_gss_4.json --image sample_pics/20260911_image.png
```

通常の設定とログを上書きせず、backendログは`.cache/sarashina-official-runtime/llama-server.log`、一時slotは同フォルダ内の`slots/`です。同じ実験ランチャーを複数同時起動せず、停止してから切り替えます。Ctrl+CでAPIとbackendの両方を停止。通常のLFM / ModernBERTへ戻すにはそのランチャーを通常どおり起動します。

### GPU数値検査とフィードバック

モデルのダウンロード前でも演算検査を実行できます。起動の許可条件ではなく、結果を集めるための任意ツールです。

```bash
python3 -m tools.verify_sarashina_fa --backend cuda --device CUDA0 \
  --output results/sarashina-rtx4090-kernels
# AMDの場合: --backend hip --device ROCm0（またはROCm1）
```

`verification.json`・`fa160-test.log`・デバイス/バージョン情報を保存します。0件・CPUだけ・未対応・失敗を合格扱いにはせず、専用検査コマンドは非ゼロ終了しますが、ランチャーの使用には影響しません。検査結果はそのGPU・ドライバー・ビルド・F16 KVについてのもので、モデル精度や速度の保証ではありません。

フィードバックにはGPU名、OS、ドライバー/CUDAまたはROCmのバージョン、ビルドコマンド、`.cache/sarashina-official-runtime/build-cuda/jev-build.json`（HIPなら`build-hip`）、検査結果、下記OFF/ON計測の`measurement.json`とbackendログを添えてください。ローカル絶対パスや入力内容を含むことがあるので公開前に確認してください。**「FA enabled」だけではGPU実行を証明しません**。graph splits、CPU fallback警告、デバイスと数値検査を合わせて確認します。

### 再検証コマンド

```bash
python3 -m unittest discover -s tests -v
python3 -m tools.verify_api --url http://127.0.0.1:18080
# 単画像の成功と既知の不合格を両方記録（現状は非ゼロ終了）
.venv-sarashina/bin/python -m tools.verify_sarashina_vision \
  --url http://127.0.0.1:18080 --output results/sarashina-vision.json

# モデル/processorのレビュー後。検査用小ファイルをまだ取得していなければ先に実行
.venv-hf/bin/python scripts/fetch_sarashina_reference.py --full

# CUDA版のencoderをTorch CPU float32と比較（HIP版ならnativeパスとdeviceを変更）
.venv-sarashina/bin/python -m tools.verify_sarashina_embeddings \
  --allow-reviewed-code \
  --native .cache/sarashina-official-runtime/build-cuda/bin/test_sarashina_embedding --device CUDA0 \
  --flash-attn 1 --output results/sarashina-embeddings
# リサイズ境界・正規化だけを比較（14合成ケース＋縮小文書があれば1ケース）
.venv-sarashina/bin/python -m tools.verify_sarashina_embeddings \
  --allow-reviewed-code \
  --native .cache/sarashina-official-runtime/build-cuda/bin/test_sarashina_embedding --device CPU \
  --pixels-only --output results/sarashina-pixels

# --fullで取得後、ローカルモデル/processor Pythonを読んでから明示的に実行許可
# 現行言語GGUFと公式tokenizerの分割差を検出し、非ゼロ終了します
.venv-sarashina/bin/python -m tools.verify_sarashina_reference \
  --allow-reviewed-code --multi-image --url http://127.0.0.1:18097 \
  --output results/sarashina-reference.json
```

どちらの参照ツールも公式ファイルのhash確認後、ローカルAutoProcessorを`trust_remote_code=True, local_files_only=True, use_fast=False`で実行します。**embedding/画素検査にも`--allow-reviewed-code`が必要**です。生成比較はさらにモデルPythonを実行し、既定で`cuda:0`へ全checkpointをロードします（`--torch-device`で変更）。`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`も指定できます。生成比較は独立した公式token IDを使用し、不一致を終了コードとJSONへ記録します。生成文の一致件数はtokenizer検査・正答率と別に報告します。

速度ツールは自分でbackendを起動/停止します。ほかの推論を停止してGPUを空けてください。隣接する`jev-build.json`からlibrary/Tensileパスを読み、明示した環境変数を優先します。既存の出力ディレクトリを再使用すると測定ログを上書きするため、比較ごとに別名を指定します。下記はCUDAの例で、HIPの場合は`build-hip`と`--device ROCm0`（またはROCm1）へ変更します。

```bash
# FA単独。モデルのみで実行可能。--dataset-casesを省略すれば追加データ取得なし
python3 -m tools.benchmark_sarashina_fa \
  --server .cache/sarashina-official-runtime/build-cuda/bin/llama-server \
  --device CUDA0 --model models/sarashina2.2-vision-3b.Q4_K_M.gguf \
  --dataset-cases --output results/sarashina-rtx4090-fa

# 画像キャッシュ単独。--workers 1で確率の逐次回帰確認
python3 -m tools.benchmark_image_cache \
  --server .cache/sarashina-official-runtime/build-cuda/bin/llama-server --device CUDA0 \
  --model sarashina --mmproj models/sarashina2.2-vision-3b.mmproj-jev-official-f16.gguf \
  --flash-attn on --image sample_pics/20260911_image.png \
  --output results/sarashina-image-cache
```

FA計測は既定29197、画像キャッシュ計測は19080/19097。別ポートは`--port` / `--backend-port`で指定します。GPUが変わった場合は速度だけでなく数値検査もやり直してください。元画像・縮小画像は同梱していません（元画像SHA256 `52aac88e27135ff840ae243216f7cd86aa8ded7bf686c337e02b19a6b04d7534`）。

## 評価の再現と記録

APIを起動した状態で、モデルごとに別の出力ディレクトリを指定します。

```bash
python3 -m tools.verify_api --url http://127.0.0.1:28080
python3 -m tools.benchmark_jglue \
  --url http://127.0.0.1:28080 --output results/sarashina-q4-jglue
python3 -m tools.evaluate_heldout_tasks \
  --url http://127.0.0.1:28080 --limit 500 --seed 0 \
  --output results/sarashina-q4-heldout
```

候補順の診断はAPIを停止し、同モデルの`run_server.sh`だけを起動して実行します（APIのslot管理と競合させないため）。

```bash
python3 -m tools.evaluate \
  --url http://127.0.0.1:28097 --output results/sarashina-q4-sanity.json
```

公式ランタイムの生ログ・応答・summaryは`results/sarashina-comparison/`、CPU fallbackの切り分けは`results/sarashina-flash-attn/`、修正後の検証は`results/sarashina-repair/`です（すべてGit対象外）。最終版の主な記録は`final-cache-{q4,q8}/`、`fa160-final-{q4,q8}/`、`final-embeddings/`、`pixels-boundaries/`、`final-{q4,q8}-gss-*.json`、`q{4,8}-fa-{jglue,heldout}/`、`regression-*.log`です。初期の失敗・中間ビルドの記録も消さずに残しています。配布されるのは本ページの要約と再実行ツールです。評価条件は[JGLUE](JGLUE.md)・[評価ツール](EVALUATION.md)も参照してください。

公式版への移行記録は`results/sarashina-official/`と`.cache/sarashina-official-review/`です。`pixels-final/`、`embeddings-{cpu,hip}/`、`reference-q{4,8}.json`、`q{4,8}-gss-*.json`、`cache-q4/`、`cache-q8-final/`、`cache-q4-serial/`に数値・全応答・hashを保存。Q8キャッシュの初回記録は別GPUの検査と同時実行したため、表にはGPUを空けた`cache-q8-final/`の再測定を採用しました。初期の画素照合失敗も残しています。

単体テスト84件、新HIPビルドでR9700 / 7900 XTXのFA各252件、Sarashina Q4/Q8のAPI契約を通過。画像精度・tokenizer診断は前述のとおり不合格を含みます。LFM / ModernBERTと通常の保存済みランタイム・モデル選択は変更していません。以前のLFM画像＋text cache・全方式APIの検証記録も保持しています。

モデルとGGUFのカードはMIT Licenseを表記しています。2026-09-21に公式MIT原文も取得し、SigLIP由来部分のApache-2.0、HFのアクセス条件、llama.cppの第三者コードを含めた[配布時の必要表記と未確認事項](SARASHINA_DISTRIBUTION.md)を整理しました。本リポジトリはモデルを同梱せず、明示的な取得・変換手順を提供します。
