# ランタイム用の第三者表記

`prepare_sarashina_release.py`がCPU / HIP / CUDAアーカイブへ必要なものを選んで同梱します。取得元・SHA256・対象backendは[`sources.json`](sources.json)に記録しています。**Jev Local独自部分のライセンスを設定するものではありません。**

- GCC 13.3: Ubuntu 24.04のcopyright、GPLv3、Runtime Library Exception。
- HIP: ROCm SDK 10.0.0のHIP / hipBLAS / rocBLAS等の原文と、ビルドしたLLVM revisionのLLVM / AMD device libraries原文。
- CUDA: CUDA 13.2.51のEULA・第三者表記、CCCLパッケージ13.2.27内のCCCL 3.2.0（revision `477f8bcb27eb80c28e18bbd97e5dde80ecfc648b`）のライセンス。`NOTICE.CCCL-headers`はインストール済みヘッダの著作権を含むコメントをそのまま集約しています。未使用ヘッダの表記も含む保守的な一覧です。

llama.cppと同梱third-partyコードのLICENSE / NOTICEは、SHA256固定のb11042ソースアーカイブから梱包時に取得します。CUDA / ROCmの共有ライブラリやドライバーは配りませんが、コンパイルで組み込まれるヘッダ・device runtime部分の表記は残します。
