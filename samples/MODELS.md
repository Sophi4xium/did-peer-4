# モデルの入手(このフォルダには同梱していません)

```sh
mkdir -p models && cd models
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip -d buffalo_l
# 使うのは buffalo_l/w600k_r50.onnx(特徴抽出, 約174MB)
```

InsightFace の事前学習済みモデルは、非商用(研究)利用に限られるのが一般的です。
利用前に InsightFace 公式の注意書きを確認してください。
