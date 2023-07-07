# YOLOPv2-ONNX-Sample
[YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)のPythonでのONNX推論サンプルです。<br>
ONNX変換自体を試したい方はColaboratoryで[Convert2ONNX.ipynb](Convert2ONNX.ipynb)を使用ください。<br>

https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/assets/37477845/ab214b79-4d08-4cc1-b6be-0f5f16501fe1


# Requirement 
* OpenCV 3.4.2 or later
* onnxruntime 1.9.0 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py
```
* --video<br>
動画ファイルの指定<br>
デフォルト：video/sample.mp4
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：weight/YOLOPv2.onnx ※ファイルが無い場合ダウンロードを行います
* --input_size<br>
モデルの入力サイズ<br>
デフォルト：640,640
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.3
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.45

# Reference
* [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
YOLOPv2-ONNX-Sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[中国・重慶　高速道路を走る](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002050453_00000)を使用しています。
