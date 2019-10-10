# Auto augment Researching
9/27〜18時ごろ動作
ランダム探索
最大Accuracy 28.9
要分析

WideResNet
スパコンで動作
1エポックに40分
GPU使っているか今度確認

---

## 参考にできそうな前処理リスト
* CNNで使われる前処理
http://iwiwi.hatenadiary.jp/entry/2016/12/31/162059

* numpyでのData augment
https://www.kumilog.net/entry/numpy-data-augmentation#Random-Crop
---
reference:
transformations.py
run.py
-->https://github.com/hongdayu/autoaugment

wide_resnet.py
->https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py#L4
