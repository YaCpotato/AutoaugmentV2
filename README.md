# 深層学習による多クラス分類の前処理の最適化
## 研究の背景
機械学習には大きめなデータセットが必要とされているが、そのデータセットは、集められたとしても、はじめから学習しやすいデータであるとは限らない。  
  
現に世のデータサイエンティスト達はデータセットを学習しやすくするために人の手を加える前処理という工程を行なっている。前処理はほぼ手作業となる場合が多く、データサイエンティスト達の業務のほとんどの時間を奪っていると言っても過言ではない。  
さて、画像の機械学習における前処理の工程を詳しく説明する。
データの中にある、学習の役に立たない画像や、ターゲットとなるオブジェクトが画像の一部分によっていたりする時に、中央に持ってきたり、拡大縮小をするといった操作を行う。そうすることで、機械の学習時にターゲットを捉えやすくする。というのが主な加工である。細かい動作になると、意図的にデータを欠落させたりするRandom ErasingやCutoutなどがある。その他、画像のCNNで使われる前処理は下記を参照。  
http://iwiwi.hatenadiary.jp/entry/2016/12/31/162059  
https://www.kumilog.net/entry/numpy-data-augmentation#Random-Crop  

上記のURLだけでも非常に多岐にわたる前処理法があり、どれを選んだら良い結果に繋がるのか確証がない。  

## 研究の目的
本研究では、この前処理を自動化する手段として、Googleが論文を出した**Autoaugment**という技術を用いて、  
論文に即し、多クラス分類問題の精度向上、最適化を図るものとする

## Autoaugmentとは
* RNN(Controller RNNという用法)を用いてデータセットに対する画像加工の方針(Policy)を予測する方法。1つのPolicyは1データセットに対して適用

* Policyの中にはSubPolicyがあり、データセットを学習する際のバッチサイズ１つにつき１つ適用

* SubPolicyにある画像加工の種類は論文では16種類, 重みは11段階, この画像加工が行われる確率として10段階

この
* 画像加工の種類
* 重み
* 確率
を、まとめて、Operationと定義する

予測したPolicyをデータセットに適用し、精度を向上させたいモデルで学習を行う。このモデルのことをChild Modelと定義する。
Child Modelが学習を終えた後のValidation AccuracyをRNNに入力すると、RNNは選んだ組み合わせの重みをValidation Accuracyに応じて更新する(学習)  

繰り返すことでValidation Accuracyが高かった組み合わせに対応した重みが大きくなり、データセットに対して最適なPolicyが予測できるようになる。

## Autoaugment論文の概要(比較したいところを重点的に)
1-PolicyにつきSubPolicyは5つ。
SubPolicyの中には2つのOperation。

Controller RNN：6層(16,11,10,16,11,10)
Controller Epochs:
hidden-units:100
ChildModel：WideResNet-40-2
Epochs:120

本番モデルは変更：WideResNet-28-10

Error rate 2.6±0.1 (Accuracy:97.4±0.1)
Base-line(今までの最高Error rateが3.9)

### WideResNet
ResNetに比べ,深さを少なくし,特徴フィルタ数を増やすことでパラメータ数を増やしたモデル。
GPUを用いての演算において,ResNetより所要時間を少なくできるうえ,より高精度かつ高速になるモデル
[WideResNetのお勉強](http://robonchu.hatenablog.com/entry/2018/12/01/185537)
### ResNet
ある層で求める最適な出力を学習するのではなく、層の入力を参照した残差関数を学習するアイデアのもと実装されたモデル。
ただ単に層を深くするだけでは精度の向上はできなかったのだが、ResNetはそれを、残差ブロック(Residual Block)とShortcut Connectionという技術を用いて限界まで層を深くすることに成功した。
[Residual Network(ResNet)の理解とチューニングのベストプラクティス](https://deepage.net/deep_learning/2016/11/30/resnet.html)
[ResNetの論文を読んだ](https://www.kumilog.net/entry/resnet-paper#Identity-vs-Projection-Shortcuts)
#### Residual BlockとShortcut Connection
Residual Blockは、畳み込み層とShortcut Connectionの2つの枝からなっている。
畳み込みに入る前に、入る前のパラメータをIdentify関数として保持しておく、実際に畳み込みが終わったものを一旦ReLU(活性化関数)にかけた後、保持しておいた畳み込み前のパラメータと足し合わせて、再度ReLU(活性化関数)にかける。
##### Plainアークテクチャ
![](https://i.imgur.com/EBxBIkn.jpg)
##### Bottleneckアーキテクチャ
![](https://i.imgur.com/fKB9ubO.jpg)
3×3の畳み込み2層を組み込む場合、1×1,3×3,1×1の３層構成にした方が、  
層の深さによる精度劣化を防ぐことができる。計算コストは同等
#### Shortcut Connection
畳み込みで学習すべき残差写像が畳み込みの前と同じ次元の場合, そのまま足す。  
次元が異なる場合は,ゼロパディングをする、もしくは畳み込み前のパラメータの線形射影を足す。
#### Optimizerの選定
[ResNet元論文](https://arxiv.org/pdf/1512.03385.pdf)
Momentum-SGDを使用
[Torchのブログ](http://torch.ch/blog/2016/02/04/resnets.html)でも以下の結果になっている
|Solver|Testing error|
|----|----|
|RMSprop|0.0697|
|Adadelta|0.0768|
|Adagrad|0.0888|
|Momentum-SGD|0.1145|

#### SGD(確率的勾配降下法)

$$
w_{t+1}  \gets w_{t} - \eta \frac{\partial E(w_{t})}{\partial w_{t}}
$$

$$
w_{t+1}  \gets w_{t} - \eta \nabla(w_{t} - E(w_{t}))
$$

wは重み、ηは学習率、tは時間、Lは交差エントロピー誤差,Eは誤差関数
∇(w-E)は確率的勾配を表す
学習率η(重みの更新の幅を制御する)を作用させ勾配∂L/∂ωの逆方向にパラメータを変化させる  
[Qiita 【機械学習】パラメータ更新の最適化](https://qiita.com/m-hayashi/items/dab9d2f61c46df0a3a0a#momentum%E6%B3%95)
##### Momentum法
パラメータの最小値を物理法則的なアプローチで解析する手法らしい。
$$
v \Leftarrow av - \eta \nabla \frac{\partial E(w_{t})}{\partial w_{t}}
$$
$$
w_{t} \Leftarrow w_{t} + v
$$
#### Momentum-SGD
通常のSGDに慣性項(Momentum)を追加したもの
$$
w_{t+1} \gets w_{t} - \eta \frac{\partial E(w_{t})}{\partial w_{t}}  + a\Delta w_{t}
$$
[Qiita: Optimizer : 深層学習における勾配法について](https://qiita.com/tokkuman/items/1944c00415d129ca0ee9#momentum-sgd)
#### Batch-Normalization
```
勾配消失・爆発を防ぐための手法
```
これが出る前は、
* 活性化関数を変更する（ReLUなど）
* ネットワークの重みの初期値を事前学習する
* 学習係数を下げる
* ネットワークの自由度を制約する（Dropoutなど）
で対応してきたが  
Batch-Normalizationは、学習プロセスを全体的に安定化させて学習速度を高めることに成功している
[DeepAge:Batch Normalization：ニューラルネットワークの学習を加速させる汎用的で強力な手法](https://deepage.net/deep_learning/2016/10/26/batch_normalization.html)
##### Normalization(正規化)
```
データの尺度を統一すること
最低０、最高１になるようにデータを加工すること
```
##### Standardization(標準化)
```
平均０、標準偏差１になるように、データを加工すること
```
##### Regularization(正則化)
```
？？？？？？
```
#### 共変量シフト
```
訓練データのサンプリングと予測データの入力の分布に偏りがあり、アルゴリズムが対応できなくなること
```
#### 内部の共変量シフト(Internal Covariate Shift)
```
隠れ層において各層とActivationごとに入力分布が変わってしまうこと
```
#### Batch-Normalizationのメリット

1. 大きな学習係数が使える
これまでのDeep Networkでは、学習係数を上げるとパラメータのscaleの問題によって、勾配消失・爆発することが分かっていた。Batch Normalizationでは、伝播中パラメータのscaleに影響を受けなくなる。結果的に学習係数を上げることができ、学習の収束速度が向上する。  
2. 正則化効果がある
L2正則化の必要性が下がる
Dropoutの必要性が下がる

##### L1,L2正則化
機械学習において、最小化がしたい関数
$$
\min f(x)
$$
に対し、L1正則化項を足すものをL1正則化、不要な説明変数をそぎ落とす次元圧縮のために用いられる。L1正則化を用いた学習ではパラメータwiが0になりやすいためです。パラメータwiが0とされた説明変数は目的変数に対して無関係であることを意味する。  
この正則化を施した線形回帰をLasso回帰という。

一方、L2正則化項を足すものをL2正則化、これを用いたモデルのパラメータの方がより滑らかで表現力に優れているため、モデルの過学習を避けるために用いられる。  
この正則化を施した線形回帰をRidge回帰という
$$
\text{L1正則化項：}\lambda \sum_{i=1}^{n} |w_{i}|
$$
$$
\text{L2正則化項：} \frac{\lambda}{2} \sum_{i=1}^{n} |w_{i}|^2
$$
$$
\text{L1正則化：} \min f(x) + \frac{\lambda}{2} \sum_{i=1}^{n} |w_{i}|^2
$$
$$
\text{L2正則化：} \min f(x) + \frac{\lambda}{2} \sum_{i=1}^{n} |w_{i}|^2
$$

[正則化の種類と目的](https://to-kei.net/neural-network/regularization/)
##### Dropout
確率でパラメータを0にする。過学習を抑える働きがあるが学習速度が遅くなる。
Batch-Normalizationと混在するとやばいからやるならBatch-Normalizationを選べという記事もあるが別にそれほどやばくなさそう。現に過学習が、BNのみより、BN + D のほうが抑えられている。ちなみに時間はBN使った方が早い

##

###### tags: `Templates` `Book`
