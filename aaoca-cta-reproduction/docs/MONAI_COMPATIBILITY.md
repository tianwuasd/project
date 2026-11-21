# MONAI真实实现兼容依据

作者环境requested为1.4.0，本机actual为1.5.2，实际使用同一MONAI SENet154实现。2026-11-21只读比对官方1.4.0源码与本地1.5.2必要路径，7份文件的19个计算定义全部一致。比对用Python AST，仅忽略docstring、类型标注和源位置；构造默认值、模块名称、参数、初始化、运算与forward均参与比较。没有创建模型、forward、加载权重、安装旧包或生成缓存文件。

|官方1.4.0文件|一致的定义|
|---|---|
|[senet.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/nets/senet.py)|SENet、SENet154，包括构造、Kaiming/BN初始化、features/logits/forward|
|[squeeze_and_excitation.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/blocks/squeeze_and_excitation.py)|ChannelSELayer、SEBlock、SEBottleneck|
|[convolutions.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/blocks/convolutions.py)|Convolution|
|[acti_norm.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/blocks/acti_norm.py)|ADN|
|[factories.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/layers/factories.py)|LayerFactory、split_args、conv_factory、batch_factory、dropout_factory、maxpooling_factory、adaptive_avgpooling_factory|
|[utils.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/layers/utils.py)|get_act_layer、get_norm_layer、get_dropout_layer|
|[convutils.py](https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/layers/convutils.py)|same_padding、stride_minus_kernel_padding|

只读核对入口为`tests/inspect_monai_compatibility.py`，官方源通过临时7897代理在内存读取，未保存原文件或计算整文件摘要。网络同参构造的参数命名与维度依据这些相同构造定义；实际加载时继续严格核对state_dict键及张量形状。固定配方为三维、单通道、单logit、3/8/36/3阶段、64组、reduction16、dropout0.3。

源码一致支持复用现有算法/API，不证明不同PyTorch、后端、设备及数值轨迹等价。运行记录同时保存`monai_requested`、`monai_actual`和兼容依据；恢复比较算法合同和真实权重，软件版本字符串不单独作为拒绝条件。模型计算、真实权重及恢复仍需独立运行许可和验收。当前没有模型forward或训练通过证据。
