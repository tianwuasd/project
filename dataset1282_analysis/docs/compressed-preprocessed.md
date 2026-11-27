# 直接读取B2ND压缩预处理

Xu的新预处理入口接受`chd-offline-npy-v1`和`chd-offline-b2nd-v1`，通过CHT现有读取器按区域读取。保持原float32 CT、有符号标签、原计划/训练统计、空间记录、病例划分和前景采样坐标。转换不执行重采样或重新归一化。

CHT代码需要包含`preprocessing/arrays.py`的CompressedArray，以及offline_store.save_case的压缩写入参数。这套接口已在`codex/chd-compressed-input-20261127`分支提供。服务器环境需能导入blosc2及原CHT预处理依赖。缺少接口时直接停止，禁止把旧NPY直接改扩展名。

先等待当前训练、预测和评价退出，再更新代码和转换。已冻结的旧任务继续使用原代码与原输入；转换后的数据和代码变化需要新的任务目录。原预测结果与评价报告保持原样。

## 转换

`scripts/compress_preprocessed.py`参数为`--preprocessed SOURCE/prepared.json --cht-code CHT_CODE --output NEW_ROOT`。默认4MiB压缩块、ZSTD级别3、单压缩线程；内存边界为100GiB与整机80%中较小者，整机可用不足20%时停止并保留已提交病例。

每个病例分块写入B2ND，重新打开后逐字节核对CT与标签、几何/监督记录和前景坐标。全部病例通过后写出新的catalog.json、prepared.json与compression-complete.json。通过`--resume`接续：已经完成的病例不重写；未完成病例重新提交。输出目录与原NPY分离，峰值磁盘额外占用为压缩输出体积，实际比例与病例相关。

加`--delete-npy-after-verify`时，在全部核验成功、目标文件身份未变且未发现活动引用后，只删除源清单内的image.npy和label.npy。原始CT/GT、模型、结果、JSON记录、locations.npz和B2ND均保留。中断的删除可用相同参数加`--resume`接续。删除NPY后，读取应使用新prepared.json；旧NPY任务不能直接续跑。若另有硬链接，移除一个文件名不一定立即释放对应磁盘空间。

服务器可执行`bash scripts/data4_server.sh compress --delete-npy-after-verify`，日志重定向至course/chd1282/compress.log。默认输出为原数据目录名追加`-b2nd`。`CHD_COMPRESSED_OUTPUT`可指定另一个独立目标。

## 切换与单例检查

设置`CHD_PREPROCESSED`为新B2ND根目录，再执行`bash scripts/data4_server.sh check`和`bash scripts/data4_server.sh short --device cpu`。短测仍固定1例，临时产物退出自动清理，日志和验收记录保留。`CHD_PREDICTION_OUTPUT`可指定新的预测结果目录，避免覆盖既有任务。

本地1例真实64³区域通过原CHT读取、Q pooled_block、数值与标签逐字节一致、几何/监督/采样坐标、重复转换、核验后NPY删除、删除后Xu读取和重复接续。约1.57MB原数组压缩为0.85MB，仅代表该小区域。未执行全量转换、Linux删除或完整六模型推理；STU的MONAI取块入口本轮未测试。
