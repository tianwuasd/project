# 复现设计 · 2026-10-07

目标：根据用户提供的 Xu 等人 2023 年论文，创建结构清晰、可训练、可推理、可测试的独立研究实现，并上传至 `tianwuasd/project/chd-ct-reproduction`。本地位于 `D:/code_project/chd-ct-reproduction`。

## 方案与边界

选择分阶段 PyTorch 实现。只搭 3D U-Net 会遗漏文章的主要贡献；直接使用其他通用分割框架会掩盖论文特有的融合和解剖特征。因此实现六个 3D 分割任务、2D U-Net 与双向 ConvLSTM、空间融合、图与形态特征、可追溯规则接口。

论文的完整决策树、阈值、初始血管标注规范和图路径细节没有全部公开；这些部分使用明确命名的近似或外部输入，不能称为作者原代码。完整临床复现还需要原始队列、标签和训练资源。

## 数据流与接口

NIfTI → RAS+ 方向与一致网格校验 → CT 强度归一化 → 双尺度 ROI → 四个 ROI 内 3D 模型 + 血池切片序列 → 兼容类别投票 → 血池约束区域扩张 → 物理单位形态/中心线特征 → 三值规则结果 JSON。

* manifest CSV 使用 `case_id,patient_id,split,image,label,initial_label`；路径相对 manifest，按患者隔离，拒绝重复和跨集合患者。
* 3D 数组轴固定为 RAS+ 的 X/Y/Z；2D 模型沿 Z 切片。输出恢复输入文件的原始轴方向与 affine。
* 标签内部编号自定义并公开：BG=0, LV=1, RV=2, LA=3, RA=4, AO=5, PA=6, MYO=7, SVC=8, IVC=9, PV=10。
* init 任务使用 BG/LV/RV/LA/RA/MYO/初始大血管七类，必须提供单独 initial_label，不能从全心标签伪造真实标注。
* 所有保存的模型包含 stage/config/seed/训练病例标识；推理检查 stage 与配置兼容性。
* 诊断是研究候选结果，支持 positive/negative/indeterminate。未实现的精确解剖定位和未公开阈值保留 unknown。

## 验证

CPU 缩小配置验证八个训练步骤（血池网络拆分为两个阶段）、七路推理、checkpoint、NIfTI 空间恢复、特征和规则输出。合成标签仅用于接口与数值验证。另测缺失类别、空 ROI、各向异性间距、患者泄漏、三值规则、互斥组和不可信 checkpoint 的失败路径。真实数据训练和临床性能不属于已完成验证。
