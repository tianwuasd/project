# data4正式数据与无血池预测

运行入口为`bash scripts/data4_server.sh <动作>`。源码位于code/dataset1282_analysis，环境位于env/dataset1282_analysis，原件位于data/dataset1282_CHD_CT_back，预处理描述位于data/chd-segmentation/preprocessed/CHD1282-all-20261125/prepared.json。模型、检查点和结果在result/chd1282；日志与验收记录在course/chd1282。路径均相对于/data4/zhangruichen。

完整1174例描述尚未写出时，所有动作直接停止，不能清理旧目录。

1. `check`调用原CHT读取器，检查1174例的病例/患者、原件引用及数组头，实际读取最多两例的标签属性。转换结果已保留原590训练统计和缺标语义。
2. `models`从原两份完成预测登记六个模型及其冻结身份，将同盘权重建立硬链接到result/models；保留原正式训练的六个最新检查点到result/checkpoints。核对新数据计划/语义，后续预测无需读取旧训练或旧预处理目录。
3. `short`固定一例source_test，执行六模型无血池/无血管修正预测及原标签评价，成功后保存course/acceptance.json；退出自动清理专用短测目录。该动作不会启动全456例。
4. `preview`显示旧input、work、718导出及已知失败短测清理范围，并检查当前模型与验收收据。其他名称含1282的目录只列出，不凭名称删除。
5. `clean`先迁移原实验结果与唯一模型状态到result/runs，按阶段把最新日志移到course/history。保留原Excel、完整原件清单与必要处理元数据，清理旧数组缓存及冗余网格文件。同当前六模型内容相同的旧权重共用硬链接，其他唯一检查点保留。原始data和当前preprocessed不删除。实际清理仅在指定Linux目录执行，拒绝本用户仍有进程引用旧目录。
6. `predict`从当前preprocessed读取并重新预测全部原冻结留出病例。六分支临时输出在病例结束后清理，仅保留最终压缩分割、质控和少量状态。`evaluate`计算真实原标签指标。相同任务续跑使用`predict --resume`，输入条目改变时使用独立输出任务。

原已发布代码归档分支为archive/chd1282-before-data4-20261125。代码更新只含源码、配置、测试和公开说明；研究数据、权重和私有验收材料不进入GitHub。

本地覆盖一例真实区域的CHT读取、输入描述切换、删除旧源后读取、crop64真权重前向及压缩输出。完整六模型验收需在服务器short动作完成；未声称本地具备其他五个完整权重。
