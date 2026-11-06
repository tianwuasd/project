# 工程适配说明

本项目采用engineering-project-reference的独立入口、共享模块、配置合同、日志状态和逐阶段验证，并按只读数据审计裁剪。环境、代码、输入数据和分析产物有独立身份；没有加入当前不需要的训练面板、模型权重、GPU供数或空目录。

模块边界：inputs/metadata解析来源，geometry/voxels/labels只计算，engine负责调度与完整提交，observability记录锁和状态，reporting只消费已提交结果并生成本地报告，maintenance仅管理代码release。没有把资源策略写入医学阈值，也没有让数据质量复核改变以后病例纳入。

稳定扩展点为CaseInput、配置schema、analysis protocol和病例结果合同。增加后续预处理时使用新协议与单独产物，不修改本轮只读审计结果或绕过输入身份。

公开工程只包含白名单源码、锁依赖、默认配置、说明及运行时生成合成资料的测试。真实路径、研究数据、Excel个体信息、运行日志和分析图表留在用户配置的私有工作区。
