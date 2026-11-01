# back：历史流程

paper-v1 是主流程迁移前的完整源代码快照，对应本地提交 fc33c8c / GitHub提交 cce42c4。包括原八阶段论文流程及当时的单U-Net ImageCHD适配、配置、入口、文档与测试。

当前工作请使用项目根目录的README。主程序不导入这里；普通安装包和默认测试不会包含此归档。

需要研究旧实现时进入paper-v1，从其中README和docs/paper-mapping.md开始。可使用已有Python环境运行该目录下的start.py；它会把归档自己的src置于导入路径首位。不要把归档包安装覆盖到当前主环境。

归档不包含原始影像、缓存、权重、虚拟环境；旧流程要求的manifest/标签规范也不会自动套用到当前七结构缓存。已在独立目录验证归档的49项测试通过。

`custom-unet-v1/` 保存这次原作者结构对齐前的自定义 U-Net 和配置；完整旧六阶段流程可在 Git 提交 64a28a4 中查看。

`preprocess-before-excel-20261013/` 保存加入 Excel 间距重采样之前的预处理入口及依赖，来源提交 `c414342`。可独立运行，说明与文件校验值见该目录的 `README.md` 和 `snapshot.json`。

`before-paper-alignment-20261013/` 保存 CHD 论文核心结构调整前的完整工作区代码、配置、测试和文档，共 125 个文件，包含刚加入的 Excel 重采样和原有服务器流程。它包含未提交修改，不只是 Git HEAD。`snapshot.json` 记录逐文件 SHA-256；2026-10-15 开始修改前已再次核验。该备份不含数据、权重、虚拟环境和更早的 back 归档。

`before-threefold-training-20261015/` 保存三折训练补齐前的完整工作区，共 134 个文件，包含已对齐的 CHD 论文主干、激活重算、Excel 重采样和原单折六阶段训练/服务器代码。逐文件 SHA-256 在 `snapshot.json`，使用说明在 `SNAPSHOT.md`；不包含数据、权重、环境与更早备份。
