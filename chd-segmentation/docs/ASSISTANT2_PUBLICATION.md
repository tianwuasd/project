# 助手2代码发布与服务器读取路径

助手2在D:/projects/project/chd-segmentation-assistant2的独立工作树维护，工作分支codex/chd-assistant2-offline-preprocess-20261115。在线流程冻结在codex/chd-assistant2-online-frozen-20261115，提交69289532a3da2c94bc72769f5726ca260f6980a7。共享发布仓库为tianwuasd/project，发布目标为main中的chd-segmentation目录。工作树分别使用分支及Git索引，发布不修改助手1和助手3工作目录。

## 发布与核验

完成实际有界验收及一次整体审阅后，在最新公开main树上整合本项目源码。助手1已发布的离线合同及其他项目保留；逐文件检查新增、修改与删除。公开文件必须符合update.py白名单及200个源码文件上限。私人AGENTS、患者分组、清单、配置、报告、权重、vendor、env和结果不进入公开树。

发布前重新查询main，使用该完整提交作为父提交，并以期望旧SHA更新main；拒绝并发推进，不强制重写。发布后重新查询main及源码子树，核对每个Git blob和其余项目。公开提交是整个共享仓库的提交，本地开发提交是平铺分割源码提交，两者分别记录。

## 切换到main的独立部署

本次发布的update.py固定读取main。它核对本地分支也是main，--force仍保留分支与活动代码锁检查。助手1默认部署参数继续为/data4/zhangruichen/code及/data4/zhangruichen/course/chd-segmentation。

助手2使用新的独立普通Git仓库/data4/zhangruichen/code/chd-segmentation-assistant2-main-repo，更新回执及代码锁目录为/data4/zhangruichen/course/chd-segmentation-assistant2。原助手2功能分支检出继续保留，不在其上调用原来的固定分支更新器。新启动命令明确使用新main检出中的入口。原任务需要其冻结代码、原环境、数据描述及检查点。

以下命令由用户在服务器执行；PUBLIC_SHA使用交付回执中的完整公开提交。本会话不登录服务器。

```bash
PUBLIC_SHA='<交付回执中的完整main提交>'
REPO=/data4/zhangruichen/code/chd-segmentation-assistant2-main-repo
RUNTIME=/data4/zhangruichen/course/chd-segmentation-assistant2
PYTHON=/data4/zhangruichen/env/chd-segmentation/bin/python

# 首次安装：目标目录必须不存在。
git clone --filter=blob:none --sparse --branch main https://github.com/tianwuasd/project.git "$REPO"
git -C "$REPO" sparse-checkout set chd-segmentation
"$PYTHON" -B "$REPO/chd-segmentation/update.py" --repo "$REPO" --runtime "$RUNTIME" --sha "$PUBLIC_SHA" --check
"$PYTHON" -B "$REPO/chd-segmentation/update.py" --repo "$REPO" --runtime "$RUNTIME" --sha "$PUBLIC_SHA" --force
"$PYTHON" -B "$REPO/chd-segmentation/run.py" --code-info
```

后续更新使用最后三条命令，PUBLIC_SHA改为相应确定提交。共享main更新器默认参数保持助手1现有用法；助手2必须保留显式repo/runtime。--force仅令本项目受管理源码使用远端内容，保留其他项目文件及暂存记录，并拒绝活动计算期间更新。更新器在安装后再次查询官方main；若远端推进则停止并报告该状态。

## 数据和运行目录

两个模型的work_root仍使用/data4/zhangruichen/course/chd-segmentation，以共享resources登记CPU、RAM及GPU额度。助手2的准备描述、来源日志和专用配置保存在其assistant2子目录。Q使用本次新增的三个阶段配置，统一run_dir为assistant2/runs/CHD1282-Q-20261115，parents指向该目录的上游best.pt；原私有包只提供冻结组别和prepare配置。Q_CONFIG_DIR可明确选择模型配置目录。MedSAM2的run_dir也使用独立assistant2/runs目录。更新回执的RUNTIME与模型资源work_root分别指定。

离线数组统一位于/data4/zhangruichen/data/chd-segmentation/preprocessed，按处理合同、折、统计来源及内容身份管理。模型训练命令见MODELS_ASSISTANT2.md，完整准备与运行脚本为scripts/CHD1282-Q-20261115_run.sh及scripts/ImageCHD-CHD1282-MedSAM2-20261115_run.sh。官方源码、依赖、权重和私有Q材料按ASSISTANT2_SERVER_FILES.md分别迁移；代码更新不会传输这些材料，也不会启动正式训练。

服务器实际部署状态待核验。只有用户执行更新并返回实际入口及回执后，才能确认服务器运行版本与当时远端main一致。
