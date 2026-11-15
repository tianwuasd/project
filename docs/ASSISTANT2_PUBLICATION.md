# 助手2代码发布与服务器读取路径

助手2在D:/projects/project/chd-segmentation-assistant2的独立工作树维护，功能及发布分支codex/chd-assistant2-offline-preprocess-20261115。在线流程冻结在codex/chd-assistant2-online-frozen-20261115，提交69289532a3da2c94bc72769f5726ca260f6980a7。该工作树与助手1共用Git对象，分别使用工作目录、分支和Git索引；助手2索引位于助手1.git/worktrees/chd-segmentation-assistant2。该本地Git仓库的根为分割源码。共享发布仓库为tianwuasd/project，发布目标子目录为chd-segmentation。

## 发布步骤

1. 在助手2分支完成修改、真实有界检查和本地提交，记录完整SHA、源码文件清单及报告范围。
2. 用户已授权仅发布助手2独立分支。读取共享仓库当前main，在该确定树上建立同名助手2发布分支；发布分支保留整个共享仓库结构，将助手2受管理源码放在chd-segmentation子目录。远端对象接口可直接创建树及提交，无需复制整套共享仓库。
3. 逐文件审查该子目录中的新增、修改与删除，保留main中其他项目文件；患者清单、分组、权重、vendor依赖、env、私有配置、运行记录和报告不进入代码分支。特别核对全部待发布提交的新增文件及内容。
4. 推送前再次读取远端发布分支与main的实际SHA，核对快进关系。存在并发更新时重新审查，不强制推送。推送后重新查询远端分支，保存实际SHA和本地源码子树对应关系。
5. 交付确定分支及完整提交，保持共享main不变。后续合并main需用户另行授权，当前任务不执行该操作。

助手2的平铺源码提交用于本地开发和来源追踪，不能直接作为共享仓库main的整个根目录发布。公开发布分支必须含chd-segmentation这一子目录。

## 服务器更新

助手2的update.py固定读取tianwuasd/project的codex/chd-assistant2-offline-preprocess-20261115分支及chd-segmentation子目录。所有下载、比较、激活和更新后远端查询使用同一分支。本地Git分支必须与其相同，--force不能绕过此要求。部署使用独立普通Git仓库/data4/zhangruichen/code/chd-segmentation-assistant2-repo和独立过程目录/data4/zhangruichen/course/chd-segmentation-assistant2，默认值也指向这两处。助手1的main更新入口及代码目录保持独立。

首次安装时在不存在的助手2目录取得指定分支；命令由用户在服务器执行，当前会话不运行：

```bash
BRANCH=codex/chd-assistant2-offline-preprocess-20261115
REPO=/data4/zhangruichen/code/chd-segmentation-assistant2-repo
git clone --filter=blob:none --sparse --branch "$BRANCH" https://github.com/tianwuasd/project.git "$REPO"
git -C "$REPO" sparse-checkout set chd-segmentation
python3 -B "$REPO/chd-segmentation/run.py" --code-info
```

后续更新固定为本次交付的完整公开提交；PUBLIC_SHA填写发布回执中的SHA：

```bash
REPO=/data4/zhangruichen/code/chd-segmentation-assistant2-repo
COURSE=/data4/zhangruichen/course/chd-segmentation-assistant2
python3 -B "$REPO/chd-segmentation/update.py" --repo "$REPO" --runtime "$COURSE" --sha "$PUBLIC_SHA" --check
python3 -B "$REPO/chd-segmentation/update.py" --repo "$REPO" --runtime "$COURSE" --sha "$PUBLIC_SHA" --force
python3 -B "$REPO/chd-segmentation/run.py" --code-info
```

默认不改有本地变更的代码；--force明确以已核验远端替换本项目受管理源码，仍保留其他项目、私有文件及活动任务保护。源文件、本项目状态、分支回执及实际入口均需核对，活动任务继续使用原代码。Q/MedSAM2的数据配方、官方权重、Q私有包和机器配置按ASSISTANT2_SERVER_FILES.md分别迁移。代码更新不启动正式训练。

本会话不登录服务器，也不据本地验收声称服务器已经更新。远端发布与服务器实际运行状态分别记录，服务器安装和运行版本等待用户核验。
