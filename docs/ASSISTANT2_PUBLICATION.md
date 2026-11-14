# 助手2代码发布与服务器读取路径

助手2在D:/projects/project/chd-segmentation-assistant2的独立工作树维护，工作分支codex/chd-segmentation-assistant2-medsam2-q-r2。该工作树与助手1共用Git对象，分别使用工作目录、分支和Git索引；助手2索引位于助手1.git/worktrees/chd-segmentation-assistant2。该本地Git仓库的根为分割源码。共享发布仓库为tianwuasd/project，发布目标子目录为chd-segmentation。

## 发布步骤

1. 在助手2分支完成修改、真实有界检查和本地提交，记录完整SHA、源码文件清单及报告范围。
2. 获得代码发布授权后，读取共享仓库当前main，在单独的共享仓库工作目录建立同名助手2发布分支。该分支保留整个共享仓库结构，将助手2受管理源码放在chd-segmentation子目录。
3. 逐文件审查该子目录中的新增、修改与删除，保留main中其他项目文件；患者清单、分组、权重、vendor依赖、env、私有配置、运行记录和报告不进入代码分支。特别核对全部待发布提交的新增文件及内容。
4. 推送前再次读取远端发布分支与main的实际SHA，核对快进关系。存在并发更新时重新审查，不强制推送。推送后重新查询远端分支，保存实际SHA和本地源码子树对应关系。
5. 将发布分支提交给审查。经授权合入共享main后，记录共享main完整SHA及其chd-segmentation子树摘要，确认与验收源码一致。

助手2的平铺源码提交用于本地开发和来源追踪，不能直接作为共享仓库main的整个根目录发布。公开发布分支必须含chd-segmentation这一子目录。

## 服务器更新

现有update.py固定读取tianwuasd/project的main及chd-segmentation子目录。助手2代码尚在独立分支时，服务器日常更新沿用当前已发布main。助手2变更经授权合入main并复核后，用户可在服务器执行既有更新命令：

```bash
python3 -B /data4/zhangruichen/code/chd-segmentation/update.py
python3 -B /data4/zhangruichen/code/chd-segmentation/run.py --code-info
```

更新目标应固定到已经复核的main提交；源文件、本项目状态、更新回执及实际入口均需核对，活动任务继续使用原代码。模型依赖、官方权重、Q私有包和机器配置按ASSISTANT2_SERVER_FILES.md分别迁移。代码更新不启动正式训练。

本会话不登录服务器，也不据本地验收声称服务器已经更新。没有用户的代码发布指令时，只完成助手2本地提交和材料说明；远端及服务器状态分别报告。
