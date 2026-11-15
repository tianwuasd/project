# Q与MedSAM2发布和服务器入口

公开仓库tianwuasd/project的main共同维护chd-segmentation源码。助手2的本地独立工作树保留在D:/projects/project/chd-segmentation-assistant2，服务器新启动任务使用/data4/zhangruichen/code/chd-segmentation共同入口；配置、任务、描述和结果按CHD1282-Q、CHD1282-MedSAM2、ImageCHD-MedSAM2命名。旧服务器目录及其冻结任务保留原位置，不自动迁移。

## 固定版本更新

先从发布回执取得完整PUBLIC_SHA和UPDATER_SHA256。旧更新器最多接收200个源码文件；本轮完整源码超过该数量，首次更新先取得并核验本轮标准库更新器，保存在course专用bootstrap目录供复用，容量小于32KiB。使用系统curl与sha256sum，在服务器执行：

```bash
PUBLIC_SHA='<发布回执中的完整main提交>'
UPDATER_SHA256='<发布回执中的update.py SHA256>'
CODE=/data4/zhangruichen/code/chd-segmentation
UPDATER=/data4/zhangruichen/course/chd-segmentation/bootstrap/update-$PUBLIC_SHA.py
mkdir -p /data4/zhangruichen/course/chd-segmentation/bootstrap
curl --noproxy '*' --proto '=https' -fL "https://raw.githubusercontent.com/tianwuasd/project/$PUBLIC_SHA/chd-segmentation/update.py" -o "$UPDATER"
printf '%s  %s\n' "$UPDATER_SHA256" "$UPDATER" | sha256sum -c -
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 -B "$UPDATER" --sha "$PUBLIC_SHA" --mirror direct --force
python3 -B "$CODE/run.py" --code-info
```

仅在curl和摘要检查都成功后执行更新命令；任一步骤失败即停止。共同更新器默认repo为/data4/zhangruichen/code、runtime为/data4/zhangruichen/course/chd-segmentation，固定读取main，源码清单上限256文件。--force保留分支、活动代码锁、路径和内容核验。回执branch为main，sha、head、origin_branch及核对时的官方main应一致。本会话不登录服务器，实际安装以用户执行回执为准。

发布前以最新main为父提交，只整合本项目受管理文件；其他项目和助手1已发布实现保留。逐文件核对Git blob，以期望旧SHA非强制更新，发布后重新查询main与源码树。患者清单、权重、依赖、报告和结果不进入公开源码。

## 一个准备入口

用户已确认原inputs与STU权重放在服务器指定目录。安装入口先核对原输入内容身份和既有权重，精确重建Q的393/197/128/456组合及A组统计来源，生成实际配置；缺少的MedSAM2官方源码和公开权重按固定版本取得并核验。普通配置通过公开代码生成。

```bash
CHD_PYTHON=/data4/zhangruichen/env/chd-segmentation/bin/python \
  bash /data4/zhangruichen/code/chd-segmentation/scripts/CHD1282-Q-MedSAM2-20261115_install.sh \
  --install-dependencies
```

服务器默认直接连接GitHub、Hugging Face和PyPI。官方文件下载使用显式空代理配置；Git及pip子进程排除已有HTTP_PROXY、HTTPS_PROXY、ALL_PROXY及大小写变体，Git单次覆盖本地代理项，pip禁用已有代理配置文件。所有设置只作用于本次进程，不修改系统、用户或仓库网络配置，不启动代理。仅明确传入--proxy时对该次准备启用指定代理。Windows本机按本机网络约定执行验证。

既有完整材料直接复用，文件不匹配或环境不相容时停止并保留原件。额外Linux依赖仅放入env/medsam2，不升级共同训练解释器的PyTorch。固定TensorDict要求PyTorch至少2.11；不相容时需要使用满足条件的独立Linux解释器。Windows依赖不迁移。

只读核对使用PYTHONPATH指定共同源码的src，并执行python -B -m chd_segmentation.delivery check --root /data4/zhangruichen。可用--inputs、--stu-weights、--med-weights、--official-root、--dependency-root、--chd-root和--image-root明确引用实际既有位置；不自动搬动旧资料。

## 数据集与模型目录

| 内容 | 相对/data4/zhangruichen的位置 |
|---|---|
| 已上传原输入 | course/chd-segmentation/inputs |
| 已上传STU权重 | data/pretrained/STU-Net/small_ep4k.model |
| Q生成的冻结输入 | course/chd-segmentation/inputs/q1282 |
| 生成的准备及模型配置 | course/chd-segmentation/configs/<数据集-模型>/20261115 |
| 准备日志 | course/chd-segmentation/prepare/<数据集-模型>/20261115 |
| 来源及离线描述 | course/chd-segmentation/prepared/<数据集-模型>/20261115 |
| 新训练及完整恢复点 | course/chd-segmentation/runs/<数据集-模型>/20261115-offline |
| 预处理数组 | data/chd-segmentation/preprocessed |
| 预测及评价 | result/chd-segmentation/<数据集-模型>/20261115-offline |
| 安装回执 | course/chd-segmentation/installations/CHD1282-Q-MedSAM2-20261115.json |

所有模型work_root继续指向共同course/chd-segmentation，CPU/GPU和内存资源统一登记。新任务使用生成配置。已有任务继续使用自己的原配置、冻结源码、数据版本和完整检查点；新入口不移动已有目录。

Q运行脚本为scripts/CHD1282-Q-20261115_run.sh，MedSAM2运行脚本为scripts/ImageCHD-CHD1282-MedSAM2-20261115_run.sh。Q阶段40000/5000/20000更新；MedSAM2官方75epoch。STU的200epoch命令不改变这些方法预算。
