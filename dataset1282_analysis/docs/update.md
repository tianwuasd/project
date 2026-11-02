# 限定子项目的代码更新

```text
python update.py --check --proxy http://127.0.0.1:7897
python update.py --proxy http://127.0.0.1:7897
python update.py --rollback FULL_ID
```

目标固定为https://github.com/tianwuasd/project.git、main分支、dataset1282_analysis子目录。更新使用隔离bare Git缓存和blob过滤，不在父仓库执行git pull，也不改其他子项目。

远端分支先解析为完整commit，之后使用该commit的树和blob身份下载。只接受白名单普通文件，拒绝链接、路径穿越、大小写冲突、缺文件、超限、协议不匹配和摘要不符。release-manifest.json校验内容一致性，并对Python文件离线语法检查；同源manifest不是独立数字签名。

通过后保存`code_versions/COMMIT/`，校验专属环境依赖，再原子切换`current.json`。代码、同版默认配置、测试和说明一起切换；本机配置、环境和结果独立保留。bootstrap在导入业务模块前选择并验证release，`run.py --version`显示实际代码、默认配置和docs来源。

默认work_root与environment锚定稳定工程home，切换release不改变默认工作区。`run.py --config task.json ...`会在选择解释器前读取任务配置。活动release缺失或损坏时，update入口回到稳定base更新器，使`--rollback`仍可使用；不会把损坏版本当作正常运行代码。

存在活动分析/更新锁、本地受管代码修改、未知远端主机锁或依赖不兼容时拒绝激活。锁核对主机和进程出生身份，不因旧心跳或同名文件直接抢锁。网络/校验失败保持旧指针。更新过程不自动安装依赖或启动数据分析。

初始源码在首次更新时保存为内容摘要命名的baseline快照，返回`baseline_snapshot`用于回滚；它是本地内容身份，不宣称是Git提交。远端release目录使用真实Git commit。回滚也验证文件、依赖和锁，并恢复同版默认配置与说明。

协议固定为bootstrap v1，顶层启动器不自动跨协议自替换。未来版本如改变bootstrap协议，当前更新器会拒绝激活，需另行设计迁移。代码更新、GitHub代码上传、真实运行产物上传为独立操作；本项目不会自动上传病例报告。
