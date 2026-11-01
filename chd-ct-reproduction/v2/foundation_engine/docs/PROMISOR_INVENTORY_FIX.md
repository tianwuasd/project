# v0.3.7：部分克隆的本地对象清单修复

用户服务器成功通过SSH取得引导脚本、main提交、目录信息和新更新器单文件，然后在运行缓存的对象检查失败：lazy fetching disabled / could not fetch 32a752a87484bbf708009485155e18a301b765da。该对象对应v0.3.6的05_src/task_update.py，尚未在此运行缓存中显式下载。失败发生在发布指针写入之前，旧版本保留。

原实现向cat-file --batch-check输入缺失promisor对象ID，查询类型也可能触发自动补取；本地检查同时禁止隐式联网，因而被拦截。本补丁保持禁网保护，改用cat-file --batch-all-objects枚举实际存储的对象；以清单差集确定要显式下载的运行文件，下载后再次枚举确认完整。不会为判断“缺什么”就查询缺失对象的类型。该方式也与AAOCA参考实现的本地对象枚举一致。依据：[Git cat-file](https://git-scm.com/docs/git-cat-file)。

113项回归108通过5条件跳过。新增故障契约测试结合真实过滤Git仓库，禁止所有缺失ID探测，保证仍能更新并排除无关对象。本机Git2.53和2.54的原查询都返回missing，没有复现服务器的相同fatal；不据此猜测服务器Git版本。二者的纯本地枚举均通过。修复后Windows HTTPS/7897实际读取59个运行文件，显式取得上述32a752对象，离线归档完成；仍不是服务器完整SSH发布验收。详见[验证JSON](PROMISOR_INVENTORY_VALIDATION.json)。

## 服务器重试

保留上次终端，f仍指向已成功下载的引导脚本、r仍为原runtime时，只需重跑：

```bash
python3 "$f" --code "$PWD" --runtime "$r" && bash start_server.sh --runtime "$r"
```

脚本会获取最新更新器；无需重复第一段SSH下载，不改已成功的密钥/代理设置，不删除缓存，也不编辑旧release或完整性清单。重开确认v0.3.7。若终端变量已失效，按[一次性升级说明](CODE_ONLY_UPDATE.md)重新指定原路径。升级不自动恢复任务或启动训练。
