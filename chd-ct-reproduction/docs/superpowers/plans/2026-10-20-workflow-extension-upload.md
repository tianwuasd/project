# 已完成任务追加与结果上传实施计划

用户已授权：已完成 workflow 可追加未执行步骤，成功步骤不能重复运行，后续结果归于原任务；更新首条校验通过的线路即停止；面板上传日志、配置、指标和模型权重到同一 GitHub。用户已确认权重也包括。

架构：保留原计划/来源校验和锁，以新的 attempt 追加原目录步骤；结果页读取扩展后的状态。上传模块只打包获准产物到 runtime 外部暂存区，使用 GitHub 官方认证的 Release 附件，先准备清单供确认，不将凭据传给公共下载镜像。更新回退与上传事务分别测试。训练器、科学参数、已有产物不变。

- [x] 追加：extension.py + resume/workflow/jobs/panel/launcher；只允许缺失步骤，完成产物校验失败时阻止而不重跑；保留历史状态与旧步骤目录，新增步骤依赖原缓存和模型；竞争/后台锁检查。
- [x] 更新：updater.py + 专项测试；首条可快进且有效的 main 提交即停止，失败才换线；官方不可达时明确版本新鲜度未知。
- [x] 上传：upload.py + 合成归档/伪 gh 测试；标准库收集日志、配置、指标、best/last 权重与清单；排除原始影像/缓存/凭据；有界切块、SHA256、预览、官方 gh 认证、失败保留待续传，成功回执。
- [x] 集成：面板选择历史任务，CLI 可复制命令，结果页包含新增产物和上传链接；兼容 0.14 及更早记录。
- [x] 验证/维护：Linux 进程与只读/续跑边界、合成流程、完整相关回归、独立审查；版本/备份与 skill 同步。项目发布提交和镜像核对结果以本次交付回执为准。

上传选择：GitHub Release 附件而非将大型权重加入代码 Git 历史。使用官方 GitHub CLI；资料已核对 https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases 和 https://cli.github.com/manual/gh_release_upload 。未连接真实服务器，不上传真实任务产物进行测试。

验证已完成：Windows 368 passed / 21 skipped；Linux 21 passed，绑定修复后追加测试 2 passed。独立审查三类问题已修复并回归。skill 已同步发布；CT 发布提交及镜像核对结果见本次交付回执。
