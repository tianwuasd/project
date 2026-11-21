# 服务器输入与运行

公开配置使用通用 `/srv/research` 示例，实际部署根通过首次 `export-inputs --server-root`明确指定，配置不公开。源码、过程、原始输入、环境及未来结果分别放根目录下code、course、data、env、result；当前裁剪在course/work，data长期正式预处理未实现。

## 最小上传

将 `private/server-inputs/manifest.json`、`project.server.json` 两份正式文件集中上传至服务器本项目course/inputs。已有手工配置保留，不用示例覆盖。

只补传清单引用且服务器缺少的CT及匹配定位标签，保留各数据根内的相对目录；复用已有同源ImageCHD文件。首次bind-inputs检查全部条目的路径与大小，后续只核对选中病例。大小相同不能证明内容相同，不做文件哈希校验。

诊断Excel可存同一inputs目录，当前检查仅接受UTF-8 CSV的 `patient_key,aaoca_present,origin_left,high_risk,reviewer,reference` 字段，确认及转换仍待完成。空白为未知。无需上传病历全文、OCR、旧报告或STU模型；作者分类权重尚未取得。

## 代码更新与依赖

update.py复用相邻CHD工程的已有更新核心，可用 `--shared-updater`指定该文件；只更新本独立子目录，不修改CHD源码或迁移其目录。首次引导需从已审核的固定Git提交取得本项目update.py，并明确指定repo、runtime及shared-updater。后续直接调用已安装的入口：

```bash
python3 -B "$CODE/update.py" --force --mirror auto --sha "$SHA"
"$PY" -B "$CODE/run.py" --code-info
```

复用既有医学影像环境，setup_env.py仅检查定位依赖，不证明完整分类环境；缺失时先核对兼容性，再通过国内镜像安装pyproject。

```bash
"$PY" -B "$CODE/setup_env.py"
"$PY" -B "$CODE/run.py" bind-inputs --config "$COURSE/inputs/project.server.json" --inputs "$COURSE/inputs/manifest.json"
"$PY" -B "$CODE/run.py" check-training --config "$COURSE/inputs/project.server.json"
```

bind-inputs不需要原Windows患者索引，不改变角色/别名，已有不同绑定时停止。需要一例裁剪时使用prepare --case并将输出写入course/logs。没有分类训练或预测入口；所需功能和未验范围详见CAPABILITIES.md。
