# 证据契约与索引行为

## 层级
`metadata_only` 只有书目；`abstract` 是数据库摘要；`reading_note` 是二手精读；`pdf_unread` 只是文件清单；`pdf_text_extract` 表示实际抽取指定页文本，仍须阅读与公式视觉核对。不同层级不自动相互升级。

索引生成 paper_id 与 document_id，保留 path、locator、sha256、source_id。JSON 定位按 1 开始的记录号；WoS 定位为该条 PT 起始行；笔记行号、PDF 物理页号与印刷页号分别说明。`excerpt` 会核验当前文件哈希，变化时要求重建。检索结果不是阅读记录。

## 主张对象
```json
{
  "claim_id": "C1",
  "text": "此处填写经实际读取支持的单一主张",
  "type": "source_fact",
  "paper_id": "来自检索的稳定ID",
  "document_id": "来自证据的ID",
  "locator": "PDF physical page 1, Abstract",
  "read_scope": "仅摘要页",
  "support": "supported / partial / conflicting / unverified",
  "limitations": "摘要不足以核验完整公式和实验细节"
}
```
推断和假设列出支持其提出的资料，但不把引用当作该新假设已获验证。computed_result 连接输入、代码、运行结果路径与哈希。未联网核查时 publication_status 为 unknown，不把 DOI 存在当作出版状态/撤稿核查。

## 去重与更新
规范 DOI 优先合并，来源题名、作者、年份、原始 DOI、UT 和定位分别保留。无 DOI 的数据库记录优先按 UT 标识，否则按题名、作者、年份组合；不将不同 UT 的同名 Editorial 强制归并。无 DOI 的本地笔记/PDF 只有唯一同题名候选时才关联，冲突时独立保留。题名近似匹配不自动合并；短文件名可能形成独立条目，需人工确认。版本关联不等于独立研究证据数量。检索同时覆盖来源题名，合并后仍能找到不同题名版本。

`build` 先完整读取来源，再写临时索引并替换已知属于本工具的数据库；输入缺失时保留旧索引并报错。数据源更新后重建，不改原始数据库。不将历史分类、创新标签或精读判断当作论文直接证据。

Windows 长路径通过 `winpath` 适配，显示和引用时移除系统前缀。输出统一 UTF-8；乱码先判断显示编码与源文件损坏，不能仅凭终端字符认定损坏。
