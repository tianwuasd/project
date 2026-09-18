# Whole Heart Explorer Implementation Plan

**Goal:** 将已批准的五步数据探索方案实现为有中文注释、可重复运行的脚本。
**Architecture:** 读取统计与绘图报告解耦，CLI 顺序扫描病例以控制内存。
**Tech Stack:** Python、NiBabel、NumPy、Matplotlib、标准库 CSV/JSON/unittest。
**Spec:** [design.md](design.md)

## Global Constraints

原始数据只读；输出不能在输入内部；不自动修复数据；不执行模型训练；统计和预览区别标注。

## Tasks

- [x] `medical.py`：文件配对、头信息、缩放读取、空间校验、强度抽样、标签计数、SHA256。
  验证：合成 NIfTI 的缩放后数值、不同体素体积、缺失配对、非整数与未知标签、无效 sform。
- [x] `charts.py`：分组统计图、三方向图像/标签/叠加图、非预期标签高亮。
  验证：非等距体素的物理显示比例，统一类别配色，不修改传入图像。
- [x] `reporting.py`：UTF-8 BOM CSV、完整 JSON、中文解释 Markdown/HTML；原件与采样范围明示。
  验证：输出病例与标签表能对应，HTML 图像内嵌，Markdown 图片链接存在。
- [x] `analyze.py`：参数、输入保护、逐病例容错、进度和只读前后核对。
  验证：合成数据端到端；106 例真实数据完整运行；查看生成图像；核对先前发现的异常。
- [x] `README.md`：安装与运行、输出解释、适用范围、复现实验环境。

本目录不是 Git 仓库。采用临时目录完成实现与验证，最后复制到用户项目；不初始化 Git，不创建分支。

验证结果见 [验证记录](../验证记录.md)。全部实现与验证完成；交付仅复制代码、文档和本次完整结果。
