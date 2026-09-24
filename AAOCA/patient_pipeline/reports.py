"""Human-readable reports; untrusted clinical strings are never executable markup."""
import csv
import html
import json
from pathlib import Path


def cell(value):
    if value is None:return ''
    if isinstance(value,(list,dict)):value=json.dumps(value,ensure_ascii=False)
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r')):return "'"+value
    return value


def write_csv(path,rows,columns=None):
    columns=columns or list(dict.fromkeys(k for r in rows for k in r)) or ['说明']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({k:cell(row.get(k)) for k in columns})


def write_html(out,patients,manifest):
    columns=['患者编号','姓名','病例关联','CT文件数','CT序列数','CT成功序列','CTA描述明确切片数','疑似CTA切片数',
             'CT数据状态','超声文件数','超声静态图数','超声视频数','超声总帧数','超声成功文件','超声数据状态','已核实PDF数']
    esc=lambda v:html.escape(str(v if v is not None else ''))
    body=''.join('<tr>'+''.join('<td>'+esc(r.get(k,''))+'</td>' for k in columns)+'</tr>' for r in patients)
    sample=manifest.get('selected_patient_ids')
    state={'running':'正在运行，以下不是本次完整结果','interrupted':'已安全停止，可续跑；以下为阶段性结果',
           'failed':'本次未完成，可修复后续跑','completed':'已完成','completed_with_issues':'已完成，有问题需要查看'}.get(manifest['status'],manifest['status'])
    counts=manifest.get('resume_counts',{})
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>病例与影像处理结果</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:32px;color:#183047;background:#f4f7fa}}h1{{font-size:26px}}.card{{background:white;padding:20px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;white-space:nowrap;background:white}}th,td{{padding:10px 14px;border:1px solid #dbe4ea;text-align:left}}th{{background:#e3edf4}}.scroll{{overflow:auto}}a{{color:#075a9c}}small{{color:#52697c}}</style>
<h1>病例与影像处理结果</h1><div class="card"><b>{'指定患者测试' if sample else '输入范围内全体患者'}</b> · 共 {len(patients)} 位患者<br>
运行状态：{esc(state)} · 模式：{esc(manifest['mode'])}<br>
本次新转换 {counts.get('images_converted',0)} 项 · 复用影像 {counts.get('images_reused',0)+counts.get('images_adopted',0)} 项 · 复用病例 {counts.get('documents_reused',0)} 份<br>
输入：{esc(manifest['input'])}</div>
<div class="card">“可读取/转换”表示文件层面的检查通过，不代表冠脉显示质量或检查覆盖范围得到医学确认。<br>
CTA依据检查描述识别；增强心脏CT单列为候选。病例按患者关联，未自动证明与影像属于同一次就诊。<br>
原始资料没有覆盖；输出含病例文字和可能的身份信息，未做正式脱敏。</div>
<p><a href="患者总表.csv">患者总表</a> · <a href="CT序列表.csv">CT序列参数</a> · <a href="超声文件表.csv">超声参数</a> · <a href="病例文档表.csv">病例文档</a> · <a href="问题清单.csv">问题清单</a></p>
<div class="scroll"><table><thead><tr>{''.join('<th>'+esc(k)+'</th>' for k in columns)}</tr></thead><tbody>{body}</tbody></table></div>
<p><small>患者编号和所有原始参数同时保存在JSON/SQLite，避免Excel自动格式转换造成编号前导零丢失。无需联网即可查看本页。</small></p></html>'''
    (out/'查看结果.html').write_text(text,encoding='utf-8')
