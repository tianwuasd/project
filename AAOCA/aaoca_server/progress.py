"""Read-only display of measured progress; no invented slice counts."""
STAGES = {'input_hash': '核验输入', 'source_recheck': '复核原文件', 'decode_load': '解压读取影像',
          'compute': '重采样', 'normalize': '强度归一化', 'save_hu': '压缩保存 HU',
          'verify_hu': '校验 HU', 'save': '压缩保存归一化影像', 'verify': '校验归一化影像'}


def bar(done, total, width=24):
    if total <= 0:
        return '[' + '-' * width + '] 等待统计'
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {done / total:6.1%} ({done}/{total})'


def lines(progress):
    total = progress.get('total', 0)
    completed, reused, failed = (progress.get(key, 0) for key in ('completed', 'reused', 'failed'))
    done = completed + reused + failed
    result = ['病例进度 ' + bar(done, total),
              f'新完成 {completed} | 复用 {reused} | 失败 {failed} | 剩余 {max(0, total - done)}']
    if failed:
        result.append('进度计入已结束的失败病例；100% 不代表全部成功。')
    active = list(progress.get('active', {}).items())
    for case_id, info in active[:8]:
        stage = STAGES.get(info.get('stage'), info.get('stage', '处理中'))
        if info.get('total_slices', 0) > 0:
            stage += ' ' + bar(info.get('completed_slices', 0), info['total_slices'], width=16) + ' 层'
        result.append(f'{case_id}：{stage}')
    if len(active) > 8:
        result.append(f'另有 {len(active) - 8} 例同时处理中。')
    if progress.get('updated_at'):
        result.append('数据更新时间：' + progress['updated_at'])
    return result
