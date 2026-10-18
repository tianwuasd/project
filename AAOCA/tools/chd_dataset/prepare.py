"""Lossless research organization of paired NIfTI CT/labels; no resampling."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import shutil
import time

import nibabel as nib
import numpy as np
try:
    from isal import igzip
    gzopen = igzip.open
except ImportError:
    gzopen = gzip.open

from archive import extract_labels, sha256, signature, stamp, write_json

VERSION = '1.1'
CHUNK_VOXELS = 1 << 20


def read_volume(path, label=False):
    """Read every voxel and reach gzip EOF/CRC without loading an entire volume."""
    im = nib.load(str(path))
    dtype = np.dtype(im.dataobj.dtype)
    if len(im.shape) != 3 or min(im.shape) < 1 or dtype.kind not in 'iuf':
        raise ValueError('Requires a nonempty real-valued 3D image')
    affine = im.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError('Nonfinite or singular affine')
    zooms = np.asarray(im.header.get_zooms(), dtype=float)
    if not np.isfinite(zooms).all() or np.any(zooms <= 0):
        raise ValueError('Invalid voxel spacing')
    slope, intercept = float(im.dataobj.slope), float(im.dataobj.inter)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError('Nonfinite intensity scaling')
    units = list(im.header.get_xyzt_units())
    warnings = ['PHYSICAL_UNITS_UNCONFIRMED'] if units[0] == 'unknown' else []
    q, qc = im.get_qform(coded=True)
    s, sc = im.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-4, rtol=0):
        raise ValueError('Active qform and sform disagree')
    if not qc and not sc:
        warnings.append('SPATIAL_TRANSFORM_CODES_UNSET')
    stats = {'shape': list(im.shape), 'dtype': str(dtype), 'header_spacing': zooms.tolist(),
             'units': units, 'affine': affine.tolist(), 'orientation': ''.join(nib.aff2axcodes(affine)),
             'qform_code': int(qc), 'sform_code': int(sc), 'slope': slope, 'intercept': intercept,
             'voxel_count': int(np.prod(im.shape)), 'warnings': warnings}
    counts = Counter()
    n, mean, m2 = 0, 0.0, 0.0
    minimum, maximum = float('inf'), float('-inf')
    with gzopen(path, 'rb') as f:
        f.seek(int(im.dataobj.offset))
        remaining = stats['voxel_count']
        while remaining:
            take = min(remaining, CHUNK_VOXELS)
            raw = f.read(take * dtype.itemsize)
            if len(raw) != take * dtype.itemsize:
                raise ValueError('Truncated voxel payload')
            a = np.frombuffer(raw, dtype=dtype)
            if slope != 1 or intercept != 0:
                a = a.astype(np.float64) * slope + intercept
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite voxel values')
            lo, hi = float(a.min()), float(a.max())
            minimum, maximum = min(minimum, lo), max(maximum, hi)
            if label:
                if lo < 0 or hi > 65535 or (a.dtype.kind == 'f' and np.any(a != np.floor(a))):
                    raise ValueError('Labels must be integer IDs from 0 through 65535')
                bins = np.bincount(a.astype(np.int64), minlength=int(hi) + 1)
                for v in np.flatnonzero(bins):
                    counts[int(v)] += int(bins[v])
            else:
                chunk_mean = float(np.mean(a, dtype=np.float64))
                chunk_m2 = float(np.var(a, dtype=np.float64)) * take
                delta = chunk_mean - mean
                m2 += chunk_m2 + delta * delta * n * take / (n + take)
                mean += delta * take / (n + take)
            n += take
            remaining -= take
        # Forces gzip trailer CRC/size validation (also detects truncated trailers).
        while f.read(8 << 20):
            pass
    stats.update(minimum=minimum, maximum=maximum)
    if label:
        stats['label_counts'] = dict(sorted(counts.items()))
        stats['foreground_voxels'] = n - counts.get(0, 0)
        if stats['foreground_voxels'] == 0:
            stats['warnings'].append('EMPTY_FOREGROUND')
    else:
        stats.update(mean=mean, std=math.sqrt(max(0, m2 / n)))
        if minimum == maximum:
            stats['warnings'].append('CONSTANT_IMAGE')
    return stats


def grid_errors(image, label):
    errors = []
    if image['shape'] != label['shape']:
        errors.append('IMAGE_LABEL_SHAPE_MISMATCH')
    if not np.allclose(image['affine'], label['affine'], atol=1e-4, rtol=0):
        errors.append('IMAGE_LABEL_AFFINE_MISMATCH')
    if image['units'][0] != label['units'][0]:
        errors.append('IMAGE_LABEL_UNITS_MISMATCH')
    return errors


def copy_verified(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    h = hashlib.sha256()
    try:
        with src.open('rb') as source, tmp.open('wb') as target:
            for chunk in iter(lambda: source.read(8 << 20), b''):
                target.write(chunk)
                h.update(chunk)
        digest = h.hexdigest()
        if sha256(tmp) != digest:
            raise IOError('Copied file SHA256 mismatch')
        os.replace(tmp, dst)
        return {'sha256': digest, **signature(dst)}
    finally:
        if tmp.exists():
            tmp.unlink()


def inventory(source, labels):
    images_by_key, labels_by_key = defaultdict(list), defaultdict(list)
    for p in sorted(Path(source).rglob('*.nii.gz')):
        if p.name.endswith('_image.nii.gz'):
            images_by_key[p.name[:-13]].append(p)
        else:
            images_by_key['unrecognized:' + p.name].append(p)
    for p in sorted(Path(labels).glob('*.nii.gz')):
        key = p.name[:-13] if p.name.endswith('_label.nii.gz') else 'unrecognized:' + p.name
        labels_by_key[key].append(p)
    rows = []
    for key in sorted(set(images_by_key) | set(labels_by_key)):
        images, labs = images_by_key[key], labels_by_key[key]
        errors = []
        if len(images) != 1:
            errors.append('MISSING_IMAGE' if not images else 'DUPLICATE_IMAGE_NAME')
        if len(labs) != 1:
            errors.append('MISSING_LABEL' if not labs else 'DUPLICATE_LABEL_NAME')
        if key.startswith('unrecognized:'):
            errors.append('UNRECOGNIZED_FILENAME')
        rows.append({'case_id': 'CHD_' + hashlib.sha256(key.encode()).hexdigest()[:16],
                     'source_key': key, 'images': [str(p.resolve()) for p in images],
                     'labels': [str(p.resolve()) for p in labs], 'pairing_errors': errors})
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Research identifier collision')
    return rows


def process_pair(row, output):
    result = {**row, 'version': VERSION, 'checked_at': stamp(), 'status': 'failed', 'errors': [], 'warnings': []}
    if row['pairing_errors']:
        result['errors'] = row['pairing_errors']
        return result
    image, label = Path(row['images'][0]), Path(row['labels'][0])
    try:
        result['source_signatures'] = [signature(image), signature(label)]
        for kind, path in [('image', image), ('label', label)]:
            try:
                result[kind] = read_volume(path, label=kind == 'label')
                result['warnings'].extend(f'{kind}:{w}' for w in result[kind]['warnings'])
            except Exception as exc:
                result['errors'].append(f'{kind}: {type(exc).__name__}: {exc}')
        grid_issues = []
        if 'image' in result and 'label' in result:
            grid_issues = grid_errors(result['image'], result['label'])
            result['grid_issues'] = grid_issues
            result['origin_difference_header_units'] = (np.asarray(result['label']['affine'])[:3, 3] - np.asarray(result['image']['affine'])[:3, 3]).tolist()
        if result['errors']:
            return result
        result['warnings'].extend(grid_issues)
        result['warnings'].append('LABEL_ANATOMICAL_NAMES_UNCONFIRMED')
        collection = 'review_required' if grid_issues else 'native'
        stage = Path(output) / 'staging' / row['case_id']
        image_name, label_name = row['case_id'] + '_0000.nii.gz', row['case_id'] + '.nii.gz'
        image_copy = copy_verified(image, stage / image_name)
        label_copy = copy_verified(label, stage / label_name)
        if [signature(image), signature(label)] != result['source_signatures']:
            raise RuntimeError('Source changed during verification/copy')
        published = []
        for kind, name, copied in [('images', image_name, image_copy), ('labels', label_name, label_copy)]:
            dest = Path(output) / collection / kind / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha256(dest) != copied['sha256']:
                raise RuntimeError('Existing output differs; refusing overwrite')
        for kind, name, copied in [('images', image_name, image_copy), ('labels', label_name, label_copy)]:
            dest = Path(output) / collection / kind / name
            os.replace(stage / name, dest)
            published.append({**copied, **signature(dest)})
        stage.rmdir()
        result.update(status='review_required' if grid_issues else ('passed_with_notes' if result['warnings'] else 'passed'), outputs=published, collection=collection)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def reusable(row, prior, rehash=False):
    if prior.get('version') != VERSION or prior.get('status') not in ['passed', 'passed_with_notes', 'review_required']:
        return False
    if row['images'] != prior.get('images') or row['labels'] != prior.get('labels'):
        return False
    try:
        if [signature(Path(p)) for p in row['images'] + row['labels']] != prior.get('source_signatures'):
            return False
        for out in prior['outputs']:
            if signature(out['path']) != {k: out[k] for k in ['path', 'bytes', 'mtime_ns']}:
                return False
            if rehash and sha256(out['path']) != out['sha256']:
                return False
        if rehash:
            for src, out in zip(row['images'] + row['labels'], prior['outputs']):
                if sha256(src) != out['sha256']:
                    return False
    except (OSError, KeyError):
        return False
    return True


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items() if k in fields})


def report(output, rows, total, started, phase='audit', status='running', reused=0, message='', inventory_total=None, scope='full'):
    output = Path(output)
    passed = sum(r['status'].startswith('passed') for r in rows)
    review = sum(r['status'] == 'review_required' for r in rows)
    errors, notes, classes, class_cases = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        errors.update(r['errors'])
        notes.update(r['warnings'])
        label_counts = r.get('label', {}).get('label_counts', {})
        classes.update({str(k): v for k, v in label_counts.items()})
        class_cases.update(str(k) for k, v in label_counts.items() if v)
    summary = {'version': VERSION, 'updated': stamp(), 'started': started, 'phase': phase, 'status': status,
               'total': total, 'inventory_total': inventory_total or total, 'scope': scope,
               'completed': len(rows), 'passed': passed, 'review_required': review, 'failed': len(rows) - passed - review,
               'reused': reused, 'message': message, 'error_counts': dict(errors), 'note_counts': dict(notes),
               'label_voxel_counts': dict(sorted(classes.items(), key=lambda x: int(x[0]))),
               'label_case_counts': dict(sorted(class_cases.items(), key=lambda x: int(x[0]))),
               'resampled': False, 'normalized': False, 'registered': False,
               'physical_geometry_confirmed': False, 'anatomical_label_names_confirmed': False}
    write_json(output / 'progress.json', summary)
    write_json(output / 'summary.json', summary)
    def table(items, first, second):
        return '<table><tr><th>' + first + '</th><th>' + second + '</th></tr>' + ''.join(
            '<tr><td>' + html.escape(str(k)) + '</td><td>' + str(v) + '</td></tr>' for k, v in items) + '</table>'
    failure_rows = ''.join('<tr><td>' + html.escape(r['case_id']) + '</td><td>' + html.escape('; '.join(r['errors'] + r.get('grid_issues', []))) + '</td></tr>' for r in rows if r['errors'] or r.get('grid_issues'))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">''' + (
        '<meta http-equiv="refresh" content="10">' if status == 'running' else '') + '''
<title>CHD 数据整理与检查</title><style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;padding:0 25px;color:#233247;background:#f5f7fa}h1{color:#173f5f}section{background:white;padding:20px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}strong{color:#096a82}progress{width:100%;height:24px}code{overflow-wrap:anywhere}.cards{font-size:22px}small{color:#65758b}</style>
<h1>CHD 数据整理与检查</h1>''' + f'<small>更新：{summary["updated"]}；完整清单 {inventory_total or total} 例；范围：{"小样本" if scope == "pilot" else "全量"}</small><section><div class="cards">已检查 {len(rows)} / {total}　网格匹配 {passed}　待复核 {review}　读取失败 {len(rows)-passed-review}</div><progress value="{len(rows)}" max="{max(1,total)}"></progress><p>阶段：{html.escape(phase)}；状态：{html.escape(status)}；本次复用：{reused}</p><p>{html.escape(message)}</p></section>' + '''
<section><strong>本次完成范围：解压、影像标注配对、完整像素与压缩流检查、空间网格对照、原字节复制及 SHA256 验证。</strong><p>没有重采样、灰度归一化、配准、标签重映射或模型训练。网格匹配表示文件可读、像素符合结构要求且影像与标签网格匹配，不等于临床标注正确。网格不一致但文件完整者保留原样在 review_required，不能直接当成已对齐训练对；native 保存网格匹配者。</p><p>CHD 头中的间距和空间单位不能据此认定为真实毫米参数；没有解剖类别说明，标签编号只作观察统计。源文件名中的 class 不作为分割类别定义。</p><p><a href="配对清单.csv">配对清单</a> · <a href="问题清单.csv">问题清单</a> · <a href="summary.json">统计 JSON</a> · <a href="manifest.json">详细清单</a></p></section><section><h2>标签编号与像素数（已完成检查范围）</h2>''' + table(sorted(classes.items(), key=lambda x: int(x[0])), '标签值', '体素数') + '</section><section><h2>待核实事项</h2>' + table(notes.items(), '说明代码', '文件/病例次数') + '<p>PHYSICAL_UNITS_UNCONFIRMED：空间单位未提供；LABEL_ANATOMICAL_NAMES_UNCONFIRMED：标签解剖名称未提供；EMPTY_FOREGROUND：标签全为背景；CONSTANT_IMAGE：影像全为相同数值；IMAGE_LABEL_AFFINE_MISMATCH：影像和标签的空间坐标变换不同；IMAGE_LABEL_SHAPE_MISMATCH：数组尺寸不同。</p></section><section><h2>失败或空间对应待复核记录</h2><table><tr><th>研究编号</th><th>具体原因</th></tr>' + failure_rows + '</table></section></html>'
    tmp = output / '查看进度.html.tmp'
    tmp.write_text(page, encoding='utf-8')
    os.replace(tmp, output / '查看进度.html')
    return summary


def run(source, output, workers=2, limit=None, force=False, rehash=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = output / 'records'
    records.mkdir(exist_ok=True)
    rows = inventory(source, output / 'extracted_labels')
    if not rows:
        raise ValueError('No NIfTI input records found')
    write_json(output / 'inventory.json', rows)
    total_inventory = len(rows)
    if limit:
        rows = rows[:limit]
    required = sum(Path(p).stat().st_size for row in rows for p in row['images'] + row['labels'])
    existing_bytes = sum(p.stat().st_size for folder in ['native', 'review_required'] for p in (output / folder).rglob('*.nii.gz'))
    if shutil.disk_usage(output).free < max(0, required - existing_bytes) + (2 << 30):
        raise OSError('Insufficient free disk space for lossless copies')
    started = stamp()
    results, pending, reused = [], [], 0
    for row in rows:
        path = records / (row['case_id'] + '.json')
        try:
            prior = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        except (OSError, ValueError):
            prior = {}
        if not force and reusable(row, prior, rehash):
            results.append(prior)
            reused += 1
        else:
            pending.append(row)
    def checkpoint(status='running', message=''):
        summary = report(output, results, len(rows), started, status=status, reused=reused, message=message, inventory_total=total_inventory, scope='pilot' if limit else 'full')
        write_json(output / 'summary.json', summary)
        write_json(output / 'manifest.json', sorted(results, key=lambda r: r['case_id']))
        write_csv(output / '配对清单.csv', sorted(results, key=lambda r: r['case_id']),
                  ['case_id', 'source_key', 'status', 'images', 'labels', 'errors', 'warnings', 'outputs'])
        issues = [{'case_id': r['case_id'], 'severity': severity, 'reason': reason}
                  for r in results for severity, key in [('error', 'errors'), ('note', 'warnings')] for reason in r[key]]
        write_csv(output / '问题清单.csv', issues, ['case_id', 'severity', 'reason'])
        return summary
    checkpoint(message='逐例完整读取像素并验证研究副本；小规模检查不代表全量完成' if limit else '逐例完整读取像素并验证研究副本')
    start_clock = time.monotonic()
    last_report = start_clock
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        def submit():
            row = next(iterator, None)
            if row is not None:
                futures[pool.submit(process_pair, row, output)] = row
        for _ in range(workers):
            submit()
        while futures:
            completed, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for fut in completed:
                row = futures.pop(fut)
                result = fut.result()
                write_json(records / (row['case_id'] + '.json'), result)
                results.append(result)
                with (output / 'events.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'time': stamp(), 'case_id': row['case_id'], 'status': result['status'], 'errors': result['errors']}, ensure_ascii=False) + '\n')
                print(f'{len(results)}/{len(rows)} {row["case_id"]} {result["status"]} elapsed={time.monotonic()-start_clock:.1f}s', flush=True)
                submit()
            if time.monotonic() - last_report >= 10:
                checkpoint(message='正在检查和复制；失败单例记录后继续')
                last_report = time.monotonic()
    return checkpoint(status='completed', message='全部计划单元已检查；请同时查看未通过项和待核实事项')


class RunLock:
    """OS lock; a leftover file after a crash is not interpreted as a live process."""
    def __init__(self, path):
        self.path = Path(path)
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open('a+b')
        self.f.seek(0)
        self.f.write(b'0')
        self.f.flush()
        self.f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self
    def __exit__(self, *args):
        self.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, help='Read local paths from project.local.json')
    parser.add_argument('--source', type=Path, help='Directory containing *_image.nii.gz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive', type=Path, help='Optional label ZIP to extract/resume first')
    parser.add_argument('--password-stdin', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force', action='store_true', help='Recheck all selected voxels')
    parser.add_argument('--rehash-resume', action='store_true', help='Rehash sources and outputs before reusing')
    args = parser.parse_args()
    if args.project_root:
        config = json.loads((args.project_root / 'project.local.json').read_text(encoding='utf-8'))
        reference = Path(config['chd_reference_root'])
        args.source = args.source or reference / 'formated_dataset1200_image_all'
        args.output = args.output or args.project_root / 'CHD'
        extraction_path = args.output / 'extraction.json'
        extraction = json.loads(extraction_path.read_text(encoding='utf-8')) if extraction_path.exists() else {}
        needs_extraction = not extraction_path.exists() or len(extraction.get('files', [])) != extraction.get('total') or bool(extraction.get('errors'))
        if needs_extraction and args.archive is None:
            args.archive = reference / 'formated_dataset1200_label_less_then_11.zip'
    if args.source is None or args.output is None:
        parser.error('Provide --project-root or both --source and --output')
    if args.workers < 1 or args.workers > 8 or (args.limit is not None and args.limit < 1):
        parser.error('workers must be 1..8; limit must be positive')
    if not args.source.is_dir():
        parser.error('source directory does not exist')
    if args.output.resolve().is_relative_to(args.source.resolve()) or args.source.resolve().is_relative_to(args.output.resolve()):
        parser.error('source and output must not contain one another')
    with RunLock(args.output / '.run.lock'):
        if args.archive:
            import sys
            import getpass
            password = (sys.stdin.readline().rstrip('\r\n') if args.password_stdin else getpass.getpass('ZIP password: ')).encode()
            extract_labels(args.archive, args.output, password)
            del password
        if not (args.output / 'extracted_labels').is_dir():
            parser.error('Extract labels first, or provide --archive')
        result = run(args.source, args.output, args.workers, args.limit, args.force, args.rehash_resume)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
