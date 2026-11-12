from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
from unittest import TestCase

import numpy as np
import psutil

from .contracts import atomic_json, sha256_file
from .online_memory import CaseMemory
from .online_prediction import OnlineHeldout
from .xu_blood_grid import AxialViews
from .xu_blood_io import case_readers
from .xu_resources import HostResourcesChanged
from .xu_targets import TargetStore


def check_integration(original, online, names, workspace, targets, event):
    train, dev = names
    x = original.read_image(train, (64,)*3)
    y = original.read_label(train, (64,)*3)
    memory = CaseMemory(x.nbytes)
    for key, value in (('image', x), ('label', y)):
        with memory.lease(key, 64*1024**2, lambda:value) as result:
            np.testing.assert_array_equal(value, result)
        assert memory.cache.currsize <= memory.limit
    assert memory.stats['evictions'] == 1
    uncached = CaseMemory(0)
    with uncached.lease('image', 64*1024**2, lambda:x) as result:
        np.testing.assert_array_equal(x, result)
    assert not len(uncached.cache)
    with TestCase().assertRaises(HostResourcesChanged):
        with uncached.lease('oversized', psutil.virtual_memory().total*2, lambda:x):
            raise AssertionError('内存不足仍然准入')
    event(stage='memory-contract', eviction=True, zero_cache=True, insufficient_memory='resource-event',
          measured_array_limit_bytes=memory.limit, peak_array_bytes=memory.stats['peak_bytes'])
    memory.clear()
    del x, y, memory, uncached

    if targets is None:
        raise ValueError('Blood 读取验收需要与该数据匹配的真实派生标签目录')
    store = TargetStore(targets, original)
    reference = AxialViews(original, train)
    z = reference.zyx[0]//2
    expected = [reference.image(i) for i in (z, z+1)]
    with case_readers(online, store, train) as (reader, target_reader):
        views = AxialViews(reader, train, target_reader)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(views.image, i) for i in (z, z+1)]
            actual = [f.result(timeout=30) for f in futures]
        for a,b in zip(expected, actual):
            np.testing.assert_array_equal(a,b)
    assert reader.closed
    online.clear()
    del reader, target_reader, views, actual, expected, reference, store
    gc.collect()
    event(stage='blood-session', case=train, slices=2, workers=2, exact=True, closed=True)

    output = workspace/'prediction-input'
    source = original._records[dev]
    base = online.run_dir/'preprocessed/Dataset282_CHDSource'
    job = dict(prepared=str(online.run_dir), cases={dev:source},
               plans=sha256_file(base/'nnUNetPlans.json'), dataset=sha256_file(base/'dataset.json'))
    atomic_json(output/'run.json', job)
    cache = OnlineHeldout(output)
    image = cache.read_image(dev, (64,)*3)
    np.testing.assert_array_equal(image, original.read_image(dev, (64,)*3))
    assert asdict(cache.geometry(dev)) == asdict(original.geometry(dev))
    label = original.read_label(dev, (64,)*3).clip(0).astype(np.uint8)
    np.testing.assert_array_equal(cache.geometry(dev).restore_label(label),
                                  original.geometry(dev).restore_label(label))
    assert list(output.iterdir()) == [output/'run.json']
    cache.memory.clear()
    event(stage='prediction-input', case=dev, labels_used_for_preprocessing=False,
          image_exact=True, native_restore_exact=True, persistent_volume_files=0)
