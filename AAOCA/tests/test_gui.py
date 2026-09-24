"""Functional Tk checks in a hidden window; never opens real clinical data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline_gui import App
spec=importlib.util.spec_from_file_location('resume_fixtures',Path(__file__).with_name('test_resume.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def wait(root,predicate,timeout=10):
    start=time.monotonic()
    while not predicate():
        root.update();time.sleep(.01)
        if time.monotonic()-start>timeout:raise AssertionError('GUI timed out')
    root.update()


with tempfile.TemporaryDirectory() as temporary:
    base=Path(temporary);source=fixtures.fixture(base);output=base/'output'
    root=tk.Tk();root.withdraw();app=App(root);app.settings=base/'settings.json'
    app.source.set(str(source));app.output.set(str(output));root.update()
    app.start();app.toggle_pause()
    wait(root,lambda:app.status.get().startswith('已暂停'))
    assert str(app.start_button['state'])=='disabled'
    assert app.pause_button['text']=='继续'
    app.stop();wait(root,lambda:not app.thread.is_alive())
    wait(root,lambda:str(app.start_button['state'])=='normal')
    assert json.loads((output/'processing_manifest.json').read_text(encoding='utf-8'))['status']=='interrupted'
    app.start();wait(root,lambda:not app.thread.is_alive())
    wait(root,lambda:str(app.start_button['state'])=='normal')
    assert float(app.overall['value'])==100
    app.start();wait(root,lambda:not app.thread.is_alive())
    wait(root,lambda:str(app.start_button['state'])=='normal')
    assert '已跳过 2 项' in app.stats.get(),app.stats.get()
    root.destroy()
print('GUI passed: hidden-window start, pause, stop while paused, resume, progress, skip counters.')
