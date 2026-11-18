import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.review import prediction_events


def main():
    path = Path(sys.argv[1]).resolve(strict=True)
    size = path.stat().st_size
    descriptor = dict(event_log=str(path), start_offset=0, end_offset=size)
    if len(sys.argv) > 2:
        mode = sys.argv[2]
        if mode == 'empty':
            descriptor['end_offset'] = 0
        elif mode == 'past_end':
            descriptor['end_offset'] = size+1
        elif mode == 'broken_start':
            descriptor['start_offset'] = 1
        elif mode == 'missing_event':
            with path.open('rb') as stream:
                line = stream.readline()
            assert json.loads(line).get('event') not in ('prediction_reused','model_case_reused','prediction','model_case_completed')
            descriptor['end_offset'] = len(line)
        prediction_events(descriptor)
        return
    events = prediction_events(descriptor)
    assert events
    for item in events:
        start, end = item['record_start_offset'], item['record_end_offset']
        with path.open('rb') as stream:
            stream.seek(start)
            assert json.loads(stream.read(end-start)) == item['record']
        one = prediction_events(dict(event_log=str(path),start_offset=start,end_offset=end))
        assert len(one) == 1 and one[0]['record'] == item['record']
    for mode in ('empty','past_end','broken_start','missing_event'):
        child = subprocess.run([sys.executable,'-B',__file__,str(path),mode],capture_output=True)
        assert child.returncode != 0
    print('ACTUAL_JSONL_RANGE_OK',dict(bytes=size,events=len(events),
        scope='existing production event reader only; not new stage3 prediction evidence'))


if __name__ == '__main__':
    main()
