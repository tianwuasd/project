"""Descriptive, reproducible audit; no dispatch-policy confirmation or wind fit."""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def quantiles(values):
    return dict(zip(['min','p05','median','p95','max'],map(float,np.quantile(values,[0,.05,.5,.95,1]))))


def integrate(values,times):
    return float(np.sum((values[1:]+values[:-1])*.5*np.diff(times)))


def gps_length(lon,lat):
    lon=np.deg2rad(lon);lat=np.deg2rad(lat)
    a=np.sin(np.diff(lat)/2)**2+np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(np.diff(lon)/2)**2
    return float(np.sum(2*6371000*np.arcsin(np.sqrt(np.clip(a,0,1)))))


def optional_scalar(text):
    try:return float(text)
    except ValueError:return None


def main():
    raw=ROOT/'data/raw/drone_flights'
    manifest=json.loads((ROOT/'data/drone_measurement_manifest.json').read_text())
    for item in manifest['files']:
        if hashlib.sha256((raw/item['name']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Source integrity changed: '+item['name'])
    params=list(csv.DictReader((raw/'parameters.csv').open(encoding='utf-8')))
    ids=[int(x['flight']) for x in params]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate flight ID')
    rows=[];issues=[]
    with zipfile.ZipFile(raw/'flights.zip') as z:
        csv_ids=sorted(int(Path(n).stem) for n in z.namelist() if n.endswith('.csv'))
        if csv_ids!=sorted(ids):raise ValueError('Parameter/measurement ID mismatch')
        for p in params:
            a=np.genfromtxt(io.BytesIO(z.read(f"flights/{p['flight']}.csv")),delimiter=',',names=True)
            numeric=np.column_stack([a[name] for name in a.dtype.names])
            t=a['time'];dt=np.diff(t)
            flags=[]
            if not np.isfinite(numeric).all():flags.append('nonfinite')
            if np.any(dt<=0):flags.append('nonincreasing_time')
            if len(t)<2:flags.append('insufficient_samples')
            if flags:
                issues.append({'flight':int(p['flight']),'flags':flags});continue
            power=a['battery_voltage']*a['battery_current']
            horizontal=np.hypot(a['velocity_x'],a['velocity_y'])
            signed_wh=integrate(power,t)/3600
            positive_wh=integrate(np.maximum(power,0),t)/3600
            row={'flight':int(p['flight']),'route':p['route'],'date_label':p['date'],'local_time':p['local_time'],
                'speed_command_mps':optional_scalar(p['speed']),'payload_g':optional_scalar(p['payload']),
                'altitude_command_m':optional_scalar(p['altitude']),'altitude_command_raw':p['altitude'],
                'samples':len(t),'duration_s':float(t[-1]-t[0]),'median_dt_s':float(np.median(dt)),'max_dt_s':float(dt.max()),
                'energy_signed_wh':signed_wh,'energy_positive_wh':positive_wh,'negative_power_correction_wh':positive_wh-signed_wh,
                'negative_current_fraction':float(np.mean(a['battery_current']<0)),
                'horizontal_distance_velocity_m':integrate(horizontal,t),
                'horizontal_distance_gps_m':gps_length(a['position_x'],a['position_y']),
                'peak_relative_altitude_m':float(a['position_z'].max()-np.median(a['position_z'][:min(10,len(t))])),
                'groundspeed_p95_mps':float(np.quantile(horizontal,.95)),
                'anemometer_median_mps':float(np.median(a['wind_speed'])),
                'current_p95_A':float(np.quantile(a['battery_current'],.95))}
            rows.append(row)
    flights=[r for r in rows if r['route'].startswith('R')]
    keys=['duration_s','energy_signed_wh','horizontal_distance_velocity_m','horizontal_distance_gps_m',
          'median_dt_s','max_dt_s','negative_power_correction_wh','peak_relative_altitude_m','groundspeed_p95_mps']
    summary={'purpose':'descriptive_model_development_only','source_manifest_sha256':hashlib.sha256((ROOT/'data/drone_measurement_manifest.json').read_bytes()).hexdigest(),
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parameter_records':len(params),'audited_records':len(rows),'issues':issues,'route_counts':dict(Counter(p['route'] for p in params)),
        'complete_flight_records':len(flights),'date_label_count':len({r['date_label'] for r in flights}),
        'complete_flight_quantiles':{k:quantiles([r[k] for r in flights]) for k in keys},
        'complete_flight_totals':{'log_duration_hours':sum(r['duration_s'] for r in flights)/3600,
            'velocity_path_km':sum(r['horizontal_distance_velocity_m'] for r in flights)/1000,
            'gps_path_km':sum(r['horizontal_distance_gps_m'] for r in flights)/1000},
        'all_record_log_duration_hours':sum(r['duration_s'] for r in rows)/3600,
        'limitations':['Time ranges include preflight/postflight logging; no phase segmentation claimed.',
            'Date labels retained without resolving README date-range conflict.',
            'Small closed triangular missions, not observed kilometer-scale parcel round trips.',
            'No charging measurements or simultaneous multi-drone common disturbances.',
            'Anemometer observations are not treated as unbiased ambient wind.']}
    out=ROOT/'development/results/flight_audit_001';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Keep the audit immutable; choose a new output version')
    with (out/'flights.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
