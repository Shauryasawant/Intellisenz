"""audit.py - quick health check of an RFL/RFM export BEFORE any modelling."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RFM_FIELDS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
              'mag_x', 'mag_y', 'mag_z']
RFL_FIELDS = [f'rfl_ch{i}' for i in range(1, 10)]

_SENSOR_ID_RE = re.compile(r'^[A-Za-z]')
_DATE_RE = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}')
_TIME_ONLY_RE = re.compile(r'^\d{1,2}:\d{2}:\d{2}$')


def _safe_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return np.nan


def parse_payload(raw_value, device_type):
    if not isinstance(raw_value, str):
        return {'parse_error': True}
    tokens = raw_value.strip().strip('"').split(',')

    idx = 0
    sensor_id = None
    if _SENSOR_ID_RE.match(tokens[0]):
        sensor_id = tokens[0]
        idx = 1

    device_time_raw = None
    if idx < len(tokens) and _DATE_RE.match(tokens[idx]):
        if ' ' in tokens[idx]:
            device_time_raw = tokens[idx]
            idx += 1
        elif idx + 1 < len(tokens) and _TIME_ONLY_RE.match(tokens[idx + 1].strip()):
            device_time_raw = f"{tokens[idx]} {tokens[idx + 1]}"
            idx += 2
        else:
            device_time_raw = tokens[idx]
            idx += 1

    numeric_tokens = tokens[idx:idx + 9]
    numeric = [_safe_float(t) for t in numeric_tokens]
    names = RFM_FIELDS if device_type == 'RFM' else RFL_FIELDS
    out = {'sensor_id': sensor_id, 'device_time_raw': device_time_raw,
           'parse_error': len(numeric_tokens) < 9}
    out.update(dict(zip(names, numeric)))

    if device_type == 'RFL':
        rest = [t for t in tokens[idx + 9:] if t != '']
        pairs = 0
        i = 0
        while i < len(rest) - 1:
            if ':' in rest[i] and 'kg' in rest[i + 1]:
                pairs += 1
                i += 2
            else:
                i += 1
        counter_ok = bool(rest) and rest[-1].strip().lstrip('-').isdigit()
        if len(rest) > 2 * pairs + (1 if counter_ok else 0):
            out['parse_error'] = True
    return out


def load_and_parse(path, device_type):
    raw = pd.read_csv(path, header=None, names=['device', 'site', 'time', 'value'],
                      quotechar='"', engine='python')
    raw = raw[raw['device'] != 'device']
    sub = raw[raw['device'] == device_type].reset_index(drop=True)
    parsed = pd.DataFrame(list(sub['value'].apply(lambda v: parse_payload(v, device_type))))
    df = pd.concat([sub[['device', 'site', 'time']], parsed], axis=1)

    df['server_time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['server_date'] = df['server_time'].dt.date
    df['server_time_of_day'] = df['server_time'].dt.time
    df['device_time'] = pd.to_datetime(df['device_time_raw'], format='%d/%m/%Y %H:%M:%S',
                                       errors='coerce')
    df['device_clock_bad'] = df['device_time'].isna() | (df['device_time'] < pd.Timestamp('2015-01-01'))
    df['sensor_id'] = df['sensor_id'].fillna('UNKNOWN')
    return df.sort_values('server_time').reset_index(drop=True)


def audit_stream(df):
    t = df['server_time'].dropna()
    dt = t.diff().dt.total_seconds().dropna()
    med = float(dt.median()) if len(dt) else float('nan')
    value_cols = [c for c in df.columns if c in RFM_FIELDS or c in RFL_FIELDS]
    all_zero = (df[value_cols].fillna(0) == 0).all(axis=1).mean() if value_cols else float('nan')
    return {
        'rows': int(len(df)),
        'parse_error_rate': float(df['parse_error'].fillna(False).astype(bool).mean()),
        'bad_device_clock_rate': float(df['device_clock_bad'].mean()),
        'all_zero_row_rate': float(all_zero),
        'time_start': str(t.min()),
        'time_end': str(t.max()),
        'time_span': str(t.max() - t.min()) if len(t) else None,
        'median_interval_s': med,
        'gaps_over_10x': int((dt > 10 * med).sum()) if med == med else 0,
        'duplicate_timestamps': int(t.duplicated().sum()),
    }


def audit_device(df, name):
    report = {'overall': audit_stream(df), 'per_sensor': {}}
    report['overall']['unknown_sensor_rate'] = float((df['sensor_id'] == 'UNKNOWN').mean())
    for sid, g in df.groupby('sensor_id'):
        report['per_sensor'][sid] = audit_stream(g)
    return report


def print_report(name, report):
    print(f"\n=== {name} ===")
    for k, v in report['overall'].items():
        print(f"  {k:24s} {v}")
    print("  --- per sensor ---")
    for sid, stats in report['per_sensor'].items():
        print(f"  {sid}: rows={stats['rows']}, span={stats['time_span']}, "
              f"interval~{stats['median_interval_s']}s, gaps={stats['gaps_over_10x']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--save-parquet', help='folder to save parsed frames as parquet')
    ap.add_argument('--json', help='write the report to this JSON file')
    args = ap.parse_args()

    full = {}
    for dev in ('RFM', 'RFL'):
        df = load_and_parse(args.csv, dev)
        if df.empty:
            print(f"\n=== {dev} === no rows")
            continue
        full[dev] = audit_device(df, dev)
        print_report(dev, full[dev])
        if args.save_parquet:
            Path(args.save_parquet).mkdir(parents=True, exist_ok=True)
            df.to_parquet(Path(args.save_parquet) / f'{dev.lower()}.parquet')
    
    if args.json:
        Path(args.json).write_text(json.dumps(full, indent=2, default=str))


if __name__ == '__main__':
    main()

#python src/intellisenz/preprocessing/audit.py influx_data.csv --save-parquet parsed_output --json audit_report.json