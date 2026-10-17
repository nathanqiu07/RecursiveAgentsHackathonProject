#!/usr/bin/env python3
"""Maintain local registration only while a verified direct node is reachable."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--manager', default='http://127.0.0.1:8002')
    args = parser.parse_args()
    path = Path(args.receipt)
    if path.stat().st_mode & 0o077:
        raise SystemExit('Receipt must be private')
    receipt = json.loads(path.read_text())
    while True:
        try:
            request = urllib.request.Request(receipt['address'] + '/v1/health', headers={'Authorization': 'Bearer ' + receipt['key']})
            with urllib.request.urlopen(request, timeout=5) as response:
                health = json.load(response)
            assert health['node_id'] == receipt['node_id']
            assert health['runtime'] == 'direct-host-systemd'
            assert health['status'] == 'ready'
            assert health['verification']['direct_execution_verified']
            body = {'id': receipt['node_id'], 'cpus': health['limits']['cpus'], 'memory_mb': health['limits']['memory_mb'], 'status': 'online'}
            request = urllib.request.Request(args.manager + '/nodes/register', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
        except Exception:
            # Let the manager lease expire; never advertise unreachable capacity.
            pass
        time.sleep(20)


if __name__ == '__main__':
    main()
