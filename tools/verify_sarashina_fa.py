#!/usr/bin/env python3
"""Optional GPU-vs-CPU FA160 checks; results never gate building or launching."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

from scripts.build_sarashina import BASE, DEFAULT_DEVICES, build_environment, run_fa160_tests
from scripts.setup_runtime import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hip', 'cuda'), default='cuda')
    parser.add_argument('--build', type=Path, help='Build directory (default: .cache/sarashina-official-runtime/build-BACKEND)')
    parser.add_argument('--device', help='Backend device to check, default: CUDA0 or ROCm0')
    parser.add_argument('--output', type=Path, help='Report directory; logs/JSON are suitable for feedback')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    build = (args.build or BASE/f'build-{args.backend}').resolve()
    if not (build/'bin/test-backend-ops').is_file():
        parser.error(f'Missing {build}/bin/test-backend-ops; build with --backend {args.backend}')
    device = args.device or DEFAULT_DEVICES[args.backend]
    output = args.output or ROOT/'results'/f'sarashina-fa160-{args.backend}-{device}'
    output.mkdir(parents=True, exist_ok=True)
    env = build_environment(build)
    manifest = build/'jev-build.json'
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
              'backend': args.backend, 'build': str(build), 'device': device,
              'build_manifest': json.loads(manifest.read_text()) if manifest.is_file() else None,
              'binaries_sha256': {p.name: sha256(p) for p in sorted((build/'bin').iterdir())
                                  if p.is_file() and (p.suffix == '.so' or p.name in ('llama-server', 'test-backend-ops'))}}
    server = str(build/'bin/llama-server')
    for name, flag in (('version', '--version'), ('devices', '--list-devices')):
        try:
            result = subprocess.run([server, flag], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=30)
            report[name] = {'returncode': result.returncode, 'output': result.stdout}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report[name] = {'error': str(exc)}
        (output/f'{name}.json').write_text(json.dumps(report[name], indent=2) + '\n')
    report['fa160'] = run_fa160_tests(build, device, output/'fa160-test.log', args.timeout)
    path = output/'verification.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    row = report['fa160']
    print(f'FA160 {device}: {row["status"]}, {row["passed"]}/{row["total"]} (expected {row["expected"]}). Report: {path}')
    print('This is a numerical kernel check, not a speed or model-accuracy result. Build/launch eligibility is unchanged.')
    if row['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
