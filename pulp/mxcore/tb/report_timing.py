"""Report one MXCore job from existing GVSoC calibration traces; no RTL comparison."""

import argparse
from collections import deque
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys


SIGNALS = {
    'clock.period': 'period',
    'soc.testbench.timing.reset': 'reset',
    'soc.dut.busy': 'busy',
    'soc.dut.timing.tcdm_req': 'req',
    'soc.dut.timing.tcdm_gnt': 'gnt',
    'soc.dut.timing.dotp_fire': 'dotp',
    'soc.dut.timing.events': 'events',
}


def read_header(stream, signals=None):
    if signals is None:
        signals = SIGNALS
    tokens = []
    for line in stream:
        tokens.extend(line.split())
        if '$enddefinitions' in tokens:
            break
    if '$enddefinitions' not in tokens:
        raise ValueError('VCD header is incomplete')
    scope, selected = [], {}
    tick_ps = None
    units = {'s': 10**12, 'ms': 10**9, 'us': 10**6, 'ns': 1000,
             'ps': 1, 'fs': Fraction(1, 1000)}
    pos = 0
    while pos < len(tokens):
        command = tokens[pos]
        end = tokens.index('$end', pos + 1)
        args = tokens[pos + 1:end]
        if command == '$scope':
            scope.append(args[1])
        elif command == '$upscope':
            scope.pop()
        elif command == '$var':
            path = '.'.join(scope + [args[3]])
            if path in signals:
                selected[args[2]] = signals[path]
        elif command == '$timescale':
            scale = ''.join(args)
            unit = scale.lstrip('0123456789')
            tick_ps = int(scale[:-len(unit)]) * Fraction(units[unit])
        pos = end + 1
    missing = set(signals.values()) - set(selected.values())
    if missing:
        raise ValueError('VCD is missing selected traces: ' + ', '.join(sorted(missing)))
    if tick_ps is None:
        raise ValueError('VCD timescale is missing')
    return selected, tick_ps


def read_changes(stream):
    tokens = (token for line in stream for token in line.split())
    now = 0
    for token in tokens:
        if token.startswith('#'):
            updated = int(token[1:])
            if updated < now:
                raise ValueError('VCD timestamps are not monotonic')
            now = updated
        elif token[0] in 'bB':
            yield now, next(tokens), token[1:]
        elif token[0] in '01xXzZ':
            yield now, token[1:], token[0]
        elif token[0] in 'rRsS':
            next(tokens)  # Unselected real/string values have a separate identifier.


def format_report(times, cycles, input_count, retired_count, latency_min, latency_max):
    """Use identical metric names and reset-relative cycle definitions for both models."""
    def at(key):
        return cycles(times[key] - times['reset']) if key in times else 'N/A'

    latency = 'N/A'
    if latency_min is not None:
        low, high = cycles(latency_min), cycles(latency_max)
        latency = low if low == high else '{}..{}'.format(low, high)
    span = cycles(times['retire_last'] - times['input_first']) if input_count else 'N/A'
    return [
        ('Busy asserted cycle', at('busy_on')),
        ('First memory request accepted cycle', at('mem_first')),
        ('First computation input cycle', at('input_first')),
        ('Last computation input cycle', at('input_last')),
        ('First retirement cycle', at('retire_first')),
        ('Last retirement cycle', at('retire_last')),
        ('Compute done cycle', at('compute_done')),
        ('Output done cycle', at('output_done')),
        ('Busy deasserted cycle', at('busy_off')),
        ('Busy duration (cycles)', cycles(times['busy_off'] - times['busy_on'])),
        ('First input → last retirement (cycles)', span),
        ('Pipeline latency', latency),
        ('Input count', input_count),
        ('Retired count', retired_count),
    ]


def summarize(path):
    states, times, pending = {}, {}, deque()
    input_count = retired_count = 0
    latency_min = latency_max = None
    period_ps = None
    with Path(path).open() as stream:
        selected, tick_ps = read_header(stream)
        for now, identifier, bits in read_changes(stream):
            name = selected.get(identifier)
            if name is None:
                continue
            old = states.get(name, 0)
            if any(ch in bits.lower() for ch in 'xz'):
                if states.get('busy'):
                    raise ValueError('Unknown value on ' + name + ' during the job')
                states[name] = 0
                continue
            value = int(bits, 2)
            states[name] = value
            if name == 'period':
                # GVSoC's clock.period trace stores picoseconds, independently of VCD timescale.
                if value > 0:
                    if period_ps is not None and value != period_ps:
                        raise ValueError('Changing clock periods are not supported by this report')
                    period_ps = value
            elif name == 'reset' and old == 1 and value == 0:
                if 'busy_on' in times:
                    raise ValueError('Reset inside a captured job is not supported')
                times['reset'] = now
            elif name == 'busy':
                if value and not old:
                    if 'busy_on' in times:
                        raise ValueError('Expected one job; VCD contains multiple busy windows')
                    times['busy_on'] = now
                elif old and not value and 'busy_on' in times:
                    times['busy_off'] = now
            elif name in ('req', 'gnt') and states.get('busy'):
                if states.get('req') and states.get('gnt'):
                    times.setdefault('mem_first', now)
            elif name == 'dotp' and states.get('busy'):
                # Do not collapse same-timestamp records: consecutive cycles contain 0 -> fire.
                rising = value & ~old
                if rising & 2:
                    if not pending:
                        raise ValueError('A retirement has no matching accepted input')
                    latency = now - pending.popleft()
                    latency_min = latency if latency_min is None else min(latency_min, latency)
                    latency_max = latency if latency_max is None else max(latency_max, latency)
                    times.setdefault('retire_first', now)
                    times['retire_last'] = now
                    retired_count += 1
                if rising & 1:
                    pending.append(now)
                    times.setdefault('input_first', now)
                    times['input_last'] = now
                    input_count += 1
            elif name == 'events' and 'busy_on' in times:
                rising = value & ~old
                if rising & 1:
                    times.setdefault('compute_done', now)
                if rising & 2:
                    times.setdefault('output_done', now)
    required = {'reset', 'busy_on', 'busy_off', 'mem_first', 'compute_done', 'output_done'}
    if not required.issubset(times) or period_ps is None:
        raise ValueError('VCD does not contain a complete job with reset, memory, and completion traces')
    if pending or input_count != retired_count:
        raise ValueError('Incomplete computation trace: input_count={} retired_count={}'.format(
            input_count, retired_count))

    def cycles(ticks):
        result = Fraction(ticks) * tick_ps / period_ps
        if result.denominator != 1:
            raise ValueError('A recorded event is not aligned to the clock period')
        return result.numerator

    return format_report(times, cycles, input_count, retired_count, latency_min, latency_max)


def file_version(path):
    try:
        info = path.stat()
        return info.st_ino, info.st_size, info.st_mtime_ns
    except FileNotFoundError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_vcd = Path(__file__).resolve().parent.parent / 'build/test/work/all.vcd'
    parser.add_argument('--vcd', type=Path, default=default_vcd)
    parser.add_argument('--save-vcd', type=Path, help='save the fresh run VCD at this path before reporting')
    parser.add_argument('--run', nargs=argparse.REMAINDER, help='run this command, then report its fresh VCD')
    args = parser.parse_args(argv)
    run_status = 0
    try:
        if args.save_vcd is not None and args.run is None:
            raise ValueError('--save-vcd requires --run to avoid saving a stale VCD')
        if args.run is not None:
            if not args.run:
                raise ValueError('--run requires a simulator command')
            before = file_version(args.vcd)
            run_status = subprocess.run(args.run).returncode
            if run_status < 0:
                run_status = 128 - run_status
            after = file_version(args.vcd)
            if after is None or after == before:
                raise ValueError('Simulator produced no fresh VCD; previous timing data was not used')
        if args.save_vcd is not None:
            args.save_vcd.parent.mkdir(parents=True, exist_ok=True)
            if args.vcd.resolve() != args.save_vcd.resolve():
                shutil.copyfile(args.vcd, args.save_vcd)
            args.vcd = args.save_vcd
            print('[VCD] ' + str(args.vcd), flush=True)
        for name, value in summarize(args.vcd):
            print('[PERF] {}: {}'.format(name, value), flush=True)
    except (OSError, ValueError, IndexError, StopIteration) as error:
        print('[PERF] ERROR: ' + str(error), file=sys.stderr)
        return run_status or 1
    return run_status


if __name__ == '__main__':
    sys.exit(main())
