"""Report one existing MXCore RTL VCD using the GVSoC [PERF] fields.

Cycle 0 is the rising edge of rst_n, matching GVSoC's reset-release origin.
Handshakes use values immediately before each rising clock edge, including
consecutive cycles with fire held high. Completion events and busy transitions
use their waveform timestamps, rather than the later TB display timestamps.
The clock period is measured from clk. This script never runs a simulator.
One complete job is required. The timing-only VCD cannot check golden accuracy
or reconstruct an unrecorded trigger: Busy duration remains busy-high to busy-low.
"""

import argparse
from collections import deque
import csv
from fractions import Fraction
from itertools import groupby
import io
import json
from pathlib import Path
import re
import sys

from report_timing import format_report, read_changes, read_header


def rtl_signals(top):
    engine = top + '.dut.i_mxcore.i_engine.i_mxcore_engine'
    return {
        top + '.clk': 'clk',
        top + '.rst_n': 'reset_n',
        top + '.busy': 'busy',
        top + '.dut.tcdm.req': 'req',
        top + '.dut.tcdm.gnt': 'gnt',
        engine + '.in_fire': 'input',
        engine + '.out_fire': 'retire',
        top + '.evt': 'events',
    }


def summarize(path, top='tb_mxcore_hwpe'):
    states, times, pending = {}, {}, deque()
    input_count = retired_count = 0
    latency_min = latency_max = None
    last_clock = period = None
    with Path(path).open() as stream:
        selected, _ = read_header(stream, rtl_signals(top))
        changes = ((now, selected[identifier], bits)
                   for now, identifier, bits in read_changes(stream)
                   if identifier in selected)
        for now, group in groupby(changes, key=lambda change: change[0]):
            updates = {}
            for _, name, bits in group:
                updates[name] = None if any(ch in bits.lower() for ch in 'xz') else int(bits, 2)
            after = dict(states)
            after.update(updates)
            if states.get('busy') == 1 or after.get('busy') == 1:
                unknown = [name for name in rtl_signals(top).values() if after.get(name) is None]
                if unknown:
                    raise ValueError('Unknown RTL signal during the job: ' + ', '.join(unknown))

            if states.get('clk') == 0 and after.get('clk') == 1:
                if last_clock is not None:
                    measured = now - last_clock
                    if period is not None and measured != period:
                        raise ValueError('Changing or missing RTL clock edges are not supported')
                    period = measured
                last_clock = now
                # RTL sequential logic sees the pre-edge values. Grouping all
                # records at this timestamp makes VCD record ordering irrelevant.
                if states.get('reset_n') == 1 and states.get('busy') == 1:
                    if states['req'] & states['gnt']:
                        times.setdefault('mem_first', now)
                    if states['input']:
                        pending.append(now)
                        times.setdefault('input_first', now)
                        times['input_last'] = now
                        input_count += 1
                    if states['retire']:
                        if not pending:
                            raise ValueError('An RTL retirement has no matching accepted input')
                        latency = now - pending.popleft()
                        latency_min = latency if latency_min is None else min(latency_min, latency)
                        latency_max = latency if latency_max is None else max(latency_max, latency)
                        times.setdefault('retire_first', now)
                        times['retire_last'] = now
                        retired_count += 1

            if states.get('reset_n') == 0 and after.get('reset_n') == 1:
                if 'busy_on' in times:
                    raise ValueError('Reset after a captured job started is not supported')
                times['reset'] = now
            elif states.get('reset_n') == 1 and after.get('reset_n') == 0 and 'busy_on' in times:
                raise ValueError('Reset after a captured job started is not supported')
            if after.get('busy') == 1 and states.get('busy') != 1:
                if 'busy_on' in times:
                    raise ValueError('Expected one job; RTL VCD contains multiple busy windows')
                times['busy_on'] = now
            elif states.get('busy') == 1 and after.get('busy') == 0:
                times['busy_off'] = now
            if 'busy_on' in times and after.get('events') is not None:
                rising = after['events'] & ~(states.get('events') or 0)
                for bit, name in ((1, 'compute_done'), (2, 'output_done')):
                    if rising & bit:
                        if name in times:
                            raise ValueError('Expected one job; multiple RTL ' + name + ' events')
                        times[name] = now
            states = after

    required = {'reset', 'busy_on', 'busy_off', 'mem_first', 'compute_done', 'output_done'}
    missing = required - set(times)
    if missing or period is None:
        raise ValueError('Incomplete RTL VCD; missing clock or events: ' + ', '.join(sorted(missing)))
    if pending or input_count != retired_count:
        raise ValueError('Incomplete computation trace: input_count={} retired_count={}'.format(
            input_count, retired_count))
    if not (times['reset'] <= times['busy_on'] <= times['mem_first'] <= times['output_done']
            and times['busy_on'] <= times['compute_done'] <= times['output_done'] <= times['busy_off']):
        raise ValueError('RTL reset, busy, memory and completion events are out of order')
    if retired_count and times['retire_last'] > times['compute_done']:
        raise ValueError('RTL compute completion precedes the last retirement')

    def cycles(ticks):
        result = Fraction(ticks, period)
        if result.denominator != 1:
            raise ValueError('An RTL event is not aligned to the reset-relative clock period')
        return result.numerator

    return format_report(times, cycles, input_count, retired_count, latency_min, latency_max)


def render(report, output_format):
    if output_format == 'json':
        return json.dumps(dict(report), ensure_ascii=False, indent=2) + '\n'
    if output_format == 'csv':
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow([name for name, value in report])
        writer.writerow([value for name, value in report])
        return stream.getvalue()
    return ''.join('[PERF] {}: {}\n'.format(name, value) for name, value in report)


def report_basename(compile_tcl):
    """Name reports from RTL compile defines, never from GVSoC make defaults."""
    names = {
        'M', 'K', 'N', 'VECTOR_SIZE', 'NPE', 'REUSE', 'SRC_FMT', 'DST_FMT',
        'PRELOAD', 'STANDALONE', 'STANDALONE_BF16', 'NO_STALLS',
        'QUANTIZE_BF16', 'QUANTIZE_MXFP8', 'QUANTIZE_MXFP8ALT', 'QUANTIZE_MXFP4',
    }
    defines = {}
    for line in Path(compile_tcl).read_text(encoding='utf-8').splitlines():
        line = line.rstrip()
        if line.endswith('\\'):
            line = line[:-1].rstrip()
        match = re.fullmatch(r'\s*\+define\+([A-Z][A-Z0-9_]*)=(.*)', line)
        if match is None or match[1] not in names:
            continue
        name, value = match[1], match[2].strip()
        if value.startswith('\\"') and value.endswith('\\"'):
            value = value[2:-2]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if name in defines and defines[name] != value:
            raise ValueError('Conflicting RTL compile definitions for ' + name)
        defines[name] = value

    def required(name):
        if name not in defines:
            raise ValueError('RTL compile configuration is missing ' + name)
        return defines[name]

    def number(name):
        value = int(required(name))
        if value <= 0:
            raise ValueError('RTL compile configuration requires positive ' + name)
        return value

    def flag(name):
        value = int(required(name))
        if value not in (0, 1):
            raise ValueError('RTL compile configuration requires 0 or 1 for ' + name)
        return value

    def data_format(name):
        value = required(name).lower()
        if value not in ('fp8', 'fp8alt', 'fp6', 'fp6alt', 'fp4', 'fp32', 'bf16', 'int8'):
            raise ValueError('Unsupported RTL format ' + value)
        return value

    m, k, n = number('M'), number('K'), number('N')
    vs, npe, reuse = number('VECTOR_SIZE'), number('NPE'), number('REUSE')
    input_format = data_format('SRC_FMT')
    output_format = None
    # Match the priority used by the RTL and GVSoC, including when several
    # quantization bits are set. DST_FMT is the accumulator, not MX output.
    for name, value in (('QUANTIZE_BF16', 'bf16'), ('QUANTIZE_MXFP8', 'fp8'),
                        ('QUANTIZE_MXFP8ALT', 'fp8alt'), ('QUANTIZE_MXFP4', 'fp4')):
        enabled = flag(name)
        if enabled and output_format is None:
            output_format = value
    if output_format is None:
        output_format = data_format('DST_FMT')
    preload, standalone, no_stalls = flag('PRELOAD'), flag('STANDALONE'), flag('NO_STALLS')
    standalone_suffix = ''
    if standalone:
        standalone_suffix = '_bf16' if flag('STANDALONE_BF16') else '_fp32'
    stem = ('IN{}_OUT{}_VS{}_NPE{}_REUSE{}_preload{}_standalone{}{}_{}'.format(
        input_format, output_format, vs, npe, reuse, preload, standalone,
        standalone_suffix, 'no_stalls' if no_stalls else 'stalls'))
    return Path('M={} K={} N={}'.format(m, k, n)) / stem


def save_reports(report, report_dir, compile_tcl):
    """Save reports together; the original VCD is only read by summarize()."""
    basename = Path(report_dir) / report_basename(compile_tcl)
    basename.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for output_format, suffix in (('text', '.txt'), ('json', '.json'), ('csv', '.csv')):
        path = basename.with_suffix(suffix)
        path.write_text(render(report, output_format), encoding='utf-8')
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcd', type=Path, required=True, help='existing RTL VCD; never reruns simulation')
    parser.add_argument('--top', default='tb_mxcore_hwpe', help='testbench scope in the VCD')
    parser.add_argument('--format', choices=('text', 'json', 'csv'), default='text')
    parser.add_argument('--output', type=Path, help='save report to this file instead of stdout')
    parser.add_argument('--report-dir', type=Path,
                        help='save TXT/JSON/CSV under this root, grouped by matrix shape and RTL configuration')
    parser.add_argument('--compile-tcl', type=Path,
                        help='RTL compile configuration for naming reports (default: compile.tcl beside the VCD)')
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.vcd.resolve():
            raise ValueError('Report output must differ from the input VCD')
        if args.output and args.report_dir:
            raise ValueError('Choose either --output or --report-dir')
        report = summarize(args.vcd, args.top)
        if args.report_dir:
            compile_tcl = args.compile_tcl or args.vcd.with_name('compile.tcl')
            for path in save_reports(report, args.report_dir, compile_tcl):
                print('[REPORT] ' + str(path), file=sys.stderr)
        result = render(report, args.format)
        if args.output:
            args.output.write_text(result, encoding='utf-8')
        else:
            sys.stdout.write(result)
    except (OSError, ValueError, IndexError, StopIteration) as error:
        print('[PERF] ERROR: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
