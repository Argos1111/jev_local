"""Human-readable CLI presentation; inference results remain plain JSON data."""
import json
import os
import sys
import unicodedata

LABELS = {'refund_requested': '返金要求', 'sentiment': '感情', 'route': '担当部署', 'churn_risk': '解約リスク'}

def value(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

def answer_type(v):
    if v is None: return 'null'
    if isinstance(v, bool): return 'boolean'
    if isinstance(v, str): return 'string'
    if isinstance(v, int): return 'integer'
    if isinstance(v, float): return 'number'
    if isinstance(v, list): return 'array'
    return 'object'


def width(s):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)

def table(headers, rows):
    rows = [[str(v).replace('\n', ' ↵ ').replace('\r', '') for v in row] for row in [headers, *rows]]
    widths = [max(width(row[i]) for row in rows) for i in range(len(headers))]
    lines = ['  '.join(cell + ' ' * (w-width(cell)) for cell,w in zip(row,widths)).rstrip() for row in rows]
    lines.insert(1, '  '.join('─'*w for w in widths))
    return '\n'.join(lines)

def percent(p):
    if 0 < p < 0.00005: return '<0.01%'
    if 0.99995 < p < 1: return '>99.99%'
    return f'{p*100:.2f}%'

def duration(ms):
    return f'{ms:,.1f} ms' if ms < 1000 else f'{ms/1000:,.2f} s'

def render(result, color=False):
    def accent(s): return f'\033[1;36m{s}\033[0m' if color else s
    def title(s): return accent(s) + '\n' + '─' * min(64, max(32, width(s)))
    if 'runs' in result:
        runs = result['runs']
        med = result['median_ms']
        lines = [title('Jev Local  /  速度比較'),
                 f"{len(runs)}回測定  ·  並列数 {result.get('workers', '—')}  ·  prompt cache {'ON' if result.get('cache_prompt') else 'OFF'}", '',
                 table(['方式', '中央値'], [['先頭トークン判定', duration(med['logprobs'])], ['JSON生成', duration(med['json'])]]),
                 '', accent(f"先頭トークン判定は {result['speedup']:.2f}倍の速度"), '',
                 table(['測定', '先頭トークン', 'JSON生成', '全項目一致'],
                       [[i+1,duration(r['logprobs']['elapsed_ms']),duration(r['json']['elapsed_ms']),'一致' if r['agreement'] else '不一致'] for i,r in enumerate(runs)])]
        for i,r in enumerate(runs):
            differences = [(k,v,r['json']['values'].get(k)) for k,v in r['logprobs']['values'].items()
                           if json.dumps(v) != json.dumps(r['json']['values'].get(k))]
            if differences:
                lines += ['', f'測定 {i+1} の判定差（先頭トークン / JSON）']
                lines += [f'  {LABELS.get(k,k)}: {value(a)} / {value(b)}' for k,a,b in differences]
        lines += ['', '方式間の一致は、正解率ではありません。']
        return '\n'.join(lines)
    if 'decisions' in result:
        lines = [title('Jev Local  /  判定結果'),
                 f"{len(result['decisions'])}項目  ·  合計 {duration(result['elapsed_ms'])}", '']
        if 'state' in result:
            lines += [accent('入力（state）'), result['state'], '']
        rows = []
        for key,d in result['decisions'].items():
            selected = next(p for p in d['probabilities'] if json.dumps(p['value']) == json.dumps(d['value']))
            rows.append([d.get('question', key), answer_type(d['value']), value(d['value']),
                         percent(selected['probability']), percent(d['confidence']) if d.get('confidence') is not None else '未評価'])
        lines += [table(['質問', '返答の型', '返答', '確率', '確信度'], rows), '',
                  '確率: 選ばれた候補の条件付き確率。確信度: 候補分布の集中度（1 − 正規化エントロピー）。正解率ではありません。']
        return '\n'.join(lines)
    if 'rows' in result and 'accuracy' in result:
        canonical = [r for r in result['rows'] if not r['reversed']]
        lines = [title('Jev Local  /  精度チェック'),
                 f"{len(canonical)}例  ·  正解率 {result['accuracy']:.1%}  ·  候補順反転で {result['reversed_choice_prediction_flips']}例が変化", '',
                 table(['指標','値'],[['Brier score',f"{result['binary_brier']:.4f}"],['NLL',f"{result['nll']:.4f}"],['順序反転の平均確率変化',f"{result['reversed_choice_mean_abs_probability_delta']:.4f}"]]), '']
        for i,r in enumerate(canonical):
            ok = r['result']['value'] == r['expected']
            lines += [f"{'✓' if ok else '✗'} {i+1}. {r['state']}",
                      f"  正解: {value(r['expected'])}  判定: {value(r['result']['value'])}  P(true): {percent(r['p_true'])}"]
        if 'json_accuracy' in result: lines += ['', f"通常JSON生成の正解率: {result['json_accuracy']:.1%}"]
        lines += ['', '手作りの少数例による診断です。一般的な精度・校正性能の評価ではありません。']
        return '\n'.join(lines)
    raise ValueError('判定結果、benchmark、sanityのJSONを指定してください')

def print_result(result, output_format='pretty'):
    if output_format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result, color=sys.stdout.isatty() and 'NO_COLOR' not in os.environ))
