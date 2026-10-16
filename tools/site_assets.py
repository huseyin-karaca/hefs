"""Generate SVG diagrams, a Matplotlib chart, and published result tables.

Run from any directory. --check verifies committed assets without writing files.
Source: Karaca et al., Digital Signal Processing 168 (2026), Tables 3 and 6.
"""
import argparse
import csv
from html import escape
from io import StringIO
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUE, INK, MUTED, LINE, TEAL = '#3355d6', '#1c2230', '#606879', '#c9ccd3', '#17816c'


def text(x, y, value, size=16, color=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def rect(x, y, w, h, fill='#f6f7f9', stroke=LINE, radius=9):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ''
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{MUTED}" stroke-width="1.6" fill="none" marker-end="url(#arrow)"{dash}/>'


def svg(width, height, title, parts):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title" font-family="Schibsted Grotesk, Arial, sans-serif">'
            f'<title id="title">{escape(title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="{MUTED}"/></marker></defs>'
            + ''.join(parts) + '</svg>\n')


def overview():
    p = [text(30, 30, 'A FORECAST, THEN A REFINEMENT', 13, MUTED, 600)]
    p += [rect(30, 58, 235, 66, '#eef1fc', '#a3b3ee'), text(48, 85, 'Dominant features', 18, BLUE, 600), text(48, 108, 'Strong initial predictors', 13, MUTED)]
    p += [rect(330, 58, 240, 66, '#eef7f4', '#a5cfc4'), text(348, 85, 'Complementary features', 17, TEAL, 600), text(348, 108, 'Additional context', 13, MUTED)]
    p += [arrow(147,124,147,172), arrow(450,124,450,302)]
    p += [rect(30,174,235,70,'#fff',BLUE),text(48,201,'Layer 1',13,BLUE,600),text(48,226,'Predict the target',19,INK,600)]
    p += [arrow(147,244,147,285),text(147,309,'Initial prediction',16,INK,500,'middle')]
    p += [arrow(220,303,328,336),rect(330,304,240,78,'#fff',TEAL),text(348,331,'Layer 2',13,TEAL,600),text(348,358,'Predict the correction',18,INK,600)]
    p += [arrow(147,325,147,432),arrow(450,382,450,432)]
    p += [rect(30,434,540,76,'#eef1fc',BLUE),text(300,464,'Refined prediction',18,BLUE,600,'middle'),text(300,489,'initial prediction × (1 + correction)',17,INK,400,'middle')]
    p += [text(300,540,'The second layer learns correction targets during training.',13,MUTED,400,'middle')]
    return svg(600,550,'HEFS: dominant features predict; complementary features refine through multiplicative correction.',p)


def flow(training):
    # Vertical composition remains readable when the figure is scaled on a phone.
    p = [text(24,28,'TRAINING' if training else 'INFERENCE',13,BLUE,600)]
    p += [rect(24,48,432,62,'#eef1fc','#a3b3ee'),text(240,74,'Dominant features → first model',17,INK,600,'middle'),text(240,97,'Produce the initial prediction',14,MUTED,400,'middle'),arrow(240,110,240,151)]
    if training:
        p += [rect(24,153,432,92,'#fff',BLUE),text(240,180,'Cost-optimization search',18,BLUE,600,'middle'),text(240,206,'Observed training targets + initial predictions',14,INK,400,'middle'),text(240,230,'Find corrections with low absolute error',14,MUTED,400,'middle'),arrow(240,245,240,292,True)]
        p += [rect(24,294,432,104,'#eef7f4','#a5cfc4'),text(240,323,'Fit the correction model',18,TEAL,600,'middle'),text(240,350,'Inputs: complementary features + first prediction',14,INK,400,'middle'),text(240,376,'Supervision: correction targets from the search',14,MUTED,400,'middle')]
        p += [text(240,430,'Save both models and any fitted scalers.',14,MUTED,400,'middle')]
        return svg(480,450,'Training constructs correction targets from observed training data, then fits the second model.',p)
    p += [rect(24,153,432,85,'#eef7f4','#a5cfc4'),text(240,181,'Complementary features + first prediction',16,INK,600,'middle'),text(240,210,'Second model → predicted correction',16,TEAL,500,'middle'),arrow(240,238,240,281)]
    p += [rect(24,283,432,78,'#eef1fc',BLUE),text(240,312,'Multiply to refine',18,BLUE,600,'middle'),text(240,338,'first prediction × (1 + predicted correction)',15,INK,400,'middle'),text(240,394,'No observed target. No correction search.',14,MUTED,400,'middle')]
    return svg(480,416,'Inference uses two learned models and multiplies the initial prediction by one plus the correction.',p)


def results(rows):
    # Stable, editable vector output from a standard scientific plotting library.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hefs-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    by_name = {r['model']: r for r in rows}
    base, hefs = by_name['Baseline LightGBM'], by_name['HEFS (K=2)']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'hefs-results',
                         'axes.labelcolor': MUTED, 'text.color': INK}):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 7.5))
        fig.subplots_adjust(left=.28, right=.88, top=.86, bottom=.09, hspace=1.12)
        labels = ['Synthetic', 'M4', 'Weather']
        ratios = [float(hefs[k+'_error']) / float(base[k+'_error']) * 100
                  for k in ['synthetic', 'm4', 'weather']]
        top.barh(range(3), [100]*3, height=.4, color='#eceef1', zorder=2)
        top.barh(range(3), ratios, height=.4, color=BLUE, zorder=3)
        top.set_yticks(range(3), labels)
        top.set_xlim(0, 112)
        top.set_ylim(2.6, -.6)
        top.set_xticks([0,25,50,75,100], ['0%', '25%', '50%', '75%', '100%'])
        top.set_xlabel('Reported error / LightGBM error', labelpad=10)
        for i, value in enumerate(ratios):
            top.text(103, i, f'{value:.1f}%', va='center', color=BLUE, fontsize=10, fontweight='bold')
        names = ['HEFS (K=2)', 'Backward Wrapper', 'Forward Wrapper']
        times = [float(by_name[n]['weather_seconds']) for n in names]
        bottom.set_xscale('log')
        bottom.set_xlim(1, 200)
        bottom.set_ylim(2.6, -.6)
        bottom.set_yticks(range(3), ['HEFS (K=2)', 'Backward selection', 'Forward selection'])
        bottom.xaxis.set_major_locator(FixedLocator([1,10,100]))
        bottom.xaxis.set_major_formatter(FixedFormatter(['1 s','10 s','100 s']))
        bottom.xaxis.set_minor_locator(NullLocator())
        bottom.set_xlabel('Mean fit time (log scale)', labelpad=10)
        for i, value in enumerate(times):
            color = BLUE if i == 0 else MUTED
            bottom.hlines(i, 1, value, color=color, linewidth=2, zorder=3)
            bottom.plot(value, i, 'o', color=color, markersize=6, zorder=4)
            bottom.annotate(f'{value:g} s', (value, i), xytext=(0, 12),
                            textcoords='offset points', ha='center', color=color)
        for ax in (top, bottom):
            ax.spines[['top','right','left','bottom']].set_visible(False)
            ax.tick_params(axis='both', length=0, pad=9, labelcolor=MUTED)
            ax.grid(axis='x', color='#e2e5ec', linewidth=.7, zorder=0)
        fig.text(.045,.96,'ERROR RELATIVE TO LIGHTGBM', color=BLUE, weight='bold', size=11)
        fig.text(.045,.925,'Lower is better · LightGBM = 100%', color=MUTED, size=10)
        fig.text(.045,.485,'WEATHER MEAN FIT TIME', color=BLUE, weight='bold', size=11)
        fig.text(.045,.45,'Seconds · logarithmic axis · lower is better', color=MUTED, size=10)
        out = StringIO()
        fig.savefig(out, format='svg', metadata={'Date': None, 'Creator': 'HEFS / Matplotlib'})
        plt.close(fig)
        result = out.getvalue().replace("font-family: 'DejaVu Sans'", "font-family: Arial, Helvetica, sans-serif")
        title = '<title id="title">Published HEFS errors relative to LightGBM and Weather fit times.</title>'
        result = result.replace('<svg ', '<svg role="img" aria-labelledby="title" ', 1)
        pos = result.index('>', result.index('<svg ')) + 1
        return result[:pos] + title + result[pos:]


def table(rows, suffix):
    head = '| Model | Synthetic | M4 | Weather |\n| --- | ---: | ---: | ---: |\n'
    for row in rows:
        cells = [row['model']] + [row[f'{d}_{suffix}'] for d in ['synthetic','m4','weather']]
        if row['model'] == 'HEFS (K=2)':
            cells = [f'**{v}**' for v in cells]
        head += '| ' + ' | '.join(cells) + ' |\n'
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    with (ROOT/'docs/data/paper_results.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    outputs = {
        'docs/assets/overview.svg': overview(),
        'docs/assets/training.svg': flow(True),
        'docs/assets/inference.svg': flow(False),
        'docs/assets/results.svg': results(rows),
        'docs/snippets/errors.md': table(rows,'error'),
        'docs/snippets/times.md': table(rows,'seconds'),
        'docs/assets/mark.svg': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#3355d6"/><path d="M8 9h7v6h9v8H8z" fill="none" stroke="white" stroke-width="2.5"/></svg>\n',
    }
    for name, content in outputs.items():
        path = ROOT/name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f'Generated asset is stale: {name}; run python tools/site_assets.py')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f'{"Verified" if args.check else "Generated"} {len(outputs)} site assets.')


if __name__ == '__main__':
    main()

