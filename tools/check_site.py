"""Dependency-free checks for the project page's local links, results and word budget."""
import csv
import html as html_lib
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


class Page(HTMLParser):
    """Collects element IDs, local link targets, and the body rows of every table."""

    def __init__(self):
        super().__init__()
        self.ids, self.links, self.tables = set(), [], []
        self.in_body, self.cell = False, None
        self.row = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate ID: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'tbody':
            self.in_body = True
            self.tables.append([])
        if self.in_body and tag == 'tr':
            self.row = []
        if self.in_body and tag in ('th', 'td'):
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if self.in_body and tag in ('th', 'td'):
            self.row.append(self.cell.strip())
            self.cell = None
        if self.in_body and tag == 'tr':
            self.tables[-1].append(self.row)
        if tag == 'tbody':
            self.in_body = False


page = Page()
page.feed((DOCS / 'index.html').read_text())

for link in page.links:
    parts = urlsplit(link)
    if parts.scheme or parts.netloc:
        continue
    if parts.path:
        path = (DOCS / unquote(parts.path)).resolve()
        assert path.is_relative_to(DOCS), f'Link outside site: {link}'
        assert path.is_file(), f'Missing asset: {link}'
    elif parts.fragment:
        assert parts.fragment in page.ids, f'Missing anchor: {link}'

# Both tables on the page transcribe `paper_results.csv`: the first its error columns,
# the second its timing columns, for the same six of the paper's nine models.
with (DOCS / 'data/paper_results.csv').open() as stream:
    published = {row['model']: row for row in csv.DictReader(stream)}
assert len(published) == 9, f'Expected 9 models in paper_results.csv, found {len(published)}'
assert len(page.tables) == 2, f'Expected 2 tables on the page, found {len(page.tables)}'

NAMES = {'HEFS (Ablation)': 'HEFS (Ablation)', 'Baseline LightGBM': 'Baseline LightGBM'}
checked = 0
for table, keys in zip(page.tables, (('synthetic_error', 'm4_error', 'weather_error'),
                                     ('synthetic_seconds', 'm4_seconds', 'weather_seconds'))):
    assert len(table) == 6, f'Expected 6 rows, found {len(table)}'
    for row in table:
        model = NAMES.get(row[0], row[0])
        assert model in published, f'Row not in paper_results.csv: {model}'
        values = [float(published[model][key]) for key in keys]
        assert values == [float(v) for v in row[1:]], f'Value mismatch on {model}'
        checked += 1

for svg in (DOCS / 'assets').glob('*.svg'):
    ET.parse(svg)

index = (DOCS / 'index.html').read_text()
for claim in ('0.956', 'cost-optimization block', '10.1016/j.dsp.2025.105528'):
    assert claim in index, f'Missing from page: {claim}'

# The page must stay inside the project-page word budget (figures, tables and BibTeX excluded).
visible = re.sub(r'(?s)<(head|script|style|svg|pre|table)\b.*?</\1>|<[^>]+>', ' ', index)
words = len(html_lib.unescape(visible).split())
assert words <= 500, f'Page has {words} visible words, budget is 500'

print(f'OK: {len(page.links)} links, {len(page.ids)} IDs, {checked} result cells across '
      f'{len(page.tables)} HTML/CSV table pairs, SVG syntax, {words} visible words.')
