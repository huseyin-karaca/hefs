"""Check the built site's local links, fragments, and image paths."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import yaml

ROOT = Path(__file__).resolve().parents[1] / 'site'
CONFIG = yaml.safe_load((ROOT.parent / 'mkdocs.yml').read_text())
BASE_PATH = urlsplit(CONFIG['site_url']).path


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path, self.ids, self.links = path, set(), []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag in ('a', 'link', 'img', 'script'):
            value = attrs.get('href') if tag in ('a', 'link') else attrs.get('src')
            if value:
                self.links.append(value)


def main():
    pages = {p.resolve(): Page(p) for p in ROOT.rglob('*.html')}
    if not pages:
        raise SystemExit('Build the site first: python -m mkdocs build --strict')
    errors, checked = [], 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            local_path = unquote(url.path)
            if local_path.startswith(BASE_PATH):
                local_path = local_path[len(BASE_PATH):]
                target = ROOT / local_path
            elif local_path.startswith('/'):
                target = ROOT / local_path.lstrip('/')
            else:
                target = path.parent / local_path if local_path else path
            target = target.resolve()
            if target.is_dir():
                target /= 'index.html'
            checked += 1
            if not target.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{path.relative_to(ROOT)}: missing fragment {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Checked {checked} local references across {len(pages)} HTML pages; no broken links or fragments.')


if __name__ == '__main__':
    main()
