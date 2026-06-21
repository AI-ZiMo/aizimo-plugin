# Search Engine Fallback for Platform-Subdomain SEO Discovery

Use this when Similarweb / 3ue Landing Pages automation is unavailable, the React input does not trigger report loading, or you need a quick first-pass candidate scan before paid-tool validation.

## Goal

Find indexed small projects hosted on platform subdomains (`vercel.app`, `netlify.app`, `pages.dev`, `github.io`) that reveal active demand patterns: daily games, quizzes, calculators, generators, trackers, converters.

## Query Patterns

Run platform-scoped searches:

```text
site:vercel.app "daily" "game"
site:vercel.app "quiz" "daily"
site:vercel.app "calculator"
site:vercel.app "generator" "AI"
site:vercel.app "tracker"
site:vercel.app "converter"
```

Repeat with other platforms:

```text
site:netlify.app "daily" "game"
site:pages.dev "calculator"
site:github.io "quiz" "daily"
```

DuckDuckGo HTML often works better than Google when Google blocks automated queries:

```bash
python3 - <<'PY'
import requests, urllib.parse, re, html
queries = [
  'site:vercel.app "daily" "game"',
  'site:vercel.app "quiz" "daily"',
  'site:vercel.app "generator" "AI"',
]
for q in queries:
    print('\n###', q)
    r = requests.get('https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(q),
                     headers={'User-Agent':'Mozilla/5.0'}, timeout=20)
    for href, title in re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>', r.text, re.S)[:10]:
        u = html.unescape(href)
        title = html.unescape(re.sub('<.*?>', '', title)).strip()
        if 'uddg=' in u:
            u = urllib.parse.unquote(re.search(r'uddg=([^&]+)', u).group(1))
        print(title, '|', u)
PY
```

## Candidate Triage

For each result, fetch title/description and check whether it has analytics/ads:

```bash
python3 - <<'PY'
import requests, re
urls = ['https://example.vercel.app/']
for url in urls:
    print('\n###', url)
    r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=15)
    txt = r.text
    title = re.search(r'<title[^>]*>(.*?)</title>', txt, re.S|re.I)
    desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', txt, re.S|re.I)
    print('status', r.status_code, 'len', len(txt))
    print('title', (title.group(1).strip() if title else '')[:120])
    print('desc', (desc.group(1).strip() if desc else '')[:200])
    print('adsense', bool(re.search('adsbygoogle|pagead|adsense', txt, re.I)))
    print('analytics', bool(re.search('gtag|googletagmanager|analytics', txt, re.I)))
PY
```

Then validate demand using autocomplete:

```bash
curl -s 'https://suggestqueries.google.com/complete/search?client=firefox&q=colordle%20game'
curl -s 'https://ac.duckduckgo.com/ac/?q=colordle+game&type=list'
```

## High-Signal Pattern: Daily Game + Answer/Hints/Archive

When autocomplete shows modifiers like these, the niche may support an SEO content matrix rather than only a game page:

```text
answer today
hints today
unlimited
archive
past answers
games like <name>
for kids / for adults / for seniors
```

Example pattern discovered from `colordle game`:

```text
colordle game
colordle answer today
colordle unlimited game
colordle hints today
today's colordle game
games like colordle
```

This implies a page matrix:

```text
Daily <Niche> Game
<Niche> Answer Today
<Niche> Hints Today
<Niche> Archive
<Niche> Unlimited
Games Like <Niche>
```

## When to Return to Similarweb

The fallback only proves indexability and search-suggestion demand. Before buying a domain or building, return to Similarweb/3ue or another keyword tool to verify traffic, rank, and landing page metrics when possible.
