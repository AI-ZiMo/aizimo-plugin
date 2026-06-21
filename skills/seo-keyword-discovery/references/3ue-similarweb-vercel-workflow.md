# 3ue Similarweb Vercel Landing Pages Workflow

Session-proven workflow for using the user's 3ue subscription to access Similarweb and run the Vercel 找词法.

## Entry path

The reliable entry point is not `similarweb.com` directly. Use the 3ue dashboard first:

1. Open `https://dash.3ue.co/zh-Hans/#/page/m/home` in the user's real Chrome.
2. Wait for the subscriptions page to load.
3. Click the **first** `打开` button under `SEO Tools` / `PRO 全球版` to launch `https://sim.3ue.co/`.
4. Use the opened `sim.3ue.co` tab for Similarweb work.

AppleScript DOM click pattern:

```applescript
set js to "(function(){ const buttons=[...document.querySelectorAll('button')]; const opens=buttons.filter(b=>(b.innerText||b.textContent||'').trim()==='打开'); if(opens[0]){ opens[0].click(); return 'clicked first open'; } return 'open button not found'; })()"
execute t javascript js
```

## Vercel Landing Pages URL

After the Similarweb SPA is loaded, direct navigation to the report URL works when the URL includes `key` and `pageFilter`:

```text
https://sim.3ue.co/#/organicsearch/pageAnalysis/landing-pages-v2/vercel.app/840/28d?key=vercel.app&pageFilter=%5B%7B%22url%22%3A%22vercel.app%22%2C%22searchType%22%3A%22domain%22%7D%5D&webSource=Total&selectedPageTab=Organic
```

If navigating to a shorter route like `/landing-pages-v2/vercel.app/999/...` shows “输入查询以查看此报告”, the app has not bound the domain query. Use the full URL above or search/select `vercel.app` from the Similarweb UI.

## Extracting and parsing table text

`document.body.innerText` returns the visible table in column chunks, not row-wise. For the Vercel landing pages table, the extracted sequence is generally:

1. URL list, 100 rows
2. Clicks/share pairs, 200 lines
3. Change values, 100 lines
4. `查看趋势` repeated
5. Average rank values, 100 lines
6. Keyword counts as `N` then `所有关键词`, 200 lines
7. Hot keywords, 100 lines

Parsing pattern:

```python
# locate first URL row, collect domains until first numeric click value
# then map chunks by fixed column lengths: clicks/share, change, rank, kw_count, hot_keyword
```

Useful output fields:

```text
rank_no,url,clicks,change,avg_rank,kw_count,hot_keyword
```

## Candidate triage heuristic

For a first-pass shortlist, prefer pages with:

- clicks >= 1K in the last 28 days
- positive growth, `新`, or `> 5,000%`
- average rank worse than ~5 if the intent is to outrank, or low keyword count if the idea is a viral/niche tool
- tool/calculator/planner/tracker/randomizer intent over piracy/adult/streaming/brand-only traffic

Good classes spotted in the session:

- Minecraft calculators/generators/planners — clear utility intent, many long-tails, monetization possible via AdSense.
- “Every X is someone’s favorite” voting/collective directory mechanics — strong social sharing, but avoid direct IP/copyright risk.
- YouTube BPM/key/song identification tools — strong global utility intent but higher implementation complexity.
- USCIS/visa trackers — high-intent, potentially monetizable, but needs careful data/legal handling.
- Game randomizers/planners — fast MVPs but trend lifecycle can be short.
