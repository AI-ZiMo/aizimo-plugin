---
name: seo-keyword-discovery
description: |
  Discover SEO keyword opportunities by monitoring platform subdomains
  (Vercel.app, Netlify.app, GitHub.io, etc.) in Similarweb landing pages,
  then analyze candidates via trend/competition/search-intent framework.
version: 1.2.0
platforms: [macos]
category: research
tags: [seo, keyword-research, vercel-find-word, overseas, competition-analysis]
---

# SEO Keyword Discovery via Platform Subdomains

Find new SEO opportunities by monitoring traffic on platform subdomains like
`*.vercel.app`, `*.netlify.app`, `*.github.io`, etc. — people who don't buy
their own domain rarely know SEO, making them easy targets.

---

## The Two-Article Foundation

This workflow combines two techniques from the Web.Cafe community:

1. **哥飞's Vercel 找词法** — Use Similarweb landing pages to find traffic
   on `.vercel.app` subdomains; these sites are often built by non-SEO-savvy
   developers using AI coding tools.
2. **野生小虎's Analysis Framework** — Three-factor evaluation (trend,
   competition, search intent) before committing to build.

---

## Step 1: Find Candidates

### Tool
- **Similarweb** (via `sim.3ue.co` proxy or `pro.similarweb.com`)
- User's preferred 3ue entry path:
  1. Open `https://dash.3ue.co/zh-Hans/#/page/m/home`
  2. In **SEO Tools**, click the first **打开** button for Similar Web (currently PRO 全球版)
  3. This opens `https://sim.3ue.co/`
  4. Navigate inside Similarweb: **关键词研究 → 着陆页**
  5. Search the platform domain, e.g. `vercel.app`
- Similarweb direct navigation pattern that worked for Vercel landing pages after opening 3ue proxy:
  `https://sim.3ue.co/#/organicsearch/pageAnalysis/landing-pages-v2/vercel.app/840/28d?key=vercel.app&pageFilter=%5B%7B%22url%22%3A%22vercel.app%22%2C%22searchType%22%3A%22domain%22%7D%5D&webSource=Total&selectedPageTab=Organic`

### Extraction pattern that worked
- Use Chrome AppleScript when logged into 3ue/Similarweb:
  - Open 3ue dashboard with `open -a "Google Chrome" "https://dash.3ue.co/zh-Hans/#/page/m/home"`
  - Click the first `打开` button via JS if needed
  - Extract `document.body.innerText` from the `sim.3ue.co` tab
- The landing page report text appears as grouped columns: URLs first, then clicks/share, change, avg rank, keyword counts, and hot keywords. Parse by locating the first URL such as `rule34dle.vercel.app`, then aligning the following 100 entries across these grouped sections.

### Platforms to Scan

| Platform | Subdomain Pattern |
|----------|-------------------|
| Vercel | `*.vercel.app` |
| Netlify | `*.netlify.app` |
| Cloudflare Pages | `*.pages.dev` |
| GitHub Pages | `*.github.io` |
| Firebase | `*.web.app` |
| Render | `*.onrender.com` |
| Railway | `*.railway.app` |

See `references/platform-subdomains.md` for the full list.

### What to Look For
- **Traffic volume** — higher is better, but even 2-5K clicks/month is viable
- **Growth trend** — positive change % is ideal, "New" or ">5,000%" means breakout
- **Keyword count** — more keywords = more search presence to capture
- **Avg. rank** — if > 5, there's room to improve

### macOS: Chrome AppleScript Extraction

The user may have Similarweb open in their own Chrome (logged in).
Use `osascript` to interact with it via "Allow JavaScript from Apple Events"
(Chrome → View → Developer → Allow JavaScript from Apple Events).

Workflow:
1. Find the correct tab with `URL contains "sim.3ue.co"` or `URL contains "similarweb"`
2. Execute JavaScript to click elements by matching text content
3. Extract `document.body.innerText` for analysis
4. For SPA navigation, use JavaScript DOM events (click, dispatchEvent)
   rather than URL hash changes, which may not trigger SPA routing.

See `references/chrome-applescript-extraction.md` for detailed patterns.

---

## Step 2: Three-Factor Analysis

For each candidate domain found, evaluate:

### 1. Trend — Is momentum real?
- Check Google Trends (24h / 4h granularity)
- Is the traffic source social media driven (unpredictable) or organic search (stable)?
- Avoid keywords that peaked 5+ days ago with no secondary trigger

### 2. Competition — Can you win?
- Search the keyword on Google
- Favorable signals:
  - First page has subpages/internal pages (not just homepages)
  - Competitor DR/backlinks are single-digit
  - No established brand names dominating the SERP
- Warning signals:
  - All premium domain suffixes registered
  - Front page is all high-DR authority sites
  - Search intent already well-served by multiple domains

### 3. Search Intent — Is there a gap?
- Look at what the current top results actually deliver
- If they miss the real user need (e.g. a "memory color quiz" served as a plain color picker), that's your gap
- The bigger the gap between what users search for and what the SERP offers, the better

Decision: **YES** if `trend is alive + competition is weak + intent gap exists`

---

## Step 2.5: Site Replication Analysis (new in v1.1)

When you find a candidate site via landing pages view, analyze its replication
potential by decomposing it into its core components. This is especially useful
for games, tools, and community sites that can be recreated with a different
data source or niche.

### Direct Site Analysis

Visit the candidate site directly via `browser_navigate` to understand it:

| Question | How to Answer |
|----------|--------------|
| What's the core mechanic? | Play/use the site; describe the loop |
| Who's the audience? | Content, language, tone |
| What features exist? | Daily challenge, streaks, leaderboards, etc. |
| What's the monetization? | Check page source for AdSense, Ko-fi, Patreon |
| Is there social virality? | Reddit links, Twitter, share buttons |
| Is it multi-language? | Language toggles in UI |
| What data powers it? | Public API, user-generated, scraping? |

### Monetization Detection

Examine the page source via `browser_console`:

```javascript
// Detect AdSense
document.querySelectorAll('script[src*="adsense"], script[src*="pagead"], ins.adsbygoogle').length

// Detect donation/patron
document.body.innerText.toLowerCase().includes('ko-fi') ||
document.body.innerText.toLowerCase().includes('patreon') ||
document.body.innerText.toLowerCase().includes('buymeacoffee')

// Detect affiliate links
document.querySelectorAll('a[href*="amazon"], a[href*="ref="]').length
```

### Decomposition Pattern

Break a successful site into replicable parts. Example from rule34dle:

```
┌──────────────────────────────────────────┐
│         Game Mechanic (generic)           │
│  "Higher or Lower" guessing game          │
├──────────────────────────────────────────┤
│         Data Source (swappable)           │
│  Rule34 API → can be swapped for:         │
│  Pokémon stats, movie box office,         │
│  Spotify plays, Wikipedia views, etc.     │
├──────────────────────────────────────────┤
│         Monetization (reusable)           │
│  Google AdSense + Ko-fi donations         │
├──────────────────────────────────────────┤
│         Viral Engine (replicable)         │
│  Daily Challenge → share score on         │
│  Reddit/Twitter → streak retention       │
└──────────────────────────────────────────┘
```

### Selecting a Clean Alternative

When the original site uses NSFW or copyrighted content, map to clean alternatives:

| Original Data | Clean Alternative |
|--------------|-------------------|
| Rule34 (adult) | Pokémon stats, movie box office |
| E621 (furry) | YouTube views, song popularity |
| Copyrighted character art | Public domain, Wikipedia data |
| Adult community | Family-friendly fandom (Pokémon, Marvel) |

### Decision Checklist

A site is worth replicating if:

- [ ] Core mechanic is generic (not patented/copyrighted)
- [ ] Data source can be changed to something clean
- [ ] Monetization is proven (ads, donations visible)
- [ ] SEO keywords > 10 and avg rank > 3 (room to improve)
- [ ] Social sharing / viral loop exists
- [ ] You can build MVP in 24-48 hours

## Step 3: Build & Launch (24h MVP) — Using Tanstarter

Use the Tanstarter template (`git clone https://github.com/yunhe-dev/Tanstarter.git <project-name>`)
to bootstrap tool/game sites quickly. Tanstarter provides TanStack Start + React 19 +
Cloudflare Workers + shadcn/ui out of the box — you strip the SaaS features and add
your tool page.

See `references/tanstarter-tool-site-quickstart.md` for the exact file-change checklist:
which configs to disable, which components to simplify, and the tabbed-tool UI pattern
(proven on Marvel Rivals Randomizer with 36 characters, 4 randomizer modes, and SEO FAQ).

Follow 野生小虎's playbook plus the May new-keyword game-site case study:
1. **Day 1 noon/evening — Find keyword and decide fast**
   - Use Similarweb landing pages to find rising platform subdomains.
   - Validate with Google Trends, SERP supply, registered domains, competitor products, and implementation complexity.
   - For social-driven hot words, decide by: old mechanic/new packaging, weak current SERP supply, low build complexity, and whether the trend is still climbing.
2. **Same night — Build a playable MVP, not a perfect clone**
   - Homepage must immediately satisfy search intent: if users search for a game, let them play on entry.
   - Differentiate from competitors with smoother onboarding, shareability, retention, or content breadth.
   - Optimize for “users can play and complete a loop within seconds/minutes”; ship before the window closes.
3. **Before launch — Set up TDH + technical SEO + analytics**
   - Title/Description/Header, FAQ, sitemap, robots.txt, canonical, GA4/GSC, favicon, mobile UX.
   - Reserve ad slots in layout early, but don't necessarily turn ads on immediately.
4. **Launch — Submit to search engines immediately**
   - Submit to Google Search Console, Bing/Yandex where relevant.
5. **Post-launch — Iterate from real data**
   - Watch bounce rate and retention; add daily high scores, achievement/result pages, share cards, badges/ranks, difficulty labels, content expansion for long-tail search.
   - Export GA/GSC CSV for AI-assisted diagnosis and iteration when API/account access is unavailable.
6. **Infrastructure must match growth**
   - Vercel free tier can be exhausted quickly by breakout traffic; consider Cloudflare Pages for static games because static requests are more forgiving.

### Monetization Lessons for Hot Game Sites

- Early stage: protect UX; avoid heavy ads before rankings and product experience stabilize.
- Traffic spike: quickly test monetization platforms/forms to close the loop.
- Stable traffic: optimize ad quality and platform mix.
- AdSense/Ezoic: better quality but slower review and stricter content standards. Avoid large numbers of near-template pages; they may be judged as low-value content.
- Monetag/Adsterra: fast approval but lower/unpredictable CPM and possible UX issues. Adsterra Social Bar/Native Banner can work, but watch for redirect/page-hijack behavior; iframe isolation may break tracking.

### Copywriting and Content Quality

- AI-written SEO copy often sounds like product documentation. For game/hot-word pages, require emotional hooks, player POV, and “would I want to play/share this?” self-review.
- Content expansion should create genuinely useful long-tail pages, not thin templated variants.

### Backlink Workflow

- First goal after launch: speed up indexing with instantly visible backlinks, even if many are nofollow.
- Build an AI-assisted backlink system: collect targets, submit, record result/status/type, classify nofollow/dofollow, and update from competitor backlink exports.
- Human should judge strategy and quality; dofollow/quality links often matter more than raw count.

---

## Pitfalls

- **Don't judge SEO fixes in 24 hours** — Google needs time to re-crawl
- **UGC (leaderboards, comments) can dilute your keywords** — isolate UGC from SEO main document
- **Don't add ads too early** — ranking isn't stable yet
- **Favicon quality matters** — Google is slow to update it in SERPs
- **Don't hand-type identifiers** — domains, API keys, project names → always copy-paste
- **Mobile UX is not optional** — check every UI change on mobile before deploying
- **Don't rely on one keyword** — multilanguage + long-tail + typo SEO as safety net
- **Google isn't the only search engine** — submit to Yandex, Bing too
- **SPA hash routing may not work** — when automating Similarweb via URL changes,
  the SPA may ignore hash-only navigation. Use JavaScript clicks on DOM elements instead.

---

## Related Skills

- `macos-computer-use` — for GUI-level desktop automation (different layer than AppleScript)
- `obsidian` — save findings as structured notes for knowledge base

## Additional References

- `references/vercel-game-opportunity-scan-2026-06-21.md` — live Vercel小游戏找词扫描：Top100游戏候选、Marvel Rivals Randomizer优先级判断、Favorite Pokémon机制借鉴、FF Resonance观察，以及 Similarweb 表格解析注意。
- `references/may-new-keyword-game-case-study.md` — 5月新词赛第3名小游戏站复盘：从 Vercel 子域名发现猜色小游戏，快速上线MVP，月12万UV；补充产品承接、数据迭代、基础设施、广告变现、外链策略等实战经验。
- `references/search-engine-fallback.md` — fallback workflow for discovering indexed platform-subdomain candidates via search engines when Similarweb/3ue Landing Pages automation does not trigger.
- `references/3ue-similarweb-vercel-workflow.md` — session-proven 3ue dashboard → sim.3ue.co entry path, full Vercel Landing Pages URL, and table parsing/triage notes.
- `references/vercel-game-opportunity-scan-2026-06.md` — concrete game/game-tool scan example from Vercel landing pages; includes candidate table and MVP decision pattern for Marvel Rivals-style randomizer opportunities.
