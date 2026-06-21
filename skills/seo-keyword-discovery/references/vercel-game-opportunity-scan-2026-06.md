# Vercel Game Opportunity Scan — June 2026

Session-specific example for applying the Vercel 找词法 to lightweight games and game-adjacent tools.

## Source

Similarweb / 3ue `vercel.app` landing pages, last 28 days, organic traffic. User asked to look specifically for small-game opportunities.

## Game / game-tool candidates found

| Rank | URL | Clicks | Change | Keywords | Hot keyword | Notes |
|---:|---|---:|---:|---:|---|---|
| 1 | `rule34dle.vercel.app` | 115.4K | -25% | 319 | `rule34dle` | Higher/lower mechanic; proven ads + Ko-fi; NSFW source, use only as mechanics reference. |
| 7 | `nuzlocke-redux.vercel.app` | 26.9K | +26% | 240 | `nuzlocke tracker` | Pokémon Nuzlocke tracker; strong utility intent but IP/data complexity. |
| 8 | `favoritepokemon.vercel.app` | 22.8K | +416% | 117 | `favorite pokemon vercel` | Deployment was paused during browser check; still useful as social/voting mechanic signal. |
| 13 | `sprunki1996.vercel.app/games/sbrunga` | 16.9K | +239% | 14 | `sbrunga` | Trend/game-specific; investigate lifecycle before building. |
| 14 | `minecraft-enchantment-order.vercel.app` | 16.5K | -10% | 1,119 | `enchant calculator minecraft` | Game calculator pattern; many long-tail keywords. |
| 23 | `neontetris.vercel.app` | 12.1K | -9.4% | 14 | `てとりす` | Pure game clone; less preferred than game utility unless differentiated. |
| 31 | `queensgame.vercel.app` | 8.5K | -5.6% | 181 | `queens game` | Puzzle game / daily game signal. |
| 47 | `rival-roulette.vercel.app` | 6.2K | +122% | 107 | `marvel rivals randomizer` | Strong MVP candidate: simple randomizer/roulette for hot game. |
| 49 | `jeffgoldblumle.vercel.app` | 6.1K | +936% | 6 | `jeffgoldblumle` | Novelty daily game; few keywords, likely trend/viral. |
| 60 | `l-randomizer.vercel.app` | 5.3K | -15% | 214 | `randomizer lol` | Game randomizer pattern. |
| 63 | `crosswordle.vercel.app` | 5.1K | +26% | 29 | `crosswordle` | Wordle-like puzzle; multilingual links and share/stats mechanics. |
| 71 | `palia-garden-planner.vercel.app` | 4.6K | +35% | 168 | `palia garden planner` | Game planner/calculator pattern; Ko-fi visible. |
| 95 | `daily-logic-puzzles.vercel.app/linkedin-games/crossclimb-answer-today` | 3.6K | -51% | 107 | `crossclimb today` | Daily answer/content template; AdSense visible; maintenance burden. |

## Page checks / monetization observations

- `rival-roulette.vercel.app`: title `Rival Roulette`, description `Randomly select your next Marvel Rivals character!`; role filters, character gallery, party randomizer, Buy Me A Coffee; no AdSense detected.
- `palia-garden-planner.vercel.app`: interactive grid planner for crops/fertilizers, layout generator, Ko-fi; game system planner pattern.
- `nuzlocke-redux.vercel.app`: tracker with New Game / Load Game / Guides; utility rather than standalone game.
- `crosswordle.vercel.app`: daily puzzle, unlimited mode, builder, share/stats, multiple languages.
- `rule34dle.vercel.app`: higher/lower guessing game; AdSense + Ko-fi detected; NSFW data source, avoid cloning directly.
- `daily-logic-puzzles...crossclimb-answer-today`: daily answer pages, archives, internal links, app funnel, AdSense.

## Decision pattern from this scan

For the user's current SEO/product goals, prefer **game-adjacent lightweight tools** over pure game clones:

1. Hot game + randomizer / roulette / challenge generator
2. Game system + planner / calculator / tracker
3. Daily puzzle answers / hints / archive pages
4. Pure Wordle/Tetris-style clone only if it has a distinctive mechanic or social loop

Best first MVP candidate from this scan: **Marvel Rivals Randomizer / Roulette**.

### Candidate MVP outline

Main keywords:
- `marvel rivals randomizer`
- `marvel rivals roulette`
- `marvel rivals wheel`
- `marvel rivals character randomizer`
- `marvel rivals team randomizer`
- `marvel rivals challenge generator`

Core features:
- Random Hero
- Random Team
- Role filter: Vanguard / Duelist / Strategist
- Challenge Generator
- Character Gallery
- Share result
- SEO FAQ

TDH draft:
- Title: `Marvel Rivals Randomizer - Random Hero, Team & Challenge Generator`
- Description: `Use this fan-made Marvel Rivals randomizer to pick a random hero, generate team comps, spin a character wheel, and create fun challenge rules.`
- H1: `Marvel Rivals Randomizer`

Risk note: Marvel is strong IP. Use fan-made/unofficial language and avoid official-looking branding/domain choices.

## Workflow note

When extracting Similarweb table text via Chrome AppleScript, the known column-chunk parse still worked for this session. For game scans, after parsing the top 100, filter URLs/hot keywords with terms like:

`game`, `games`, `dle`, `wordle`, `tetris`, `nuzlocke`, `pokemon`, `roulette`, `randomizer`, `puzzle`, `crossword`, `tier-list`, `minecraft`, `planner`, `tracker`, `challenge`, `answer today`.
