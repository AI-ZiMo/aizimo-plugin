# Vercel小游戏找词实战扫描 — 2026-06-21

## 数据源

- Similarweb / 3ue `vercel.app` Landing Pages
- 时间窗口：最近 28 天，页面显示 `As of Jun 18`
- 目标：从 `*.vercel.app` Top 100 自然着陆页里筛小游戏、游戏工具、可快速MVP化的新词机会。

## 本轮候选

| 排名 | 页面 | 28天点击 | 变化 | 关键词数 | 热词 |
|---:|---|---:|---:|---:|---|
| 1 | `rule34dle.vercel.app` | 48.9K | -27% | 272 | rule34dle |
| 5 | `favoritepokemon.vercel.app` | 17.8K | +305% | 107 | favorite pokemon vercel |
| 6 | `nuzlocke-redux.vercel.app` | 17.2K | +30% | 198 | nuzlocke tracker |
| 7 | `sprunki1996.vercel.app/games/sbrunga` | 16.9K | +239% | 14 | sbrunga |
| 9 | `minecraft-enchantment-order.vercel.app` | 14.9K | -4.9% | 894 | enchant calculator minecraft |
| 15 | `neontetris.vercel.app` | 12.1K | -9.4% | 14 | てとりす |
| 16 | `tomolife1.vercel.app` | 11.5K | +745% | 52 | tomodachi life |
| 17 | `nyt-wordle-app.vercel.app` | 11.3K | +324% | 25 | wordle |
| 35 | `ff-resonance.vercel.app` | 5.2K | 新 | 15 | ffレゾナンス |
| 36 | `rival-roulette.vercel.app` | 5.2K | +107% | 97 | marvel rivals randomizer |
| 41 | `l-randomizer.vercel.app` | 4.7K | -4.7% | 192 | lol randomizer |
| 42 | `queensgame.vercel.app` | 4.7K | -3.1% | 157 | queens game |
| 56 | `palia-garden-planner.vercel.app` | 3.7K | +20% | 130 | palia garden planner |
| 94 | `daily-logic-puzzles.vercel.app/linkedin-games/queens-answer-today` | 2.7K | +113% | 44 | queens solution today |
| 96 | `minecraft-shapes.vercel.app` | 2.6K | +53% | 94 | minecraft shapes |
| 98 | `rockpaperscissors-ai.vercel.app` | 2.6K | -53% | 125 | rock paper scissors |

## 推荐优先级

### 1. Marvel Rivals Randomizer / Roulette

最适合快速MVP。

信号：
- `rival-roulette.vercel.app` 28天自然点击约 5.2K，增长 +107%。
- 关键词数约 97，热词为 `marvel rivals randomizer`。
- 原站核心功能很简单：随机选择 Marvel Rivals 角色，支持职业筛选、角色图库、Party Randomizer、Buy Me A Coffee；未见明显 AdSense。
- SERP 供给主要是通用 Spin Wheel、Perchance、Reddit/YouTube 讨论，垂直产品供给不强。

MVP方向：
- 不要只做“复制 Rival Roulette”。
- 做成 `Marvel Rivals Randomizer`：Hero Randomizer + Team Randomizer + Challenge Generator。
- 首页必须直接可玩：随机英雄、按职业筛选、排除不想玩的英雄。
- 差异化：6人队伍随机、2/3/6人组队分配、挑战生成、结果分享图。

TDH草案：
- Title: `Marvel Rivals Randomizer - Pick a Random Hero, Team, or Challenge`
- Description: `Use this Marvel Rivals randomizer to pick a random hero, generate a team, spin a challenge wheel, and make your next match more fun.`
- H1: `Marvel Rivals Randomizer`
- H2: `Random Marvel Rivals Hero`, `Marvel Rivals Team Randomizer`, `Marvel Rivals Challenge Generator`, `Marvel Rivals Character Wheel`

### 2. Favorite Pokémon Picker / “Every Pokémon is someone’s favorite”机制

信号强，但直接追原词偏晚。

观察：
- `favoritepokemon.vercel.app` 在 Similarweb 中 17.8K 点击、+305%。
- 打开时 Vercel 部署已暂停，但项目已迁到 `favoritepokemon.app`。
- 新站包含 Declare / Game / Explore / Pokédex / Stats，机制是选择最多6个最喜欢的 Pokémon、写理由、下载分享卡、探索粉丝声明数据。

启发：
- 可复制的是“每个 X 都是某人的最爱”这一集体声明 + 分享卡 + 数据看板机制。
- 不建议直接做 Pokémon 同质站；更适合迁移到其他 IP/圈层/对象集合。

### 3. FF Resonance / Final Fantasy Soul Analysis

信号新，但需要谨慎判断日语市场和 IP 风险。

观察：
- `ff-resonance.vercel.app` 显示为新，5.2K 点击，热词 `ffレゾナンス`。
- 打开后是日语为主的非官方粉丝心理测试/灵魂分析站：`FINAL FANTASY SOUL ANALYSIS`，支持 JP/EN，含诊断、排行榜、角色图鉴、更新信息。
- 可借鉴的是“角色/系列粉丝人格测试 + 分享结果 + 排行榜”。

## 排除/谨慎项

- `rule34dle`：NSFW/成人内容风险高，不建议直接做。可借鉴 Higher-or-Lower 机制，替换为干净数据源。
- `tomolife1.vercel.app`：打开是 Tomodachi Life 下载站，含 sponsor app 安装弹窗；Nintendo IP + 下载诱导风险高，不建议。
- `queens game / queens solution today`：需求明确，但 SERP 已有多个专门站和攻略站，偏每日答案内容运营，不是最佳轻MVP。
- `palia garden planner`：健康游戏工具，但已有多个专业工具站，且需要理解 Palia 种植规则，复杂度高于 randomizer。

## Similarweb解析注意

本次页面抽取时，`document.body.innerText` 的 Top100 表格顺序为：URL列表 → 点击量/占比对 → 变化 → `查看趋势`重复 → 关键词数/`所有关键词`对 → 热词。没有稳定出现“平均排名”列；解析脚本不要假设一定存在 avg rank。
