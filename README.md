# yunhe-skills

> 中文文档请看：[`README.zh-CN.md`](./README.zh-CN.md)

A curated collection of practical OpenClaw skills maintained by **yunhe-dev**.

This repository currently focuses on image-generation workflows and X (Twitter) content assets.

## Included Skills

### 1) `nano-banana-clawapi`
Generate images with the BLT-compatible Nano Banana endpoint, now defaulting to `clawapi.co`.

**What it supports**
- Text-to-image
- Multiple aspect ratios (`1:1`, `16:9`, `21:9`, etc.)
- Multiple sizes (`1K`, `2K`, `4K`, `512px`)
- Sync mode and async mode with task polling
- Optional reference images

**Main script**
- `skills/nano-banana-clawapi/scripts/main.ts`

### 2) `Openclaw-X-article-cover-generator`
Generate OpenClaw-themed X article cover images with a fixed composition.

**Composition rules**
- Lobster logo subject on the **right 1/4**
- Text area on the **left 3/4**
- Keep the lobster subject identity consistent with the reference image
- If user asks for 5:2, use **21:9** as the nearest supported ratio

**Main script**
- `skills/Openclaw-X-article-cover-generator/scripts/generate_cover.py`

## Requirements

- Python 3.10+
- `uv` (recommended runner)
- Valid API keys in environment variables

```bash
export BLT_API_KEY="your-key"
export BLT_API_BASE_URL="https://clawapi.co"
```

## Quick Start

```bash
# nano-banana-clawapi
node ./skills/nano-banana-clawapi/scripts/main.ts \
  --prompt "A futuristic city skyline at sunset" \
  --ratio 16:9 \
  --size 2K \
  --output demo.png
```

```bash
# Openclaw-X-article-cover-generator
uv run ./skills/Openclaw-X-article-cover-generator/scripts/generate_cover.py \
  --title "Zero-Threshold OpenClaw" \
  --reference "https://example.com/reference.jpg" \
  --output x-cover.png \
  --size 2K \
  --async-mode
```

## Notes

- This repo is actively evolving.
- Skill docs are in each skill folder (`SKILL.md`).
- If a packaged release misses binary assets, prefer URL-based references for reproducibility.
