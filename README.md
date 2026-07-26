# yunhe-skills

> 中文文档请看：[`README.zh-CN.md`](./README.zh-CN.md)

A curated collection of practical OpenClaw skills maintained by **yunhe-dev**.

This repository currently focuses on image-generation workflows and X (Twitter) content assets.

## Included Skills

### 1) `gpt-image-sub2api`
Generate and edit GPT-Image-2 images through `https://sub2api26.zeabur.app/`.

**What it supports**
- Text-to-image and image editing
- URL, data URI, and local reference images
- Common aspect ratios and `1k` / `2k` / `4k` resolution presets
- Local PNG, JPEG, or WebP output

**Main script**
- `skills/gpt-image-sub2api/scripts/generate.js`

**First-time setup**

```bash
mkdir -p ~/.sub2api
echo '{"api_key":"PASTE_YOUR_KEY_HERE","base_url":"https://sub2api26.zeabur.app/"}' > ~/.sub2api/config.json
chmod 600 ~/.sub2api/config.json
```

If the config is missing or invalid, the script prints the setup URL and the correct command for macOS/Linux or Windows. Once configured, later generations run directly.

### 2) `nano-banana-clawapi`
Generate images with the BLT-compatible Nano Banana endpoint, now defaulting to `clawapi.co`.

**What it supports**
- Text-to-image
- Multiple aspect ratios (`1:1`, `16:9`, `21:9`, etc.)
- Multiple sizes (`1K`, `2K`, `4K`, `512px`)
- Sync mode and async mode with task polling
- Optional reference images

**Main script**
- `skills/nano-banana-clawapi/scripts/main.ts`

### 3) `Openclaw-X-article-cover-generator`
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
- Node.js 18+
- `uv` (recommended runner)
- Provider credentials configured as described by each skill

```bash
# nano-banana-clawapi only
export BLT_API_KEY="your-key"
export BLT_API_BASE_URL="https://clawapi.co"
```

## Quick Start

```bash
# gpt-image-sub2api
node ./skills/gpt-image-sub2api/scripts/generate.js \
  --prompt "A cinematic robot walking through Shanghai in the rain" \
  --size 16:9 \
  --resolution 2k \
  --output gpt-image.png
```

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
