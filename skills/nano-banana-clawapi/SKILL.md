---
name: nano-banana-clawapi
description: Generate images with clawapi using prompts to create images. Supports text-to-image, reference images, sync/async generation, task polling, and multi-image async generation workflows (run multiple tasks in parallel and poll by task IDs), plus aspect ratios and size presets. Use when the user asks to generate, create, draw, or edit images.
version: 1.0.0
metadata:
  openclaw:
    requires:
      anyBins:
        - node
---

# Image Generation (clawapi)

Official image generation via `clawapi.co`.

## Script Directory

**Agent Execution**:
1. `{baseDir}` = this SKILL.md file's directory
2. Script path = `{baseDir}/scripts/main.ts`
3. Execute with: `node {baseDir}/scripts/main.ts ...`

## Step 0: Load Credentials ⛔ BLOCKING

This step must complete before generation.

Check env files (priority: project -> user):

```bash
test -f .yunhe-skills/.env && echo "project"
test -f "$HOME/.yunhe-skills/.env" && echo "user"
```

Expected variables:

- `BLT_API_KEY` (required)
- `BLT_API_BASE_URL` (optional, default `https://clawapi.co`)

Load priority:

1. CLI args (`--api-key`)
2. Existing process env
3. `<cwd>/.yunhe-skills/.env`
4. `~/.yunhe-skills/.env`

Notes:

- If `BLT_API_BASE_URL` is `https://clawapi.co`, it auto-resolves to `https://clawapi.co/v1`.
- If `BLT_API_BASE_URL` already contains path (for example `/v1`), it is used as-is.

## Usage

```bash
# Basic text-to-image
node {baseDir}/scripts/main.ts \
  --prompt "A futuristic city skyline at sunset" \
  --output out.png

# With ratio and size
node {baseDir}/scripts/main.ts \
  --prompt "A cinematic mountain landscape" \
  --ratio 16:9 \
  --size 2K \
  --output landscape.png

# With reference image(s)
node {baseDir}/scripts/main.ts \
  --prompt "Keep style, turn into night scene" \
  --reference ./source.png \
  --output edited.png

# Async generation (recommended for slow/complex tasks)
node {baseDir}/scripts/main.ts \
  --prompt "Highly detailed sci-fi concept art" \
  --ratio 21:9 \
  --size 4K \
  --async-mode \
  --output concept.png

# Poll existing task
node {baseDir}/scripts/main.ts --check-task TASK_ID
```

## Options

| Option | Description |
|--------|-------------|
| `--prompt <text>`, `-p` | Prompt text (required unless `--check-task`) |
| `--output <path>`, `-o` | Output path. If omitted, a timestamp filename is auto-generated |
| `--ratio <ratio>`, `--ar <ratio>`, `-r` | Aspect ratio, default `1:1` |
| `--size <size>`, `-s` | Size preset: `512px`, `1K`, `2K`, `4K` (default `1K`) |
| `--async-mode`, `--async`, `-a` | Enable async generation + polling |
| `--reference <files...>`, `--ref <files...>` | Reference image paths (repeatable) |
| `--api-key <key>`, `-k` | Override API key |
| `--check-task <task_id>` | Query async task status and return JSON |
| `--webhook <url>` | Optional webhook for async task callbacks |
| `--help`, `-h` | Show usage help |

## Aspect Ratios

Supported ratios:

- `1:1`
- `4:3`, `3:4`
- `16:9`, `9:16`
- `2:3`, `3:2`
- `4:5`, `5:4`
- `21:9`
- `1:4`, `4:1`, `1:8`, `8:1`

## Size Presets

| Preset | Typical Dimensions | Use Case |
|--------|--------------------|----------|
| `512px` | ~512x512 | Fast drafts |
| `1K` | ~1024 edge | Default quality |
| `2K` | ~2048 edge | Higher quality |
| `4K` | ~4096 edge | Maximum detail (usually async) |

## Sync vs Async Guidance

Use sync mode when:

- Prompt is simple
- You need quick feedback
- `1K` or `2K` is enough

Use async mode when:

- Prompt is complex
- You request `4K` or large ratios
- Sync call may time out

If sync mode times out, rerun with `--async-mode`.

## Multi-image Async Generation

When users need multiple outputs:

1. Start multiple async tasks (one command per prompt/image target) using `--async-mode`.
2. Capture each returned `task_id`.
3. Poll each task with `--check-task <task_id>` until completion.
4. Download/save output per task target path.

Recommended usage pattern (parallel tasks):

```bash
node {baseDir}/scripts/main.ts --prompt "Prompt A" --async-mode --output out-a.png
node {baseDir}/scripts/main.ts --prompt "Prompt B" --async-mode --output out-b.png
node {baseDir}/scripts/main.ts --prompt "Prompt C" --async-mode --output out-c.png
```

Then poll:

```bash
node {baseDir}/scripts/main.ts --check-task TASK_ID_A
node {baseDir}/scripts/main.ts --check-task TASK_ID_B
node {baseDir}/scripts/main.ts --check-task TASK_ID_C
```

## Provider/API Behavior

- API endpoint: `POST {base}/images/generations`
- Async status endpoint: `GET {base}/images/tasks/{task_id}`
- Model: `gemini-3.1-flash-image-preview`
- Payload fields include:
  - `prompt`
  - `aspect_ratio`
  - `image_size`
  - `response_format`
  - optional `image` (reference image Data URLs)

## Error Handling

| Error | Meaning | Action |
|------|---------|--------|
| `No API key provided` | No key found in CLI/env/.env | Set `BLT_API_KEY` or pass `--api-key` |
| `Invalid aspect ratio` | Unsupported ratio value | Use supported ratio list above |
| `Request failed: 401/403` | Invalid key or permission issue | Rotate key / verify account permissions |
| `Request timeout after 120s` | Sync request took too long | Retry with `--async-mode` |
| `Reference image not found` | Local file path invalid | Fix file path |
| `Task failed` | Async job failed server-side | Retry with revised prompt or lower size |

## Workflow Recommendations

Draft-to-final workflow:

1. Draft with `--size 1K --ratio 1:1`
2. Improve prompt
3. Final render with target ratio and `2K/4K`
4. If final render is slow, switch to async

Reference-image workflow:

1. Start with one reference image
2. Keep prompt explicit about what to preserve/change
3. Increase size after style/content is stable

## Security Notes

- Never commit `BLT_API_KEY` into git-tracked files.
- Store secrets in `~/.yunhe-skills/.env` or CI secrets.
- If a key is exposed in chat/logs, rotate it immediately.
