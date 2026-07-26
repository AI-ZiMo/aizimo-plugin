---
name: gpt-image-sub2api
description: 使用 Sub2API 的 GPT-Image-2 接口生成或编辑图片，支持文生图、URL 或本地参考图、常用宽高比及 1k/2k/4k 分辨率。当用户说“生图”“画图”“生成图片”“gpt-image”“Image2 生图”“帮我画”“用 GPT 画”“把这张图改成”“参考这张图”或提出 AI 图片生成、图片编辑请求时使用。
---

# GPT-Image-2 via Sub2API

通过 `https://sub2api26.zeabur.app/` 调用 `gpt-image-2`。脚本会把比例与分辨率转换为像素尺寸，调用 Sub2API 的 OpenAI 兼容图片接口，并把结果保存为本地图片。

## 前置检查

1. 运行 `node -v`，确认 Node.js 18 或更高版本可用。
2. 检查 `~/.sub2api/config.json` 是否存在，并确认 `api_key` 非空且不是示例值。
3. 若未配置，引导用户访问 `https://sub2api26.zeabur.app/` 登录并创建 API Key，然后执行：

```bash
mkdir -p ~/.sub2api
echo '{"api_key":"PASTE_YOUR_KEY_HERE","base_url":"https://sub2api26.zeabur.app/"}' > ~/.sub2api/config.json
chmod 600 ~/.sub2api/config.json
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.sub2api" | Out-Null
'{"api_key":"PASTE_YOUR_KEY_HERE","base_url":"https://sub2api26.zeabur.app/"}' | Set-Content "$env:USERPROFILE\.sub2api\config.json"
```

脚本也会自行执行相同检查。若缺少配置、JSON 无效或 Key 仍是示例值，会直接显示注册网址、对应系统的配置命令和配置文件位置。配置一次后，后续调用直接生成。

## 生成流程

将本文件目录作为 `SKILL_DIR`，调用：

```bash
node "$SKILL_DIR/scripts/generate.js" \
  --prompt "用户的提示词" \
  --size "1:1" \
  --resolution "2k" \
  --output "output.png"
```

成功后向用户展示保存的本地图片，并说明实际尺寸。

### 判断模式

- 只有文字描述：调用 `/v1/images/generations`。
- 有参考图片：传一个或多个 `--image-url`，调用 `/v1/images/edits`。
- `--image-url` 支持 HTTP/HTTPS URL、data URI 和本地文件路径；本地文件由脚本自动转为 data URI。
- 最多传入 16 张参考图。

```bash
node "$SKILL_DIR/scripts/generate.js" \
  --prompt "把这张图改成水彩风格" \
  --image-url "./source.png" \
  --size "1:1" \
  --resolution "2k" \
  --output "watercolor.png"
```

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prompt` | 必填 | 生图或编辑提示词 |
| `--size` | `1:1` | `auto`、`1:1`、`3:2`、`2:3`、`4:3`、`3:4`、`5:4`、`4:5`、`16:9`、`9:16`、`2:1`、`1:2`、`21:9`、`9:21` |
| `--resolution` | `2k` | `1k`、`2k`、`4k` |
| `--image-url` | 无 | 参考图，可重复传入，最多 16 张 |
| `--output` | 自动命名 | 本地输出路径 |
| `--model` | `gpt-image-2` | 图片模型 |
| `--quality` | `high` | `low`、`medium`、`high`、`auto` |
| `--format` | `png` | `png`、`jpeg`、`webp` |

## 参数选择

- 无明确要求：`--size 1:1 --resolution 2k`
- 横屏/宽屏：`--size 16:9`
- 竖屏/手机壁纸：`--size 9:16`
- 海报：`--size 2:3`
- “高清”或“4K”：`--resolution 4k`
- “快速”或“省钱”：`--resolution 1k`

4K 只允许 `16:9`、`9:16`、`2:1`、`1:2`、`21:9`、`9:21`。其他比例请求 4K 时，脚本自动降为 2k 并在 stderr 提示。

## 输出与错误处理

- 默认请求 `b64_json`，也兼容服务返回 URL。
- 成功时 stdout 只输出最终图片的绝对路径，进度与实际像素尺寸写入 stderr。
- 401/403：检查 Key、分组及图片生成权限。
- 402：检查余额或订阅。
- `No available compatible accounts`：当前分组没有可用的 GPT-Image-2 上游账号。
- 超时：稍后重试，或先改为 `1k`。

## 安全

- Key 只保存在用户目录的 `~/.sub2api/config.json`，不要写入项目或提交到 Git。
- macOS/Linux 上将配置文件权限设为 `600`。
- 测试输出放在仓库外或明确的输出目录，避免把生成图片误提交到 Git。
