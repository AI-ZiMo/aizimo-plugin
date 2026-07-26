---
name: gpt-image-sub2api
description: 使用 Sub2API 的 GPT-Image-2 接口生成或编辑图片，支持文生图、URL 或本地参考图、常用宽高比及 1k/2k/4k 分辨率。当用户说“生图”“画图”“生成图片”“gpt-image”“Image2 生图”“帮我画”“用 GPT 画”“把这张图改成”“参考这张图”或提出 AI 图片生成、图片编辑请求时使用。
---

# GPT-Image-2 via Sub2API

通过 `https://sub2api26.zeabur.app/` 调用 `gpt-image-2`。脚本会把比例与分辨率转换为像素尺寸，调用 Sub2API 的 OpenAI 兼容图片接口，并把结果保存为本地图片。

## 前置检查

1. 运行 `node -v`，确认 Node.js 18 或更高版本可用。
2. 检查 API Key。读取优先级：
   1. 命令行 `--api-key`
   2. 环境变量 `SUB2API_API_KEY`
   3. 项目配置 `<cwd>/.yunhe-skills/.env`
   4. 用户配置 `~/.yunhe-skills/.env`
3. 不要把 Key 写入仓库。若未配置，引导用户写入用户级配置：

```bash
mkdir -p ~/.yunhe-skills
echo 'SUB2API_API_KEY=your-key' >> ~/.yunhe-skills/.env
```

可用 `SUB2API_BASE_URL` 覆盖默认 Base URL。传入域名根地址时，脚本自动补全 `/v1`。

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
| `--api-key` | 无 | 临时覆盖 API Key；不要写入日志或仓库 |
| `--base-url` | 默认服务 | 临时覆盖 Base URL |

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

- 不打印、不提交、不持久化用户临时提供的 Key。
- 测试输出放在仓库外或明确的输出目录，避免把生成图片误提交到 Git。
