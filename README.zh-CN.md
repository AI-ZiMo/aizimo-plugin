# yunhe-skills（中文说明）

> English README: [`README.md`](./README.md)

这是由 **yunhe-dev** 维护的实用技能集合，专注于内容创作与发布工作流。

## 已包含技能

### 图像生成

#### `gpt-image-sub2api`
通过 `https://sub2api26.zeabur.app/` 使用 GPT-Image-2 生成和编辑图片。

**支持能力**
- 文生图和参考图编辑
- URL、data URI 和本地参考图
- 常用宽高比与 `1k` / `2k` / `4k` 分辨率
- 保存为本地 PNG、JPEG 或 WebP

**主脚本**: `skills/gpt-image-sub2api/scripts/generate.js`

**首次配置**

```bash
mkdir -p ~/.sub2api
echo '{"api_key":"PASTE_YOUR_KEY_HERE","base_url":"https://sub2api26.zeabur.app/"}' > ~/.sub2api/config.json
chmod 600 ~/.sub2api/config.json
```

若配置缺失或无效，脚本会自动显示注册网址，以及适用于 macOS/Linux 或 Windows 的配置命令。配置一次后，后续可以直接生图。

#### `nano-banana-clawapi`
基于 BLT 兼容接口的 Nano Banana 生图技能，默认走 `clawapi.co`。

**支持能力**
- 文生图
- 多种宽高比（`1:1`、`16:9`、`21:9` 等）
- 多种分辨率（`1K`、`2K`、`4K`、`512px`）
- 同步 / 异步任务轮询
- 可选参考图输入

**主脚本**: `skills/nano-banana-clawapi/scripts/main.ts`

#### `Openclaw-X-article-cover-generator`
用于生成 OpenClaw 主题的 X 文章封面图（固定构图）。

**构图规则**
- 龙虾主体在画面**右侧 1/4**
- 文字区域在画面**左侧 3/4**
- 保持参考图中的龙虾主体形象一致
- 用户要求 5:2 时，用 **21:9** 近似

**主脚本**: `skills/Openclaw-X-article-cover-generator/scripts/generate_cover.py`

#### `cover-image`
文章封面图生成器，支持 5 维度定制（类型、配色、渲染、文字、氛围）。

**特点**
- 9 种配色方案：warm, elegant, cool, dark, earth, vivid, pastel, mono, retro
- 6 种渲染风格：flat-vector, hand-drawn, painterly, digital, pixel, chalk
- 多种宽高比：16:9, 2.35:1, 4:3, 3:2, 1:1, 3:4

#### `article-illustrator`
文章插图生成器，自动分析文章结构，识别需要配图的位置，生成风格一致的插图。

**特点**
- 6 种插图类型：infographic, scene, flowchart, comparison, framework, timeline
- 20+ 种视觉风格
- Type × Style 自由组合

#### `xhs-images`
小红书信息图系列生成器，将内容拆解为 1-10 张卡通风格图片。

**特点**
- 10 种视觉风格：cute, fresh, warm, bold, minimal, retro, pop, notion, chalkboard, study-notes
- 8 种布局：sparse, balanced, dense, list, comparison, flow, mindmap, quadrant
- 3 种策略：故事驱动型、信息密集型、视觉优先型

### 内容发布

#### `post-to-wechat`
微信公众号内容发布，支持 API 和浏览器两种方式。

**支持能力**
- 文章发布（HTML/Markdown/纯文本）
- 图文发布（最多 9 张图片）
- 多种主题样式
- 自动生成元数据

#### `post-to-x`
X (Twitter) 内容发布，使用真实 Chrome 浏览器绕过反自动化检测。

**支持能力**
- 普通推文（文本 + 最多 4 张图片）
- 视频推文（MP4/MOV/WebM）
- 引用推文
- X Articles 长文发布（Markdown）
- 首次运行需手动登录，会话持久化

### 工具类

#### `oss-upload`
阿里云 OSS 文件上传工具，支持上传文件并生成临时或永久访问链接。

**支持能力**
- 文件上传到阿里云 OSS
- 生成带签名的临时访问链接
- 设置为公开访问的永久链接
- 自动检测 Content-Type

## 环境要求

- Python 3.10+
- Node.js 18+
- `uv`（推荐）
- 按各 skill 的说明配置对应凭证

```bash
# 仅用于 nano-banana-clawapi
export BLT_API_KEY="你的密钥"
export BLT_API_BASE_URL="https://clawapi.co"
```

## nano-banana-clawapi 环境配置

支持通过 `.env` 文件配置 API key，无需每次设置环境变量：

**加载优先级**（高→低）：
1. 命令行参数 `--api-key`
2. 系统环境变量 `BLT_API_KEY`
3. 项目级配置 `<cwd>/.yunhe-skills/.env`
4. 用户级配置 `~/.yunhe-skills/.env`

**配置方法**：

```bash
# 用户级配置（推荐，一次配置全局可用）
mkdir -p ~/.yunhe-skills
echo "BLT_API_KEY=your-api-key" > ~/.yunhe-skills/.env

# 项目级配置
mkdir -p .yunhe-skills
echo "BLT_API_KEY=your-api-key" > .yunhe-skills/.env
```

## 快速开始

```bash
# gpt-image-sub2api
node ./skills/gpt-image-sub2api/scripts/generate.js \
  --prompt "雨夜上海街头行走的电影感机器人" \
  --size 16:9 \
  --resolution 2k \
  --output gpt-image.png
```

```bash
# nano-banana-clawapi
node ./skills/nano-banana-clawapi/scripts/main.ts \
  --prompt "未来城市黄昏天际线" \
  --ratio 16:9 \
  --size 2K \
  --output demo.png
```

```bash
# Openclaw-X-article-cover-generator
uv run ./skills/Openclaw-X-article-cover-generator/scripts/generate_cover.py \
  --title "零门槛上手 OpenClaw" \
  --reference "https://example.com/reference.jpg" \
  --output x-cover.png \
  --size 2K \
  --async-mode
```

## 说明

- 仓库会持续迭代。
- 每个技能目录下有独立文档（`SKILL.md`）。
- 如果打包平台丢失二进制图片，建议优先使用 URL 参考图。
