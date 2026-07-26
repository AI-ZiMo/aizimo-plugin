#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");

const DEFAULT_BASE_URL = "https://sub2api26.zeabur.app/";
const DEFAULT_MODEL = "gpt-image-2";
const DEFAULT_TIMEOUT_MS = 6 * 60 * 1000;

const RATIOS = new Set([
  "auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5",
  "16:9", "9:16", "2:1", "1:2", "21:9", "9:21",
]);
const RESOLUTIONS = new Set(["1k", "2k", "4k"]);
const FORMATS = new Set(["png", "jpeg", "webp"]);
const QUALITIES = new Set(["low", "medium", "high", "auto"]);
const VALID_4K_RATIOS = new Set(["16:9", "9:16", "2:1", "1:2", "21:9", "9:21"]);

function usage() {
  console.log(`Usage:
  node scripts/generate.js --prompt "A cinematic robot" --output robot.png
  node scripts/generate.js --prompt "Watercolor style" --image-url source.png --output edited.png

Options:
  --prompt <text>          Image prompt (required)
  --size <ratio>          Aspect ratio (default: 1:1)
  --resolution <value>    1k | 2k | 4k (default: 2k)
  --image-url <source>    URL, data URI, or local image; repeatable (max 16)
  --output <path>         Output path (default: generated timestamp filename)
  --model <name>          Model (default: gpt-image-2)
  --quality <value>       low | medium | high | auto (default: high)
  --format <value>        png | jpeg | webp (default: png)
  --api-key <key>         Temporary API key override
  --base-url <url>        Base URL override
  --help                  Show this help

Environment:
  SUB2API_API_KEY         API key
  SUB2API_BASE_URL        Base URL (default: https://sub2api26.zeabur.app/)`);
}

function parseArgs(argv) {
  const args = {
    prompt: "",
    size: "1:1",
    resolution: "2k",
    imageUrls: [],
    output: "",
    model: DEFAULT_MODEL,
    quality: "high",
    format: "png",
    apiKey: "",
    baseUrl: "",
    help: false,
  };
  const valueOptions = new Map([
    ["--prompt", "prompt"], ["--size", "size"], ["--resolution", "resolution"],
    ["--output", "output"], ["--model", "model"], ["--quality", "quality"],
    ["--format", "format"], ["--api-key", "apiKey"], ["--base-url", "baseUrl"],
  ]);

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }
    if (arg === "--image-url") {
      const value = argv[++i];
      if (!value) throw new Error("--image-url requires a value");
      args.imageUrls.push(value);
      continue;
    }
    const key = valueOptions.get(arg);
    if (!key) throw new Error(`Unknown option: ${arg}`);
    const value = argv[++i];
    if (!value) throw new Error(`${arg} requires a value`);
    args[key] = value;
  }
  return args;
}

function parseEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return {};
  const result = {};
  for (const rawLine of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const index = line.indexOf("=");
    if (index < 1) continue;
    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    result[key] = value;
  }
  return result;
}

function loadConfig(args) {
  const project = parseEnvFile(path.join(process.cwd(), ".yunhe-skills", ".env"));
  const user = parseEnvFile(path.join(os.homedir(), ".yunhe-skills", ".env"));
  const apiKey = args.apiKey || process.env.SUB2API_API_KEY || project.SUB2API_API_KEY || user.SUB2API_API_KEY;
  const baseUrl = args.baseUrl || process.env.SUB2API_BASE_URL || project.SUB2API_BASE_URL || user.SUB2API_BASE_URL || DEFAULT_BASE_URL;
  return { apiKey: apiKey?.trim(), baseUrl: normalizeBaseUrl(baseUrl) };
}

function normalizeBaseUrl(value) {
  const url = new URL(String(value).trim());
  const pathname = url.pathname.replace(/\/+$/, "");
  url.pathname = pathname || "/v1";
  return url.toString().replace(/\/+$/, "");
}

function validateArgs(args) {
  if (!args.prompt.trim()) throw new Error("--prompt is required");
  if (!RATIOS.has(args.size)) throw new Error(`Unsupported size ratio: ${args.size}`);
  args.resolution = args.resolution.toLowerCase();
  args.format = args.format.toLowerCase();
  args.quality = args.quality.toLowerCase();
  if (!RESOLUTIONS.has(args.resolution)) throw new Error(`Unsupported resolution: ${args.resolution}`);
  if (!FORMATS.has(args.format)) throw new Error(`Unsupported format: ${args.format}`);
  if (!QUALITIES.has(args.quality)) throw new Error(`Unsupported quality: ${args.quality}`);
  if (args.imageUrls.length > 16) throw new Error("At most 16 reference images are supported");
  if (args.resolution === "4k" && !VALID_4K_RATIOS.has(args.size)) {
    process.stderr.write(`4k does not support ${args.size}; downgraded to 2k\n`);
    args.resolution = "2k";
  }
}

function dimensionsFor(ratio, resolution) {
  if (ratio === "auto") return "auto";
  const [widthRatio, heightRatio] = ratio.split(":").map(Number);
  let longEdge = resolution === "1k" ? 1024 : resolution === "2k" ? 2048 : 4096;
  if (resolution === "4k" && (ratio === "16:9" || ratio === "9:16")) longEdge = 3840;
  let width;
  let height;
  if (widthRatio >= heightRatio) {
    width = longEdge;
    height = Math.round(longEdge * heightRatio / widthRatio);
  } else {
    height = longEdge;
    width = Math.round(longEdge * widthRatio / heightRatio);
  }
  return `${width}x${height}`;
}

function mimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  if (ext === ".gif") return "image/gif";
  return "image/png";
}

function normalizeImageSource(source) {
  if (/^https?:\/\//i.test(source) || /^data:image\//i.test(source)) return source;
  const resolved = path.resolve(source);
  if (!fs.existsSync(resolved)) throw new Error(`Reference image not found: ${source}`);
  const data = fs.readFileSync(resolved).toString("base64");
  return `data:${mimeType(resolved)};base64,${data}`;
}

function defaultOutput(prompt, format) {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const slug = prompt.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-").replace(/^-|-$/g, "").slice(0, 32) || "image";
  return path.resolve(`${stamp}-${slug}.${format === "jpeg" ? "jpg" : format}`);
}

async function requestImage(url, apiKey, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Sub2API returned invalid JSON (HTTP ${response.status})`);
  }
  if (!response.ok) {
    const message = data?.error?.message || data?.message || text || response.statusText;
    throw new Error(`Sub2API HTTP ${response.status}: ${message}`);
  }
  return data;
}

async function saveResult(result, outputPath, format) {
  const item = Array.isArray(result?.data) ? result.data[0] : null;
  if (!item) throw new Error(`No image data in response: ${JSON.stringify(result)}`);
  let bytes;
  if (typeof item.b64_json === "string" && item.b64_json) {
    bytes = Buffer.from(item.b64_json, "base64");
  } else if (typeof item.url === "string" && item.url) {
    if (item.url.startsWith("data:image/")) {
      const comma = item.url.indexOf(",");
      bytes = Buffer.from(item.url.slice(comma + 1), "base64");
    } else {
      const response = await fetch(item.url, { signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS) });
      if (!response.ok) throw new Error(`Image download failed: HTTP ${response.status}`);
      bytes = Buffer.from(await response.arrayBuffer());
    }
  } else {
    throw new Error(`No b64_json or URL in response: ${JSON.stringify(result)}`);
  }
  const finalPath = path.resolve(outputPath || defaultOutput("image", format));
  fs.mkdirSync(path.dirname(finalPath), { recursive: true });
  fs.writeFileSync(finalPath, bytes);
  return finalPath;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  validateArgs(args);
  const { apiKey, baseUrl } = loadConfig(args);
  if (!apiKey) throw new Error("No API key. Set SUB2API_API_KEY, use --api-key, or configure .yunhe-skills/.env");

  const pixelSize = dimensionsFor(args.size, args.resolution);
  const references = args.imageUrls.map(normalizeImageSource);
  const endpoint = references.length ? "/images/edits" : "/images/generations";
  const payload = {
    model: args.model,
    prompt: args.prompt.trim(),
    n: 1,
    size: pixelSize,
    quality: args.quality,
    output_format: args.format,
    response_format: "b64_json",
  };
  if (references.length) payload.images = references.map((image_url) => ({ image_url }));

  process.stderr.write(`Submitting ${references.length ? "image edit" : "image generation"}: ${args.size}, ${args.resolution}, ${pixelSize}\n`);
  const result = await requestImage(`${baseUrl}${endpoint}`, apiKey, payload);
  const output = args.output || defaultOutput(args.prompt, args.format);
  const finalPath = await saveResult(result, output, args.format);
  process.stderr.write(`Saved: ${finalPath}\n`);
  console.log(finalPath);
}

if (require.main === module) {
  main().catch((error) => {
    const message = error?.name === "TimeoutError" ? "Request timed out after 6 minutes" : error.message;
    console.error(`Error: ${message}`);
    process.exit(1);
  });
}

module.exports = { dimensionsFor, normalizeBaseUrl, parseArgs, validateArgs };
