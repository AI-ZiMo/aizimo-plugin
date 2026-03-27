import path from "node:path";
import process from "node:process";
import { homedir } from "node:os";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";

const DEFAULT_BASE_URL = "https://clawapi.co";
const DEFAULT_MODEL = "gemini-3.1-flash-image-preview";
const DEFAULT_ASPECT_RATIO = "1:1";
const DEFAULT_IMAGE_SIZE = "1K";
const DEFAULT_WAIT_SECONDS = 300;
const REQUEST_TIMEOUT_MS = 120000;
const POLL_INTERVAL_MS = 5000;

const VALID_ASPECT_RATIOS = [
  "4:3",
  "3:4",
  "16:9",
  "9:16",
  "2:3",
  "3:2",
  "1:1",
  "4:5",
  "5:4",
  "21:9",
  "1:4",
  "4:1",
  "8:1",
  "1:8",
] as const;

const VALID_SIZES = ["512px", "1K", "2K", "4K"] as const;

type ImageSize = (typeof VALID_SIZES)[number];

type CliArgs = {
  prompt: string | null;
  output: string | null;
  ratio: string;
  size: ImageSize;
  asyncMode: boolean;
  referenceImages: string[];
  apiKey: string | null;
  checkTask: string | null;
  webhook: string | null;
  help: boolean;
};

type ImagePayload = {
  url?: string;
  b64_json?: string;
  b64?: string;
};

type ImageBinary = {
  bytes: Uint8Array;
  ext: string;
};

function printUsage(): void {
  console.log(`Usage:
  node scripts/main.ts --prompt "A cat wearing sunglasses" --output cat.png
  node scripts/main.ts --prompt "A futuristic skyline" --ratio 16:9 --size 2K --output skyline.png
  node scripts/main.ts --prompt "A surreal city" --async-mode --output city.png
  node scripts/main.ts --check-task TASK_ID

Options:
  -p, --prompt <text>        Image prompt
  -o, --output <path>        Output file path (default: generated timestamp filename)
  -r, --ratio, --ar <ratio>  Aspect ratio (default: 1:1)
  -s, --size <size>          Image size: 512px | 1K | 2K | 4K (default: 1K)
  -a, --async, --async-mode  Use async generation
  --ref, --reference <...>   Reference image path(s)
  -k, --api-key <key>        BLT API key
  --check-task <id>          Check async task status
  --webhook <url>            Webhook for async notifications
  -h, --help                 Show help

Environment variables:
  BLT_API_KEY                API key
  BLT_API_BASE_URL           Base URL override (default: https://clawapi.co; auto-resolves to /v1)

Env file load order: CLI args > process.env > <cwd>/.yunhe-skills/.env > ~/.yunhe-skills/.env`);
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    prompt: null,
    output: null,
    ratio: DEFAULT_ASPECT_RATIO,
    size: DEFAULT_IMAGE_SIZE,
    asyncMode: false,
    referenceImages: [],
    apiKey: null,
    checkTask: null,
    webhook: null,
    help: false,
  };

  const positional: string[] = [];

  const takeMany = (startIndex: number): { values: string[]; next: number } => {
    const values: string[] = [];
    let index = startIndex + 1;
    while (index < argv.length) {
      const value = argv[index]!;
      if (value.startsWith("-")) break;
      values.push(value);
      index++;
    }
    return { values, next: index - 1 };
  };

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }

    if (arg === "--prompt" || arg === "-p") {
      const value = argv[++index];
      if (!value) throw new Error(`Missing value for ${arg}`);
      args.prompt = value;
      continue;
    }

    if (arg === "--output" || arg === "--image" || arg === "-o") {
      const value = argv[++index];
      if (!value) throw new Error(`Missing value for ${arg}`);
      args.output = value;
      continue;
    }

    if (arg === "--ratio" || arg === "--ar" || arg === "-r") {
      const value = argv[++index];
      if (!value) throw new Error(`Missing value for ${arg}`);
      args.ratio = value;
      continue;
    }

    if (arg === "--size" || arg === "-s") {
      const value = argv[++index];
      if (!value) throw new Error(`Missing value for ${arg}`);
      if (!VALID_SIZES.includes(value as ImageSize)) {
        throw new Error(`Invalid image size: ${value}`);
      }
      args.size = value as ImageSize;
      continue;
    }

    if (arg === "--async-mode" || arg === "--async" || arg === "-a") {
      args.asyncMode = true;
      continue;
    }

    if (arg === "--ref" || arg === "--reference") {
      const { values, next } = takeMany(index);
      if (values.length === 0) throw new Error(`Missing files for ${arg}`);
      args.referenceImages.push(...values);
      index = next;
      continue;
    }

    if (arg === "--api-key" || arg === "-k") {
      const value = argv[++index];
      if (!value) throw new Error(`Missing value for ${arg}`);
      args.apiKey = value;
      continue;
    }

    if (arg === "--check-task") {
      const value = argv[++index];
      if (!value) throw new Error("Missing value for --check-task");
      args.checkTask = value;
      continue;
    }

    if (arg === "--webhook") {
      const value = argv[++index];
      if (!value) throw new Error("Missing value for --webhook");
      args.webhook = value;
      continue;
    }

    if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}`);
    }

    positional.push(arg);
  }

  if (!args.prompt && positional.length > 0) {
    args.prompt = positional.join(" ");
  }

  return args;
}

async function loadEnvFile(filePath: string): Promise<Record<string, string>> {
  try {
    const content = await readFile(filePath, "utf8");
    const env: Record<string, string> = {};

    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;

      const index = trimmed.indexOf("=");
      if (index === -1) continue;

      const key = trimmed.slice(0, index).trim();
      let value = trimmed.slice(index + 1).trim();

      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      env[key] = value;
    }

    return env;
  } catch {
    return {};
  }
}

async function loadEnv(): Promise<void> {
  const homeEnv = await loadEnvFile(path.join(homedir(), ".yunhe-skills", ".env"));
  const cwdEnv = await loadEnvFile(path.join(process.cwd(), ".yunhe-skills", ".env"));

  for (const [key, value] of Object.entries(cwdEnv)) {
    if (!process.env[key]) process.env[key] = value;
  }

  for (const [key, value] of Object.entries(homeEnv)) {
    if (!process.env[key]) process.env[key] = value;
  }
}

function getApiKey(cliValue: string | null): string | null {
  return cliValue?.trim() || process.env.BLT_API_KEY?.trim() || null;
}

function getApiBaseUrl(): string {
  const normalized = (process.env.BLT_API_BASE_URL || DEFAULT_BASE_URL).trim().replace(/\/+$/g, "");
  const parsed = new URL(normalized);

  if (parsed.pathname === "" || parsed.pathname === "/") {
    parsed.pathname = "/v1";
  }

  return parsed.toString().replace(/\/+$/g, "");
}

function validateAspectRatio(value: string): void {
  if (!VALID_ASPECT_RATIOS.includes(value as (typeof VALID_ASPECT_RATIOS)[number])) {
    throw new Error(
      `Invalid aspect ratio "${value}". Valid ratios: ${VALID_ASPECT_RATIOS.join(", ")}`
    );
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function generateFilename(prompt: string): string {
  const timestamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const snippet = prompt
    .split(/\s+/)
    .slice(0, 4)
    .join("-")
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 30);

  return `${timestamp}-${snippet || "image"}`;
}

function normalizeOutputPath(output: string | null, prompt: string): string {
  const target = output || generateFilename(prompt);
  const resolved = path.resolve(target);
  return resolved;
}

function getMimeType(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
  if (extension === ".webp") return "image/webp";
  if (extension === ".gif") return "image/gif";
  return "image/png";
}

async function loadReferenceImages(referencePaths: string[]): Promise<string[]> {
  const images: string[] = [];

  for (const referencePath of referencePaths) {
    await access(referencePath);
    const bytes = await readFile(referencePath);
    const mimeType = getMimeType(referencePath);
    images.push(`data:${mimeType};base64,${bytes.toString("base64")}`);
  }

  return images;
}

async function fetchJson(url: string, init: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      throw new Error(`Request timeout after ${REQUEST_TIMEOUT_MS / 1000}s: ${url}`);
    }
    throw error;
  }
  const text = await response.text();

  if (!response.ok) {
    const details = text ? `\nResponse: ${text}` : "";
    throw new Error(`Request failed: ${response.status} ${response.statusText}${details}`);
  }

  if (!text) return {};

  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Response was not valid JSON: ${text}`);
  }
}

async function createImageRequest(
  prompt: string,
  apiKey: string,
  ratio: string,
  size: ImageSize,
  referenceImages: string[],
  asyncMode: boolean,
  webhook: string | null,
): Promise<unknown> {
  const url = new URL(`${getApiBaseUrl()}/images/generations`);
  if (asyncMode) url.searchParams.set("async", "true");
  if (webhook) url.searchParams.set("webhook", webhook);

  const payload: Record<string, unknown> = {
    model: DEFAULT_MODEL,
    prompt,
    aspect_ratio: ratio,
    image_size: size,
    response_format: "url",
  };

  if (referenceImages.length > 0) {
    payload.image = referenceImages;
  }

  return fetchJson(url.toString(), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function checkTaskStatus(taskId: string, apiKey: string): Promise<unknown> {
  return fetchJson(`${getApiBaseUrl()}/images/tasks/${taskId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
  });
}

function findImagePayload(value: unknown): ImagePayload | null {
  if (Array.isArray(value)) {
    for (const item of value) {
      const payload = findImagePayload(item);
      if (payload) return payload;
    }
    return null;
  }

  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  const url = typeof record.url === "string" && record.url ? record.url : undefined;
  const b64Json =
    typeof record.b64_json === "string" && record.b64_json ? record.b64_json : undefined;
  const b64 = typeof record.b64 === "string" && record.b64 ? record.b64 : undefined;

  if (url || b64Json || b64) {
    return { url, b64_json: b64Json, b64 };
  }

  for (const key of ["data", "result", "output", "response"]) {
    if (key in record) {
      const payload = findImagePayload(record[key]);
      if (payload) return payload;
    }
  }

  return null;
}

function extractTaskId(value: unknown): string | null {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    for (const item of value) {
      const taskId = extractTaskId(item);
      if (taskId) return taskId;
    }
    return null;
  }
  if (typeof value !== "object") return null;

  const record = value as Record<string, unknown>;
  for (const key of ["task_id", "taskId", "id", "data"]) {
    if (!(key in record)) continue;
    const taskId = extractTaskId(record[key]);
    if (taskId) return taskId;
  }

  return null;
}

function extensionFromMimeType(mimeType: string | null): string | null {
  if (!mimeType) return null;
  const lower = mimeType.toLowerCase();
  if (lower.includes("image/jpeg") || lower.includes("image/jpg")) return ".jpg";
  if (lower.includes("image/png")) return ".png";
  if (lower.includes("image/webp")) return ".webp";
  if (lower.includes("image/gif")) return ".gif";
  return null;
}

function extensionFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    const ext = path.extname(parsed.pathname).toLowerCase();
    if (ext === ".jpg" || ext === ".jpeg") return ".jpg";
    if (ext === ".png" || ext === ".webp" || ext === ".gif") return ext;
    return null;
  } catch {
    return null;
  }
}

function decodeDataUrl(value: string): ImageBinary | null {
  const match = value.match(/^data:(image\/[^;]+);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) return null;
  const ext = extensionFromMimeType(match[1]!) || ".png";
  const bytes = Uint8Array.from(Buffer.from(match[2]!, "base64"));
  return { bytes, ext };
}

async function bytesFromImagePayload(payload: ImagePayload): Promise<ImageBinary> {
  if (payload.b64_json) {
    return {
      bytes: Uint8Array.from(Buffer.from(payload.b64_json, "base64")),
      ext: ".png",
    };
  }

  if (payload.b64) {
    return {
      bytes: Uint8Array.from(Buffer.from(payload.b64, "base64")),
      ext: ".png",
    };
  }

  if (!payload.url) {
    throw new Error("No image data found in response");
  }

  const inline = decodeDataUrl(payload.url);
  if (inline) return inline;

  const response = await fetch(payload.url, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  if (!response.ok) {
    throw new Error(`Failed to download image: ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get("content-type");
  const ext = extensionFromMimeType(contentType) || extensionFromUrl(payload.url) || ".png";
  return { bytes: new Uint8Array(await response.arrayBuffer()), ext };
}

function withOutputExtension(outputPath: string, ext: string): string {
  if (path.extname(outputPath)) return outputPath;
  return `${outputPath}${ext}`;
}

async function saveImageFromResponse(result: unknown, outputPath: string): Promise<string> {
  const payload = findImagePayload(result);
  if (!payload) {
    throw new Error(`No image data found in response: ${JSON.stringify(result, null, 2)}`);
  }

  const binary = await bytesFromImagePayload(payload);
  const finalPath = withOutputExtension(outputPath, binary.ext);
  await writeFile(finalPath, binary.bytes);
  return finalPath;
}

function getTaskData(result: unknown): Record<string, unknown> {
  if (!result || typeof result !== "object") return {};
  const record = result as Record<string, unknown>;
  if (record.data && typeof record.data === "object" && !Array.isArray(record.data)) {
    return record.data as Record<string, unknown>;
  }
  return record;
}

function isSuccessStatus(status: string): boolean {
  return ["SUCCESS", "COMPLETED", "DONE"].includes(status.toUpperCase());
}

function isFailureStatus(status: string): boolean {
  return ["FAILURE", "FAILED", "ERROR", "CANCELLED"].includes(status.toUpperCase());
}

async function waitForAsyncTask(taskId: string, apiKey: string, outputPath: string): Promise<string> {
  console.log(`⏳ Waiting for async task: ${taskId}`);
  console.log(`   Checking every 5 seconds (max ${DEFAULT_WAIT_SECONDS}s)...`);

  const deadline = Date.now() + DEFAULT_WAIT_SECONDS * 1000;

  while (Date.now() < deadline) {
    const result = await checkTaskStatus(taskId, apiKey);
    const taskData = getTaskData(result);
    const status = String(taskData.status || "UNKNOWN");

    if (isSuccessStatus(status)) {
      console.log("✅ Task completed");
      return saveImageFromResponse(taskData, outputPath);
    }

    if (isFailureStatus(status)) {
      const reason =
        (typeof taskData.fail_reason === "string" && taskData.fail_reason) ||
        (typeof taskData.error === "string" && taskData.error) ||
        (typeof taskData.message === "string" && taskData.message) ||
        "Unknown error";
      throw new Error(`Task failed: ${reason}`);
    }

    const progress =
      typeof taskData.progress === "string" || typeof taskData.progress === "number"
        ? ` (${taskData.progress})`
        : "";

    console.log(`   Status: ${status}${progress}`);
    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error(
    `Timeout after ${DEFAULT_WAIT_SECONDS}s. Check later with: node scripts/main.ts --check-task ${taskId}`
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  await loadEnv();

  const apiKey = getApiKey(args.apiKey);
  if (!apiKey) {
    throw new Error(
      "No API key provided. Use --api-key, BLT_API_KEY, or ~/.yunhe-skills/.env / <cwd>/.yunhe-skills/.env"
    );
  }

  if (args.checkTask) {
    const result = await checkTaskStatus(args.checkTask, apiKey);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (!args.prompt) {
    throw new Error("--prompt is required unless you are using --check-task");
  }

  validateAspectRatio(args.ratio);

  const outputPath = normalizeOutputPath(args.output, args.prompt);
  await mkdir(path.dirname(outputPath), { recursive: true });

  const referenceImages = await loadReferenceImages(args.referenceImages);

  if (referenceImages.length > 0) {
    console.log(`📎 Loaded ${referenceImages.length} reference image(s)`);
  }

  console.log("🎨 Generating image...");
  console.log(
    `   Prompt: ${args.prompt.length > 60 ? `${args.prompt.slice(0, 60)}...` : args.prompt}`
  );
  console.log(`   Ratio: ${args.ratio}`);
  console.log(`   Size: ${args.size}`);
  console.log(`   Mode: ${args.asyncMode ? "Async" : "Sync"}`);
  console.log(`   Base URL: ${getApiBaseUrl()}`);
  console.log(`   Output: ${outputPath}`);
  console.log("");

  const result = await createImageRequest(
    args.prompt,
    apiKey,
    args.ratio,
    args.size,
    referenceImages,
    args.asyncMode,
    args.webhook,
  );

  let finalPath = outputPath;

  if (args.asyncMode) {
    const taskId = extractTaskId(result);
    if (!taskId) {
      throw new Error(`Unexpected async response: ${JSON.stringify(result, null, 2)}`);
    }
    console.log(`📝 Task ID: ${taskId}`);
    finalPath = await waitForAsyncTask(taskId, apiKey, outputPath);
  } else {
    finalPath = await saveImageFromResponse(result, outputPath);
  }

  console.log(`✅ Image saved: ${finalPath}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`❌ ${message}`);
  process.exit(1);
});
