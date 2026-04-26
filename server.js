import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import { GoogleGenAI } from '@google/genai';
import puppeteer from 'puppeteer';
import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import { existsSync, mkdirSync } from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
if (!existsSync(path.join(__dirname, 'output'))) mkdirSync(path.join(__dirname, 'output'));

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const GEMINI_IMAGE_MODEL = 'gemini-3-pro-image-preview';

const STYLE = {
  gray50: '#FAFAFA',
  gray900: '#171717',
};

// Load logo SVG at startup — embedded inline to avoid Puppeteer path issues
const ASSETS_DIR = path.join(__dirname, 'public', 'assets');
const rawLogoSvg = await fs.readFile(path.join(ASSETS_DIR, 'logo.svg'), 'utf-8');
// Add fill="currentColor" so text paths inherit the container's color
const logoSvg = rawLogoSvg.replace('<svg ', '<svg fill="currentColor" ');

// ── Figma-derived layout constants (node 1:57, 1200×630) ──────────────────
// Title text:   left=100  top=76   width=547  (Pretendard Bold 60px / lh 1.4)
// Illustration: left=670  top=90   width=430  height=450
// Logo:         left=100  top=491  width=300  (height auto by SVG aspect ratio)

// Determine focal accent color based on thumbnail bgColor
function resolveFocalColor(bgColor) {
  const r = parseInt(bgColor.slice(1, 3), 16);
  const g = parseInt(bgColor.slice(3, 5), 16);
  const b = parseInt(bgColor.slice(5, 7), 16);
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const chroma = max - min;
  let hue = 0;
  if (chroma > 0) {
    if (max === r) hue = (((g - b) / chroma) % 6) * 60;
    else if (max === g) hue = ((b - r) / chroma + 2) * 60;
    else hue = ((r - g) / chroma + 4) * 60;
    if (hue < 0) hue += 360;
  }
  const sat = max === 0 ? 0 : chroma / max;
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  const isOrangeish = hue >= 10 && hue <= 50 && sat > 0.4 && lum > 0.3;
  return isOrangeish ? '#0075FF' : '#FF6F1F';
}

// Fixed style rules appended to every image generation request
const STYLE_RULES = `[Fixed Image Style Rule]
Create a branded editorial blog thumbnail illustration.

Solid pure black background (#000000).
Fill the main subject with {focalColor} as a solid color. All other elements use white or light gray outline only — no fills.
Connect separate elements with a dashed line, dotted arc, or simple arrow.
Bold, clean, rounded white linework.
Large objects with generous negative space. Readable at thumbnail size.
Scatter a few small decorative accents in empty areas — mix dots, plus signs, diamonds, and short lines in varied sizes (some larger, some tiny).

No text, labels, letters, numbers, dense diagrams, thin technical lines, realistic rendering, 3D, gradients, shadows, or glow.`;

const LAYOUTS = [
  '좌우 대비형: 캔버스를 좌우로 나눠 왼쪽과 오른쪽에 각각 큰 오브젝트를 배치. 두 오브젝트가 서로 대면하는 구도.',
  '흐름형: 왼쪽→중앙→오른쪽 순서로 3개 요소를 일렬로 배치. 방향성과 순서가 느껴지는 구도.',
  '클로즈업형: 하나의 큰 오브젝트가 캔버스 대부분을 채움. 보조 요소는 작게 한두 개만 구석에 배치.',
  '상하 구조형: 위쪽에 크고 지배적인 요소, 아래쪽에 작은 보조 요소 1~2개.',
  '대각선형: 좌상단에서 우하단 방향으로 요소들을 대각선으로 배치.',
  '좌편향형: 주요 오브젝트를 왼쪽에 크게 배치하고 오른쪽은 넓은 여백으로 남김.',
];

function pickLayout() {
  return LAYOUTS[Math.floor(Math.random() * LAYOUTS.length)];
}

const CONCEPT_SYSTEM = `You are a visual art director. Given a blog title, content, and a fixed layout, output a JSON object with exactly one field: "prompt".

[Goal]
Select icons that visually represent this specific article, then place them according to the given layout.
This prompt decides what to draw — not the visual style.

[Icon Selection Rule]
Choose 1 main icon and 1–2 supporting icons appropriate for the given layout.
Icons must be concrete, drawable objects extracted directly from this article's specific content or domain.
The selection must feel unique to this article — a different article should produce different icons.
Avoid generic icons that could appear in any article regardless of its content.

The scene must contain no text, letters, numbers, or labels as visual elements.

[Prompt Writing Rule]
Write the "prompt" in Korean, 60–80 words.
Follow the given layout exactly — describe icon names, positions, and sizes according to it.
Specify which icon is the main subject (it will be filled with {focalColor}). All other icons are outline only.
Mention a visual connector (dashed line, dotted arc, or arrow) linking the main and supporting icons.
Do not describe background, lighting, rendering style, or visual effects.

[Examples]
Example 1 —
Input:
  배경색: #171717
  제목: AI가 대신 써준 사업계획서, 여기에 내 경험을 더했다
  레이아웃: 좌우 대비형: 캔버스를 좌우로 나눠 왼쪽과 오른쪽에 각각 큰 오브젝트를 배치. 두 오브젝트가 서로 대면하는 구도.
  본문: AI가 만든 논리 구조 + 사람의 현장 경험 = 최고 품질
Output:
{"prompt": "좌우 대비형. 왼쪽에 로봇 아이콘(#FF6F1F solid fill + 흰 outline, 캔버스의 40%). 오른쪽에 연필을 든 손 아이콘(흰 outline, 캔버스의 35%). 두 오브젝트 사이 중앙에 흰 점선 화살표(←→)로 연결. 빈 공간에 흰 점·플러스·다이아몬드를 크고 작게 섞어 산재."}

Example 2 —
Input:
  배경색: #171717
  제목: 블로그 글쓰기 3단계 루틴
  레이아웃: 흐름형: 왼쪽→중앙→오른쪽 순서로 3개 요소를 일렬로 배치. 방향성과 순서가 느껴지는 구도.
  본문: 아이디어 수집 → 초고 작성 → 퇴고 순서로 글이 완성됨
Output:
{"prompt": "흐름형. 왼쪽에 전구 아이콘(#FF6F1F solid fill + 흰 outline, 캔버스의 25%). 중앙에 노트+펜 아이콘(흰 outline, 캔버스의 30%). 오른쪽에 돋보기 아이콘(흰 outline, 캔버스의 25%). 세 아이콘 사이를 흰 화살표(→)로 순서대로 연결. 빈 공간에 흰 점·플러스·짧은 선을 크고 작게 섞어 산재."}

Output valid JSON only. No markdown, no explanation.`;

async function buildIllustrationPrompt(title, content, bgColor) {
  const focalColor = resolveFocalColor(bgColor);
  const systemPrompt = CONCEPT_SYSTEM.replaceAll('{focalColor}', focalColor);

  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: systemPrompt },
      {
        role: 'user',
        content: `배경색: ${bgColor}\n제목: ${title}\n\n[고정 레이아웃]\n${pickLayout()}\n\n본문:\n${content.slice(0, 3000)}`,
      },
    ],
    max_tokens: 1000,
    response_format: { type: 'json_object' },
  });

  const parsed = JSON.parse(response.choices[0].message.content);
  const styleRules = STYLE_RULES.replaceAll('{focalColor}', focalColor);
  return { prompt: parsed.prompt, focalColor, styleRules };
}

// Remove near-black pixels → transparent (tolerance: 0–255 color distance)
async function removeBlackBackground(base64, tolerance = 40) {
  const buffer = Buffer.from(base64, 'base64');
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const dist = Math.sqrt(r * r + g * g + b * b); // distance from black (0,0,0)
    if (dist < tolerance) data[i + 3] = 0;
  }

  const result = await sharp(data, {
    raw: { width: info.width, height: info.height, channels: 4 },
  }).png().toBuffer();

  return result.toString('base64');
}

async function generateIllustrationOpenAI(prompt, model) {
  console.log(`[OpenAI:${model}] prompt:`, prompt.slice(0, 80));
  const response = await openai.images.generate({
    model,
    prompt,
    n: 1,
    size: '1024x1024',
    quality: 'high',
  });
  return removeBlackBackground(response.data[0].b64_json);
}

async function generateIllustrationGemini(prompt) {
  console.log('[Gemini] prompt:', prompt.slice(0, 80));
  const response = await gemini.models.generateContent({
    model: GEMINI_IMAGE_MODEL,
    contents: prompt,
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
    },
  });
  const imagePart = response.candidates[0].content.parts.find(p => p.inlineData);
  if (!imagePart) throw new Error('Gemini가 이미지를 반환하지 않았습니다.');
  return removeBlackBackground(imagePart.inlineData.data);
}

async function generateIllustration(prompt, model = 'gpt-image-1') {
  return model === 'gemini'
    ? generateIllustrationGemini(prompt)
    : generateIllustrationOpenAI(prompt, model);
}

// Returns { markColor, logoTextColor } based on bgColor
function resolveLogoColors(bgColor, textColor) {
  const r = parseInt(bgColor.slice(1, 3), 16);
  const g = parseInt(bgColor.slice(3, 5), 16);
  const b = parseInt(bgColor.slice(5, 7), 16);

  // Luminance (0 = black, 1 = white)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

  // Hue: check if background is orange-ish
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const chroma = max - min;
  let hue = 0;
  if (chroma > 0) {
    if (max === r) hue = (((g - b) / chroma) % 6) * 60;
    else if (max === g) hue = ((b - r) / chroma + 2) * 60;
    else hue = ((r - g) / chroma + 4) * 60;
    if (hue < 0) hue += 360;
  }
  const saturation = max === 0 ? 0 : chroma / max;
  const isOrangeish = hue >= 10 && hue <= 50 && saturation > 0.4 && luminance > 0.3;

  return {
    // On orange bg → white mark; otherwise keep original orange
    markColor: isOrangeish ? 'white' : null,
    // On light bg → dark text for logo; respect user's textColor otherwise
    logoTextColor: luminance > 0.6 ? '#171717' : textColor,
  };
}

function buildThumbnailHTML({ title, bgColor, textColor, illustrationBase64 }) {
  const titleHTML = title
    .split('\n')
    .map(line => `<span>${line}</span>`)
    .join('<br>');

  const { markColor, logoTextColor } = resolveLogoColors(bgColor, textColor);
  const logoHtml = markColor
    ? logoSvg.replace(/fill="#ff9819"/gi, `fill="${markColor}"`).replace(/fill="#ff6f1f"/gi, `fill="${markColor}"`)
    : logoSvg;

  // All values are Figma coords × 2 — canvas is natively 2400×1260px
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/static/pretendard.css" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    width: 2400px;
    height: 1260px;
    overflow: hidden;
    background: ${bgColor};
    font-family: 'Pretendard', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    --fill-0: ${textColor};
  }

  .canvas {
    position: relative;
    width: 2400px;
    height: 1260px;
  }

  /* Title: Figma (100,76,547w,60px) × 2 */
  .title {
    position: absolute;
    left: 200px;
    top: 152px;
    width: 1094px;
    color: ${textColor};
    font-size: 120px;
    font-weight: 700;
    line-height: 1.4;
    letter-spacing: -0.02em;
  }

  /* Illustration: Figma (670,90,430w,450h) × 2 */
  .illustration-zone {
    position: absolute;
    left: 1340px;
    top: 180px;
    width: 860px;
    height: 900px;
  }

  .illustration-zone img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }

  /* Logo: Figma (100,491,300w) × 2 */
  .logo {
    position: absolute;
    left: 200px;
    top: 982px;
    width: 600px;
    color: ${logoTextColor};
  }

  .logo svg {
    width: 100%;
    height: auto;
    display: block;
  }
</style>
</head>
<body>
<div class="canvas">

  <div class="title">${titleHTML}</div>

  <div class="illustration-zone">
    <img src="data:image/png;base64,${illustrationBase64}" alt="">
  </div>

  <div class="logo">${logoHtml}</div>

</div>
</body>
</html>`;
}

// Step 1a: Generate prompt
app.post('/api/prompt', async (req, res) => {
  try {
    const { title, content, bgColor = STYLE.gray900 } = req.body;
    if (!content?.trim()) return res.status(400).json({ error: '본문을 입력해주세요.' });

    console.log('[Prompt] title:', title?.slice(0, 40));
    const { prompt, focalColor, styleRules } = await buildIllustrationPrompt(title || '', content, bgColor);
    console.log('[Prompt]', prompt);
    res.json({ prompt, focalColor, styleRules });
  } catch (err) {
    console.error('[Error]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Step 1b: Generate illustration
app.post('/api/illustration', async (req, res) => {
  try {
    const { prompt, bgColor = STYLE.gray900, model = 'openai' } = req.body;
    if (!prompt?.trim()) return res.status(400).json({ error: '프롬프트를 입력해주세요.' });

    const focalColor = resolveFocalColor(bgColor);
    const styleRules = STYLE_RULES.replaceAll('{focalColor}', focalColor);
    const finalPrompt = `${prompt}\n\n${styleRules}`;
    console.log('[Illustration]', finalPrompt.slice(0, 120));
    const illustrationBase64 = await generateIllustration(finalPrompt, model);
    res.json({ illustrationBase64 });
  } catch (err) {
    console.error('[Error]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Step 2: Render thumbnail only (fast, Puppeteer only)
app.post('/api/render', async (req, res) => {
  try {
    const {
      title,
      bgColor = STYLE.gray900,
      textColor = STYLE.gray50,
      illustrationBase64,
    } = req.body;

    if (!title?.trim()) return res.status(400).json({ error: '제목을 입력해주세요.' });
    if (!illustrationBase64) return res.status(400).json({ error: '일러스트를 먼저 생성해주세요.' });

    console.log('[Render] title:', title.slice(0, 40));

    const html = buildThumbnailHTML({ title, bgColor, textColor, illustrationBase64 });

    const browser = await puppeteer.launch({
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 2400, height: 1260 });
    await page.setContent(html, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 600));
    const screenshot = await page.screenshot({ type: 'png' });
    await browser.close();

    const filename = `thumbnail_${Date.now()}.png`;
    await fs.writeFile(path.join(__dirname, 'output', filename), screenshot);
    console.log('[Output]', filename);

    res.json({ image: Buffer.from(screenshot).toString('base64'), filename });
  } catch (err) {
    console.error('[Error]', err.message);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`서버 시작: http://localhost:${PORT}`);
});
