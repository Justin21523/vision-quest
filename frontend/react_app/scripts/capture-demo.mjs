import { chromium } from '@playwright/test'
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'

const root = path.resolve(new URL('.', import.meta.url).pathname, '../../..')
const appDir = path.join(root, 'frontend/react_app')
const outDir = path.join(root, 'docs/demo')
const screenshotDir = path.join(outDir, 'screenshots')
const videoDir = path.join(outDir, 'demo')

await fs.mkdir(screenshotDir, { recursive: true })
await fs.mkdir(videoDir, { recursive: true })

const server = spawn('npm', ['run', 'dev', '--', '--host', '127.0.0.1', '--port', '4177'], {
  cwd: appDir,
  stdio: 'ignore',
  shell: process.platform === 'win32',
})

const browser = await chromium.launch()
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
    recordVideo: { dir: videoDir, size: { width: 1440, height: 980 } },
  })
  const page = await context.newPage()
  await page.goto('http://127.0.0.1:4177/', { waitUntil: 'networkidle' })
  await page.screenshot({ path: path.join(screenshotDir, '01-console-overview.png'), fullPage: true })

  for (const [index, label] of ['VQA', 'RAG', 'Agent', 'Adventure', 'DAG'].entries()) {
    await page.getByRole('button', { name: label }).click()
    await page.waitForTimeout(350)
    await page.screenshot({
      path: path.join(screenshotDir, `${String(index + 2).padStart(2, '0')}-${label.toLowerCase()}.png`),
      fullPage: true,
    })
  }

  await page.getByRole('button', { name: 'Run active module' }).click()
  await page.waitForTimeout(700)
  await page.screenshot({ path: path.join(screenshotDir, '07-recordable-state.png'), fullPage: true })

  const video = page.video()
  await context.close()
  const videoPath = await video.path()
  await fs.copyFile(videoPath, path.join(videoDir, 'demo-tour.webm'))
  await fs.copyFile(path.join(screenshotDir, '01-console-overview.png'), path.join(outDir, 'cover.png'))
} finally {
  await browser.close()
  server.kill('SIGTERM')
}
