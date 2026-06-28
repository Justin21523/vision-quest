import { chromium } from '@playwright/test'
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'

const root = path.resolve(new URL('.', import.meta.url).pathname, '../../..')
const appDir = path.join(root, 'frontend/react_app')
const outDir = path.join(root, 'docs/demo')
const screenshotDir = path.join(outDir, 'screenshots')
const videoDir = path.join(outDir, 'demo')

await fs.rm(screenshotDir, { recursive: true, force: true })
await fs.rm(videoDir, { recursive: true, force: true })
await fs.mkdir(screenshotDir, { recursive: true })
await fs.mkdir(videoDir, { recursive: true })

const server = spawn('npm', ['run', 'dev', '--', '--host', '127.0.0.1', '--port', '4177'], {
  cwd: appDir,
  stdio: 'ignore',
  shell: process.platform === 'win32',
})

async function waitForServer(url, timeoutMs = 15000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
  }
  throw new Error(`Timed out waiting for ${url}`)
}

const browser = await chromium.launch()
try {
  const demoUrl = 'http://127.0.0.1:4177/'
  await waitForServer(demoUrl)
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
    recordVideo: { dir: videoDir, size: { width: 1440, height: 980 } },
  })
  const page = await context.newPage()
  await page.goto(demoUrl, { waitUntil: 'networkidle' })
  await page.screenshot({ path: path.join(screenshotDir, '01-workspace-overview.png'), fullPage: true })

  await page.getByRole('button', { name: 'Run next step' }).click()
  await page.waitForTimeout(250)
  await page.getByRole('button', { name: 'Run next step' }).click()
  await page.waitForTimeout(250)
  await page.screenshot({ path: path.join(screenshotDir, '02-pipeline-running.png'), fullPage: true })

  const views = [
    ['Vision Lab', '03-vision-lab.png'],
    ['Knowledge Base', '04-knowledge-base.png'],
    ['Agent Trace', '05-agent-trace.png'],
    ['Adventure State', '06-adventure-state.png'],
    ['API Inspector', '07-api-inspector.png'],
    ['System Monitor', '08-system-monitor.png'],
  ]

  for (const [label, file] of views) {
    await page.getByRole('button', { name: label }).click()
    await page.waitForTimeout(350)
    await page.screenshot({ path: path.join(screenshotDir, file), fullPage: true })
  }

  await page.getByRole('button', { name: 'Workspace' }).click()
  await page.getByRole('button', { name: 'Game UI Screenshot Review' }).click()
  await page.waitForTimeout(500)
  await page.screenshot({ path: path.join(screenshotDir, '09-scenario-game-ui.png'), fullPage: true })

  const video = page.video()
  await context.close()
  const videoPath = await video.path()
  await fs.copyFile(videoPath, path.join(videoDir, 'demo-tour.webm'))
  await fs.rm(videoPath, { force: true })
  await fs.copyFile(path.join(screenshotDir, '01-workspace-overview.png'), path.join(outDir, 'cover.png'))
} finally {
  await browser.close()
  server.kill('SIGTERM')
}
