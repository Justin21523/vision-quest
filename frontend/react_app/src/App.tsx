import { useMemo, useState } from 'react'
import {
  Activity,
  Bot,
  Camera,
  Database,
  GitBranch,
  Image as ImageIcon,
  MessageSquare,
  Play,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react'
import './App.css'

type StepId = 'vision' | 'vqa' | 'rag' | 'agent' | 'game' | 'history'

const steps: Array<{ id: StepId; label: string; icon: JSX.Element }> = [
  { id: 'vision', label: 'Caption', icon: <Camera size={16} /> },
  { id: 'vqa', label: 'VQA', icon: <ImageIcon size={16} /> },
  { id: 'rag', label: 'RAG', icon: <Database size={16} /> },
  { id: 'agent', label: 'Agent', icon: <Bot size={16} /> },
  { id: 'game', label: 'Adventure', icon: <Sparkles size={16} /> },
  { id: 'history', label: 'DAG', icon: <GitBranch size={16} /> },
]

const sampleAnswers: Record<StepId, string> = {
  vision: 'Demo caption: uploaded image, 1280x720px, dominant cyan and slate palette, likely a dashboard or gameplay scene.',
  vqa: 'Q: What is the important object? A: The highlighted control panel is the active decision point in this demo frame.',
  rag: 'Top retrieval: Mock-safe mode keeps the UI, API contracts, and data flow demonstrable without GPU model weights.',
  agent: 'Agent plan: inspect image metadata -> retrieve project context -> update the adventure state -> save branch event.',
  game: 'The archive gate opens. Health remains 100, mana drops to 45, and the player gains access_glyph.',
  history: 'DAG branch recorded: Caption -> VQA -> RAG -> Agent -> Adventure state update.',
}

function App() {
  const [active, setActive] = useState<StepId>('vision')
  const [query, setQuery] = useState('What should the agent inspect first?')
  const [log, setLog] = useState<string[]>([
    'System booted in mock-safe mode.',
    'Seed data loaded: 3 knowledge chunks, 1 adventure scenario, 1 DAG branch.',
  ])

  const stats = useMemo(
    () => [
      ['API groups', '8'],
      ['Demo mode', 'Mock-safe'],
      ['Build target', 'GitHub Pages'],
      ['Smoke tests', '7 passing'],
    ],
    [],
  )

  const runStep = (id = active) => {
    setActive(id)
    setLog((current) => [`${steps.find((s) => s.id === id)?.label}: ${sampleAnswers[id]}`, ...current].slice(0, 6))
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="VisionQuest demo workspace">
        <aside className="rail">
          <div className="brand">
            <div className="brand-mark">
              <Sparkles size={22} />
            </div>
            <div>
              <h1>VisionQuest</h1>
              <p>Multi-modal AI demo console</p>
            </div>
          </div>

          <div className="mode-pill">
            <ShieldCheck size={16} />
            <span>Mock-safe demo mode</span>
          </div>

          <nav className="steps" aria-label="Demo modules">
            {steps.map((step) => (
              <button
                key={step.id}
                className={active === step.id ? 'step is-active' : 'step'}
                onClick={() => runStep(step.id)}
              >
                {step.icon}
                <span>{step.label}</span>
              </button>
            ))}
          </nav>

          <div className="interview-card">
            <p className="eyebrow">Interview focus</p>
            <strong>One API shape, two execution modes.</strong>
            <span>GPU inference is optional; the demo keeps workflows, payloads, screenshots, and video stable.</span>
          </div>
        </aside>

        <section className="product">
          <header className="topbar">
            <div>
              <p className="eyebrow">Portfolio demo</p>
              <h2>Unified Vision, Retrieval, Agent, and Adventure Pipeline</h2>
            </div>
            <button className="primary-action" onClick={() => runStep()}>
              <Play size={16} />
              Run active module
            </button>
          </header>

          <section className="hero-console">
            <div className="visual-panel">
              <div className="image-stage">
                <div className="scan-line" />
                <div className="asset-card">
                  <Upload size={28} />
                  <span>sample_dashboard.png</span>
                </div>
                <div className="hotspot one">VQA</div>
                <div className="hotspot two">RAG</div>
              </div>
              <div className="caption-strip">
                <Camera size={16} />
                <span>{sampleAnswers[active]}</span>
              </div>
            </div>

            <div className="control-panel">
              <label htmlFor="query">Natural language task</label>
              <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} />
              <div className="answer-box">
                <MessageSquare size={18} />
                <p>
                  {query.includes('first')
                    ? 'The agent should inspect the visible control panel, then retrieve safety context before changing game state.'
                    : sampleAnswers[active]}
                </p>
              </div>
              <div className="stat-grid">
                {stats.map(([label, value]) => (
                  <div className="stat" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="lower-grid">
            <div className="panel">
              <div className="panel-title">
                <Database size={18} />
                <h3>Knowledge Evidence</h3>
              </div>
              <table>
                <tbody>
                  <tr>
                    <th>Retrieved chunk</th>
                    <td>mock-safe mode design</td>
                  </tr>
                  <tr>
                    <th>Source</th>
                    <td>README architecture notes</td>
                  </tr>
                  <tr>
                    <th>Confidence</th>
                    <td>0.91 demo score</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="panel">
              <div className="panel-title">
                <GitBranch size={18} />
                <h3>Operation Timeline</h3>
              </div>
              <ol className="timeline">
                {log.map((entry, index) => (
                  <li key={`${entry}-${index}`}>{entry}</li>
                ))}
              </ol>
            </div>

            <div className="panel telemetry">
              <div className="panel-title">
                <Activity size={18} />
                <h3>System Telemetry</h3>
              </div>
              <div className="bars">
                <span style={{ '--value': '68%' } as React.CSSProperties}>Frontend build</span>
                <span style={{ '--value': '84%' } as React.CSSProperties}>API smoke</span>
                <span style={{ '--value': '72%' } as React.CSSProperties}>Demo coverage</span>
              </div>
            </div>
          </section>
        </section>
      </section>
    </main>
  )
}

export default App
