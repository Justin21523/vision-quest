import { useMemo, useState } from 'react'
import {
  Activity,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  FileJson,
  GitBranch,
  Image as ImageIcon,
  Play,
  Plus,
  Radio,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react'
import { apiEndpoints, scenarios, systemMetrics, type Scenario, type ViewId, views } from './demoData'
import './App.css'

const pipeline = ['caption', 'vqa', 'rag', 'agent', 'adventure', 'history'] as const

function App() {
  const [activeView, setActiveView] = useState<ViewId>('workspace')
  const [scenarioId, setScenarioId] = useState<Scenario['id']>('museum')
  const [completed, setCompleted] = useState(2)
  const [query, setQuery] = useState('What should the agent inspect before changing state?')
  const [note, setNote] = useState('Mock-safe knowledge note: reviewer can inspect every payload without model weights.')

  const scenario = scenarios.find((item) => item.id === scenarioId) ?? scenarios[0]
  const activeStep = pipeline[Math.min(completed, pipeline.length - 1)]
  const latestVqa = scenario.vqa[0]
  const totalLatency = scenario.agentSteps.reduce((sum, step) => sum + step.ms, 0) + 184

  const apiPayload = useMemo(
    () => ({
      request: {
        scenario: scenario.id,
        file: scenario.fileName,
        prompt: scenario.prompt,
        question: query,
        mode: 'mock-safe',
      },
      response: {
        caption: scenario.caption,
        objects: scenario.objects,
        rag_results: scenario.rag,
        agent_steps: scenario.agentSteps,
        game_state: scenario.game,
        latency_ms: totalLatency,
      },
    }),
    [query, scenario, totalLatency],
  )

  const runPipeline = () => {
    setCompleted((current) => (current >= pipeline.length ? 1 : current + 1))
  }

  return (
    <main className="app-shell">
      <section className="workspace-shell">
        <aside className="rail">
          <div className="brand">
            <div className="brand-mark">
              <Sparkles size={22} />
            </div>
            <div>
              <h1>VisionQuest</h1>
              <p>Multimodal AI operations console</p>
            </div>
          </div>

          <div className="mode-pill">
            <ShieldCheck size={16} />
            <span>Mock-safe public demo</span>
          </div>

          <nav className="view-nav" aria-label="Demo views">
            {views.map((view) => (
              <NavButton
                key={view.id}
                view={view}
                active={activeView === view.id}
                onClick={() => setActiveView(view.id)}
              />
            ))}
          </nav>

          <div className="side-card">
            <p className="eyebrow">Reviewer path</p>
            <strong>Select a scenario, run the pipeline, inspect payloads, then open the generated media.</strong>
          </div>
        </aside>

        <section className="product">
          <header className="topbar">
            <div>
              <p className="eyebrow">Portfolio demo v2</p>
              <h2>Realistic Multimodal Workflow With Inspectable State</h2>
            </div>
            <button className="primary-action" onClick={runPipeline}>
              <Play size={16} />
              {completed >= pipeline.length ? 'Restart pipeline' : 'Run next step'}
            </button>
          </header>

          <section className="scenario-strip">
            {scenarios.map((item) => (
              <button
                key={item.id}
                className={item.id === scenario.id ? 'scenario-card is-active' : 'scenario-card'}
                onClick={() => {
                  setScenarioId(item.id)
                  setCompleted(2)
                }}
              >
                <span>{item.domain}</span>
                <strong>{item.title}</strong>
                <small>{item.fileName}</small>
              </button>
            ))}
          </section>

          <section className="pipeline">
            {pipeline.map((step, index) => (
              <div key={step} className={index < completed ? 'pipe-step done' : index === completed ? 'pipe-step current' : 'pipe-step'}>
                {index < completed ? <CheckCircle2 size={16} /> : <Clock3 size={16} />}
                <span>{step}</span>
              </div>
            ))}
          </section>

          {activeView === 'workspace' && (
            <section className="main-grid">
              <VisualWorkbench scenario={scenario} activeStep={activeStep} />
              <ControlPanel scenario={scenario} query={query} setQuery={setQuery} latestVqa={latestVqa} />
              <EvidencePanel scenario={scenario} />
              <TimelinePanel scenario={scenario} completed={completed} />
              <SystemSummary totalLatency={totalLatency} />
            </section>
          )}

          {activeView === 'vision' && <VisionLab scenario={scenario} />}
          {activeView === 'knowledge' && <KnowledgeBase scenario={scenario} note={note} setNote={setNote} />}
          {activeView === 'agent' && <AgentTrace scenario={scenario} />}
          {activeView === 'adventure' && <AdventureState scenario={scenario} />}
          {activeView === 'api' && <ApiInspector payload={apiPayload} />}
          {activeView === 'system' && <SystemMonitor />}
        </section>
      </section>
    </main>
  )
}

function NavButton({
  view,
  active,
  onClick,
}: {
  view: (typeof views)[number]
  active: boolean
  onClick: () => void
}) {
  const Icon = view.icon
  return (
    <button className={active ? 'nav-item is-active' : 'nav-item'} onClick={onClick}>
      <Icon size={16} />
      <span>{view.label}</span>
    </button>
  )
}

function VisualWorkbench({ scenario, activeStep }: { scenario: Scenario; activeStep: string }) {
  return (
    <div className="panel visual-workbench">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Image workspace</p>
          <h3>{scenario.fileName}</h3>
        </div>
        <span className="status-chip">{activeStep}</span>
      </div>
      <div className="image-stage">
        <div className="scan-line" />
        <div className="mock-image" style={{ background: `linear-gradient(135deg, ${scenario.palette.join(', ')})` }}>
          <Upload size={34} />
          <strong>{scenario.title}</strong>
          <span>{scenario.domain}</span>
        </div>
        {scenario.objects.slice(0, 3).map((object, index) => (
          <span key={object} className={`hotspot hotspot-${index + 1}`}>
            {object}
          </span>
        ))}
      </div>
      <div className="caption-strip">
        <ImageIcon size={16} />
        <span>{scenario.caption}</span>
      </div>
    </div>
  )
}

function ControlPanel({
  scenario,
  query,
  setQuery,
  latestVqa,
}: {
  scenario: Scenario
  query: string
  setQuery: (value: string) => void
  latestVqa: Scenario['vqa'][number]
}) {
  return (
    <div className="panel control-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Task input</p>
          <h3>Natural language control</h3>
        </div>
        <Radio size={18} />
      </div>
      <label htmlFor="query">Reviewer prompt</label>
      <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} />
      <div className="answer-box">
        <Sparkles size={18} />
        <p>{query.toLowerCase().includes('before') ? latestVqa.answer : scenario.prompt}</p>
      </div>
      <div className="metric-row">
        <div>
          <span>Confidence</span>
          <strong>{Math.round(latestVqa.confidence * 100)}%</strong>
        </div>
        <div>
          <span>Objects</span>
          <strong>{scenario.objects.length}</strong>
        </div>
        <div>
          <span>Branch</span>
          <strong>{scenario.game.branch}</strong>
        </div>
      </div>
    </div>
  )
}

function EvidencePanel({ scenario }: { scenario: Scenario }) {
  return (
    <div className="panel">
      <div className="panel-head compact">
        <Database size={18} />
        <h3>RAG Evidence</h3>
      </div>
      <div className="evidence-list">
        {scenario.rag.map((item) => (
          <article key={item.source}>
            <div>
              <strong>{item.source}</strong>
              <span>{Math.round(item.score * 100)}%</span>
            </div>
            <p>{item.chunk}</p>
            <small>{item.reason}</small>
          </article>
        ))}
      </div>
    </div>
  )
}

function TimelinePanel({ scenario, completed }: { scenario: Scenario; completed: number }) {
  const events = [
    `Loaded ${scenario.fileName}`,
    `Caption extracted ${scenario.objects.length} objects`,
    `VQA answered: ${scenario.vqa[0].answer}`,
    `RAG matched ${scenario.rag[0].source}`,
    `Agent opened branch ${scenario.game.branch}`,
    `Adventure state updated with ${scenario.game.inventory[scenario.game.inventory.length - 1]}`,
  ].slice(0, Math.max(2, completed + 1))

  return (
    <div className="panel">
      <div className="panel-head compact">
        <GitBranch size={18} />
        <h3>Operation Timeline</h3>
      </div>
      <ol className="timeline">
        {events.map((entry) => (
          <li key={entry}>{entry}</li>
        ))}
      </ol>
    </div>
  )
}

function SystemSummary({ totalLatency }: { totalLatency: number }) {
  return (
    <div className="panel telemetry">
      <div className="panel-head compact">
        <Activity size={18} />
        <h3>Run Summary</h3>
      </div>
      <div className="summary-grid">
        <div><span>Latency</span><strong>{totalLatency}ms</strong></div>
        <div><span>Mode</span><strong>mock-safe</strong></div>
        <div><span>API groups</span><strong>8</strong></div>
        <div><span>Tests</span><strong>7 passing</strong></div>
      </div>
    </div>
  )
}

function VisionLab({ scenario }: { scenario: Scenario }) {
  return (
    <section className="detail-grid">
      <VisualWorkbench scenario={scenario} activeStep="vision-lab" />
      <div className="panel">
        <div className="panel-head compact">
          <ImageIcon size={18} />
          <h3>Detected Objects</h3>
        </div>
        <div className="tag-cloud">
          {scenario.objects.map((object) => <span key={object}>{object}</span>)}
        </div>
        <h4>Visual Q&A</h4>
        {scenario.vqa.map((item) => (
          <article className="qa-card" key={item.question}>
            <strong>{item.question}</strong>
            <p>{item.answer}</p>
            <small>{Math.round(item.confidence * 100)}% confidence</small>
          </article>
        ))}
      </div>
    </section>
  )
}

function KnowledgeBase({ scenario, note, setNote }: { scenario: Scenario; note: string; setNote: (value: string) => void }) {
  return (
    <section className="detail-grid">
      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Knowledge entry</p>
            <h3>Add reviewer note</h3>
          </div>
          <Plus size={18} />
        </div>
        <textarea value={note} onChange={(event) => setNote(event.target.value)} />
        <button className="secondary-action">
          <Search size={16} />
          Query evidence
        </button>
      </div>
      <EvidencePanel scenario={scenario} />
    </section>
  )
}

function AgentTrace({ scenario }: { scenario: Scenario }) {
  return (
    <section className="panel full">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Tool execution</p>
          <h3>Agent Reasoning Trace</h3>
        </div>
        <Bot size={20} />
      </div>
      <div className="agent-steps">
        {scenario.agentSteps.map((step, index) => (
          <article key={step.tool}>
            <span>{index + 1}</span>
            <div>
              <strong>{step.tool}</strong>
              <p>Input: {step.input}</p>
              <p>Output: {step.output}</p>
            </div>
            <small>{step.ms}ms</small>
          </article>
        ))}
      </div>
    </section>
  )
}

function AdventureState({ scenario }: { scenario: Scenario }) {
  return (
    <section className="detail-grid">
      <div className="panel adventure-card">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Branch {scenario.game.branch}</p>
            <h3>Adventure State</h3>
          </div>
          <Sparkles size={20} />
        </div>
        <p className="narrative">{scenario.game.narrative}</p>
        <div className="metric-row">
          <div><span>Health</span><strong>{scenario.game.health}</strong></div>
          <div><span>Mana</span><strong>{scenario.game.mana}</strong></div>
          <div><span>Inventory</span><strong>{scenario.game.inventory.length}</strong></div>
        </div>
      </div>
      <div className="panel">
        <div className="panel-head compact">
          <ChevronRight size={18} />
          <h3>Next Choices</h3>
        </div>
        <div className="choice-list">
          {scenario.game.choices.map((choice) => <button key={choice}>{choice}</button>)}
        </div>
      </div>
    </section>
  )
}

function ApiInspector({ payload }: { payload: object }) {
  return (
    <section className="detail-grid api-grid">
      <div className="panel">
        <div className="panel-head compact">
          <FileJson size={18} />
          <h3>Endpoints</h3>
        </div>
        <div className="endpoint-list">
          {apiEndpoints.map((endpoint) => <code key={endpoint}>{endpoint}</code>)}
        </div>
      </div>
      <div className="panel">
        <div className="panel-head compact">
          <FileJson size={18} />
          <h3>Current Payload</h3>
        </div>
        <pre>{JSON.stringify(payload, null, 2)}</pre>
      </div>
    </section>
  )
}

function SystemMonitor() {
  return (
    <section className="panel full">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Deployment evidence</p>
          <h3>System Monitor</h3>
        </div>
        <Activity size={20} />
      </div>
      <div className="monitor-grid">
        {systemMetrics.map((metric) => (
          <article key={metric.label}>
            <div>
              <strong>{metric.label}</strong>
              <span>{metric.detail}</span>
            </div>
            <div className="bar"><span style={{ width: `${metric.value}%` }} /></div>
            <small>{metric.value}%</small>
          </article>
        ))}
      </div>
    </section>
  )
}

export default App
