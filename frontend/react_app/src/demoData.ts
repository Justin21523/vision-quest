import {
  Bot,
  Camera,
  Database,
  FileJson,
  type LucideIcon,
  Map,
  Monitor,
  Sparkles,
} from 'lucide-react'

export type ViewId = 'workspace' | 'vision' | 'knowledge' | 'agent' | 'adventure' | 'api' | 'system'

export type ScenarioId = 'museum' | 'game-ui' | 'research' | 'dashboard'

export interface Scenario {
  id: ScenarioId
  title: string
  fileName: string
  domain: string
  prompt: string
  palette: string[]
  caption: string
  objects: string[]
  vqa: Array<{ question: string; answer: string; confidence: number }>
  rag: Array<{ source: string; chunk: string; score: number; reason: string }>
  agentSteps: Array<{ tool: string; input: string; output: string; ms: number }>
  game: {
    branch: string
    narrative: string
    health: number
    mana: number
    inventory: string[]
    choices: string[]
  }
}

export const views: Array<{ id: ViewId; label: string; icon: LucideIcon }> = [
  { id: 'workspace', label: 'Workspace', icon: Sparkles },
  { id: 'vision', label: 'Vision Lab', icon: Camera },
  { id: 'knowledge', label: 'Knowledge Base', icon: Database },
  { id: 'agent', label: 'Agent Trace', icon: Bot },
  { id: 'adventure', label: 'Adventure State', icon: Map },
  { id: 'api', label: 'API Inspector', icon: FileJson },
  { id: 'system', label: 'System Monitor', icon: Monitor },
]

export const scenarios: Scenario[] = [
  {
    id: 'museum',
    title: 'Museum Archive Intake',
    fileName: 'archive_display_042.png',
    domain: 'Digital archive',
    prompt: 'Identify the artifact panel, retrieve conservation context, and suggest the next safe action.',
    palette: ['#0f172a', '#38bdf8', '#f59e0b', '#e2e8f0'],
    caption:
      'A museum archive workstation showing a cyan-lit artifact panel, metadata fields, and an amber warning tag near the access controls.',
    objects: ['artifact panel', 'metadata table', 'warning tag', 'operator controls'],
    vqa: [
      { question: 'What object should be inspected first?', answer: 'The amber warning tag beside the artifact panel.', confidence: 0.91 },
      { question: 'Is the scene safe to proceed?', answer: 'Proceed only after retrieving the conservation note.', confidence: 0.84 },
    ],
    rag: [
      {
        source: 'conservation-playbook.md',
        chunk: 'Amber tags indicate restricted handling until humidity and provenance checks are confirmed.',
        score: 0.93,
        reason: 'Matches warning tag and artifact-handling terms.',
      },
      {
        source: 'archive-api-contract.md',
        chunk: 'Vision evidence should be linked to a branch id before the agent updates state.',
        score: 0.87,
        reason: 'Explains why the DAG is updated after RAG retrieval.',
      },
    ],
    agentSteps: [
      { tool: 'visual_analysis', input: 'archive_display_042.png', output: 'Detected artifact panel and warning tag.', ms: 126 },
      { tool: 'knowledge_retrieval', input: 'restricted artifact handling', output: 'Found conservation-playbook.md score 0.93.', ms: 88 },
      { tool: 'state_update', input: 'safe action policy', output: 'Created branch archive-042-review.', ms: 42 },
    ],
    game: {
      branch: 'archive-042-review',
      narrative: 'The archive gate stays locked until the conservation note is acknowledged. A safe scan route appears on the console.',
      health: 100,
      mana: 45,
      inventory: ['old_map', 'conservation_note'],
      choices: ['Confirm humidity check', 'Ask curator agent', 'Open branch history'],
    },
  },
  {
    id: 'game-ui',
    title: 'Game UI Screenshot Review',
    fileName: 'adventure_hud_state.png',
    domain: 'Interactive narrative',
    prompt: 'Read the HUD, answer what changed, and record the branch in the adventure history.',
    palette: ['#111827', '#ef4444', '#3b82f6', '#fbbf24'],
    caption:
      'A game HUD with health, mana, inventory, and a dialogue choice panel. The inventory area highlights a newly acquired access glyph.',
    objects: ['health meter', 'mana meter', 'inventory item', 'choice panel'],
    vqa: [
      { question: 'Which player resource changed?', answer: 'Mana decreased while the access glyph was added to inventory.', confidence: 0.89 },
      { question: 'What is the recommended next move?', answer: 'Use the access glyph to inspect the sealed terminal.', confidence: 0.86 },
    ],
    rag: [
      {
        source: 'game-state-schema.md',
        chunk: 'Inventory changes should emit a state_delta and a history DAG edge.',
        score: 0.95,
        reason: 'Matches inventory and state update behavior.',
      },
      {
        source: 'demo-scenario-guide.md',
        chunk: 'Adventure state is designed as a visible proof of agent decisions.',
        score: 0.82,
        reason: 'Connects game UI to portfolio evidence.',
      },
    ],
    agentSteps: [
      { tool: 'hud_parser', input: 'adventure_hud_state.png', output: 'health=100, mana=45, item=access_glyph.', ms: 104 },
      { tool: 'branch_writer', input: 'inventory delta', output: 'Wrote edge rag-context -> adventure-state.', ms: 51 },
      { tool: 'choice_ranker', input: 'available actions', output: 'Recommended sealed terminal inspection.', ms: 69 },
    ],
    game: {
      branch: 'terminal-access-glyph',
      narrative: 'The access glyph pulses in the inventory. The sealed terminal recognizes the mark and reveals a second route.',
      health: 100,
      mana: 45,
      inventory: ['old_map', 'access_glyph'],
      choices: ['Inspect terminal', 'Save branch', 'Query RAG memory'],
    },
  },
  {
    id: 'research',
    title: 'Research Diagram QA',
    fileName: 'rag_pipeline_diagram.png',
    domain: 'AI architecture',
    prompt: 'Summarize the diagram, cite the matching architecture note, and explain the retrieval path.',
    palette: ['#0b1120', '#22c55e', '#60a5fa', '#f8fafc'],
    caption:
      'A retrieval pipeline diagram with ingestion, vector index, citation ranking, and answer synthesis blocks connected left to right.',
    objects: ['ingestion block', 'vector index', 'citation ranker', 'answer synthesis'],
    vqa: [
      { question: 'What is the data flow direction?', answer: 'The flow moves from ingestion to index, retrieval, citation ranking, and answer synthesis.', confidence: 0.94 },
      { question: 'Which module explains evidence?', answer: 'The citation ranker explains why each chunk was selected.', confidence: 0.9 },
    ],
    rag: [
      {
        source: 'README.md#data-flow',
        chunk: 'The demo follows image input, caption, VQA, RAG evidence, agent planning, state update, and DAG logging.',
        score: 0.96,
        reason: 'Directly matches the diagram modules.',
      },
      {
        source: 'api-contract.md',
        chunk: 'RAG responses include source, chunk, score, and reason fields for reviewer inspection.',
        score: 0.9,
        reason: 'Matches API inspector output.',
      },
    ],
    agentSteps: [
      { tool: 'diagram_reader', input: 'rag_pipeline_diagram.png', output: 'Extracted 4 pipeline blocks.', ms: 118 },
      { tool: 'citation_lookup', input: 'data-flow modules', output: 'Matched README.md#data-flow.', ms: 76 },
      { tool: 'answer_synthesizer', input: '2 evidence chunks', output: 'Generated reviewer-facing explanation.', ms: 92 },
    ],
    game: {
      branch: 'rag-diagram-review',
      narrative: 'The agent converts the architecture diagram into a reviewable evidence chain and stores it as a reusable note.',
      health: 100,
      mana: 50,
      inventory: ['architecture_note', 'citation_map'],
      choices: ['Open API inspector', 'Compare evidence', 'Export summary'],
    },
  },
  {
    id: 'dashboard',
    title: 'Product Dashboard Audit',
    fileName: 'ops_dashboard_snapshot.png',
    domain: 'Ops dashboard',
    prompt: 'Detect health indicators, retrieve runbook context, and produce an action summary.',
    palette: ['#020617', '#34d399', '#fb7185', '#93c5fd'],
    caption:
      'An operations dashboard showing healthy services, one warning queue, a model slot table, and deployment status cards.',
    objects: ['service cards', 'warning queue', 'model slot table', 'deployment status'],
    vqa: [
      { question: 'Which component needs attention?', answer: 'The warning queue should be checked before publishing a new run.', confidence: 0.88 },
      { question: 'Are deployments healthy?', answer: 'The static demo and portfolio pages are marked healthy.', confidence: 0.92 },
    ],
    rag: [
      {
        source: 'deployment-runbook.md',
        chunk: 'Pages deployment should be verified with curl checks for HTML, media, JS, and CSS assets.',
        score: 0.91,
        reason: 'Matches deployment status and public asset checks.',
      },
      {
        source: 'smoke-test-report.md',
        chunk: 'Mock-safe backend smoke tests cover health, chat, caption, VQA, and docs endpoints.',
        score: 0.85,
        reason: 'Explains current test coverage.',
      },
    ],
    agentSteps: [
      { tool: 'telemetry_scan', input: 'ops_dashboard_snapshot.png', output: 'Detected warning queue and green deployment cards.', ms: 132 },
      { tool: 'runbook_lookup', input: 'deployment verification', output: 'Found curl asset checklist.', ms: 73 },
      { tool: 'report_writer', input: 'telemetry + runbook', output: 'Produced action summary.', ms: 61 },
    ],
    game: {
      branch: 'ops-dashboard-audit',
      narrative: 'The platform stays online. The agent queues a media verification task and marks the branch ready for reviewer walkthrough.',
      health: 96,
      mana: 52,
      inventory: ['deployment_badge', 'curl_checklist'],
      choices: ['Verify media', 'Open system monitor', 'Record walkthrough'],
    },
  },
]

export const systemMetrics = [
  { label: 'Frontend build', value: 100, detail: 'Vite static bundle' },
  { label: 'Backend smoke', value: 100, detail: '7 pytest checks' },
  { label: 'Demo assets', value: 94, detail: 'screenshots + WebM' },
  { label: 'GPU dependency', value: 0, detail: 'not required' },
]

export const apiEndpoints = [
  'POST /api/v1/caption/',
  'POST /api/v1/vqa/',
  'GET /api/v1/rag/query',
  'POST /api/v1/agent/chat',
  'POST /api/v1/game/act',
  'GET /api/v1/history/dag',
]
