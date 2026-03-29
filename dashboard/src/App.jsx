import { useState, useEffect, useCallback, useRef } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import ForceGraph2D from 'react-force-graph-2d'

const API = 'http://localhost:3001'

// ── Color Maps ────────────────────────────────────────────────────────────

const TYPE_COLORS = {
  observation: '#c9a227',
  task: '#3a8ec9',
  idea: '#3aad6e',
  reference: '#8a7f6e',
  person_note: '#c94a3a',
  instruction: '#c084fc',
}

const PROJECT_COLORS = {
  SABLE: '#c9a227',
  CORTEX: '#3a8ec9',
  GameForge: '#3aad6e',
  'Contact Front!': '#c94a3a',
  'Cultivated Learning': '#c084fc',
  'infra-toolkit': '#8a7f6e',
}

const REL_COLORS = {
  supports: '#3aad6e',
  contradicts: '#c94a3a',
  elaborates: '#c9a227',
  depends_on: '#3a8ec9',
  caused_by: '#c084fc',
  related: '#8a7f6e',
  supersedes: '#e8ddc4',
}

// ── Dashboard Page ────────────────────────────────────────────────────────

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/stats`).then(r => r.json()).then(setStats).catch(() => {})
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {})
  }, [])

  if (!stats) return <div className="loading">Loading</div>

  const maxType = Math.max(...Object.values(stats.type_counts))
  const maxRel = Math.max(...Object.values(stats.rel_counts))

  return (
    <div>
      <div className="page-header">
        <h2>COGNITIVE ARCHITECTURE DASHBOARD</h2>
        <div className="description">Project PARALLAX — Three perspectives, one truth</div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.n_thoughts}</div>
          <div className="stat-label">Thoughts</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.n_links}</div>
          <div className="stat-label">Links</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.n_projects}</div>
          <div className="stat-label">Projects</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.isolated}</div>
          <div className="stat-label">Isolated</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.avg_degree}</div>
          <div className="stat-label">Avg Degree</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.max_degree}</div>
          <div className="stat-label">Max Degree</div>
        </div>
      </div>

      {health && (
        <div className="card">
          <div className="card-header">System Status</div>
          <div style={{ display: 'flex', gap: 24 }}>
            {Object.entries(health).map(([k, v]) => {
              let statusClass = 'error'
              if (v === 'ok') statusClass = 'ok'
              else if (v === 'offline') statusClass = 'offline'
              return (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className={`status status-${statusClass}`} />
                  <span style={{ color: 'var(--dim)', fontSize: 11, textTransform: 'uppercase' }}>{k}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <div className="two-col">
        <div className="card">
          <div className="card-header">Thought Types</div>
          {Object.entries(stats.type_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([type, count]) => (
              <div key={type} className="bar-row">
                <span className="bar-label">{type}</span>
                <div className="bar" style={{ width: `${(count / maxType) * 100}%`, background: TYPE_COLORS[type] || 'var(--gold)' }} />
                <span className="bar-value">{count}</span>
              </div>
            ))}
        </div>

        <div className="card">
          <div className="card-header">Link Types</div>
          {Object.entries(stats.rel_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([type, count]) => (
              <div key={type} className="bar-row">
                <span className="bar-label">{type}</span>
                <div className="bar" style={{ width: `${(count / maxRel) * 100}%`, background: REL_COLORS[type] || 'var(--gold)' }} />
                <span className="bar-value">{count}</span>
              </div>
            ))}
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-header">Projects</div>
          {Object.entries(stats.project_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([name, count]) => (
              <div key={name} className="bar-row">
                <span className="bar-label">{name}</span>
                <div className="bar" style={{ width: `${(count / Math.max(...Object.values(stats.project_counts))) * 100}%`, background: PROJECT_COLORS[name] || 'var(--gold)' }} />
                <span className="bar-value">{count}</span>
              </div>
            ))}
        </div>

        <div className="card">
          <div className="card-header">Top Topics</div>
          {stats.top_topics.slice(0, 10).map(([topic, count]) => (
            <div key={topic} className="bar-row">
              <span className="bar-label" style={{ minWidth: 160 }}>{topic}</span>
              <span className="bar-value">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Graph Page ────────────────────────────────────────────────────────────

function GraphView() {
  const [graphData, setGraphData] = useState(null)
  const [selected, setSelected] = useState(null)
  const [colorBy, setColorBy] = useState('type')
  const graphRef = useRef()

  useEffect(() => {
    fetch(`${API}/api/graph`)
      .then(r => r.json())
      .then(data => {
        const nodeSet = new Set(data.nodes.map(n => n.id))
        const edges = data.edges.filter(e => nodeSet.has(e.source) && nodeSet.has(e.target))

        // Pre-compute degree ONCE so we don't recalculate every frame
        const degreeMap = {}
        edges.forEach(e => {
          degreeMap[e.source] = (degreeMap[e.source] || 0) + 1
          degreeMap[e.target] = (degreeMap[e.target] || 0) + 1
        })
        const nodes = data.nodes.map(n => ({
          ...n,
          _degree: degreeMap[n.id] || 0,
          // Pre-compute size: isolated=2, low=3, medium=5, hub=8+
          _size: Math.max(2, Math.min(12, 2 + Math.sqrt(degreeMap[n.id] || 0) * 1.8)),
        }))
        setGraphData({ nodes, links: edges })
      })
      .catch(() => {})
  }, [])

  const getNodeColor = useCallback((node) => {
    if (colorBy === 'project') return PROJECT_COLORS[node.project] || '#2a2520'
    return TYPE_COLORS[node.type] || '#8a7f6e'
  }, [colorBy])

  if (!graphData) return <div className="loading">Loading graph</div>

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2>KNOWLEDGE GRAPH</h2>
          <div className="description">
            {graphData.nodes.length} nodes, {graphData.links.length} edges
          </div>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className={`btn ${colorBy === 'type' ? 'btn-gold' : ''}`} onClick={() => setColorBy('type')}>Type</button>
          <button className={`btn ${colorBy === 'project' ? 'btn-gold' : ''}`} onClick={() => setColorBy('project')}>Project</button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 0, height: 'calc(100vh - 140px)' }}>
        {/* Graph canvas */}
        <div style={{ flex: 1, position: 'relative', background: '#0a0908', border: '1px solid #2a2520', borderRadius: 4, overflow: 'hidden' }}>
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            width={typeof globalThis.window !== 'undefined' ? globalThis.window.innerWidth - 220 - 280 - 40 : 800}
            height={typeof globalThis.window !== 'undefined' ? globalThis.window.innerHeight - 140 : 600}
            nodeCanvasObjectMode={() => 'replace'}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const r = node._size
              ctx.beginPath()
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
              ctx.fillStyle = getNodeColor(node)
              ctx.fill()

              if (selected?.id === node.id) {
                ctx.strokeStyle = '#c9a227'
                ctx.lineWidth = 1.5
                ctx.stroke()
              }

              if (globalScale > 2) {
                const label = (node.content || '').slice(0, 30)
                ctx.font = `${Math.max(3, 10 / globalScale)}px JetBrains Mono`
                ctx.fillStyle = '#8a7f6e'
                ctx.textAlign = 'center'
                ctx.fillText(label, node.x, node.y + r + 4)
              }
            }}
            nodePointerAreaPaint={(node, color, ctx, globalScale) => {
              // Hit area must be large enough to click at any zoom level
              // At low zoom (globalScale < 1), nodes are tiny — need bigger hit area
              const minHit = 12 / Math.max(globalScale, 0.3)
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x, node.y, Math.max(node._size, minHit), 0, 2 * Math.PI)
              ctx.fill()
            }}
            linkColor={link => {
              const rel = link.relation || 'related'
              return (REL_COLORS[rel] || '#2a2520') + '30'
            }}
            linkWidth={0.4}
            backgroundColor="#0a0908"
            onNodeClick={(node) => setSelected(node)}
            enableNodeDrag={true}
            cooldownTicks={100}
            warmupTicks={50}
          />

          {/* Legend — bottom left, inside canvas */}
          <div style={{
            position: 'absolute', bottom: 10, left: 10,
            background: 'rgba(10,9,8,0.92)', border: '1px solid #2a2520',
            padding: '6px 10px', borderRadius: 4, fontSize: 10, zIndex: 5,
          }}>
            {Object.entries(colorBy === 'project' ? PROJECT_COLORS : TYPE_COLORS).map(([name, color]) => (
              <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                <div style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
                <span style={{ color: '#8a7f6e', whiteSpace: 'nowrap' }}>{name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Detail panel — fixed sidebar on right, always visible */}
        <div style={{
          width: 280, flexShrink: 0, background: '#151311',
          borderLeft: '1px solid #2a2520', padding: 16,
          overflowY: 'auto', height: '100%',
        }}>
          {selected ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span style={{ fontSize: 10, color: '#c9a227', letterSpacing: 1, textTransform: 'uppercase' }}>
                  {selected.type || 'thought'}
                </span>
                <span role="button" tabIndex={0} onClick={() => setSelected(null)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSelected(null); }} style={{ cursor: 'pointer', color: '#8a7f6e', fontSize: 14 }}>x</span>
              </div>
              <div style={{ fontSize: 12, color: '#c8bda0', lineHeight: 1.6, marginBottom: 12, wordBreak: 'break-word' }}>
                {selected.content}
              </div>
              <div style={{ fontSize: 10, color: '#8a7f6e', lineHeight: 1.8 }}>
                {selected.project && <div>Project: <span style={{ color: '#c8bda0' }}>{selected.project}</span></div>}
                {selected.topics?.length > 0 && <div>Topics: <span style={{ color: '#c8bda0' }}>{selected.topics.join(', ')}</span></div>}
                <div>Degree: <span style={{ color: '#c8bda0' }}>{selected._degree} links</span></div>
                <div>Created: <span style={{ color: '#c8bda0' }}>{selected.created}</span></div>
                <div style={{ marginTop: 6, fontSize: 9, wordBreak: 'break-all' }}>{selected.id}</div>
              </div>
            </>
          ) : (
            <div style={{ color: '#8a7f6e', fontSize: 11, padding: '20px 0' }}>
              Click a node to see details
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── GNN Page ──────────────────────────────────────────────────────────────

function GNNView() {
  const [gnnStats, setGnnStats] = useState(null)
  const [contradictions, setContradictions] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/gnn/stats`).then(r => r.json()).then(setGnnStats).catch(() => {})
    fetch(`${API}/api/gnn/contradictions`).then(r => r.json()).then(setContradictions).catch(() => {})
  }, [])

  return (
    <div>
      <div className="page-header">
        <h2>PILLAR 1 — STRUCTURAL REASONING</h2>
        <div className="description">EdgeConditionedGAT — Graph Neural Network</div>
      </div>

      {gnnStats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{gnnStats.num_nodes?.toLocaleString()}</div>
            <div className="stat-label">Nodes</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{gnnStats.num_edges?.toLocaleString()}</div>
            <div className="stat-label">Edges</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{gnnStats.model_params?.toLocaleString()}</div>
            <div className="stat-label">Parameters</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{gnnStats.hidden_dim}</div>
            <div className="stat-label">Hidden Dim</div>
          </div>
        </div>
      )}

      {contradictions?.contradictions && (
        <div className="card">
          <div className="card-header">Belief Tensions Detected</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Probability</th>
                <th>Current Type</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {contradictions.contradictions.slice(0, 15).map((c) => (
                <tr key={`${c.contradiction_probability}-${c.current_relation}-${c.is_known_contradiction}`}>
                  <td style={{ color: c.contradiction_probability > 0.8 ? 'var(--danger)' : 'var(--text)' }}>
                    {c.contradiction_probability}
                  </td>
                  <td>{c.current_relation}</td>
                  <td>{c.is_known_contradiction ? 'Known' : 'NEW'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!gnnStats && (
        <div className="card">
          <div className="card-header">GNN Server Status</div>
          <div style={{ color: 'var(--danger)' }}>
            GNN server offline. Start it: <code>python gnn_server.py --port 5070</code>
          </div>
        </div>
      )}
    </div>
  )
}

// ── App Shell ─────────────────────────────────────────────────────────────

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <h1>SABLE</h1>
            <div className="subtitle">COGNITIVE ARCHITECTURE</div>
          </div>
          <nav>
            <NavLink to="/" end>Dashboard</NavLink>
            <NavLink to="/graph">Knowledge Graph</NavLink>
            <NavLink to="/gnn">Structural Reasoning</NavLink>
          </nav>
          <div className="sidebar-footer">
            <div className="status-row">
              <span className="status status-ok" /> Project PARALLAX
            </div>
            <div className="status-row" style={{ color: 'var(--dim)', fontSize: 9 }}>
              Three perspectives, one truth
            </div>
          </div>
        </aside>

        <main className="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/graph" element={<GraphView />} />
            <Route path="/gnn" element={<GNNView />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
