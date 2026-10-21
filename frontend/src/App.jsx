import { useState, useEffect, useRef } from 'react'

const COLORS = { accepted: '#1D9E75', human: '#E8890C', dropped: '#E24B4A' }
const CLASS_COLORS = { Car: '#2f7fd3', Pedestrian: '#d0457a', Cyclist: '#5a9a1f' }   // used in the downloaded picture
const EMPTY_TEXT =
  'Pick an image (or drop one here) and press Label. Green boxes are accepted by the agent, ' +
  'orange ones wait for your decision, red ones were dropped.'
// ── Small components ──────────────────────────────────────────────────────────

function Spinner() {
  return (
    <svg className="spinning" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  )
}

function Rect({ box, color, hot, dim, dash }) {
  const [x1, y1, x2, y2] = box
  return (
    <rect
      className={`overlay-rect ${dim ? 'dim' : ''}`}
      x={x1} y={y1} width={x2 - x1} height={y2 - y1}
      stroke={color}
      strokeDasharray={dash}
      style={hot ? { fill: color, fillOpacity: 0.2, strokeWidth: 4 } : undefined}
    />
  )
}

// A small label on the picture: what the box is, and how sure the agent was.
function Tag({ d, color, result, front }) {
  const [x1, y1] = d.box
  const inside = y1 / result.height < 0.07            // too close to the top edge: put the tag inside the box
  return (
    <div className={`tag ${inside ? 'inside' : ''} ${front ? 'front' : ''}`}
      style={{ left: `${(x1 / result.width) * 100}%`, top: `${(y1 / result.height) * 100}%`, background: color }}>
      {d.class} {Math.round(d.confidence * 100)}%
    </div>
  )
}

function Stat({ n, label }) {
  return (
    <div className="stat">
      <div className="stat-n">{n}</div>
      <div className="stat-l">{label}</div>
    </div>
  )
}

function TraceLine({ line }) {
  const cut = line.indexOf(':')
  const agent = line.slice(0, cut)
  return (
    <div className="trace-line">
      <span className={`trace-agent agent-${agent}`}>{agent}</span>
      <span>{line.slice(cut + 1).trim()}</span>
    </div>
  )
}

function saveBlob(blob, filename) {
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(link.href), 1000)
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [upload, setUpload] = useState(null)           // { file, url }
  const [dragging, setDragging] = useState(false)
  const [opts, setOpts] = useState({ accept_above: 0.6, drop_below: 0.2, min_height: 0, zoom: false, critic: false })
  const [result, setResult] = useState(null)           // the API's answer
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [decisions, setDecisions] = useState({})       // index in for_human -> 'yes' | 'no'
  const [show, setShow] = useState({ accepted: true, human: true, dropped: false, labels: true })
  const [hot, setHot] = useState(null)                 // index of the box under the mouse
  const fileRef = useRef(null)

  useEffect(() => {
    fetch('/health').catch(() => setError('Cannot reach the API. Start it with:  uvicorn api:app --reload'))
  }, [])

  const setOpt = (key, value) => setOpts(o => ({ ...o, [key]: value }))

  function chooseFile(file) {
    if (!file) return
    if (!file.type.startsWith('image/')) { setError('That file is not an image.'); return }
    if (upload) URL.revokeObjectURL(upload.url)
    setUpload({ file, url: URL.createObjectURL(file) })
    setResult(null)
    setDecisions({})
    setHot(null)
    setError('')
  }

  async function label() {
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', upload.file)
      const res = await fetch(`/label-upload?${new URLSearchParams(opts)}`, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `The API answered with an error (${res.status}).`)
      }
      setResult(await res.json())
      setDecisions({})
      setHot(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const decide = (i, value) =>
    setDecisions(d => {
      const next = { ...d }
      if (next[i] === value) delete next[i]
      else next[i] = value
      return next
    })

  const forHuman = result ? result.for_human : []
  const approved = forHuman.filter((_, i) => decisions[i] === 'yes')
  const pending = forHuman.filter((_, i) => !decisions[i])
  const finalLabels = result ? [...result.accepted, ...approved] : []     // the agent's boxes + the ones you approved

  function downloadJson() {
    const labels = [
      ...result.accepted.map(d => ({ ...d, decided_by: 'agent' })),
      ...approved.map(d => ({ ...d, decided_by: 'human' })),
    ]
    const body = { image: result.image, width: result.width, height: result.height, labels }
    saveBlob(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }), `${result.image}_labels.json`)
  }

  function downloadPicture() {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = result.width
      canvas.height = result.height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, result.width, result.height)
      const lineWidth = Math.max(2, Math.round(result.width / 400))
      const fontSize = Math.max(12, Math.round(result.width / 70))
      ctx.lineWidth = lineWidth
      ctx.font = `600 ${fontSize}px Inter, sans-serif`
      ctx.textBaseline = 'top'
      for (const d of finalLabels) {
        const [x1, y1, x2, y2] = d.box
        const color = CLASS_COLORS[d.class] || COLORS.accepted
        ctx.strokeStyle = color
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        const w = ctx.measureText(d.class).width + 8
        const h = fontSize + 6
        const top = y1 - h >= 0 ? y1 - h : y1
        ctx.fillStyle = color
        ctx.fillRect(x1, top, w, h)
        ctx.fillStyle = '#fff'
        ctx.fillText(d.class, x1 + 4, top + 3)
      }
      canvas.toBlob(blob => saveBlob(blob, `${result.image}_labelled.png`), 'image/png')
    }
    img.src = upload.url
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    chooseFile(e.dataTransfer.files[0])
  }

  return (
    <div>
      {/* ── Top bar ── */}
      <div className="topbar">
        <div className="brand"><span className="brand-dot" />Auto-Labeler</div>

        <div className="field">
          <button className="btn" onClick={() => fileRef.current.click()}>{upload ? 'Change image' : 'Choose image'}</button>
          {upload && <span>{upload.file.name}</span>}
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={e => { chooseFile(e.target.files[0]); e.target.value = '' }} />
        </div>

        <label className="field">Accept above
          <input type="number" step="0.05" min="0" max="1" value={opts.accept_above}
            onChange={e => setOpt('accept_above', Number(e.target.value))} />
        </label>
        <label className="field">Drop below
          <input type="number" step="0.05" min="0" max="1" value={opts.drop_below}
            onChange={e => setOpt('drop_below', Number(e.target.value))} />
        </label>
        <label className="field">Min height (px)
          <input type="number" step="5" min="0" value={opts.min_height}
            onChange={e => setOpt('min_height', Number(e.target.value))} />
        </label>
        <label className="check">
          <input type="checkbox" checked={opts.zoom} onChange={e => setOpt('zoom', e.target.checked)} />
          Zoom retry
        </label>
        <label className="check"
          title="Claude looks at each orange box and rejects the ones that are not real objects (posters, reflections, blurry blobs). It never accepts a box by itself.">
          <input type="checkbox" checked={opts.critic} onChange={e => setOpt('critic', e.target.checked)} />
          Claude double-check
        </label>
        {opts.critic && <span className="cost-note">uses API credit (~$0.001 per uncertain box)</span>}

        <div className="spacer" />
        <button className="btn btn-primary" onClick={label} disabled={loading || !upload}>
          {loading ? <Spinner /> : null}{loading ? 'Labelling…' : 'Label'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="main">
        {/* ── Image with boxes ── */}
        <div>
          <div className={`card stage ${dragging ? 'drag' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}>
            {!upload ? (
                <div className="stage-empty">{EMPTY_TEXT}</div>
            ) : (
              <>
                <div className="stage-inner">
                  <img src={upload.url} alt={upload.file.name} />
                  {result && (
                    <svg viewBox={`0 0 ${result.width} ${result.height}`} preserveAspectRatio="none">
                      {show.dropped && result.dropped.map((d, i) =>
                        <Rect key={`d${i}`} box={d.box} color={COLORS.dropped} dim />)}
                      {show.accepted && result.accepted.map((d, i) =>
                        <Rect key={`a${i}`} box={d.box} color={COLORS.accepted} />)}
                      {show.accepted && approved.map((d, i) =>
                        <Rect key={`y${i}`} box={d.box} color={COLORS.accepted} dash="5 3" />)}
                      {show.human && forHuman.map((d, i) => !decisions[i] &&
                        <Rect key={`h${i}`} box={d.box} color={COLORS.human} hot={hot === i} />)}
                      {hot !== null && decisions[hot] && (
                        <Rect box={forHuman[hot].box} color={decisions[hot] === 'yes' ? COLORS.accepted : COLORS.dropped} hot />)}
                    </svg>
                  )}
                  {result && show.labels && (
                    <>
                      {show.accepted && result.accepted.map((d, i) =>
                        <Tag key={`ta${i}`} d={d} color={COLORS.accepted} result={result} />)}
                      {show.accepted && approved.map((d, i) =>
                        <Tag key={`ty${i}`} d={d} color={COLORS.accepted} result={result} />)}
                      {show.human && forHuman.map((d, i) => !decisions[i] &&
                        <Tag key={`th${i}`} d={d} color={COLORS.human} result={result} front={hot === i} />)}
                    </>
                  )}
                </div>
                {result ? (
                  <div className="legend">
                    {[['accepted', 'Accepted by the agent'], ['human', 'Needs your decision'], ['dropped', 'Dropped']].map(([key, text]) => (
                      <label key={key}>
                        <input type="checkbox" checked={show[key]}
                          onChange={e => setShow(s => ({ ...s, [key]: e.target.checked }))} />
                        <span className="swatch" style={{ background: COLORS[key] }} />{text}
                      </label>
                    ))}
                    <label>
                      <input type="checkbox" checked={show.labels}
                        onChange={e => setShow(s => ({ ...s, labels: e.target.checked }))} />
                      Show labels
                    </label>
                    <span style={{ marginLeft: 'auto' }}>dashed green = approved by you</span>
                  </div>
                ) : (
                  <div className="stage-note">
                    {loading
                      ? <div className="loading"><Spinner />Running the agent… the first request loads the models (about 10 s).</div>
                      : <div className="empty-note">Press Label to run the agent on this image.</div>}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* ── Side panel ── */}
        <div>
          {result && (
            <>
              <div className="card animate-slide">
                <div className="stats">
                  <Stat n={finalLabels.length} label="labelled" />
                  <Stat n={pending.length} label="to review" />
                  <Stat n={result.dropped.length} label="dropped" />
                  <Stat n={`$${result.claude_cost.toFixed(3)}`} label="Claude" />
                </div>
                <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <button className="btn" onClick={downloadPicture}>Download labelled image (PNG)</button>
                  <button className="btn" onClick={downloadJson}>Download labels (JSON)</button>
                </div>
              </div>

              <div className="card animate-slide">
                <p className="card-title">Your review ({approved.length} approved, {forHuman.length - pending.length - approved.length} rejected)</p>
                {forHuman.length === 0 ? (
                  <div className="empty-note">Nothing needs a human on this image.</div>
                ) : (
                  <div className="review-scroll">
                    {forHuman
                      .map((d, i) => ({ d, i }))
                      .sort((a, b) => b.d.confidence - a.d.confidence)
                      .map(({ d, i }) => (
                        <div key={i}
                          className={`review-row ${hot === i ? 'hot' : ''} ${decisions[i] ? 'done' : ''}`}
                          onMouseEnter={() => setHot(i)} onMouseLeave={() => setHot(null)}>
                          <div>
                            <div className="row-top">
                              <span className={`pill ${d.class}`}>{d.class}</span>
                              <span className="mono">{Math.round(d.confidence * 100)}%</span>
                              <span className="mono">{Math.round(d.box[3] - d.box[1])} px high</span>
                            </div>
                            <div className="meter"><div style={{ width: `${d.confidence * 100}%` }} /></div>
                          </div>
                          <div className="row-actions">
                            <button className={`icon-btn ok ${decisions[i] === 'yes' ? 'on' : ''}`}
                              title="Accept this box" onClick={() => decide(i, 'yes')}>✓</button>
                            <button className={`icon-btn no ${decisions[i] === 'no' ? 'on' : ''}`}
                              title="Reject this box" onClick={() => decide(i, 'no')}>✕</button>
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>

              <div className="card animate-slide">
                <p className="card-title">What the agent did</p>
                {result.trace.map((line, i) => <TraceLine key={i} line={line} />)}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}