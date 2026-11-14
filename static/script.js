/*
 * FlowTrace - Interactive Dashboard Script
 * Vanilla JS + SVG graph rendering for Swiss-minimal UI.
 * Features: trace list, hierarchical SVG node-link diagram on strict grid,
 * node inspector with prompt/response and tool data.
 */

let currentTraceId = null;
let currentRootStep = null;
let selectedStepId = null;

const API_BASE = '/api';

/**
 * Load recent traces and render them in the sidebar list.
 * Uses HTMX-style direct DOM updates.
 */
async function loadTraces() {
    try {
        const res = await fetch(`${API_BASE}/traces?limit=40`);
        if (!res.ok) throw new Error('Failed to fetch traces');
        const traces = await res.json();

        const container = document.getElementById('traces-list');
        if (!container) return;

        if (traces.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#ccc">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2" />
                    </svg>
                    <div class="text-xs tracking-widest">NO TRACES RECORDED YET</div>
                    <div class="text-[10px] mt-6 max-w-[180px] mx-auto">Run an instrumented agent or POST to /api/traces</div>
                </div>
            `;
            return;
        }

        let html = '';
        for (const t of traces) {
            const active = t.id === currentTraceId ? 'active' : '';
            const date = new Date(t.start);
            const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const status = t.success
                ? `<span class="status-ok">OK</span>`
                : `<span class="status-err">ERR</span>`;

            html += `
                <div class="trace-item ${active}" onclick="selectTrace('${t.id}')">
                    <div class="flex items-baseline justify-between">
                        <div class="font-semibold tracking-tight">${t.name}</div>
                        ${status}
                    </div>
                    <div class="trace-meta flex justify-between mt-2">
                        <div>${t.id.substring(0, 8)}</div>
                        <div>${time}</div>
                    </div>
                    <div class="text-xs text-gray-500 mt-3">
                        ${Math.round(t.duration_ms)}&nbsp;ms &nbsp;•&nbsp; ${t.total_tokens} tokens
                    </div>
                </div>
            `;
        }
        container.innerHTML = html;
    } catch (err) {
        console.error('loadTraces failed', err);
    }
}

/**
 * Select a trace by ID, fetch its graph data and render the SVG.
 */
async function selectTrace(traceId) {
    currentTraceId = traceId;
    await loadTraces(); // refresh active state

    try {
        const res = await fetch(`${API_BASE}/traces/${traceId}/graph`);
        if (!res.ok) {
            if (res.status === 404) throw new Error('Trace not found');
            throw new Error('Failed to load graph');
        }
        const rootStep = await res.json();
        currentRootStep = rootStep;
        renderGraph(rootStep);

        // Reset inspector
        const inspector = document.getElementById('inspector-content');
        if (inspector) {
            inspector.innerHTML = `
                <div class="flex flex-col justify-center items-center h-80 text-center opacity-40">
                    <div class="w-9 h-9 border border-current rounded mb-6"></div>
                    <div class="text-sm font-medium">NODE INSPECTOR</div>
                    <div class="text-xs mt-8 max-w-[160px]">Click any node in the graph to view raw prompts, responses, tool outputs and metadata</div>
                </div>
            `;
        }
    } catch (e) {
        console.error(e);
        const container = document.getElementById('graph-container');
        if (container) container.innerHTML = `<div class="text-red-600 p-8">Could not load trace graph.</div>`;
    }
}

/**
 * Render interactive SVG node-link diagram using strict hierarchical grid layout.
 * Nodes are positioned with even spacing per subtree. Supports click to inspect.
 */
function renderGraph(root) {
    const container = document.getElementById('graph-container');
    if (!container) return;
    container.innerHTML = '';

    if (!root) {
        container.innerHTML = `<div class="p-12 text-red-500">No graph data available</div>`;
        return;
    }

    const SVG_W = 920;
    const SVG_H = 640;
    const LEVEL_GAP = 118;
    const NODE_W = 152;
    const NODE_H = 64;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', SVG_W);
    svg.setAttribute('height', SVG_H);
    svg.setAttribute('viewBox', `0 0 ${SVG_W} ${SVG_H}`);
    svg.setAttribute('class', 'shadow-none');

    const positions = new Map();

    // Recursive layout: center children under parent with even horizontal spread
    function layout(step, depth = 0, centerX = SVG_W / 2, availWidth = SVG_W * 0.82) {
        const y = 68 + depth * LEVEL_GAP;
        positions.set(step.id, { x: centerX, y, step });

        const kids = step.children || [];
        if (kids.length === 0) return;

        const childWidth = availWidth / Math.max(kids.length, 1);
        const startX = centerX - (availWidth / 2) + (childWidth / 2);

        for (let i = 0; i < kids.length; i++) {
            const childX = startX + i * childWidth;
            layout(kids[i], depth + 1, childX, childWidth * 0.92);
        }
    }

    layout(root);

    // Draw connecting edges first (so they sit behind nodes)
    function drawEdges(step) {
        const p = positions.get(step.id);
        if (!p) return;

        const children = step.children || [];
        for (const child of children) {
            const cp = positions.get(child.id);
            if (!cp) continue;

            const x1 = p.x;
            const y1 = p.y + NODE_H / 2;
            const x2 = cp.x;
            const y2 = cp.y - NODE_H / 2 - 6;

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', String(x1));
            line.setAttribute('y1', String(y1));
            line.setAttribute('x2', String(x2));
            line.setAttribute('y2', String(y2));
            line.setAttribute('class', 'edge');
            svg.appendChild(line);

            drawEdges(child);
        }
    }
    drawEdges(root);

    // Draw nodes with labels
    function drawNodes(step) {
        const pos = positions.get(step.id);
        if (!pos) return;

        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('data-id', step.id);
        g.style.cursor = 'pointer';

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const rx = pos.x - NODE_W / 2;
        const ry = pos.y - NODE_H / 2;
        rect.setAttribute('x', String(rx));
        rect.setAttribute('y', String(ry));
        rect.setAttribute('width', String(NODE_W));
        rect.setAttribute('height', String(NODE_H));
        rect.setAttribute('rx', '3');
        rect.setAttribute('class', `node node-${step.kind || 'step'}`);
        g.appendChild(rect);

        // Primary label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', String(pos.x));
        label.setAttribute('y', String(pos.y - 6));
        label.setAttribute('class', 'text-label');
        let name = step.name || 'Step';
        if (name.length > 19) name = name.substring(0, 16) + '…';
        label.textContent = name;
        g.appendChild(label);

        // Kind indicator
        const kindEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        kindEl.setAttribute('x', String(pos.x));
        kindEl.setAttribute('y', String(pos.y + 19));
        kindEl.setAttribute('class', 'text-kind');
        kindEl.textContent = (step.kind || 'STEP').toUpperCase();
        g.appendChild(kindEl);

        // Click handler
        g.addEventListener('click', (ev) => {
            ev.stopImmediatePropagation();
            onNodeClick(step.id);
        });

        svg.appendChild(g);

        // Recurse into children
        for (const child of (step.children || [])) {
            drawNodes(child);
        }
    }

    drawNodes(root);

    // Trace title at top of diagram
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    title.setAttribute('x', String(SVG_W / 2));
    title.setAttribute('y', '34');
    title.setAttribute('text-anchor', 'middle');
    title.setAttribute('fill', '#111111');
    title.setAttribute('font-size', '17');
    title.setAttribute('font-weight', '600');
    title.setAttribute('letter-spacing', '-0.02em');
    title.textContent = root.name || 'Workflow Trace';
    svg.appendChild(title);

    container.appendChild(svg);
}

/**
 * Find a step by ID in the tree and display rich metadata in the inspector.
 */
function onNodeClick(stepId) {
    if (!currentRootStep) return;
    selectedStepId = stepId;

    function find(node) {
        if (node.id === stepId) return node;
        for (const c of (node.children || [])) {
            const found = find(c);
            if (found) return found;
        }
        return null;
    }

    const step = find(currentRootStep);
    if (!step) return;

    const inspector = document.getElementById('inspector-content');
    if (!inspector) return;

    const content = step.content || {};
    let html = `
        <div class="mb-8">
            <span onclick="window.loadTraces()" class="inline-flex items-center gap-2 px-5 py-1.5 bg-black text-white text-xs font-medium cursor-pointer hover:bg-zinc-800">
                ← BACK TO LIST
            </span>
        </div>

        <div class="flex items-center gap-x-3 mb-4">
            <span class="px-4 py-1 text-[10px] font-mono tracking-widest border border-current">${(step.kind || 'step').toUpperCase()}</span>
            <span class="font-mono text-xs text-gray-400">${step.id.substring(0, 8)}</span>
        </div>

        <h2 class="text-[27px] leading-none font-semibold tracking-tighter mb-7">${step.name}</h2>
    `;

    if (step.duration_ms) {
        html += `
            <div class="metadata-row">
                <div class="metadata-label">DURATION</div>
                <div class="font-medium">${step.duration_ms} ms</div>
            </div>
        `;
    }

    if (step.kind === 'llm') {
        html += `
            <div class="mt-10">
                <div class="panel-title">PROMPT</div>
                <pre>${escapeHtml(content.prompt || '(no prompt)')}</pre>
            </div>

            <div class="mt-10">
                <div class="panel-title">RESPONSE</div>
                <pre>${escapeHtml(content.response || '(no response)')}</pre>
            </div>

            <div class="hairline"></div>

            <div class="grid grid-cols-2 gap-x-8 text-sm">
                <div>
                    <span class="metadata-label">TOKENS</span>
                    <div class="font-semibold text-xl mt-1">${content.tokens || 0}</div>
                </div>
                ${content.latency_ms ? `
                <div>
                    <span class="metadata-label">LATENCY</span>
                    <div class="font-semibold text-xl mt-1">${content.latency_ms} ms</div>
                </div>` : ''}
            </div>
        `;
    } else if (step.kind === 'tool') {
        html += `
            <div class="mt-10">
                <div class="panel-title">ARGUMENTS</div>
                <pre>${escapeHtml(JSON.stringify(content.arguments || content.args || {}, null, 2))}</pre>
            </div>
            <div class="mt-10">
                <div class="panel-title">OUTPUT</div>
                <pre>${escapeHtml(JSON.stringify(content.output || content.result || {}, null, 2))}</pre>
            </div>
        `;
    } else if (Object.keys(content).length > 0) {
        html += `
            <div class="mt-10">
                <div class="panel-title">METADATA</div>
                <pre>${escapeHtml(JSON.stringify(content, null, 2))}</pre>
            </div>
        `;
    } else {
        html += `<div class="mt-16 text-xs text-gray-400 italic">No additional data attached to this step.</div>`;
    }

    inspector.innerHTML = html;
}

function escapeHtml(text) {
    if (typeof text !== 'string') return String(text);
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/** Initialize everything on page load */
function init() {
    loadTraces();

    // Periodic refresh when no trace is selected
    setInterval(() => {
        if (!currentTraceId) loadTraces();
    }, 7000);

    // Keyboard support
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            loadTraces();
        }
    });

    console.log('%cFlowTrace UI ready — Swiss minimal style', 'color:#e30613;font-family:monospace;font-size:10px');
}

// Public API for inline event handlers and HTMX patterns
window.loadTraces = loadTraces;
window.selectTrace = selectTrace;
window.renderGraph = renderGraph;
window.onNodeClick = onNodeClick;

window.onload = init;
