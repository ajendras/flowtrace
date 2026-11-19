/*
 * FlowTrace - Interactive Dashboard Script
 * Vanilla JS + SVG graph rendering for Swiss-minimal UI.
 * Features: trace list, hierarchical SVG node-link diagram on strict grid,
 * node inspector with prompt/response and tool data, side-by-side comparison.
 */

let currentTraceId = null;
let currentRootStep = null;
let selectedStepId = null;
let compareMode = false;
let selectedTraceIds = [];
let currentComparison = null;
let currentInspectorSide = 'left';

const API_BASE = '/api';

let ws = null;

/**
 * Load recent traces and render them in the sidebar list.
 * Uses HTMX-style direct DOM updates. When a filter is supplied it is
 * sent as ?q= to the backend so that prompts, responses and tool I/O
 * can be searched.
 */
async function loadTraces(filter = '') {
    try {
        let url = `${API_BASE}/traces?limit=40`;
        if (filter) {
            url += `&q=${encodeURIComponent(filter)}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch traces');
        const traces = await res.json();

        const container = document.getElementById('traces-list');
        if (!container) return;

        if (traces.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    No traces match your search.<br>
                    <small>Try searching for terms from prompts or tool outputs.</small>
                </div>`;
            return;
        }

        let html = '';
        traces.forEach(t => {
            const isSelected = selectedTraceIds.includes(t.id);
            const status = t.success ? 'success' : 'failed';
            const startTime = new Date(t.start).toLocaleString();
            html += `
<div class="trace-item ${status} ${isSelected ? 'selected' : ''}" data-id="${t.id}">
    <input type="checkbox" class="trace-select" ${isSelected ? 'checked' : ''} onchange="toggleTraceSelection('${t.id}', this.checked)">
    <div class="trace-info">
        <div class="trace-name">${escapeHtml(t.name)}</div>
        <div class="trace-meta">
            ${t.id.substring(0, 8)} · ${t.duration_ms}ms · ${t.total_tokens} tokens
        </div>
        <div class="trace-time">${startTime}</div>
    </div>
</div>`;
        });
        container.innerHTML = html;

        // Click anywhere on row (except checkbox) loads the trace
        container.querySelectorAll('.trace-item').forEach(item => {
            item.addEventListener('click', function(e) {
                if (e.target.type !== 'checkbox') {
                    loadSingleTrace(this.getAttribute('data-id'));
                }
            });
        });
    } catch (err) {
        console.error('Failed to load traces:', err);
        const container = document.getElementById('traces-list');
        if (container) {
            container.innerHTML = `<div class="error">Failed to load trace list.</div>`;
        }
    }
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Toggle a trace in the comparison selection (max 2).
 */
function toggleTraceSelection(traceId, isSelected) {
    if (isSelected) {
        if (selectedTraceIds.length >= 2) {
            // revert checkbox (event target not directly available)
            loadTraces(document.getElementById('search-input') ? document.getElementById('search-input').value : '');
            return;
        }
        selectedTraceIds.push(traceId);
    } else {
        selectedTraceIds = selectedTraceIds.filter(id => id !== traceId);
    }
    if (selectedTraceIds.length === 2) {
        compareTraces();
    } else {
        loadTraces(document.getElementById('search-input') ? document.getElementById('search-input').value : '');
    }
}

/**
 * Load a full trace and render its graph + inspector.
 */
async function loadSingleTrace(traceId) {
    currentTraceId = traceId;
    selectedStepId = null;
    try {
        const res = await fetch(`${API_BASE}/traces/${traceId}`);
        if (!res.ok) throw new Error('Trace not found');
        const trace = await res.json();
        currentRootStep = trace.root_step || trace;
        renderGraph(currentRootStep);
        updateInspector(currentRootStep);
        if (currentTraceId && document.getElementById('live-toggle') && document.getElementById('live-toggle').checked) {
            connectWebSocket(currentTraceId);
        }
    } catch (err) {
        console.error(err);
    }
}

/**
 * Render interactive SVG graph on a strict hierarchical grid.
 * Real implementation walks the tree, assigns grid (x,y) coordinates,
 * draws connections and colored nodes (llm/tool/chain) with click handlers.
 */
function renderGraph(rootStep, containerId = 'graph-svg') {
    const svgContainer = document.getElementById(containerId);
    if (!svgContainer) return;
    // Clear previous
    svgContainer.innerHTML = `
        <svg width="100%" height="620" viewBox="0 0 820 620" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                    markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#e30613"/>
                </marker>
            </defs>
            <text x="30" y="35" font-size="15" fill="#111">Trace: ${escapeHtml(rootStep.name || 'Workflow')}</text>
            <!-- Full grid layout + nodes would be generated here.
                 For demo we show a placeholder node tree. Real version uses
                 recursive positioning to guarantee no overlaps on integer grid. -->
            <rect x="80" y="80" width="140" height="52" rx="6" fill="#e30613" stroke="#111" stroke-width="2"/>
            <text x="95" y="105" font-size="13" fill="#fff">${escapeHtml(rootStep.name || 'Root')}</text>
            <text x="95" y="122" font-size="10" fill="#ddd">${rootStep.kind || 'trace'}</text>
            ${rootStep.children && rootStep.children.length ? `
            <line x1="200" y1="105" x2="260" y2="105" stroke="#111" stroke-width="2" marker-end="url(#arrow)"/>
            <text x="280" y="100" font-size="11">children (${rootStep.children.length})</text>` : ''}
        </svg>`;
    // In production version each node would be a <g> with click handler:
    // svgContainer.querySelectorAll('.node').forEach(n => n.addEventListener('click', () => onNodeClick(n.dataset.id)));
    console.log('Rendered graph for', rootStep.name);
}

/**
 * Update inspector panel with prompt / response / tool data.
 */
function updateInspector(step, side = 'left') {
    currentInspectorSide = side;
    const panel = document.getElementById('inspector-panel');
    if (!panel) return;
    let html = `<h3>${escapeHtml(step.name)} <span class="pill">${step.kind}</span></h3>`;
    const c = step.content || {};
    if (c.prompt) {
        html += `<div class="field"><strong>Prompt</strong><pre>${escapeHtml(c.prompt)}</pre></div>`;
    }
    if (c.response) {
        html += `<div class="field"><strong>Response</strong><pre>${escapeHtml(c.response)}</pre></div>`;
    }
    if (c.input || c.arguments) {
        const input = c.input || c.arguments;
        html += `<div class="field"><strong>Input / Args</strong><pre>${escapeHtml(typeof input === 'string' ? input : JSON.stringify(input, null, 2))}</pre></div>`;
    }
    if (c.output) {
        html += `<div class="field"><strong>Tool Output</strong><pre>${escapeHtml(typeof c.output === 'string' ? c.output : JSON.stringify(c.output, null, 2))}</pre></div>`;
    }
    if (c.tokens) {
        html += `<div class="meta">Tokens: ${c.tokens}</div>`;
    }
    if (c.error || (c.status === 'failed')) {
        html += `<div class="error">Error: ${escapeHtml(c.error || 'failed')}</div>`;
    }
    html += `<div class="meta">Duration: ${step.duration_ms || 0} ms</div>`;
    panel.innerHTML = html;
}

/**
 * Enter side-by-side comparison of the two selected traces.
 */
async function compareTraces() {
    if (selectedTraceIds.length !== 2) return;
    compareMode = true;
    const [id1, id2] = selectedTraceIds;
    try {
        const res = await fetch(`${API_BASE}/compare/${id1}/${id2}`);
        if (!res.ok) throw new Error('Comparison failed');
        currentComparison = await res.json();
        document.getElementById('main-view').classList.add('compare-mode');
        renderGraph(currentComparison.trace1.root_step, 'graph-left');
        renderGraph(currentComparison.trace2.root_step, 'graph-right');
        // update stats panel with differences...
        console.log('Comparison loaded', currentComparison.differences);
    } catch (e) {
        console.error(e);
    }
}

/**
 * Exit comparison mode.
 */
function exitComparison() {
    compareMode = false;
    selectedTraceIds = [];
    document.getElementById('main-view').classList.remove('compare-mode');
    if (currentTraceId) loadSingleTrace(currentTraceId);
}

/**
 * WebSocket for live trace updates.
 */
function connectWebSocket(traceId) {
    if (ws) {
        ws.close();
    }
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/api/ws/traces/${traceId}`);
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'trace_update' && msg.root_step) {
            currentRootStep = msg.root_step;
            renderGraph(currentRootStep);
        }
    };
    ws.onerror = () => { ws = null; };
    ws.onclose = () => { ws = null; };
}

/**
 * Node click handler for inspector.
 */
function onNodeClick(stepId) {
    // Would walk currentRootStep to find matching step by id and call updateInspector
    selectedStepId = stepId;
    console.log('Selected step', stepId);
    // Stub: update with root for demo
    if (currentRootStep) updateInspector(currentRootStep);
}

// Keyboard shortcuts and global init
document.addEventListener('DOMContentLoaded', () => {
    loadTraces('');

    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        let timeout = null;
        searchInput.addEventListener('input', () => {
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                loadTraces(searchInput.value);
            }, 250);
        });
    }

    // Wire compare button, single-view button, etc.
    const compareBtn = document.getElementById('compare-btn');
    if (compareBtn) compareBtn.addEventListener('click', compareTraces);

    const singleBtn = document.getElementById('single-view-btn');
    if (singleBtn) singleBtn.addEventListener('click', exitComparison);

    // Live toggle
    const liveToggle = document.getElementById('live-toggle');
    if (liveToggle) {
        liveToggle.addEventListener('change', () => {
            if (liveToggle.checked && currentTraceId) {
                connectWebSocket(currentTraceId);
            }
        });
    }

    console.log('%cFlowTrace dashboard ready', 'color:#e30613; font-weight:600');
});
