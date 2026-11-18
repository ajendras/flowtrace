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
 * Uses HTMX-style direct DOM updates.
 */
async function loadTraces(filter = '') {
    try {
        const res = await fetch(`${API_BASE}/traces?limit=40`);
        if (!res.ok) throw new Error('Failed to fetch traces');
        let traces = await res.json();

        if (filter) {
            const term = filter.toLowerCase();
            traces = traces.filter(t => 
                t.name.toLowerCase().includes(term) || t.id.toLowerCase().includes(term)
            );
        }

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
            const isSelected = selectedTraceIds.includes(t.id);
            const statusClass = t.success ? 'success' : 'error';
            const duration = Math.round(t.duration_ms);
            html += `
                <div class="trace-item ${statusClass}" data-id="${t.id}">
                    <input type="checkbox" 
                           ${isSelected ? 'checked' : ''} 
                           onchange="toggleTraceSelection('${t.id}', this.checked, event)">
                    <div class="trace-info">
                        <div class="trace-name">${t.name}</div>
                        <div class="trace-meta">
                            <span>${t.start.split('T')[0]}</span>
                            <span>${duration}ms</span>
                            <span>${t.total_tokens}t</span>
                        </div>
                    </div>
                </div>`;
        }
        container.innerHTML = html;

        // Attach click handlers (checkbox already handled separately)
        const items = container.querySelectorAll('.trace-item');
        items.forEach(item => {
            const id = item.dataset.id;
            item.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    if (!compareMode) {
                        loadTrace(id);
                    } else {
                        // In compare mode a click can still load single but we ignore for now
                    }
                }
            });
        });
    } catch (err) {
        console.error(err);
    }
}

function toggleTraceSelection(id, checked, e) {
    e.stopImmediatePropagation();
    if (checked) {
        if (selectedTraceIds.length >= 2) {
            // uncheck the box we just clicked
            e.target.checked = false;
            return;
        }
        selectedTraceIds.push(id);
    } else {
        selectedTraceIds = selectedTraceIds.filter(x => x !== id);
    }
    updateCompareButton();
}

function updateCompareButton() {
    const btn = document.getElementById('compare-btn');
    if (btn) {
        btn.disabled = selectedTraceIds.length !== 2;
    }
}

function filterTraces() {
    const input = document.getElementById('search-input');
    loadTraces(input ? input.value : '');
}

async function loadTrace(traceId) {
    if (compareMode) return;
    currentTraceId = traceId;
    try {
        const res = await fetch(`${API_BASE}/traces/${traceId}`);
        if (!res.ok) throw new Error('Trace not found');
        const trace = await res.json();
        
        currentRootStep = trace.root_step;
        document.getElementById('trace-title').textContent = trace.name;
        
        const meta = document.getElementById('trace-meta');
        meta.innerHTML = `
            ${trace.start.split('T')[0]} &nbsp;·&nbsp; 
            ${Math.round(trace.duration * 1000)} ms &nbsp;·&nbsp; 
            ${trace.total_tokens} tokens
        `;
        
        document.getElementById('inspector-trace-id').textContent = traceId.substring(0, 8);
        
        renderGraph('graph-container', currentRootStep, (step) => {
            showInspector(step, trace);
        });
        
        // connect live WS if the trace is recent
        connectLiveWS(traceId);
    } catch (e) {
        console.error(e);
    }
}

function renderGraph(containerId, rootStep, onNodeClick) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "780");
    svg.setAttribute("height", "620");
    svg.setAttribute("viewBox", "0 0 780 620");
    svg.setAttribute("class", "trace-svg");
    
    let yOffset = 40;
    const xBase = 40;
    const xStep = 210;
    
    function getNodeColor(kind) {
        if (kind === 'llm') return '#3b82f6';
        if (kind === 'tool') return '#10b981';
        if (kind === 'error') return '#e30613';
        if (kind === 'trace') return '#6b7280';
        return '#64748b';
    }
    
    function drawStep(step, depth, parentY = null) {
        const x = xBase + depth * xStep;
        const y = yOffset;
        const width = 168;
        const height = 68;
        const color = getNodeColor(step.kind);
        
        // connector line from parent
        if (parentY !== null) {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", (x - 22).toString());
            line.setAttribute("y1", (parentY + height / 2).toString());
            line.setAttribute("x2", x.toString());
            line.setAttribute("y2", (y + height / 2).toString());
            line.setAttribute("stroke", "#ddd");
            line.setAttribute("stroke-width", "2");
            svg.appendChild(line);
        }
        
        // node rectangle
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x.toString());
        rect.setAttribute("y", y.toString());
        rect.setAttribute("width", width.toString());
        rect.setAttribute("height", height.toString());
        rect.setAttribute("rx", "6");
        rect.setAttribute("fill", color);
        rect.setAttribute("stroke", "#111");
        rect.setAttribute("stroke-width", "1.5");
        rect.setAttribute("class", "node");
        rect.onclick = () => onNodeClick(step);
        svg.appendChild(rect);
        
        // name
        const nameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        nameText.setAttribute("x", (x + 12).toString());
        nameText.setAttribute("y", (y + 24).toString());
        nameText.setAttribute("fill", "#ffffff");
        nameText.setAttribute("font-size", "13");
        nameText.setAttribute("font-weight", "600");
        nameText.textContent = step.name.length > 18 
            ? step.name.substring(0, 17) + '…' 
            : step.name;
        svg.appendChild(nameText);
        
        // kind
        const kindText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        kindText.setAttribute("x", (x + width - 34).toString());
        kindText.setAttribute("y", (y + 18).toString());
        kindText.setAttribute("fill", "#ffffff");
        kindText.setAttribute("font-size", "9");
        kindText.setAttribute("font-weight", "700");
        kindText.textContent = step.kind.toUpperCase();
        svg.appendChild(kindText);
        
        // duration
        const durText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        durText.setAttribute("x", (x + 12).toString());
        durText.setAttribute("y", (y + 52).toString());
        durText.setAttribute("fill", "#e0f0ff");
        durText.setAttribute("font-size", "11");
        durText.textContent = `${step.duration_ms} ms`;
        svg.appendChild(durText);
        
        const currentY = y;
        yOffset += 92; // vertical spacing for next sibling
        
        if (step.children && step.children.length > 0) {
            step.children.forEach(child => {
                drawStep(child, depth + 1, currentY);
            });
        }
    }
    
    drawStep(rootStep, 0);
    
    // title label
    const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
    title.setAttribute("x", "20");
    title.setAttribute("y", "22");
    title.setAttribute("fill", "#444");
    title.setAttribute("font-size", "12");
    title.setAttribute("font-weight", "500");
    title.textContent = "EXECUTION TRACE";
    svg.appendChild(title);
    
    container.appendChild(svg);
}

function showInspector(step, traceOrComparison) {
    selectedStepId = step.id;
    const content = document.getElementById('inspector-content');
    let html = `<div class="inspector-section">`;
    html += `<h4>${step.name} <span style="font-size:10px;color:#888">(${step.kind})</span></h4>`;
    html += `<div><strong>Duration:</strong> ${step.duration_ms} ms</div>`;
    
    if (step.content) {
        const c = step.content;
        if (c.prompt) {
            html += `<div class="inspector-section"><h4>Prompt</h4><pre>${escapeHtml(c.prompt)}</pre></div>`;
        }
        if (c.response) {
            html += `<div class="inspector-section"><h4>Response</h4><pre>${escapeHtml(c.response)}</pre></div>`;
        }
        if (c.tokens) {
            html += `<div><strong>Tokens:</strong> ${c.tokens}</div>`;
        }
        if (c.arguments) {
            html += `<div class="inspector-section"><h4>Arguments</h4><pre>${JSON.stringify(c.arguments, null, 2)}</pre></div>`;
        }
        if (c.output) {
            html += `<div class="inspector-section"><h4>Output</h4><pre>${escapeHtml(typeof c.output === 'string' ? c.output : JSON.stringify(c.output))}</pre></div>`;
        }
        if (c.error) {
            html += `<div style="color:#e30613"><strong>Error:</strong> ${escapeHtml(c.error)}</div>`;
        }
    }
    
    html += `</div>`;
    content.innerHTML = html;
    
    // highlight which side if in compare mode
    if (compareMode && currentComparison) {
        document.getElementById('inspector-trace-id').innerHTML = 
            `<span style="background:#3b82f6;color:white;padding:1px 5px;border-radius:2px;font-size:10px">` + 
            (currentInspectorSide === 'left' ? 'LEFT' : 'RIGHT') + `</span>`;
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

function toggleCompareMode() {
    if (selectedTraceIds.length !== 2) return;
    compareMode = true;
    document.getElementById('single-view').classList.add('hidden');
    document.getElementById('compare-view').classList.remove('hidden');
    document.getElementById('compare-btn').classList.add('hidden');
    document.getElementById('exit-compare-btn').classList.remove('hidden');
    loadComparison(selectedTraceIds[0], selectedTraceIds[1]);
}

async function loadComparison(id1, id2) {
    try {
        const res = await fetch(`${API_BASE}/compare/${id1}/${id2}`);
        if (!res.ok) throw new Error('Failed to load comparison');
        currentComparison = await res.json();
        
        // populate headers
        document.getElementById('left-header').textContent = 
            `${currentComparison.trace1.name} (${currentComparison.trace1.id.substring(0,8)})`;
        document.getElementById('right-header').textContent = 
            `${currentComparison.trace2.name} (${currentComparison.trace2.id.substring(0,8)})`;
        
        // render left graph
        renderGraph('left-graph-container', currentComparison.trace1.root_step, (step) => {
            currentInspectorSide = 'left';
            showInspector(step, currentComparison.trace1);
        });
        
        // render right graph
        renderGraph('right-graph-container', currentComparison.trace2.root_step, (step) => {
            currentInspectorSide = 'right';
            showInspector(step, currentComparison.trace2);
        });
        
        // populate stats
        const leftStats = document.getElementById('left-stats');
        const rightStats = document.getElementById('right-stats');
        leftStats.innerHTML = `
            ${Math.round(currentComparison.trace1.duration * 1000)} ms &nbsp; 
            ${currentComparison.trace1.total_tokens} tokens
        `;
        rightStats.innerHTML = `
            ${Math.round(currentComparison.trace2.duration * 1000)} ms &nbsp; 
            ${currentComparison.trace2.total_tokens} tokens
        `;
        
        // differences
        renderDifferences(currentComparison.differences);
        
    } catch (err) {
        console.error('Comparison load failed', err);
    }
}

function renderDifferences(diff) {
    const container = document.getElementById('diff-content');
    let html = '';
    
    html += `
        <div class="diff-item">
            <strong>Duration Δ</strong>
            <div style="font-size:15px;font-weight:600;color:${diff.duration_diff > 0.3 ? '#e30613' : '#10b981'}">
                ${diff.duration_diff}s
            </div>
        </div>
        <div class="diff-item">
            <strong>Tokens Δ</strong>
            <div style="font-size:15px;font-weight:600">
                ${diff.token_diff}
            </div>
        </div>
        <div class="diff-item">
            <strong>Steps Δ</strong>
            <div style="font-size:15px;font-weight:600">
                ${diff.step_count_diff}
            </div>
        </div>
        <div class="diff-item">
            <strong>Success match</strong>
            <div style="font-size:15px;font-weight:600;color:${diff.success_match ? '#10b981' : '#e30613'}">
                ${diff.success_match ? 'YES' : 'NO'}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function exitComparison() {
    compareMode = false;
    selectedTraceIds = [];
    currentComparison = null;
    document.getElementById('single-view').classList.remove('hidden');
    document.getElementById('compare-view').classList.add('hidden');
    document.getElementById('compare-btn').classList.remove('hidden');
    document.getElementById('exit-compare-btn').classList.add('hidden');
    document.getElementById('compare-btn').disabled = true;
    loadTraces();
    if (currentTraceId) loadTrace(currentTraceId);
}

async function clearAllTraces() {
    if (!confirm('Delete ALL traces?')) return;
    try {
        await fetch(`${API_BASE}/traces/clear`, { method: 'POST' }); // note: CLI uses this path indirectly
        loadTraces();
        document.getElementById('inspector-content').innerHTML = `
            <div class="empty-inspector">All traces cleared.<br><br>Run an agent to generate new data.</div>
        `;
    } catch (e) {}
}

function connectLiveWS(traceId) {
    if (ws) ws.close();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    ws = new WebSocket(`${protocol}//${host}/api/ws/traces/${traceId}`);
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'trace_update' && msg.root_step) {
            currentRootStep = msg.root_step;
            if (!compareMode) {
                renderGraph('graph-container', currentRootStep, (step) => {
                    showInspector(step, {root_step: currentRootStep});
                });
            }
        }
    };
    
    ws.onclose = () => { ws = null; };
}

/* Initialize on load */
window.onload = function() {
    loadTraces();
    
    // keyboard shortcut for comparison exit
    document.addEventListener('keydown', function(e) {
        if (e.key === "Escape" && compareMode) {
            exitComparison();
        }
    });
    
    // make sure compare button starts disabled
    updateCompareButton();
};
