const statusEl = document.getElementById("status");
const statusDot = document.getElementById("status-dot");
const sessionEl = document.getElementById("session-id");
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message");
const newSessionBtn = document.getElementById("new-session");
const toolsToggle = document.getElementById("toggle-tools");
const fastModeToggle = document.getElementById("fast-mode");
const sendButton = formEl.querySelector("button");

// RAG Info Card elements
const ragInfoCard = document.getElementById("rag-info-card");
const ragInfoToggle = document.getElementById("rag-info-toggle");
const ragInfoSummary = document.getElementById("rag-info-summary");
const ragInfoSources = document.getElementById("rag-info-sources");

let socket;
let sessionId = sessionStorage.getItem("profectus_session_id");
let pendingBubble = null;
let pendingTextEl = null;
let streamingBubble = null;
let streamingBody = null;
let streamingParser = null;
let streamingFullText = "";
let streamingEmittedText = "";
let streamingQueuedTextLength = 0;
let streamingQueue = [];
let streamingTimer = null;
let streamingFinalizeText = null;
let streamingFinalizeMeta = null;
let streamingFinalizeDeadline = 0;
const STREAM_TOKEN_DELAY_MS = 24;
const STREAM_TOKENS_PER_TICK = 3;
const STREAM_FINALIZE_TIMEOUT_MS = 2500;
let reconnectTimer = null;
let reconnectDelayMs = 1000;

const TOOL_TOGGLE_KEY = "profectus_show_tools";
const FAST_MODE_KEY = "profectus_fast_mode";
const savedToggle = sessionStorage.getItem(TOOL_TOGGLE_KEY);
if (savedToggle !== null) {
  toolsToggle.checked = savedToggle === "true";
}
if (fastModeToggle) {
  const savedFastMode = sessionStorage.getItem(FAST_MODE_KEY);
  if (savedFastMode !== null) {
    fastModeToggle.checked = savedFastMode === "true";
  }
}
// Visual indicator for debug mode
if (toolsToggle.checked) {
  document.body.classList.add("debug-mode");
}

// Simple loading state
let loadingInterval = null;

// Fetch and display RAG index stats
async function loadRagInfo() {
  if (!ragInfoSummary || !ragInfoSources) return;

  try {
    const response = await fetch("/index-stats");
    if (!response.ok) throw new Error("Failed to fetch stats");
    const stats = await response.json();

    const total = stats.total_entries || 0;
    const sources = stats.sources || {};

    // Summary text
    const summary = total > 0
      ? `${total.toLocaleString()} content chunks indexed`
      : "No content indexed";
    ragInfoSummary.textContent = summary;

    // Build source tags
    const sourceIcons = {
      youtube: { icon: "📹", label: "videos" },
      docs: { icon: "📄", label: "docs" },
      blog: { icon: "📝", label: "posts" },
      legal: { icon: "⚖️", label: "legal" }
    };

    ragInfoSources.innerHTML = "";
    for (const [key, { icon, label }] of Object.entries(sourceIcons)) {
      const count = sources[key];
      if (count > 0) {
        const tag = document.createElement("span");
        tag.className = "rag-source-tag";
        tag.innerHTML = `${icon} <strong>${count}</strong> ${label}`;
        ragInfoSources.appendChild(tag);
      }
    }
  } catch (err) {
    ragInfoSummary.textContent = "Unable to load info";
    ragInfoSources.innerHTML = "";
  }
}

// Toggle RAG info card
function toggleRagInfo() {
  ragInfoCard.classList.toggle("expanded");
}

function renderMarkdown(text) {
  const raw = text || "";
  if (window.marked && typeof window.marked.parse === "function") {
    return sanitizeHtml(window.marked.parse(raw, { breaks: true }));
  }
  return sanitizeHtml(escapeHtml(raw).replace(/\n/g, "<br>"));
}

function splitStreamingBlocks(text) {
  const raw = text || "";
  const blocks = raw.split(/\n{2,}/);
  if (blocks.length <= 1) {
    return { stable: "", pending: raw };
  }
  return {
    stable: blocks.slice(0, -1).join("\n\n"),
    pending: blocks[blocks.length - 1],
  };
}

function renderStreamingMarkdown(text) {
  const { stable, pending } = splitStreamingBlocks(text);
  let html = "";
  if (stable) {
    html += renderMarkdown(stable);
  }
  if (pending) {
    const pendingHtml = escapeHtml(pending).replace(/\n/g, "<br>");
    const pendingBlock = `<p>${pendingHtml}</p>`;
    html += pendingBlock;
  }
  return html || "";
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function sanitizeHtml(html) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, "text/html");
  const allowed = new Set([
    "A",
    "B",
    "BLOCKQUOTE",
    "BR",
    "CODE",
    "EM",
    "I",
    "LI",
    "OL",
    "P",
    "PRE",
    "STRONG",
    "UL",
  ]);
  const walker = document.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
  const toRemove = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!allowed.has(node.tagName)) {
      toRemove.push(node);
      continue;
    }
    [...node.attributes].forEach((attr) => {
      const name = attr.name.toLowerCase();
      if (name.startsWith("on")) {
        node.removeAttribute(attr.name);
      }
      if (node.tagName === "A" && name === "href") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener");
      }
    });
  }
  toRemove.forEach((node) => node.replaceWith(...node.childNodes));
  return doc.body.innerHTML;
}

window.profectusSanitizeHtml = sanitizeHtml;

function setStatus(label, state) {
  statusEl.textContent = label;
  statusDot.classList.remove("ok", "warn");
  if (state === "ok") {
    statusDot.classList.add("ok");
  } else if (state === "warn") {
    statusDot.classList.add("warn");
  }
}

function generateUUID() {
  // Simple UUID v4 generator
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function setSessionId(id) {
  sessionId = id || null;
  if (sessionId) {
    sessionStorage.setItem("profectus_session_id", sessionId);
    // Show only first segment to user (before first dash), with ellipsis if longer
    const firstSegment = sessionId.split('-')[0];
    sessionEl.textContent = firstSegment + (sessionId.includes('-') ? '...' : '');
  } else {
    sessionStorage.removeItem("profectus_session_id");
    sessionEl.textContent = "new";
  }
}

function formatDurationMs(msValue) {
  const ms = Number(msValue);
  if (!Number.isFinite(ms)) return "";
  const secs = ms / 1000;
  return Number.isInteger(secs) ? `${secs}s` : `${secs.toFixed(1)}s`;
}

function formatStepLabel(name) {
  if (!name) return "";
  return name
    .replace(/^tool_adk_/, "")
    .replace(/^tool_/, "")
    .replace(/_total$/, "")
    .replace(/_/g, " ");
}

function addMessage(role, text, meta = {}) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;

  const body = document.createElement("div");
  body.className = "bubble-body";
  if (role === "assistant") {
    body.innerHTML = renderMarkdown(text);
  } else {
    body.textContent = text;
  }
  bubble.appendChild(body);

  if (role === "assistant") {
    appendAssistantMeta(bubble, meta);
  }

  chatEl.appendChild(bubble);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function appendAssistantMeta(bubble, meta) {
  if (!bubble) return;
  const existingMeta = bubble.querySelector(".meta");
  if (existingMeta) existingMeta.remove();
  const existingPanel = bubble.querySelector(".tool-info");
  if (existingPanel) existingPanel.remove();

  const metaItems = [];
  if (meta.requestId) metaItems.push(`req ${meta.requestId.slice(0, 8)}`);
  if (meta.verdict) metaItems.push(meta.verdict.toLowerCase());
  if (meta.escalationId) metaItems.push("escalation logged");
  if (metaItems.length) {
    const metaLine = document.createElement("div");
    metaLine.className = "meta";
    metaLine.textContent = metaItems.join(" • ");
    bubble.appendChild(metaLine);
  }

  if (meta.requestId || meta.verdict || meta.escalationId || meta.timing || meta.trace) {
    const details = document.createElement("details");
    details.className = "tool-info";
    details.open = toolsToggle.checked;

    const summary = document.createElement("summary");
    summary.textContent = "debug info";
    details.appendChild(summary);

    const content = document.createElement("div");
    content.className = "tool-info-content";

    if (meta.requestId) {
      content.innerHTML += `<div><span>Request ID</span><code>${meta.requestId}</code></div>`;
    }
    if (meta.verdict) {
      let verdictText = meta.verdict;
      if (typeof meta.confidence === "number") {
        const pct = Math.round(meta.confidence * 100);
        verdictText = `${verdictText} (${pct}%)`;
      }
      content.innerHTML += `<div><span>Verdict</span><code>${verdictText}</code></div>`;
    }
    if (meta.classification && (meta.classification.type || meta.classification.reason)) {
      const typeLabel = meta.classification.type
        ? String(meta.classification.type).toUpperCase()
        : "";
      const reasonLabel = meta.classification.reason
        ? String(meta.classification.reason).replace(/_/g, " ")
        : "";
      const label = reasonLabel ? (typeLabel ? `${typeLabel} - ${reasonLabel}` : reasonLabel) : typeLabel;
      content.innerHTML += `<div><span>Classification</span><code>${label}</code></div>`;
    }
    if (meta.escalationId) {
      content.innerHTML += `<div><span>Escalation ID</span><code>${meta.escalationId}</code></div>`;
    }
    if (meta.timing) {
      const timingDiv = document.createElement("div");
      timingDiv.className = "timing-section";
      const totalText = formatDurationMs(meta.timing.total_ms);
      if (totalText) {
        timingDiv.innerHTML = `<div class="timing-total">Total ${totalText}</div>`;
      }

      const stepEntries = meta.timing.steps && typeof meta.timing.steps === "object"
        ? Object.entries(meta.timing.steps)
        : [];
      if (stepEntries.length) {
        stepEntries.sort((a, b) => b[1] - a[1]);
        const topEntries = stepEntries.slice(0, 3);
        const remaining = stepEntries.length - topEntries.length;
        const topText = topEntries
          .map(([name, ms]) => `${formatStepLabel(name)} ${formatDurationMs(ms)}`.trim())
          .filter(Boolean)
          .join(" | ");
        if (topText) {
          const stepsLine = document.createElement("div");
          stepsLine.className = "timing-steps";
          stepsLine.textContent = remaining > 0 ? `${topText} (+${remaining} more)` : topText;
          timingDiv.appendChild(stepsLine);
        }
      }

      if (meta.timing.breakdown && stepEntries.length > 3) {
        const more = document.createElement("details");
        more.className = "timing-more";
        const summary = document.createElement("summary");
        summary.textContent = "All timing steps";
        more.appendChild(summary);
        const breakdown = document.createElement("div");
        breakdown.className = "timing-breakdown";
        const converted = meta.timing.breakdown.replace(/(\d+)ms/g, (_, p1) => {
          const secs = parseInt(p1) / 1000;
          return Number.isInteger(secs) ? `${secs}s` : `${secs.toFixed(1)}s`;
        });
        breakdown.textContent = converted;
        more.appendChild(breakdown);
        timingDiv.appendChild(more);
      } else if (meta.timing.breakdown && stepEntries.length > 0) {
        const breakdown = document.createElement("div");
        breakdown.className = "timing-breakdown";
        const converted = meta.timing.breakdown.replace(/(\d+)ms/g, (_, p1) => {
          const secs = parseInt(p1) / 1000;
          return Number.isInteger(secs) ? `${secs}s` : `${secs.toFixed(1)}s`;
        });
        breakdown.textContent = converted;
        timingDiv.appendChild(breakdown);
      }
      content.appendChild(timingDiv);
    }
    if (meta.trace) {
      if (meta.trace.steps && meta.trace.steps.length) {
        const toolList = document.createElement("ul");
        toolList.className = "tool-steps";
        meta.trace.steps.forEach((step) => {
          const li = document.createElement("li");
          li.textContent = step;
          toolList.appendChild(li);
        });
        content.appendChild(toolList);
      }
    }

    if (!meta.trace && meta.timing) {
      const noTraceDiv = document.createElement("div");
      noTraceDiv.innerHTML = `<span>Trace</span><code>No steps (trivial query?)</code>`;
      content.appendChild(noTraceDiv);
    }

    if (content.children.length) {
      details.appendChild(content);
    }

    bubble.dataset.debugHtml = details.outerHTML;

    if (toolsToggle.checked) {
      bubble.appendChild(details);
    }
  }
}

function addPending() {
  clearPending();
  pendingBubble = document.createElement("div");
  pendingBubble.className = "bubble assistant pending";

  pendingTextEl = document.createElement("div");
  pendingTextEl.className = "loading-text";
  pendingTextEl.textContent = toolsToggle.checked ? "Starting request..." : "Thinking...";
  pendingBubble.appendChild(pendingTextEl);

  const dots = document.createElement("div");
  dots.className = "typing";
  dots.innerHTML = "<span></span><span></span><span></span>";
  pendingBubble.appendChild(dots);

  chatEl.appendChild(pendingBubble);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearPending() {
  if (pendingBubble) {
    pendingBubble.remove();
    pendingBubble = null;
    pendingTextEl = null;
  }
}

function ensureDebugPlaceholder(bubble) {
  if (!bubble || bubble.querySelector(".tool-info")) {
    return;
  }
  const details = document.createElement("details");
  details.className = "tool-info tool-info-loading";
  details.open = false;
  const summary = document.createElement("summary");
  summary.textContent = "debug info";
  details.appendChild(summary);
  bubble.appendChild(details);
}

function createStreamingParser(target) {
  if (window.Semidown) {
    return new window.Semidown(target);
  }
  return null;
}

function tokenizeForStreaming(text) {
  if (!text) return [];
  const tokens = text.match(/\S+\s*/g);
  return tokens ? tokens : [text];
}

function stopStreamingTick() {
  if (streamingTimer) {
    clearInterval(streamingTimer);
    streamingTimer = null;
  }
}

function scheduleStreamingTick() {
  if (streamingTimer) return;
  streamingTimer = setInterval(() => {
    if (!streamingParser) {
      stopStreamingTick();
      return;
    }
    let count = 0;
    while (count < STREAM_TOKENS_PER_TICK && streamingQueue.length) {
      const token = streamingQueue.shift();
      if (token) {
        streamingParser.write(token);
        streamingEmittedText += token;
        streamingQueuedTextLength = Math.max(0, streamingQueuedTextLength - token.length);
        count += 1;
      }
    }
    if (!streamingQueue.length) {
      stopStreamingTick();
      if (streamingFinalizeText) {
        const complete = streamingEmittedText.length >= streamingFinalizeText.length;
        const deadlineHit = streamingFinalizeDeadline && Date.now() > streamingFinalizeDeadline;
        if (complete || deadlineHit) {
          finalizeStreamingNow();
        }
      }
    }
  }, STREAM_TOKEN_DELAY_MS);
}

function resetStreamingParser() {
  if (streamingParser) {
    streamingParser.destroy();
  }
  streamingParser = createStreamingParser(streamingBody);
  streamingFullText = "";
  streamingEmittedText = "";
  streamingQueuedTextLength = 0;
  streamingQueue = [];
  streamingFinalizeText = null;
  streamingFinalizeMeta = null;
  streamingFinalizeDeadline = 0;
  stopStreamingTick();
}

function writeStreamingText(text) {
  if (!streamingBody) return;
  if (!streamingParser) {
    streamingBody.innerHTML = renderStreamingMarkdown(text);
    return;
  }
  if (streamingFullText && !text.startsWith(streamingFullText)) {
    resetStreamingParser();
  }
  streamingFullText = text;
  const pendingLength = streamingEmittedText.length + streamingQueuedTextLength;
  if (pendingLength > text.length) {
    resetStreamingParser();
  }
  const delta = text.slice(streamingEmittedText.length + streamingQueuedTextLength);
  if (delta) {
    const tokens = tokenizeForStreaming(delta);
    tokens.forEach((token) => {
      streamingQueue.push(token);
      streamingQueuedTextLength += token.length;
    });
    scheduleStreamingTick();
  }
}

function addStreaming(text) {
  if (streamingBubble) {
    updateStreaming(text);
    return;
  }
  streamingBubble = document.createElement("div");
  streamingBubble.className = "bubble assistant streaming";

  streamingBody = document.createElement("div");
  streamingBody.className = "bubble-body";
  streamingBody.innerHTML = "";
  streamingBubble.appendChild(streamingBody);

  chatEl.appendChild(streamingBubble);
  chatEl.scrollTop = chatEl.scrollHeight;

  if (toolsToggle.checked) {
    ensureDebugPlaceholder(streamingBubble);
  }

  streamingParser = createStreamingParser(streamingBody);
  streamingFullText = "";
  streamingEmittedText = "";
  streamingQueuedTextLength = 0;
  streamingQueue = [];
  streamingFinalizeText = null;
  streamingFinalizeMeta = null;
  streamingFinalizeDeadline = 0;
  if (text) {
    writeStreamingText(text);
  }
}

function updateStreaming(text) {
  if (!streamingBubble || !streamingBody) {
    addStreaming(text);
    return;
  }
  writeStreamingText(text || "");
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearStreaming() {
  if (streamingBubble) {
    streamingBubble.remove();
    streamingBubble = null;
    streamingBody = null;
  }
  if (streamingParser) {
    streamingParser.destroy();
    streamingParser = null;
  }
  streamingFullText = "";
  streamingEmittedText = "";
  streamingQueuedTextLength = 0;
  streamingQueue = [];
  streamingFinalizeText = null;
  streamingFinalizeMeta = null;
  streamingFinalizeDeadline = 0;
  stopStreamingTick();
}

function finalizeStreaming(text, meta) {
  if (!streamingBubble || !streamingBody) {
    addMessage("assistant", text, meta);
    return;
  }
  const rawFinalText = typeof text === "string" ? text : "";
  const finalText = rawFinalText.trim().length ? rawFinalText : "";
  const hasStreamed = streamingFullText && streamingFullText.trim().length;
  const safeFinalText = finalText || (hasStreamed ? streamingFullText : "");
  streamingFinalizeText = safeFinalText;
  streamingFinalizeMeta = meta;
  streamingFinalizeDeadline = Date.now() + STREAM_FINALIZE_TIMEOUT_MS;
  if (safeFinalText && streamingFullText !== safeFinalText) {
    writeStreamingText(safeFinalText);
  }
  if (!streamingQueue.length) {
    finalizeStreamingNow();
  } else {
    scheduleStreamingTick();
  }
}

function finalizeStreamingNow() {
  if (!streamingBubble || !streamingBody) {
    if (streamingFinalizeText) {
      addMessage("assistant", streamingFinalizeText, streamingFinalizeMeta || {});
    }
    clearStreaming();
    return;
  }
  stopStreamingTick();
  const finalText = streamingFinalizeText || "";
  const finalMeta = streamingFinalizeMeta || {};
  let normalizedFinalText = (finalText || "").trim().length ? finalText : "";
  if (!normalizedFinalText && streamingFullText && streamingFullText.trim().length) {
    normalizedFinalText = streamingFullText;
  }
  if (!normalizedFinalText) {
    normalizedFinalText = "(no response text)";
  }
  const canReuseStream =
    streamingParser &&
    normalizedFinalText &&
    streamingFullText &&
    streamingFullText === normalizedFinalText &&
    streamingEmittedText.length >= normalizedFinalText.length;
  if (streamingParser) {
    streamingParser.end();
    if (!canReuseStream) {
      streamingParser.destroy();
    }
    streamingParser = null;
  }
  if (!canReuseStream) {
    streamingBody.innerHTML = renderMarkdown(normalizedFinalText);
  }
  streamingBubble.classList.remove("streaming");
  appendAssistantMeta(streamingBubble, finalMeta);
  streamingBubble = null;
  streamingBody = null;
  streamingFullText = "";
  streamingEmittedText = "";
  streamingQueuedTextLength = 0;
  streamingQueue = [];
  streamingFinalizeText = null;
  streamingFinalizeMeta = null;
  streamingFinalizeDeadline = 0;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
}

function updatePending(text) {
  if (!pendingBubble) return;
  if (pendingTextEl) {
    pendingTextEl.textContent = text;
  }
}

// Toggle debug panels on existing messages
function updateDebugVisibility() {
  const bubbles = document.querySelectorAll(".bubble.assistant");
  bubbles.forEach((bubble) => {
    const existingPanel = bubble.querySelector(".tool-info");
    if (toolsToggle.checked) {
      if (!existingPanel) {
        if (bubble.classList.contains("streaming")) {
          ensureDebugPlaceholder(bubble);
        } else if (bubble.dataset.debugHtml) {
          const metaLine = bubble.querySelector(".meta");
          if (metaLine) {
            const template = document.createElement("template");
            template.innerHTML = bubble.dataset.debugHtml;
            const details = template.content.firstChild;
            metaLine.after(details);
          }
        }
      }
    } else {
      // Remove debug panels when toggle is off (but keep data attribute)
      const panel = bubble.querySelector(".tool-info");
      if (panel) panel.remove();
    }
  });

  if (pendingBubble) {
    if (toolsToggle.checked) {
      if (pendingTextEl) {
        pendingTextEl.textContent = pendingTextEl.textContent || "Starting request...";
      }
    } else {
      if (pendingTextEl) {
        pendingTextEl.textContent = "Thinking...";
      }
    }
  }
}

function scheduleReconnect() {
  if (reconnectTimer) {
    return;
  }
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    reconnectDelayMs = Math.min(reconnectDelayMs * 1.5, 8000);
    connect();
  }, reconnectDelayMs);
}

function connect() {
  setStatus("Connecting...", "warn");
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws`;
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    reconnectDelayMs = 1000;
    setStatus("Connected", "ok");
    loadRagInfo(); // Load index stats when connected
  };

  socket.onclose = () => {
    setStatus("Disconnected", "warn");
    scheduleReconnect();
  };

  socket.onerror = () => {
    setStatus("Connection error", "warn");
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "log") {
      const rawLevel = (data.level || "log").toString().toLowerCase();
      const level = rawLevel.toUpperCase();
      const message = data.message || "";
      const isError = rawLevel === "error" || rawLevel === "warn" || rawLevel === "warning";
      if (!isError && !toolsToggle.checked) {
        return;
      }
      addMessage("system", `[${level}] ${message}`);
      return;
    }

    // Handle progress updates
    if (data.type === "progress") {
      if (!streamingBubble) {
        updatePending(data.message || "Looking up...");
      }
      return;
    }

    if (data.type === "stream") {
      const streamText = data.text || "";
      if (streamText.trim().length) {
        clearPending();
        addStreaming(streamText);
      }
      return;
    }

    // Handle final response
    clearPending();
    setBusy(false);

    if (data.session_id) {
      setSessionId(data.session_id);
    }
    if ("answer" in data) {
      const rawAnswer = typeof data.answer === "string" ? data.answer : "";
      const answerText = rawAnswer.trim().length ? rawAnswer : "(no response text)";
      const meta = {
        requestId: data.request_id,
        verdict: data.verdict,
        confidence: data.confidence,
        classification: data.classification,
        escalationId: data.escalation_id,
        trace: data.trace,
        timing: data.timing,
      };
      if (streamingBubble) {
        finalizeStreaming(answerText, meta);
      } else {
        addMessage("assistant", answerText, meta);
      }
    }
    if (data.error) {
      clearStreaming();
      addMessage("assistant", `Error: ${data.error}`);
    }
  };
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) {
    return;
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addMessage("system", "Still reconnecting. Please try again in a moment.");
    return;
  }
  addMessage("user", message);
  setBusy(true);
  addPending();
  socket.send(
    JSON.stringify({
      message,
      session_id: sessionId,
      debug: toolsToggle.checked,
      fast_mode: Boolean(fastModeToggle && fastModeToggle.checked),
    })
  );
  inputEl.value = "";
});

toolsToggle.addEventListener("change", () => {
  sessionStorage.setItem(TOOL_TOGGLE_KEY, toolsToggle.checked ? "true" : "false");
  if (toolsToggle.checked) {
    document.body.classList.add("debug-mode");
  } else {
    document.body.classList.remove("debug-mode");
  }
  updateDebugVisibility();
});

if (fastModeToggle) {
  fastModeToggle.addEventListener("change", () => {
    sessionStorage.setItem(FAST_MODE_KEY, fastModeToggle.checked ? "true" : "false");
  });
}

newSessionBtn.addEventListener("click", () => {
  setSessionId(generateUUID());
  chatEl.innerHTML = "";
  clearStreaming();
  addMessage("system", "New session started. Ask your next question.");
});

// RAG Info card toggle
if (ragInfoToggle) {
  ragInfoToggle.addEventListener("click", toggleRagInfo);
}

// Initialize session ID - generate a new one if none exists
if (!sessionId) {
  sessionId = generateUUID();
}
setSessionId(sessionId);
connect();
loadRagInfo(); // Also try to load stats on page load

// About modal
const aboutModal = document.getElementById("about-modal");
const aboutClose = document.querySelector(".about-close");
const aboutBtn = document.getElementById("about-btn");

function showAbout() {
  if (aboutModal) {
    aboutModal.classList.add("show");
  }
}

function hideAbout() {
  if (aboutModal) {
    aboutModal.classList.remove("show");
  }
}

if (aboutModal) {
  aboutClose.addEventListener("click", hideAbout);
  aboutModal.addEventListener("click", (e) => {
    if (e.target === aboutModal) {
      hideAbout();
    }
  });
  // Close on ESC key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && aboutModal.classList.contains("show")) {
      hideAbout();
    }
  });
}

if (aboutBtn) {
  aboutBtn.addEventListener("click", showAbout);
}

// ========================================
// IMMERSIVE MODE TOGGLE
// ========================================

// Wait for DOM to be ready before setting up immersive toggle
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupImmersiveToggle);
} else {
  setupImmersiveToggle();
}

function setupImmersiveToggle() {
  const IMMERSIVE_KEY = "profectus_immersive_mode";
  const immersiveToggle = document.getElementById("immersive-toggle");

  // Restore immersive mode state from sessionStorage
  if (sessionStorage.getItem(IMMERSIVE_KEY) === "true") {
    document.body.classList.add("immersive-mode");
  }

  if (immersiveToggle) {
    immersiveToggle.addEventListener("click", () => {
      const isImmersive = document.body.classList.toggle("immersive-mode");
      sessionStorage.setItem(IMMERSIVE_KEY, isImmersive ? "true" : "false");

      // Scroll chat to bottom when toggling immersive mode
      if (chatEl) {
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    });
  }
}
