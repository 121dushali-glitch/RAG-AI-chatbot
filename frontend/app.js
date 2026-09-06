// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;

// ── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let chatHistory = []; // [{role, content}]
let hasDocuments = false;
let isStreaming = false;

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const menuBtn = document.getElementById('menuBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const sourceList = document.getElementById('sourceList');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const addUrlBtn = document.getElementById('addUrlBtn');
const progressWrap = document.getElementById('progressWrap');
const progressLabel = document.getElementById('progressLabel');
const resetBtn = document.getElementById('resetBtn');
const messagesEl = document.getElementById('messages');
const welcomeCard = document.getElementById('welcomeCard');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const statusBadge = document.getElementById('statusBadge');
const sourcesDrawer = document.getElementById('sourcesDrawer');
const sourcesContent = document.getElementById('sourcesContent');
const drawerClose = document.getElementById('drawerClose');
const toast = document.getElementById('toast');

// ── Init ─────────────────────────────────────────────────────────────────────
init();

async function init() {
  await createNewSession();
  bindEvents();
}

async function createNewSession() {
  try {
    const res = await fetch(`${API_BASE}/api/session/new`, { method: 'POST' });
    const data = await res.json();
    sessionId = data.session_id;
  } catch (e) {
    showToast('Could not connect to backend. Is the server running?', 'error');
  }
}

// ── Event Bindings ───────────────────────────────────────────────────────────
function bindEvents() {
  // Mobile sidebar toggle
  menuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));

  // New session
  newSessionBtn.addEventListener('click', resetSession);
  resetBtn.addEventListener('click', resetSession);

  // File upload via click
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

  // Drag & drop
  ['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    })
  );
  dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));

  // URL ingestion
  addUrlBtn.addEventListener('click', handleAddUrl);
  urlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleAddUrl();
  });

  // Chat input
  questionInput.addEventListener('input', autoGrow);
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  sendBtn.addEventListener('click', sendMessage);

  // Sources drawer
  drawerClose.addEventListener('click', () => (sourcesDrawer.hidden = true));
}

function autoGrow() {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
}

// ── File Upload ──────────────────────────────────────────────────────────────
async function handleFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;

  for (const file of files) {
    await uploadOneFile(file);
  }
  fileInput.value = '';
}

async function uploadOneFile(file) {
  showProgress(`Processing "${file.name}"…`);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);

  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Upload failed');
    }

    addSourceToList(file.name, getFileIcon(file.name));
    markDocumentsReady();
    showToast(`"${file.name}" indexed — ${data.chunks_indexed} chunks added.`, 'success');
  } catch (err) {
    showToast(`Failed to process "${file.name}": ${err.message}`, 'error');
  } finally {
    hideProgress();
  }
}

async function handleAddUrl() {
  const url = urlInput.value.trim();
  if (!url) return;

  try {
    new URL(url);
  } catch {
    showToast('Please enter a valid URL (including https://)', 'error');
    return;
  }

  showProgress(`Fetching "${url}"…`);

  try {
    const res = await fetch(`${API_BASE}/api/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, session_id: sessionId }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Could not fetch URL');
    }

    addSourceToList(new URL(url).hostname, '🌐');
    markDocumentsReady();
    showToast(`URL indexed — ${data.chunks_indexed} chunks added.`, 'success');
    urlInput.value = '';
  } catch (err) {
    showToast(`Failed: ${err.message}`, 'error');
  } finally {
    hideProgress();
  }
}

function getFileIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const icons = {
    pdf: '📄', docx: '📝', doc: '📝',
    xlsx: '📊', xls: '📊', csv: '📊',
    pptx: '📽️', ppt: '📽️',
    txt: '📃', md: '📃',
  };
  return icons[ext] || '📁';
}

function addSourceToList(name, icon) {
  const emptyHint = sourceList.querySelector('.empty-hint');
  if (emptyHint) emptyHint.remove();

  const li = document.createElement('li');
  li.className = 'source-item';
  li.innerHTML = `<span class="source-icon">${icon}</span><span>${escapeHtml(name)}</span>`;
  sourceList.appendChild(li);
}

function markDocumentsReady() {
  hasDocuments = true;
  questionInput.disabled = false;
  sendBtn.disabled = false;
  questionInput.placeholder = 'Ask a question about your documents…';
  statusBadge.textContent = '✓ Documents loaded';
  statusBadge.classList.add('ready');
}

function showProgress(label) {
  progressLabel.textContent = label;
  progressWrap.hidden = false;
}

function hideProgress() {
  progressWrap.hidden = true;
}

// ── Chat ─────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || isStreaming) return;
  if (!hasDocuments) {
    showToast('Please upload a document or add a URL first.', 'error');
    return;
  }

  welcomeCard.style.display = 'none';

  appendMessage('user', question);
  chatHistory.push({ role: 'user', content: question });

  questionInput.value = '';
  autoGrow();
  questionInput.disabled = true;
  sendBtn.disabled = true;
  isStreaming = true;

  const aiRow = appendMessage('ai', '');
  const bubble = aiRow.querySelector('.bubble');
  bubble.innerHTML = '<span class="cursor"></span>';

  let fullAnswer = '';

  try {
    const historyParam = encodeURIComponent(
      JSON.stringify(chatHistory.slice(-7, -1)) // last 6 prior turns, excluding current
    );
    const url = `${API_BASE}/api/chat/stream?session_id=${sessionId}&question=${encodeURIComponent(
      question
    )}&history=${historyParam}`;

    const eventSource = new EventSource(url);

    await new Promise((resolve, reject) => {
      eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
          eventSource.close();
          resolve();
          return;
        }
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            fullAnswer += `\n\n⚠️ ${data.error}`;
          } else if (data.token) {
            fullAnswer += data.token;
          }
          bubble.innerHTML = escapeHtml(fullAnswer) + '<span class="cursor"></span>';
          scrollToBottom();
        } catch {
          /* ignore parse errors on heartbeat lines */
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        resolve();
      };
    });

    bubble.innerHTML = escapeHtml(fullAnswer || 'No response generated.');
    chatHistory.push({ role: 'assistant', content: fullAnswer });

    // Add "View sources" button
    const sourceBtn = document.createElement('button');
    sourceBtn.className = 'source-btn';
    sourceBtn.innerHTML = `📎 View sources`;
    sourceBtn.addEventListener('click', () => showSources(question));
    aiRow.appendChild(sourceBtn);
  } catch (err) {
    bubble.textContent = `Error: ${err.message}`;
  } finally {
    isStreaming = false;
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
    scrollToBottom();
  }
}

function appendMessage(role, content) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : 'Assistant';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;

  row.appendChild(label);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function showSources(question) {
  try {
    const res = await fetch(
      `${API_BASE}/api/sources?session_id=${sessionId}&question=${encodeURIComponent(question)}`
    );
    const data = await res.json();

    sourcesContent.innerHTML = '';
    if (!data.sources || !data.sources.length) {
      sourcesContent.innerHTML = '<p style="color:var(--text-muted)">No sources found.</p>';
    } else {
      data.sources.forEach((src, i) => {
        const div = document.createElement('div');
        div.className = 'source-excerpt';
        div.innerHTML = `
          <div class="source-excerpt-label">Excerpt ${i + 1} — ${escapeHtml(src.source)} (relevance: ${(src.score * 100).toFixed(0)}%)</div>
          <div>${escapeHtml(src.text)}</div>
        `;
        sourcesContent.appendChild(div);
      });
    }
    sourcesDrawer.hidden = false;
  } catch (err) {
    showToast('Could not load sources.', 'error');
  }
}

// ── Reset ────────────────────────────────────────────────────────────────────
async function resetSession() {
  if (sessionId) {
    try {
      await fetch(`${API_BASE}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch {
      /* ignore */
    }
  }

  // Reset UI state
  chatHistory = [];
  hasDocuments = false;
  messagesEl.innerHTML = '';
  messagesEl.appendChild(welcomeCard);
  welcomeCard.style.display = 'block';
  sourceList.innerHTML = '<li class="empty-hint">No documents yet</li>';
  questionInput.disabled = true;
  sendBtn.disabled = true;
  questionInput.placeholder = 'Ask a question about your documents…';
  statusBadge.textContent = 'No documents loaded';
  statusBadge.classList.remove('ready');
  sourcesDrawer.hidden = true;
  sidebar.classList.remove('open');

  await createNewSession();
  showToast('Session reset. Upload new documents to begin.', 'success');
}

// ── Utilities ────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

let toastTimer = null;
function showToast(message, type = '') {
  toast.textContent = message;
  toast.className = `toast ${type}`;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.hidden = true), 3500);
}
