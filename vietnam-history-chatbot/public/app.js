const DEMO_QUESTIONS = [
  {
    group: 'Sự kiện',
    label: 'Ý nghĩa Điện Biên Phủ',
    query: 'Chiến thắng Điện Biên Phủ 1954 có ý nghĩa gì?',
    capability: 'Vì sao đây là bước ngoặt quan trọng?',
    demoId: 'demo_dbp',
  },
  {
    group: 'Dòng thời gian',
    label: 'Cách mạng Tháng Tám',
    query: 'Tóm tắt các mốc chính của Cách mạng Tháng Tám 1945.',
    capability: 'Theo dõi diễn biến chính năm 1945',
    demoId: 'demo_cmtt',
  },
  {
    group: 'So sánh',
    label: 'Genève và Paris',
    query: 'So sánh Hiệp định Genève 1954 và Hiệp định Paris 1973.',
    capability: 'Hai hiệp định khác nhau ở điểm nào?',
    fallbackKey: 'comparison_geneve_paris',
  },
  {
    group: 'Nguồn tư liệu',
    label: 'Sự kiện ngày 30/4',
    query: 'Nguồn nào cho thấy 30/4/1975 là mốc kết thúc chiến tranh ở Việt Nam?',
    capability: 'Xem nguồn hỗ trợ cho một mốc lịch sử',
    demoId: 'demo_saigon',
  },
  {
    group: 'Tìm hiểu thêm',
    label: 'Ý nghĩa Hiệp định Genève',
    query: 'Hiệp định Genève 1954 có ý nghĩa gì?',
    followup: 'Nó khác Hiệp định Paris 1973 thế nào?',
    capability: 'Hỏi tiếp để so sánh với Paris 1973',
    demoId: 'demo_17',
    fallbackKey: 'followup_geneve_paris',
  },
  {
    group: 'Tra cứu',
    label: 'Chiến dịch Điện Biên Phủ',
    query: 'chien dich dien bien phu co y nghia gi',
    capability: 'Có thể nhập câu hỏi không dấu',
    demoId: 'demo_dbp',
  },
  {
    group: 'Nhân vật và tổ chức',
    label: 'Việt Minh',
    query: 'Việt Minh có vai trò như thế nào trong Cách mạng Tháng Tám 1945?',
    capability: 'Tìm hiểu vai trò của một tổ chức',
  },
];

const LOCAL_FIXTURES = {
  comparison_geneve_paris: {
    answer: '**A - Hiệp định Genève 1954:**\n- Genève 1954 gắn với kết thúc chiến tranh Đông Dương và đặt ra khuôn khổ tạm thời cho vấn đề Việt Nam sau năm 1954 [1].\n\n**B - Hiệp định Paris 1973:**\n- Paris 1973 gắn với việc chấm dứt chiến tranh, lập lại hòa bình ở Việt Nam và quá trình Mỹ rút quân [2].\n\n**Khác nhau chính:**\n- Genève thuộc bối cảnh chống Pháp và chia cắt tạm thời, còn Paris thuộc bối cảnh chiến tranh Việt Nam giai đoạn chống Mỹ [1][2].',
    citations: [
      source('[1]', 'Hiệp định Genève 1954: nội dung, giới hạn và hệ quả chia cắt', 'SYN_TREATY_008', 'SRC_0009, SRC_0005', 'Nguồn tổng hợp về Hiệp định Genève 1954, hệ quả chia cắt và bối cảnh sau chiến tranh Đông Dương.'),
      source('[2]', 'Hiệp định Paris 1973 và việc Mỹ rút quân khỏi Việt Nam', 'SYN_TREATY_011', 'SRC_0011, SRC_0005', 'Nguồn tổng hợp về Hiệp định Paris 1973, chấm dứt chiến tranh và việc Mỹ rút quân.'),
    ],
    intent: 'comparison',
    retrieval_mode: 'verified_local_fixture_backup',
  },
  followup_geneve_paris: {
    answer: '**A - Hiệp định Genève 1954:**\n- Genève 1954 là thỏa thuận gắn với kết thúc chiến tranh Đông Dương và đặt ra các vấn đề chính trị-quân sự sau 1954 [1].\n\n**B - Hiệp định Paris 1973:**\n- Paris 1973 là thỏa thuận về chấm dứt chiến tranh, lập lại hòa bình tại Việt Nam và gắn với việc Mỹ rút quân [2].\n\n**Kết luận ngắn:** Câu hỏi follow-up “Nó” được hiểu là Hiệp định Genève 1954; điểm khác chính nằm ở bối cảnh lịch sử, bên tham gia và hệ quả trực tiếp của hai hiệp định [1][2].',
    citations: [
      source('[1]', 'Hiệp định Genève 1954: nội dung, giới hạn và hệ quả chia cắt', 'SYN_TREATY_008', 'SRC_0009, SRC_0005', 'Nguồn hỗ trợ phần Hiệp định Genève 1954 và hệ quả sau năm 1954.'),
      source('[2]', 'Hiệp định Paris 1973 và việc Mỹ rút quân khỏi Việt Nam', 'SYN_TREATY_011', 'SRC_0011, SRC_0005', 'Nguồn hỗ trợ phần Hiệp định Paris 1973 và quá trình Mỹ rút quân.'),
    ],
    intent: 'followup_comparison',
    retrieval_mode: 'verified_local_fixture_backup',
    rewritten_query: 'Hiệp định Genève 1954 khác Hiệp định Paris 1973 thế nào?',
  },
  oos_gold: {
    answer: 'Câu hỏi này nằm ngoài phạm vi chatbot lịch sử Việt Nam. Mình không có nguồn nội bộ để trả lời về giá vàng hôm nay, nên không tạo citation cho nội dung này. Bạn có thể hỏi một câu liên quan đến lịch sử Việt Nam, ví dụ: “Chiến thắng Điện Biên Phủ 1954 có ý nghĩa gì?”',
    citations: [],
    intent: 'out_of_scope',
    retrieval_mode: 'policy_guard',
    safety_mode: 'safe_out_of_scope',
  },
};

let sessionId = `web-demo-${Date.now()}`;
let isSending = false;
let activeCitations = [];
let activeDebug = {};
let pendingFollowup = '';
let runtimeMode = 'local_no_cloud';
const DEFAULT_LOCAL_DATA_PROFILE = 'cloud_primary_final';
const DEFAULT_API_DATA_PROFILE = 'cloud_primary_final';
const DEFAULT_DATA_PROFILE = DEFAULT_LOCAL_DATA_PROFILE;
let dataProfile = DEFAULT_LOCAL_DATA_PROFILE;

document.addEventListener('DOMContentLoaded', () => {
  renderDemoList();
  renderSuggestions();
  bindEvents();
  renderRuntimeMode();
  checkHealth();
});

function source(marker, title, docId, sourceId, snippet) {
  return { marker, title, doc_id: docId, source_id: sourceId, snippet, metadata: { demo_source: true } };
}

function bindEvents() {
  const input = el('chatInput');
  el('btnSend').addEventListener('click', sendFromInput);
  el('btnNewChat').addEventListener('click', resetChat);
  el('btnDebug').addEventListener('click', () => {
    openSources();
    el('technicalDetails').open = true;
  });
  el('btnSourcesMobile').addEventListener('click', openSources);
  el('btnCloseSources').addEventListener('click', closeSources);
  el('sourceBackdrop').addEventListener('click', closeSources);
  el('modeLocal').addEventListener('click', () => setRuntimeMode('local_no_cloud'));
  el('modeApiFast').addEventListener('click', () => setRuntimeMode('api_9router_fast'));
  el('btnAbout').addEventListener('click', () => {
    openSources();
    el('technicalDetails').open = true;
    el('evaluationPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  el('btnFollowup').addEventListener('click', () => {
    if (pendingFollowup) sendMessage(pendingFollowup, { fallbackKey: 'followup_geneve_paris' });
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendFromInput();
    }
  });
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      input.focus();
    }
    if (event.key === 'Escape') closeSources();
  });
  document.querySelectorAll('[data-fill]').forEach((button) => {
    button.addEventListener('click', () => {
      input.value = button.dataset.fill || '';
      input.focus();
    });
  });
}

function setRuntimeMode(mode) {
  runtimeMode = mode;
  dataProfile = mode === 'api_9router_fast' ? DEFAULT_API_DATA_PROFILE : DEFAULT_LOCAL_DATA_PROFILE;
  renderRuntimeMode();
}

function renderRuntimeMode() {
  const isApi = runtimeMode === 'api_9router_fast';
  el('modeLocal').classList.toggle('active', !isApi);
  el('modeApiFast').classList.toggle('active', isApi);
  el('modeLocal').setAttribute('aria-pressed', String(!isApi));
  el('modeApiFast').setAttribute('aria-pressed', String(isApi));
  document.body.dataset.runtimeMode = runtimeMode;
  el('runtimeModeDescription').textContent = isApi
    ? 'AI trực tuyến diễn giải từ nguồn tư liệu đã tìm thấy.'
    : 'Tìm trong bộ tư liệu và trả lời kèm nguồn trích dẫn.';
  el('composerHint').textContent = isApi
    ? 'Diễn giải AI sử dụng nguồn đã truy xuất.'
    : 'Tra cứu cục bộ, kèm nguồn tư liệu.';
  el('healthRuntimeMode').textContent = isApi ? 'Diễn giải AI trực tuyến' : 'Xử lý cục bộ';
  el('healthRuntimeMode').className = isApi ? 'warn' : 'ok';
}

function renderDemoList() {
  el('demoList').innerHTML = DEMO_QUESTIONS.map((question) => `
    <button class="demo-row" type="button" data-query="${escapeAttr(question.query)}">
      <strong>${escapeHtml(question.label)}</strong>
      <small>${escapeHtml(question.group)} - ${escapeHtml(question.capability)}</small>
    </button>
  `).join('');
  document.querySelectorAll('.demo-row').forEach((button) => {
    button.addEventListener('click', () => {
      const item = DEMO_QUESTIONS.find((q) => q.query === button.dataset.query);
      if (item) sendMessage(item.query, item);
    });
  });
}

function renderSuggestions() {
  const visible = DEMO_QUESTIONS.slice(0, 6);
  el('suggestionGrid').innerHTML = visible.map((question) => `
    <button class="demo-card" type="button" data-query="${escapeAttr(question.query)}">
      <span>${escapeHtml(question.group)}</span>
      <strong>${escapeHtml(question.label)}</strong>
      <p>${escapeHtml(question.query)}</p>
    </button>
  `).join('');
  document.querySelectorAll('.demo-card').forEach((button) => {
    button.addEventListener('click', () => {
      const item = DEMO_QUESTIONS.find((q) => q.query === button.dataset.query);
      if (item) sendMessage(item.query, item);
    });
  });
}

async function checkHealth() {
  try {
    const response = await fetch('/api/web-demo-health', { cache: 'no-store' });
    const health = response.ok ? await response.json() : null;
    el('healthBackend').textContent = health?.backend === 'ok' ? 'Sẵn sàng' : 'Chưa xác định';
    el('healthOllama').textContent = health?.ollama === 'ok' ? 'Sẵn sàng' : (health?.ollama || 'Chưa xác định');
    el('healthRetrieval').textContent = health?.retrieval === 'ok' ? 'Sẵn sàng' : (health?.retrieval || 'Chưa xác định');
  } catch {
    el('healthBackend').textContent = 'Mất kết nối';
    el('healthOllama').textContent = 'Chưa xác định';
    el('healthRetrieval').textContent = 'Chưa xác định';
  }
}

function sendFromInput() {
  const input = el('chatInput');
  const query = input.value.trim();
  if (!query || isSending) return;
  const item = DEMO_QUESTIONS.find((q) => normalize(q.query) === normalize(query));
  input.value = '';
  input.style.height = 'auto';
  sendMessage(query, item || guessQuestionMeta(query));
}

async function sendMessage(query, meta = {}) {
  if (isSending) return;
  isSending = true;
  el('btnSend').disabled = true;
  el('emptyState')?.remove();
  appendUser(query);
  const thinking = showThinking();
  pendingFollowup = meta.followup || '';
  updateFollowupBar();

  const started = performance.now();
  try {
    const endpoint = runtimeMode === 'api_9router_fast' ? '/api/9router-fast-chat' : '/api/local-hybrid-chat';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: query,
        session_id: sessionId,
        demo_mode: true,
        return_debug: true,
        force_local_hybrid: true,
        runtime_mode: runtimeMode,
        data_profile: dataProfile,
        force_cloud_llm_final: runtimeMode === 'api_9router_fast',
      }),
    });
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => null);
      if (runtimeMode === 'api_9router_fast' && errorPayload) {
        const mapped = buildApiModeErrorResponse(query, Math.round(performance.now() - started));
        mapped.debug.provider_status = response.status;
        mapped.debug.data_profile = errorPayload.debug?.data_profile || dataProfile;
        mapped.debug.served_by = errorPayload.debug?.served_by || null;
        finishAssistant(thinking, mapped);
        return;
      }
      throw new Error('runtime unavailable');
    }
    const raw = await response.json();
    const mapped = normalizeRuntimeResponse(raw, query, meta, Math.round(performance.now() - started));
    finishAssistant(thinking, mapped);
  } catch {
    const mapped = runtimeMode === 'api_9router_fast'
      ? buildApiModeErrorResponse(query, Math.round(performance.now() - started))
      : buildFixtureResponse(query, meta, Math.round(performance.now() - started));
    finishAssistant(thinking, mapped);
  } finally {
    isSending = false;
    el('btnSend').disabled = false;
  }
}

function normalizeRuntimeResponse(raw, query, meta, latencyMs) {
  const citations = mapCitations(raw.citations || raw.sources || []);
  const isSafety = detectSafety(raw.answer || '', raw.mode, raw._capabilityDecision);
  return {
    answer: raw.answer || raw.response || '',
    citations: isSafety ? [] : citations,
    debug: {
      original_query: query,
      normalized_query: normalizeForDebug(query, meta),
      rewritten_query: raw.debug?.rewritten_query || raw.metadata?.memory?.effective_query || raw.metadata?.memory?.rewrite?.rewritten_query || null,
      runtime_mode: raw.debug?.runtime_mode || runtimeMode,
      data_profile: raw.debug?.data_profile || dataProfile,
      intent: raw.debug?.routing?.intent || raw._debugTrace?.routing?.intent || inferIntent(query, meta),
      retrieval_mode: raw.debug?.retrieval_mode || raw.mode || 'hybrid_rrf',
      rrf_k: 30,
      safety_mode: isSafety ? 'safe_out_of_scope_or_clarification' : 'none',
      answer_generator: raw.debug?.answer_generator || (runtimeMode === 'api_9router_fast' ? '9router_api' : 'local_ollama_or_template'),
      provider_model: raw.debug?.provider_model,
      provider_base_url: raw.debug?.provider_base_url,
      local_llm_model: 'qwen2.5:3b-instruct',
      latency_ms: latencyMs,
      retrieval_latency_ms: raw.debug?.retrieval_latency_ms,
      generation_latency_ms: raw.debug?.generation_latency_ms,
      chunks_count: citations.length,
      cloud_api_calls: raw.debug?.cloud_api_calls || 0,
      cloud_llm_calls: raw.debug?.cloud_llm_calls || 0,
      bm25_used: raw.debug?.bm25_used,
      query_embedding_generated: raw.debug?.query_embedding_generated,
      vector_used: raw.debug?.vector_used,
      faiss_used: raw.debug?.faiss_used,
      rrf_used: raw.debug?.rrf_used,
      bm25_fallback: raw.debug?.bm25_fallback,
      fallback_reason: raw.debug?.fallback_reason,
      local_llm_called: raw.debug?.local_llm_called,
      local_embedding_model: raw.debug?.local_embedding_model,
      cloud_embedding_calls: raw.debug?.cloud_embedding_calls || 0,
      external_network_calls: raw.debug?.external_network_calls || 0,
      local_retrieval: raw.debug?.local_retrieval,
      api_used_for_answer_generation_only: raw.debug?.api_used_for_answer_generation_only,
      context_only_guard_issues: raw.debug?.context_only_guard_issues,
      render_mode: raw.debug?.render_mode,
      candidate_corpus_used: raw.debug?.candidate_corpus_used,
      candidate_index_used: raw.debug?.candidate_index_used,
      served_by: raw.debug?.served_by,
      adapter_endpoint: raw.debug?.adapter_endpoint || (runtimeMode === 'api_9router_fast' ? '/api/9router-fast-chat' : '/api/local-hybrid-chat'),
    },
    status: { answerable: !isSafety, safe: true, no_cloud: runtimeMode === 'local_no_cloud' },
    warnings: raw.warnings || [],
    confidence: raw.confidence || 'medium',
  };
}

function buildApiModeErrorResponse(query, latencyMs) {
  return {
    answer: 'Chế độ diễn giải AI hiện chưa sẵn sàng. Bạn vẫn có thể chọn “Tra cứu” để tiếp tục tìm hiểu từ nguồn tư liệu.',
    citations: [],
    debug: {
      original_query: query,
      normalized_query: query,
      rewritten_query: null,
      runtime_mode: 'api_9router_fast',
      intent: 'runtime_error',
      retrieval_mode: 'api_fast_unavailable',
      safety_mode: 'none',
      answer_generator: '9router_api',
      latency_ms: latencyMs,
      chunks_count: 0,
      cloud_api_calls: 0,
      cloud_llm_calls: 0,
      cloud_embedding_calls: 0,
      external_network_calls: 0,
      local_retrieval: false,
      api_used_for_answer_generation_only: false,
      adapter_endpoint: '/api/9router-fast-chat',
    },
    status: { answerable: false, safe: true, no_cloud: false, api_fast_mode: true },
    warnings: ['Dịch vụ diễn giải AI đang tạm thời không kết nối được.'],
    confidence: 'low',
  };
}

function buildFixtureResponse(query, meta, latencyMs) {
  const fixture = LOCAL_FIXTURES[meta.fallbackKey] || LOCAL_FIXTURES.oos_gold;
  const safety = fixture.safety_mode || 'none';
  return {
    answer: fixture.answer,
    citations: fixture.citations,
    debug: {
      original_query: query,
      normalized_query: normalizeForDebug(query, meta),
      rewritten_query: fixture.rewritten_query || null,
      intent: fixture.intent || inferIntent(query, meta),
      retrieval_mode: fixture.retrieval_mode || 'verified_local_fixture_backup',
      rrf_k: 30,
      safety_mode: safety,
      local_llm_model: 'qwen2.5:3b-instruct',
      latency_ms: latencyMs,
      chunks_count: fixture.citations.length,
      cloud_api_calls: 0,
      adapter_endpoint: 'local verified fallback',
    },
    status: { answerable: safety === 'none', safe: true, no_cloud: true },
    warnings: ['Không thể truy cập dịch vụ trả lời trong lượt này; hệ thống đang hiển thị phản hồi dự phòng từ nguồn cục bộ.'],
    confidence: safety === 'none' ? 'medium' : 'high',
  };
}

function mapCitations(citations) {
  return citations.slice(0, 6).map((citation, index) => ({
    marker: citation.marker || `[${index + 1}]`,
    title: citation.title || `Nguồn ${index + 1}`,
    source_id: Array.isArray(citation.source_ids) ? citation.source_ids.join(', ') : citation.source_id || citation.record_id || '',
    doc_id: citation.doc_id || citation.record_id || '',
    snippet: citation.snippet || citation.relevance || citation.excerpt || '',
    url: citation.url || null,
    metadata: {
      doc_source: citation.doc_source,
      doc_type: citation.doc_type,
      year: citation.year,
      sources: citation.sources,
    },
  }));
}

function finishAssistant(thinking, response) {
  thinking.remove();
  activeCitations = response.citations || [];
  activeDebug = response.debug || {};
  appendAssistant(response);
  renderSources(response);
  renderDebug(response);
}

function appendUser(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message user';
  wrapper.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  el('messages').appendChild(wrapper);
  scrollMessages();
}

function appendAssistant(response) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  const safety = response.debug?.safety_mode && response.debug.safety_mode !== 'none';
  wrapper.innerHTML = `
    <div class="bubble">
      <div class="answer-content">${formatAnswer(response.answer)}</div>
      ${safety ? '<div class="safety-card">Câu hỏi này không có nguồn phù hợp trong phạm vi tư liệu hiện có.</div>' : ''}
      <div class="message-meta">
        <span class="mini-pill ${safety ? 'amber' : 'blue'}">${safety ? 'Không có nguồn phù hợp' : `${activeCitations.length || response.citations?.length || 0} nguồn tham khảo`}</span>
        <span class="mini-pill ${response.debug?.runtime_mode === 'api_9router_fast' ? 'amber' : 'green'}">${response.debug?.runtime_mode === 'api_9router_fast' ? 'Diễn giải AI trực tuyến' : 'Xử lý cục bộ'}</span>
        <span class="mini-pill">${Math.round(response.debug?.latency_ms || 0)} ms</span>
        ${response.warnings?.length ? '<span class="mini-pill amber">Có lưu ý</span>' : ''}
      </div>
      ${response.warnings?.length ? `<div class="safety-card">${response.warnings.map(escapeHtml).join('<br>')}</div>` : ''}
    </div>`;
  el('messages').appendChild(wrapper);
  wrapper.querySelectorAll('.citation-marker').forEach((button) => {
    button.addEventListener('click', () => highlightSource(button.dataset.marker));
  });
  scrollMessages();
}

function showThinking() {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant thinking';
  const started = performance.now();
  const phases = ['Đang chuẩn hóa câu hỏi', 'Đang truy xuất nguồn', 'Đang tổng hợp câu trả lời'];
  wrapper.innerHTML = `
    <div class="bubble">
      <div class="thinking-lines">
        <div class="thinking-line"><span id="thinkingPhase">${phases[0]}</span><span class="dots"><i></i><i></i><i></i></span></div>
        <div class="thinking-line"><span>Thời gian chờ</span><strong id="thinkingTimer">0s</strong></div>
        <div id="latencyNote" class="latency-note" hidden>Mô hình đang chạy cục bộ nên có thể mất thêm vài giây.</div>
      </div>
    </div>`;
  el('messages').appendChild(wrapper);
  const interval = setInterval(() => {
    if (!document.body.contains(wrapper)) {
      clearInterval(interval);
      return;
    }
    const elapsed = Math.floor((performance.now() - started) / 1000);
    el('thinkingTimer').textContent = `${elapsed}s`;
    el('thinkingPhase').textContent = phases[Math.min(2, Math.floor(elapsed / 5))];
    if (elapsed > 20) el('latencyNote').hidden = false;
  }, 500);
  scrollMessages();
  return wrapper;
}

function renderSources(response) {
  const sources = response.citations || [];
  const safety = response.debug?.safety_mode && response.debug.safety_mode !== 'none';
  if (!sources.length) {
    el('sourceCards').innerHTML = safety
      ? '<div class="empty-panel">Không có nguồn phù hợp để trích dẫn cho câu hỏi này.</div>'
      : '<div class="empty-panel">Nguồn trích dẫn sẽ hiển thị tại đây.</div>';
    el('btnSourcesMobile').textContent = 'Xem nguồn';
    return;
  }
  el('btnSourcesMobile').textContent = `Xem nguồn (${sources.length})`;
  el('sourceCards').innerHTML = sources.map((sourceCard) => `
    <article class="source-card" data-marker="${escapeAttr(sourceCard.marker)}">
      <div class="source-card-head">
        <span class="source-marker">${escapeHtml(sourceCard.marker)}</span>
        <div>
          <h4 class="source-title">${escapeHtml(displaySourceTitle(sourceCard.title))}</h4>
        </div>
      </div>
      <div class="source-snippet">${escapeHtml(sourceCard.snippet || 'Chưa có trích đoạn hiển thị.')}</div>
    </article>
  `).join('');
}

function renderDebug(response) {
  const debug = response.debug || {};
  const rows = [
    ['Câu hỏi gốc', debug.original_query || 'Chưa có câu hỏi'],
    ['Câu hỏi chuẩn hóa', debug.normalized_query || 'n/a'],
    ['Chế độ trả lời', debug.runtime_mode === 'api_9router_fast' ? 'AI trực tuyến' : 'Cục bộ'],
    ['Mục đích câu hỏi', debug.intent || 'n/a'],
    ['Cách truy xuất', debug.retrieval_mode || 'n/a'],
    ['Tìm kiếm từ khóa', debug.bm25_used === true ? 'Có' : debug.bm25_used === false ? 'Không' : 'n/a'],
    ['Tìm kiếm ngữ nghĩa', debug.vector_used === true ? 'Có' : debug.vector_used === false ? 'Không' : 'n/a'],
    ['Kết hợp kết quả', debug.rrf_used === true ? 'Có' : debug.rrf_used === false ? 'Không' : 'n/a'],
    ['Nguồn trả về', debug.chunks_count ?? 0],
    ['Thời gian', `${debug.latency_ms || 0} ms`],
    ['Gọi dịch vụ trực tuyến', debug.cloud_api_calls || 0],
  ];
  el('debugPanel').innerHTML = rows.map(([key, value]) => `
    <div class="debug-row"><span>${escapeHtml(String(key))}</span><strong>${escapeHtml(String(value ?? 'n/a'))}</strong></div>
  `).join('');
}

function highlightSource(marker) {
  document.querySelectorAll('.citation-marker').forEach((button) => {
    button.classList.toggle('active', button.dataset.marker === marker);
  });
  document.querySelectorAll('.source-card').forEach((card) => {
    const selected = card.dataset.marker === marker;
    card.classList.toggle('highlight', selected);
    if (selected) {
      openSources();
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
}

function formatAnswer(answer) {
  const escaped = escapeHtml(answer || '');
  const withBold = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  const withCitations = withBold.replace(/\[(\d+)\]/g, (_, number) => {
    const marker = `[${number}]`;
    return `<button class="citation-marker" type="button" data-marker="${marker}">${marker}</button>`;
  });
  return withCitations
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
    .join('');
}

function updateFollowupBar() {
  const bar = el('followupBar');
  const button = el('btnFollowup');
  if (!pendingFollowup) {
    bar.hidden = true;
    button.textContent = '';
    return;
  }
  bar.hidden = false;
  button.textContent = pendingFollowup;
}

function resetChat() {
  sessionId = `web-demo-${Date.now()}`;
  pendingFollowup = '';
  activeCitations = [];
  activeDebug = {};
  el('messages').innerHTML = `
    <div class="empty-state" id="emptyState">
      <section class="welcome-hero" aria-label="Sử Việt">
        <div class="welcome-copy">
          <div class="empty-kicker">Lịch sử Việt Nam | 1930 - 1975</div>
          <h2>Khám phá lịch sử qua tư liệu</h2>
          <p>Sự kiện, nhân vật, hiệp định và những chuyển biến định hình một thời kỳ.</p>
        </div>
      </section>
      <section class="discovery-prompts" aria-label="Câu hỏi khởi đầu">
        <div class="discovery-heading">
          <h3>Bắt đầu từ một câu hỏi</h3>
          <p>Chọn chủ đề hoặc tự nhập điều bạn muốn tìm hiểu.</p>
        </div>
        <div class="suggestion-grid" id="suggestionGrid"></div>
      </section>
    </div>`;
  renderSuggestions();
  renderSources({ citations: [], debug: { safety_mode: 'none' } });
  renderDebug({ debug: { original_query: '', normalized_query: '', retrieval_mode: 'hybrid_rrf / demo adapter', rrf_k: 30, safety_mode: 'none', cloud_api_calls: 0 } });
  updateFollowupBar();
}

function openSources() {
  el('sourcePanel').classList.add('open');
  el('sourceBackdrop').classList.add('open');
}

function closeSources() {
  el('sourcePanel').classList.remove('open');
  el('sourceBackdrop').classList.remove('open');
}

function guessQuestionMeta(query) {
  if (/giá vàng|weather|waterloo|bitcoin/i.test(query)) return { fallbackKey: 'oos_gold' };
  if (/so sánh|khác/i.test(query) && /gen/i.test(query) && /paris/i.test(query)) return { fallbackKey: 'comparison_geneve_paris' };
  if (/dienn|dicch|dien bien|điện biên/i.test(query)) return { demoId: 'demo_dbp' };
  return {};
}

function normalizeRuntimeText(text) {
  return (text || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
}

function normalize(text) {
  return normalizeRuntimeText(text).replace(/\s+/g, ' ').trim();
}

function normalizeForDebug(query, meta) {
  if (meta.group === 'No-accent' || /chien dich dien bien phu/i.test(query)) return 'Chiến dịch Điện Biên Phủ';
  if (meta.group === 'Typo' || /dicch|dienn|bin phu/i.test(query)) return 'Chiến dịch Điện Biên Phủ';
  if (/cach mang thang tam|cách mạng tháng tám/i.test(query)) return 'Cách mạng Tháng Tám 1945';
  if (/gene/i.test(query) && /paris/i.test(query)) return 'Hiệp định Genève 1954 / Hiệp định Paris 1973';
  return query.toLowerCase();
}

function inferIntent(query, meta) {
  if (meta?.group === 'Citation') return 'citation_source';
  if (meta?.group === 'Comparison' || /so sánh|khác/i.test(query)) return 'comparison';
  if (meta?.group === 'Timeline' || /mốc|timeline|tóm tắt/i.test(query)) return 'timeline';
  if (meta?.group === 'Safety' || /giá vàng/i.test(query)) return 'out_of_scope';
  return 'fact';
}

function detectSafety(answer, mode, decision) {
  return decision?.policy === 'REFUSE_OOS'
    || decision?.policy === 'CLARIFY'
    || mode === 'oos'
    || mode === 'clarification'
    || /ngoài phạm vi|không có nguồn|không đủ dữ liệu|cụ thể hơn|chưa rõ/i.test(answer || '');
}

function scrollMessages() {
  const messages = el('messages');
  messages.scrollTop = messages.scrollHeight;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#096;');
}

function displaySourceTitle(title) {
  const cleaned = String(title || 'Nguồn tham khảo')
    .replace(/^Nguồn\s+Stage[\w-]+:\s*/i, '')
    .replace(/^timeline\b/i, 'Dòng thời gian')
    .trim();
  return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : 'Nguồn tham khảo';
}

