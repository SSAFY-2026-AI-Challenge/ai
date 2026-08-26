const rosterBody = document.querySelector("#roster-body");
const rosterStatus = document.querySelector("#roster-status");
const dossier = document.querySelector("#dossier");
const simForm = document.querySelector("#sim-form");
const simResult = document.querySelector("#sim-result");

const FEATURE_META = [
  { key: "spending_trend", label: "지출 추이", invert: true, format: (v) => Number(v).toFixed(2) },
  { key: "homework_rate", label: "숙제 완료율", invert: false, format: (v) => pct(v) },
  { key: "attendance_rate", label: "출결률", invert: false, format: (v) => pct(v) },
];

function pct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function barWidth(key, value) {
  return Math.min(100, Number(value) * 100);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : "요청에 실패했습니다.");
  }
  return data;
}

function renderBars(features) {
  return FEATURE_META.map((meta) => {
    const value = features[meta.key];
    const width = barWidth(meta.key, value);
    const warn = meta.invert && width > 45;
    return `<div class="bar-row">
      <span><b>${meta.label}</b><i>${meta.format(value)}</i></span>
      <div class="track"><i class="${warn ? "warn" : ""}" style="width:${width}%"></i></div>
    </div>`;
  }).join("");
}

function renderReport(payload, extraTitle) {
  const recs = (payload.recommendations || [])
    .map(
      (item) => `<li>
        <strong>${item.recommendation_type}</strong>
        <div>${item.content}</div>
        <small>${item.expected_effect}</small>
      </li>`
    )
    .join("");
  return `
    <div class="seal">
      <div>
        <small>${extraTitle || payload.name}</small>
        <strong>${payload.credit_score}</strong>
        <div class="grade">${payload.grade} ${payload.grade_code}</div>
      </div>
    </div>
    <p>${payload.summary}</p>
    <div class="bars">${renderBars(payload.features)}</div>
    <h3>이렇게 해보세요</h3>
    <ul class="recs">${recs}</ul>
  `;
}

function renderRoster(rows) {
  rosterBody.innerHTML = rows
    .map(
      (row, index) => `<tr data-id="${row.member_id}">
        <td>${index + 1}</td>
        <td>${row.name}</td>
        <td>${row.job || "-"}</td>
        <td>${row.credit_score}</td>
        <td><span class="grade">${row.grade}</span></td>
      </tr>`
    )
    .join("");
  rosterStatus.textContent = `${rows.length}명의 신용점수를 딥러닝 모델로 계산했습니다.`;
}

async function loadStatus() {
  try {
    const status = await fetchJson("/v1/status");
    document.querySelector("#mode-pill").textContent = status.using_database ? "DB 연동" : "오프라인 체험";
    document.querySelector("#model-pill").textContent = status.model_loaded ? "MLP 준비됨" : "모델 없음";
  } catch (error) {
    document.querySelector("#mode-pill").textContent = "오류";
    document.querySelector("#model-pill").textContent = error.message;
  }
}

async function loadClass() {
  rosterStatus.textContent = "채점하는 중…";
  try {
    const rows = await fetchJson("/v1/class");
    renderRoster(rows);
    if (rows[0]) showStudent(rows[0].member_id);
  } catch (error) {
    rosterStatus.innerHTML = `<span class="error">${error.message}</span>`;
  }
}

async function showStudent(memberId) {
  [...rosterBody.querySelectorAll("tr")].forEach((row) => {
    row.classList.toggle("is-selected", row.dataset.id === String(memberId));
  });
  dossier.innerHTML = `<p class="hint">리포트를 여는 중…</p>`;
  try {
    const payload = await fetchJson(`/v1/score/member/${memberId}`, { method: "POST" });
    dossier.innerHTML = renderReport(payload);
  } catch (error) {
    dossier.innerHTML = `<p class="error">${error.message}</p>`;
  }
}

function readSim() {
  const data = new FormData(simForm);
  const spending = Number(data.get("spending_trend"));
  const homework = Number(data.get("homework_rate"));
  const attendance = Number(data.get("attendance_rate"));
  document.querySelector("#v-spending").textContent = spending.toFixed(2);
  document.querySelector("#v-homework").textContent = pct(homework);
  document.querySelector("#v-attendance").textContent = pct(attendance);
  return {
    spending_trend: spending,
    homework_rate: homework,
    attendance_rate: attendance,
  };
}

let simTimer = 0;
async function runSim() {
  const body = readSim();
  try {
    const payload = await fetchJson("/v1/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    simResult.innerHTML = renderReport({ ...payload, name: "가상 학생" }, "시뮬레이터");
  } catch (error) {
    simResult.innerHTML = `<p class="error">${error.message}</p>`;
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((item) => item.classList.remove("is-active"));
    tab.classList.add("is-active");
    document.querySelector(`#panel-${tab.dataset.tab}`).classList.add("is-active");
    if (tab.dataset.tab === "simulator") runSim();
  });
});

rosterBody.addEventListener("click", (event) => {
  const row = event.target.closest("tr");
  if (row) showStudent(row.dataset.id);
});

document.querySelector("#rescore-btn").addEventListener("click", async () => {
  rosterStatus.textContent = "학급 전체를 다시 채점하는 중…";
  try {
    await fetchJson("/v1/score/class", { method: "POST" });
    await loadClass();
  } catch (error) {
    rosterStatus.innerHTML = `<span class="error">${error.message}</span>`;
  }
});

simForm.addEventListener("input", () => {
  readSim();
  clearTimeout(simTimer);
  simTimer = setTimeout(runSim, 180);
});

loadStatus();
loadClass();
readSim();
