#!/usr/bin/env bash
# GCE 셀프호스티드 러너에서 코드 동기화, 학습, 서비스 기동을 수행합니다.
set -euo pipefail

APP_DIR="${CREDIT_AI_HOME:-$HOME/credit-ai}"
PORT="${APP_PORT:-48765}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[credit-ai] %s\n' "$*"
}

ensure_path() {
  export PATH="${HOME}/.local/bin:${PATH}"
}

ensure_uv() {
  ensure_path
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  log "uv 설치"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ensure_path
}

sync_code() {
  mkdir -p "${APP_DIR}/models" "${APP_DIR}/logs"
  if [[ "$(cd "${SOURCE_DIR}" && pwd)" == "$(cd "${APP_DIR}" && pwd)" ]]; then
    log "소스와 배포 디렉터리가 같습니다. 복사를 건너뜁니다."
    return
  fi
  log "코드를 ${APP_DIR} 로 복사"
  tar -C "${SOURCE_DIR}" \
    --exclude .git \
    --exclude .venv \
    --exclude .env \
    --exclude logs \
    --exclude models/credit_mlp.pt \
    -cf - . | tar -C "${APP_DIR}" -xf -
}

ensure_env() {
  if [[ -f "${APP_DIR}/.env" ]]; then
    log "기존 .env 를 유지합니다."
    return
  fi
  log "배포용 .env 생성 (APP_MODE=db)"
  sed 's/^APP_MODE=offline$/APP_MODE=db/' "${SOURCE_DIR}/.env.example" > "${APP_DIR}/.env"
}

sync_deps() {
  ensure_uv
  cd "${APP_DIR}"
  log "의존성 설치"
  uv python install 3.12
  uv sync --no-dev
}

model_missing() {
  [[ ! -f "${APP_DIR}/models/credit_mlp.pt" ]]
}

run_train() {
  ensure_uv
  cd "${APP_DIR}"
  log "모델 학습 시작"
  uv run credit-ai train "$@"
  log "모델 학습 완료"
}

stop_serve() {
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  systemctl --user stop credit-ai >/dev/null 2>&1 || true
  if [[ -f "${APP_DIR}/credit-ai.pid" ]]; then
    local pid
    pid="$(cat "${APP_DIR}/credit-ai.pid" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "기존 프로세스 종료 (pid ${pid})"
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${APP_DIR}/credit-ai.pid"
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
}

start_systemd() {
  local uid dest
  uid="$(id -u)"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
  mkdir -p "${HOME}/.config/systemd/user"
  dest="${HOME}/.config/systemd/user/credit-ai.service"
  sed "s|__APP_DIR__|${APP_DIR}|g" "${APP_DIR}/deploy/credit-ai.service" > "${dest}"
  systemctl --user daemon-reload >/dev/null 2>&1 || return 1
  systemctl --user enable --now credit-ai >/dev/null 2>&1 || return 1
  log "systemd user 서비스로 기동"
  wait_healthy
}

start_serve() {
  ensure_uv
  cd "${APP_DIR}"
  mkdir -p "${APP_DIR}/logs"
  stop_serve
  if start_systemd; then
    return
  fi
  log "서비스 기동 (0.0.0.0:${PORT})"
  nohup uv run credit-ai serve --host 0.0.0.0 --port "${PORT}" \
    >> "${APP_DIR}/logs/serve.log" 2>&1 &
  echo $! > "${APP_DIR}/credit-ai.pid"
  wait_healthy
}

wait_healthy() {
  local i
  for i in {1..40}; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "헬스체크 성공"
      curl -sS "http://127.0.0.1:${PORT}/v1/status" || true
      printf '\n'
      return 0
    fi
    sleep 2
  done
  log "헬스체크 실패. 최근 로그:"
  tail -n 80 "${APP_DIR}/logs/serve.log" || true
  journalctl --user -u credit-ai -n 80 --no-pager 2>/dev/null || true
  return 1
}

cmd_deploy() {
  local force_train="${FORCE_TRAIN:-false}"
  sync_code
  ensure_env
  sync_deps
  if [[ "${force_train}" == "true" ]] || model_missing; then
    run_train
  else
    log "학습된 모델이 있어 학습을 건너뜁니다. 강제 학습은 FORCE_TRAIN=true 입니다."
  fi
  start_serve
}

cmd_train() {
  sync_code
  ensure_env
  sync_deps
  run_train "$@"
  start_serve
}

cmd_restart() {
  start_serve
}

usage() {
  cat <<'EOF'
사용법:
  bash deploy/gcp.sh deploy     # 코드 배포, 모델 없으면 학습, 서비스 기동
  bash deploy/gcp.sh train ...  # 학습 후 서비스 재시작 (credit-ai train 인자 전달)
  bash deploy/gcp.sh restart    # 서비스만 재시작
EOF
}

main() {
  local cmd="${1:-deploy}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
  case "${cmd}" in
    deploy) cmd_deploy ;;
    train) cmd_train "$@" ;;
    restart) cmd_restart ;;
    -h|--help|help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
