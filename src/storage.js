const KEY = "chatgpt_clone";

export function loadState() {
  const raw = localStorage.getItem(KEY);
  return raw
    ? JSON.parse(raw)
    : { conversations: {}, activeId: null };
}

export function saveState(state) {
  localStorage.setItem(KEY, JSON.stringify(state));
}
