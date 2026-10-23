// Read-only client. Only GET requests and an inbound-only WebSocket exist.
export type Dict = Record<string, any>;

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: "GET" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

export const api = {
  experiments: () => get<Dict[]>("/api/experiments"),
  experiment: (id: string) => get<Dict>(`/api/experiments/${id}`),
  bars: (id: string) => get<{ bars: number[][] }>(`/api/experiments/${id}/bars`),
  candidates: (id: string) => get<Dict[]>(`/api/experiments/${id}/candidates`),
  chromosomes: (id: string) => get<Dict[]>(`/api/experiments/${id}/chromosomes`),
  final: (id: string) => get<{ final: Dict | null; holdout: Dict[] }>(`/api/experiments/${id}/final`),
};

export function streamEvents(id: string, onEvent: (e: Dict) => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws: WebSocket | null = null;
  let last = 0;
  let closed = false;
  const connect = () => {
    ws = new WebSocket(`${proto}://${location.host}/ws/${id}?after=${last}`);
    ws.onmessage = (m) => {
      const e = JSON.parse(m.data);
      last = e.seq;
      onEvent(e);
    };
    ws.onclose = () => {
      if (!closed) setTimeout(connect, 2000); // reconnect resumes from last seq
    };
  };
  connect();
  return () => {
    closed = true;
    ws?.close();
  };
}

export const fmt = (v: unknown, d = 4): string => {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(d);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
};
