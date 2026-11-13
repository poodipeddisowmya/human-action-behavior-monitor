import { useEffect, useRef } from "react";

export type WsMessage =
  | { type: "frame_annotations"; annotations: any[];
      caption?: { text: string | null; scene: string | null } | null }
  | { type: "event"; person_id: string; label: string;
      start_time: string; end_time: string; duration_s: number; confidence: number };

// Reconnects on its own. A dropped socket (server restart, a tunnel or
// Wi-Fi blip) used to leave the dashboard silently frozen until a reload.
export function useWebSocket(onMessage: (msg: WsMessage) => void) {
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    let closed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => { delay = 1000; };
      ws.onmessage = (e) => {
        try { cbRef.current(JSON.parse(e.data)); } catch {}
      };
      ws.onclose = () => {
        if (closed) return;
        retry = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 15000);
      };
    };
    connect();

    const ping = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) ws.send("ping");
    }, 25000);

    return () => {
      closed = true;
      clearInterval(ping);
      clearTimeout(retry);
      ws?.close();
    };
  }, []);
}
