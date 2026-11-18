export interface PersonSummary {
  id: string;
  first_seen: string;
  last_seen: string;
}

export interface ActionHistoryEntry {
  id?: number;
  label: string;
  start_time: string;
  end_time: string;
  duration_s: number;
  confidence: number;
  evidence?: string[];       // proof image URLs (start, middle, end)
}

export interface PersonDetail extends PersonSummary {
  history: ActionHistoryEntry[];
}

export interface EventMessage {
  id?: number;
  person_id: string;
  label: string;
  start_time: string;
  end_time: string;
  duration_s: number;
  confidence: number;
  evidence?: string[];       // proof image URLs (start, middle, end)
}

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export interface Clip {
  filename: string;
  name: string;
  description: string;
  size_kb: number;
}
export interface Camera {
  filename: string;      // "camera:0"
  name: string;
  description: string;
}
export interface SourcesResponse {
  clips: Clip[];
  cameras?: Camera[];
  current: string | null;
  browser_camera?: string;   // "browser:camera" — the viewer's own webcam
}

export interface Metrics {
  version: string;
  device: string;
  gpu: string | null;
  playback_fps?: number;
  detect_ms?: number;
  tracked?: number;
  gpu_mem_mb?: number;
  gpu_mem_total_mb?: number;
  pipeline_alive: boolean;
  detector?: string;
}

// Recorded CPU-vs-GPU comparison (/api/showcase).
export interface ShowcaseSide {
  video: string;
  share: number | null;      // fraction of on-screen labels that were correct
  first_s: number | null;    // seconds until the correct label first appeared
  fps: number | null;
  people: number | null;     // people tracked per frame, average
  top: Record<string, number>;
}
export interface ShowcaseClip {
  name: string;
  expected: string | null;
  featured: boolean;
  left: ShowcaseSide | null;
  right: ShowcaseSide;
  extra?: Record<string, ShowcaseSide | null>;   // further devices, by key
  new_activity?: boolean;    // recorded on the GPUs only (added after the CPU run)
}
export interface ShowcaseTotals {
  clear: number;
  appeared: number;          // clips where the correct label was shown at all
  mean_share: number | null;
  median_first_s: number | null;
  median_fps: number | null;
}
export interface Showcase {
  available: boolean;
  left: { tag: string; sub: string };
  right: { tag: string; sub: string };
  recorded: string;
  seconds_per_clip: number;
  labelled_clips: number;
  totals: { left: ShowcaseTotals; right: ShowcaseTotals };
  // Further devices recorded on the same clips (scripts/add_showcase_side.py).
  extra?: { key: string; tag: string; sub: string; short: string; recorded: string | null; totals: ShowcaseTotals }[];
  // Activities added after the CPU recording, totalled per GPU side.
  new_activities?: { clips: number; totals: Record<string, ShowcaseTotals> } | null;
  clips: ShowcaseClip[];
}

export const BROWSER_CAMERA = "browser:camera";
export const isLive = (s: string | null | undefined) =>
  !!s && (s.startsWith("camera:") || s === BROWSER_CAMERA);

export const api = {
  listPersons: () => json<PersonSummary[]>("/api/persons"),
  getPerson:   (id: string) => json<PersonDetail>(`/api/persons/${id}`),
  recentEvents: (limit = 50) => json<EventMessage[]>(`/api/events?limit=${limit}`),
  videoUrl:    "/api/stream/video",

  metrics: () => json<Metrics>("/api/metrics"),
  showcase: () => json<Showcase>("/api/showcase"),
  showcaseVideo: (side: string, file: string) =>
    `/api/showcase/video/${side}/${encodeURIComponent(file)}`,
  listSources: () => json<SourcesResponse>("/api/sources"),
  switchSource: async (filename: string) => {
    const r = await fetch("/api/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },
  uploadSource: async (file: File, onProgress?: (pct: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    // fetch doesn't expose upload progress, so use XHR when a progress cb is given.
    if (!onProgress) {
      const r = await fetch("/api/sources/upload", { method: "POST", body: form });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    }
    return new Promise<any>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/sources/upload");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
      xhr.onload = () => xhr.status < 400
        ? resolve(JSON.parse(xhr.responseText))
        : reject(new Error(`${xhr.status} ${xhr.statusText}`));
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.send(form);
    });
  },
};
