import { useEffect, useState } from "react";
import { Cpu, Gauge } from "lucide-react";
import { api, Metrics } from "../services/api";

// "NVIDIA Tesla T4" -> "Tesla T4"
const shortGpu = (name: string) => name.replace(/^NVIDIA\s+(GeForce\s+)?/i, "");

// What the system is running on and how fast, in the header. Answers the
// first question asked in every demo ("is this real time, on what?") with
// live numbers rather than a claim.
export default function SystemStatus() {
  const [m, setM] = useState<Metrics | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const poll = () => api.metrics()
      .then((r) => { setM(r); setDown(false); })
      .catch(() => setDown(true));
    poll();
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, []);

  if (down) {
    return <span className="text-xs font-medium text-danger">Server unreachable — retrying…</span>;
  }
  if (!m) return null;

  const gpu = m.gpu ? shortGpu(m.gpu) : null;
  return (
    <div className="flex items-center gap-3 text-xs text-muted" title={m.detector ? `Detector: ${m.detector}` : undefined}>
      <span className={"flex items-center gap-1.5 rounded-md border px-2 py-1 " +
        (gpu ? "border-good/40 text-good" : "border-border")}>
        <Cpu size={13} /> {gpu ? `GPU · ${gpu}` : "CPU"}
      </span>
      {m.playback_fps ? (
        <span className="flex items-center gap-1.5">
          <Gauge size={13} /> {m.playback_fps.toFixed(0)} fps
          {m.detect_ms ? ` · ${m.detect_ms.toFixed(0)} ms` : ""}
        </span>
      ) : null}
      <span>v{m.version}</span>
    </div>
  );
}
