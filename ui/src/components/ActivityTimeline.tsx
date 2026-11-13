import { format } from "date-fns";
import { ActionHistoryEntry } from "../services/api";
import clsx from "clsx";

const LABEL_COLOR: Record<string, string> = {
  standing: "bg-slate-500",
  walking:  "bg-accent",
  running:  "bg-warn",
  sitting:  "bg-good",
  bending:  "bg-purple-500",
  lying:    "bg-cyan-500",
  falling:  "bg-danger",
  picking_up: "bg-pink-500",
};

export default function ActivityTimeline({ history }: { history: ActionHistoryEntry[] }) {
  if (history.length === 0) {
    return <div className="p-4 text-sm text-muted">No activity recorded yet.</div>;
  }
  return (
    <div className="max-h-[380px] overflow-auto">
      <ul className="p-4 space-y-3">
        {history.map((h, i) => (
          <li key={i} className="flex items-start gap-3">
            <span className={clsx("mt-1.5 w-2.5 h-2.5 rounded-full flex-shrink-0",
                                  LABEL_COLOR[h.label] ?? "bg-slate-500")} />
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-medium capitalize">{h.label.replace("_", " ")}</span>
                <span className="text-xs text-muted font-mono">
                  {format(new Date(h.start_time), "HH:mm:ss")}
                </span>
              </div>
              <div className="text-xs text-muted mt-0.5">
                {h.duration_s.toFixed(1)}s · {Math.round(h.confidence * 100)}% confidence
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
