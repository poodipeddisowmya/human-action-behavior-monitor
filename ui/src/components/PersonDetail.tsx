import { useEffect, useState } from "react";
import { api, PersonDetail as PD } from "../services/api";
import ActivityTimeline from "./ActivityTimeline";
import { format } from "date-fns";

export default function PersonDetail({ personId }: { personId: string }) {
  const [detail, setDetail] = useState<PD | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api.getPerson(personId).then((d) => { if (!cancelled) setDetail(d); }).catch(() => {});
    load();
    const t = setInterval(load, 3000);
    return () => { cancelled = true; clearInterval(t); };
  }, [personId]);

  if (!detail) {
    return <div className="card p-4 text-sm text-muted">Loading {personId}…</div>;
  }

  const current = detail.history[0];
  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-baseline gap-2">
          <h2 className="font-mono text-lg font-semibold">{detail.id}</h2>
          <span className="text-xs text-muted">
            first seen {format(new Date(detail.first_seen), "HH:mm:ss")}
          </span>
        </div>
      </div>

      {current && (
        <div className="px-4 py-4 border-b border-border grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-muted text-xs uppercase tracking-wide">Current</div>
            <div className="mt-1 font-medium capitalize">{current.label}</div>
          </div>
          <div>
            <div className="text-muted text-xs uppercase tracking-wide">Confidence</div>
            <div className="mt-1 font-medium">{Math.round(current.confidence * 100)}%</div>
          </div>
          <div>
            <div className="text-muted text-xs uppercase tracking-wide">Last seen</div>
            <div className="mt-1 font-medium">
              {format(new Date(detail.last_seen), "HH:mm:ss")}
            </div>
          </div>
        </div>
      )}

      <ActivityTimeline history={detail.history} personId={detail.id} />
    </div>
  );
}
