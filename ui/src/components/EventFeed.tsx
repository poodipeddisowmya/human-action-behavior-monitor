import { useMemo, useState } from "react";
import { format } from "date-fns";
import clsx from "clsx";
import { EventMessage } from "../services/api";
import { NEW_LABELS } from "../services/clips";
import EvidenceViewer, { EvidenceThumbs } from "./EvidenceViewer";

const ALERTS = ["falling down", "falling"];
const SEVERITY: Record<string, string> = {
  falling: "text-danger font-semibold",
  "falling down": "text-danger font-semibold",
  lying: "text-warn",
};

type Filter = "all" | "new" | "named" | "alerts";
const FILTERS: { key: Filter; label: string; test: (e: EventMessage) => boolean }[] = [
  { key: "all", label: "All", test: () => true },
  { key: "new", label: "New activities", test: (e) => NEW_LABELS.includes(e.label) },
  { key: "named", label: "Known people", test: (e) => !!e.identity },
  { key: "alerts", label: "Alerts", test: (e) => ALERTS.includes(e.label) },
];

interface Props {
  events: EventMessage[];
  onSelect: (id: string) => void;
}

export default function EventFeed({ events, onSelect }: Props) {
  const [open, setOpen] = useState<{ event: EventMessage; index: number } | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const counts = useMemo(
    () => Object.fromEntries(FILTERS.map((f) => [f.key, events.filter(f.test).length])) as Record<Filter, number>,
    [events],
  );
  const shown = useMemo(() => events.filter(FILTERS.find((f) => f.key === filter)!.test), [events, filter]);

  return (
    <div id="activity-log" className="card scroll-mt-24">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div>
          <h2 className="font-semibold">Activity log</h2>
          <p className="text-xs text-muted">Each activity with who did it and proof photos. Click a photo to open it.</p>
        </div>
        <div className="flex flex-wrap gap-1" role="tablist" aria-label="Filter the log">
          {FILTERS.map((f) => (
            <button key={f.key} role="tab" aria-selected={filter === f.key} onClick={() => setFilter(f.key)}
              className={clsx(
                "rounded-full border px-2.5 py-1 text-xs transition",
                filter === f.key ? "border-accent bg-accent/15 text-accent" : "border-border text-muted hover:text-slate-100",
              )}>
              {f.label} <span className="tabular-nums opacity-70">{counts[f.key]}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="max-h-[460px] overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-surface text-left text-xs uppercase text-muted">
            <tr>
              <th className="px-4 py-2 font-medium">Person</th>
              <th className="px-4 py-2 font-medium">Activity</th>
              <th className="px-4 py-2 font-medium">Started</th>
              <th className="px-4 py-2 font-medium">Duration</th>
              <th className="px-4 py-2 font-medium">Confidence</th>
              <th className="px-4 py-2 font-medium">Proof</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {shown.map((e, i) => (
              <tr key={e.id ?? i} onClick={() => onSelect(e.person_id)}
                className={clsx("cursor-pointer hover:bg-surface", ALERTS.includes(e.label) && "bg-danger/5")}>
                <td className="px-4 py-2">
                  <div className={e.identity ? "font-medium" : "text-muted"}>{e.identity ?? "Unknown"}</div>
                  <div className="font-mono text-[11px] text-muted">{e.person_id}</div>
                </td>
                <td className={clsx("px-4 py-2 capitalize", SEVERITY[e.label])}>
                  {e.label.replace("_", " ")}
                  {NEW_LABELS.includes(e.label) && (
                    <span className="ml-1.5 rounded bg-accent/20 px-1 align-middle text-[10px] font-semibold not-italic text-accent">NEW</span>
                  )}
                </td>
                <td className="px-4 py-2 font-mono text-muted">
                  {format(new Date(e.start_time), "HH:mm:ss")}
                  {e.source && <div className="font-sans text-[11px] truncate max-w-[10rem]" title={e.source}>{e.source}</div>}
                </td>
                <td className="px-4 py-2 tabular-nums text-muted">{e.duration_s.toFixed(1)} s</td>
                <td className="px-4 py-2 tabular-nums text-muted">{Math.round(e.confidence * 100)}%</td>
                <td className="px-4 py-1.5">
                  <EvidenceThumbs event={e} onOpen={(index) => setOpen({ event: e, index })} />
                </td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-muted">
                  {events.length === 0 ? "Waiting for the first activity…" : "Nothing logged of this kind yet."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {open && <EvidenceViewer event={open.event} start={open.index} onClose={() => setOpen(null)} />}
    </div>
  );
}
