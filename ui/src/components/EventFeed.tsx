import { format } from "date-fns";
import { EventMessage } from "../services/api";
import clsx from "clsx";

const SEVERITY: Record<string, string> = {
  falling: "text-danger",
  lying:   "text-warn",
};

interface Props {
  events: EventMessage[];
  onSelect: (id: string) => void;
}

export default function EventFeed({ events, onSelect }: Props) {
  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h2 className="font-semibold">Recent events</h2>
        <span className="text-xs text-muted">{events.length}</span>
      </div>
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-muted bg-surface sticky top-0">
            <tr>
              <th className="px-4 py-2 font-medium">Person</th>
              <th className="px-4 py-2 font-medium">Action</th>
              <th className="px-4 py-2 font-medium">Start</th>
              <th className="px-4 py-2 font-medium">Duration</th>
              <th className="px-4 py-2 font-medium">Conf.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {events.map((e, i) => (
              <tr
                key={i}
                onClick={() => onSelect(e.person_id)}
                className="hover:bg-surface cursor-pointer"
              >
                <td className="px-4 py-2 font-mono">{e.person_id}</td>
                <td className={clsx("px-4 py-2 capitalize", SEVERITY[e.label])}>
                  {e.label.replace("_", " ")}
                </td>
                <td className="px-4 py-2 text-muted font-mono">
                  {format(new Date(e.start_time), "HH:mm:ss")}
                </td>
                <td className="px-4 py-2 text-muted">{e.duration_s.toFixed(1)}s</td>
                <td className="px-4 py-2 text-muted">{Math.round(e.confidence * 100)}%</td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-6 text-center text-muted">
                  Waiting for the first event…
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
