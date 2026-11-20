import { useState, useMemo } from "react";
import { Search } from "lucide-react";
import clsx from "clsx";
import { formatDistanceToNow } from "date-fns";
import { PersonSummary } from "../services/api";

interface Props {
  persons: PersonSummary[];
  names: Record<string, string>;   // track ID -> enrolled name, when recognised
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function PersonList({ persons, names, selected, onSelect }: Props) {
  const [q, setQ] = useState("");
  // Track IDs restart with every video or camera session, so the full table
  // fills up with past sessions' people: show the last 10 minutes by default.
  const [all, setAll] = useState(false);
  const recent = useMemo(
    () => persons.filter(p => Date.now() - new Date(p.last_seen).getTime() < 10 * 60_000),
    [persons],
  );
  const filtered = useMemo(
    () => (all ? persons : recent).filter(p =>
      `${p.id} ${names[p.id] ?? ""}`.toLowerCase().includes(q.toLowerCase())),
    [persons, recent, all, q, names],
  );

  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h2 className="font-semibold">People {all ? `(${persons.length})` : `seen recently (${recent.length})`}</h2>
        <button onClick={() => setAll(!all)} className="text-xs text-accent hover:underline">
          {all ? "Recent only" : "Show all"}
        </button>
      </div>
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by name or ID…"
            className="w-full bg-surface border border-border rounded-lg pl-9 pr-3 py-2 text-sm outline-none focus:border-accent"
          />
        </div>
      </div>
      <ul className="max-h-[360px] overflow-auto divide-y divide-border">
        {filtered.map((p) => (
          <li
            key={p.id}
            onClick={() => onSelect(p.id)}
            className={clsx(
              "px-4 py-3 cursor-pointer hover:bg-surface transition",
              selected === p.id && "bg-surface border-l-2 border-accent",
            )}
          >
            <div className="flex justify-between items-baseline">
              <span>
                <span className={names[p.id] ? "font-medium" : "text-muted"}>{names[p.id] ?? "Unknown"}</span>
                <span className="ml-2 font-mono text-xs text-muted">{p.id}</span>
              </span>
              <span className="text-xs text-muted">
                {formatDistanceToNow(new Date(p.last_seen))} ago
              </span>
            </div>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="px-4 py-6 text-center text-sm text-muted">
            {all ? "No people yet." : "No one in the last 10 minutes."}
          </li>
        )}
      </ul>
    </div>
  );
}
