import { useState, useMemo } from "react";
import { Search } from "lucide-react";
import clsx from "clsx";
import { formatDistanceToNow } from "date-fns";
import { PersonSummary } from "../services/api";

interface Props {
  persons: PersonSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function PersonList({ persons, selected, onSelect }: Props) {
  const [q, setQ] = useState("");
  const filtered = useMemo(
    () => persons.filter(p => p.id.toLowerCase().includes(q.toLowerCase())),
    [persons, q],
  );

  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h2 className="font-semibold">People ({persons.length})</h2>
      </div>
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by ID…"
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
              <span className="font-mono font-medium">{p.id}</span>
              <span className="text-xs text-muted">
                {formatDistanceToNow(new Date(p.last_seen))} ago
              </span>
            </div>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="px-4 py-6 text-center text-sm text-muted">No people yet.</li>
        )}
      </ul>
    </div>
  );
}
