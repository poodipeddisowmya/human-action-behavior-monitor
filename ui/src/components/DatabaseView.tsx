import { useCallback, useEffect, useRef, useState } from "react";
import { format } from "date-fns";
import clsx from "clsx";
import { Database, HardDrive, Images, Radio, Rows3, UserCheck, Users } from "lucide-react";
import { api, DatabaseRow, DatabaseStatus, EventMessage } from "../services/api";
import { useWebSocket } from "../hooks/useWebSocket";
import EvidenceViewer, { EvidenceThumbs } from "./EvidenceViewer";

// Someone in view right now and what they are doing: an activity that is
// still going on and is written to the database when it ends.
interface Open { person: string; name: string | null; label: string; since: number }

const size = (b: number | null | undefined) =>
  b == null ? "—" : b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(1)} GB`
    : b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${Math.round(b / 1024)} kB`;

function Tile({ icon: Icon, label, value, sub }: { icon: any; label: string; value: string | number; sub?: string }) {
  return (
    <div className="card px-4 py-3">
      <div className="flex items-center gap-2 text-xs text-muted"><Icon size={14} className="text-accent" />{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-muted truncate" title={sub}>{sub}</div>}
    </div>
  );
}

/** The database filling up as the camera watches: for showing a client what
 *  is stored, when, and where. Mounted only while its tab is open. */
export default function DatabaseView() {
  const [db, setDb] = useState<DatabaseStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fresh, setFresh] = useState<Set<number>>(new Set());
  const [open, setOpen] = useState<Record<string, Open>>({});
  const [now, setNow] = useState(Date.now());
  const [viewing, setViewing] = useState<{ event: EventMessage; index: number } | null>(null);
  const seen = useRef<Set<number> | null>(null);
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(() => {
    api.database(30).then((d) => {
      // Rows not in the previous answer were just written: flash them.
      if (seen.current) {
        const added = d.rows.filter((r) => !seen.current!.has(r.id)).map((r) => r.id);
        if (added.length) {
          setFresh((f) => new Set([...f, ...added]));
          setTimeout(() => setFresh((f) => { const n = new Set(f); added.forEach((i) => n.delete(i)); return n; }), 4000);
        }
      }
      seen.current = new Set(d.rows.map((r) => r.id));
      setDb(d);
      setError(null);
    }).catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);              // sizes, counts, other writers
    const c = setInterval(() => setNow(Date.now()), 1000);
    return () => { clearInterval(t); clearInterval(c); };
  }, [load]);

  useWebSocket((msg) => {
    if (msg.type === "event") {
      // A row was just written: show it now, not at the next poll. Several
      // people finishing at once share one reload.
      if (pending.current) clearTimeout(pending.current);
      pending.current = setTimeout(load, 250);
    } else if (msg.type === "frame_annotations") {
      setOpen((prev) => {
        const next: Record<string, Open> = {};
        for (const a of msg.annotations) {
          if (!a.label || String(a.label).startsWith("detecting")) continue;
          const person = `P${String(a.person_id).padStart(3, "0")}`;
          const was = prev[person];
          next[person] = was && was.label === a.label
            ? { ...was, name: a.name ?? was.name }
            : { person, name: a.name ?? null, label: a.label, since: Date.now() };
        }
        return next;
      });
    }
  });

  const openList = Object.values(open).sort((a, b) => a.since - b.since);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2"><Database size={20} className="text-accent" /> Database, live</h2>
          <p className="text-sm text-muted max-w-3xl">
            Each activity is written as one row the moment it ends (the person starts doing something
            else, or leaves), with who did it, when, for how long, how sure, which camera, and
            2–3 proof photos. New rows flash as they arrive.
          </p>
        </div>
        {db && (
          <div className="text-xs text-muted text-right">
            <div><span className="text-slate-200">{db.database.engine}</span> · {db.database.file ?? "server database"}</div>
            <div>Photos in {db.evidence.folder}, kept {db.evidence.retention_days ? `${db.evidence.retention_days} days` : "forever"}</div>
          </div>
        )}
      </div>

      {error && <div className="card px-4 py-3 text-sm text-danger">Database not reachable: {error}</div>}

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <Tile icon={Rows3} label="Activities stored" value={db?.tables.actions ?? "—"} sub="rows in actions" />
        <Tile icon={Radio} label="Written, last minute" value={db?.written_last_minute ?? "—"}
          sub={db ? `${db.written_last_hour} in the last hour` : undefined} />
        <Tile icon={Users} label="People tracked" value={db?.tables.persons ?? "—"} sub="rows in persons" />
        <Tile icon={UserCheck} label="Named activities" value={db?.named_actions ?? "—"}
          sub={db ? `${db.tables.known_people} enrolled, ${db.tables.face_samples} face photos` : undefined} />
        <Tile icon={Images} label="Proof photos" value={db?.evidence.photos ?? "—"} sub={size(db?.evidence.bytes)} />
        <Tile icon={HardDrive} label="Database size" value={size(db?.database.bytes)}
          sub={db?.last_write ? `last write ${format(new Date(db.last_write), "HH:mm:ss")}` : "nothing written yet"} />
      </div>

      <div className="grid grid-cols-12 gap-6">
        <section className="col-span-12 xl:col-span-5 space-y-6">
          <div className="card overflow-hidden">
            <div className="px-4 py-3 border-b border-border font-semibold">What the camera sees</div>
            <div className="bg-black aspect-video">
              <img src={api.videoUrl} alt="live video" className="w-full h-full object-contain" />
            </div>
          </div>
          <div className="card">
            <div className="px-4 py-3 border-b border-border">
              <h3 className="font-semibold">Being recorded now</h3>
              <p className="text-xs text-muted">Activities still going on. Each becomes a row when it ends.</p>
            </div>
            <ul className="divide-y divide-border text-sm">
              {openList.length === 0 && <li className="px-4 py-4 text-muted">Nobody in view.</li>}
              {openList.map((o) => (
                <li key={o.person} className="px-4 py-2 flex items-center gap-3">
                  <span className="live-dot shrink-0" />
                  <span className="font-mono text-xs text-muted w-12">{o.person}</span>
                  <span className="w-24 truncate">{o.name ?? <span className="text-muted">Unknown</span>}</span>
                  <span className="capitalize flex-1">{o.label}</span>
                  <span className="tabular-nums text-muted text-xs">{Math.max(0, Math.round((now - o.since) / 1000))} s</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="col-span-12 xl:col-span-7 card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Table <span className="font-mono text-accent">actions</span>, newest first</h3>
              <p className="text-xs text-muted">As stored. The same rows are served by <span className="font-mono">GET /api/events</span> and pushed live over <span className="font-mono">/ws</span>.</p>
            </div>
          </div>
          <div className="max-h-[640px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10 bg-surface text-left text-xs text-muted font-mono">
                <tr>
                  {["id", "written", "person", "activity", "duration", "confidence", "camera", "photos"].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {db?.rows.length === 0 && (
                  <tr><td colSpan={8} className="px-4 py-8 text-center text-muted">The table is empty. Rows appear here as activities end.</td></tr>
                )}
                {db?.rows.map((r: DatabaseRow) => (
                  <tr key={r.id} className={clsx("border-t border-border transition-colors duration-1000",
                    fresh.has(r.id) ? "bg-accent/15" : "hover:bg-elevated")}>
                    <td className="px-3 py-2 font-mono text-xs text-muted">{r.id}</td>
                    <td className="px-3 py-2 font-mono text-xs">{format(new Date(r.end_time), "HH:mm:ss")}</td>
                    <td className="px-3 py-2">
                      {r.identity ? <span className="font-medium">{r.identity}</span> : <span className="text-muted">Unknown</span>}
                      <span className="ml-1 font-mono text-[11px] text-muted">{r.person_id}</span>
                    </td>
                    <td className={clsx("px-3 py-2 capitalize", r.label.startsWith("fall") && "text-danger font-semibold")}>{r.label}</td>
                    <td className="px-3 py-2 tabular-nums text-muted">{r.duration_s.toFixed(1)} s</td>
                    <td className="px-3 py-2 tabular-nums text-muted">{Math.round(r.confidence * 100)}%</td>
                    <td className="px-3 py-2 text-xs text-muted truncate max-w-[9rem]" title={r.source ?? undefined}>{r.source ?? "—"}</td>
                    <td className="px-3 py-1.5"><EvidenceThumbs event={r} onOpen={(index) => setViewing({ event: r, index })} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
      {viewing && <EvidenceViewer event={viewing.event} start={viewing.index} onClose={() => setViewing(null)} />}
    </div>
  );
}
