import { useEffect, useState } from "react";
import { format } from "date-fns";
import { ChevronLeft, ChevronRight, ImageOff, UserCheck, X } from "lucide-react";
import { api, EventMessage, KnownPerson } from "../services/api";
import { knownPeopleChanged } from "./KnownPeople";

// Name the face in this proof image: an enrolled person, or a new one.
function EnrolFace({ event, index }: { event: EventMessage; index: number }) {
  const [people, setPeople] = useState<KnownPerson[]>([]);
  const [choice, setChoice] = useState<string>("");
  const [newName, setNewName] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { api.knownPeople().then(setPeople).catch(() => {}); }, []);
  if (event.id == null) return null;

  const enrol = async () => {
    setBusy(true); setMsg(null);
    let created: KnownPerson | null = null;
    try {
      let id = Number(choice);
      if (choice === "new") {
        created = await api.addKnownPerson(newName.trim());
        id = created.id;
      }
      await api.enrolFromEvidence(id, event.id!, index);
      const who = created?.name ?? people.find((p) => p.id === id)?.name;
      setMsg({ ok: true, text: `Enrolled as ${who}. They will be named from now on.` });
      knownPeopleChanged.dispatchEvent(new Event("changed"));
    } catch (e) {
      if (created) await api.deleteKnownPerson(created.id).catch(() => {});   // nothing enrolled: undo
      setMsg({ ok: false, text: (e as Error).message });
    }
    setBusy(false);
  };
  const ready = choice && (choice !== "new" || newName.trim());
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-border px-4 py-2 text-xs">
      <UserCheck size={14} className="text-muted" />
      <span className="text-muted">This face is</span>
      <select value={choice} onChange={(e) => setChoice(e.target.value)}
        className="rounded border border-border bg-surface px-2 py-1 outline-none focus:border-accent">
        <option value="">choose…</option>
        {people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        <option value="new">new person…</option>
      </select>
      {choice === "new" && (
        <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Name"
          className="w-32 rounded border border-border bg-surface px-2 py-1 outline-none focus:border-accent" />
      )}
      <button onClick={enrol} disabled={!ready || busy}
        className="rounded bg-accent/90 px-2.5 py-1 font-medium text-white disabled:opacity-40">
        {busy ? "Enrolling…" : "Enrol face"}
      </button>
      {msg && <span className={msg.ok ? "text-good" : "text-warn"}>{msg.text}</span>}
    </div>
  );
}

// Proof images of one logged activity: the start, the middle and the end of
// it, as saved by the pipeline (src/storage/evidence.py).

export function EvidenceThumbs({ event, onOpen }: {
  event: EventMessage; onOpen: (index: number) => void;
}) {
  const urls = event.evidence ?? [];
  if (urls.length === 0) return <span className="text-muted text-xs">—</span>;
  return (
    <div className="flex gap-1">
      {urls.map((u, i) => (
        <button key={u} onClick={(e) => { e.stopPropagation(); onOpen(i); }}
          aria-label={`Open proof image ${i + 1} of ${urls.length}`}
          className="h-9 w-9 overflow-hidden rounded border border-border hover:border-accent focus:outline-none focus:ring-1 focus:ring-accent">
          <Thumb src={u} />
        </button>
      ))}
    </div>
  );
}

function Thumb({ src }: { src: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return <span className="flex h-full w-full items-center justify-center bg-elevated text-muted"><ImageOff size={12} /></span>;
  }
  return <img src={src} loading="lazy" alt="" onError={() => setFailed(true)}
    className="h-full w-full object-cover" />;
}

export default function EvidenceViewer({ event, start, onClose }: {
  event: EventMessage; start: number; onClose: () => void;
}) {
  const urls = event.evidence ?? [];
  const [i, setI] = useState(start);
  const [failed, setFailed] = useState(false);
  const go = (d: number) => { setFailed(false); setI((x) => (x + d + urls.length) % urls.length); };

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  const when = ["start", "middle", "end"];
  const stage = urls.length === 3 ? when[i] : urls.length === 2 ? ["start", "end"][i] : "";
  return (
    <div role="dialog" aria-modal="true" aria-label="Activity proof images"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}>
      <div className="card w-full max-w-3xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
          <div className="min-w-0 text-sm">
            {event.identity
              ? <><span className="font-semibold">{event.identity}</span> <span className="font-mono text-muted">({event.person_id})</span></>
              : <span className="font-mono">{event.person_id}</span>}
            <span className="mx-2 text-muted">·</span>
            <span className="font-medium capitalize">{event.label}</span>
            <span className="mx-2 text-muted">·</span>
            <span className="text-muted tabular-nums">
              {format(new Date(event.start_time), "d MMM yyyy, HH:mm:ss")} · {event.duration_s.toFixed(1)} s ·{" "}
              {Math.round(event.confidence * 100)}%
            </span>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded p-1 text-muted hover:text-slate-100">
            <X size={18} />
          </button>
        </div>
        <div className="relative bg-black">
          {failed ? (
            <div className="flex aspect-video items-center justify-center gap-2 text-sm text-muted">
              <ImageOff size={16} /> Image no longer kept (past the retention period)
            </div>
          ) : (
            <img key={urls[i]} src={urls[i]} alt={`${event.label}, ${stage}`} onError={() => setFailed(true)}
              className="mx-auto block max-h-[70vh] w-auto object-contain" />
          )}
          {urls.length > 1 && (
            <>
              <button onClick={() => go(-1)} aria-label="Previous image"
                className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-black/60 p-1.5 text-slate-100 hover:bg-black/80">
                <ChevronLeft size={20} />
              </button>
              <button onClick={() => go(1)} aria-label="Next image"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-black/60 p-1.5 text-slate-100 hover:bg-black/80">
                <ChevronRight size={20} />
              </button>
            </>
          )}
        </div>
        <div className="flex items-center justify-between px-4 py-2 text-xs text-muted">
          <span>{stage ? `${stage[0].toUpperCase()}${stage.slice(1)} of the activity` : "Proof image"}</span>
          <span className="tabular-nums">{i + 1} / {urls.length}</span>
        </div>
        {!failed && <EnrolFace event={event} index={i} />}
      </div>
    </div>
  );
}
