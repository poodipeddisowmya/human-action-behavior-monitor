import { useEffect, useRef, useState } from "react";
import { Check, ScanFace, ShieldCheck } from "lucide-react";
import { api, FaceOnScreen, KnownPerson } from "../services/api";
import { knownPeopleChanged } from "./KnownPeople";

// People in view whose face is clear but who are not enrolled: name one and
// they are recognised from then on, here and on any later video. Their faces
// are held in memory by the server only while they are in view; nothing is
// stored until someone is named (with their consent).

function NameFace({ face, people, v, onDone }: {
  face: FaceOnScreen; people: KnownPerson[]; v: number;
  onDone: (text: string) => void;
}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const save = async () => {
    const n = name.trim().replace(/\s+/g, " ");
    if (!n) return;
    setBusy(true); setErr(null);
    const existing = people.find((p) => p.name.toLowerCase() === n.toLowerCase());
    let created: KnownPerson | null = null;
    try {
      const id = existing?.id ?? (created = await api.addKnownPerson(n)).id;
      const r = await api.enrolFromVideo(id, face.track);
      const who = existing?.name ?? created!.name;
      const parts = [`${face.person_id} is ${who}`, `${r.added.length} face photo${r.added.length === 1 ? "" : "s"} added`];
      if (r.events_named) parts.push(`${r.events_named} earlier ${r.events_named === 1 ? "activity" : "activities"} named`);
      knownPeopleChanged.dispatchEvent(new Event("changed"));
      onDone(parts.join(" · "));
    } catch (e) {
      if (created) await api.deleteKnownPerson(created.id).catch(() => {});   // nothing enrolled: undo
      setErr((e as Error).message);
    }
    setBusy(false);
  };

  return (
    <li className="flex gap-3 px-4 py-3">
      <img src={api.faceOnScreenUrl(face.track, v)} alt={`Face of ${face.person_id}`}
        className="h-14 w-14 shrink-0 rounded-lg border border-border object-cover" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-sm"><span className="text-muted">Unknown</span> <span className="font-mono text-xs text-muted">{face.person_id}</span></span>
          <span className="text-[11px] text-muted tabular-nums" title="Clear faces held for this person">
            {face.faces} clear face{face.faces === 1 ? "" : "s"}
          </span>
        </div>
        <form className="mt-1.5 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); save(); }}>
          <input value={name} onChange={(e) => setName(e.target.value)} list="known-names"
            placeholder="Name, or pick someone enrolled" aria-label={`Name ${face.person_id}`}
            className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent" />
          <button type="submit" disabled={!name.trim() || busy}
            className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white disabled:opacity-40">
            {busy ? "Saving…" : "Name"}
          </button>
        </form>
        {err && <div className="mt-1 text-xs text-warn">{err}</div>}
      </div>
    </li>
  );
}

export default function FacesOnScreen() {
  const [faces, setFaces] = useState<FaceOnScreen[]>([]);
  const [people, setPeople] = useState<KnownPerson[]>([]);
  const [done, setDone] = useState<string | null>(null);
  const [v, setV] = useState(0);              // refreshes the face images
  const timer = useRef<number>();

  useEffect(() => {
    const loadPeople = () => api.knownPeople().then(setPeople).catch(() => {});
    const poll = () => {
      // Only while the live view is on screen.
      if (!document.hidden && document.getElementById("faces-on-screen")?.offsetParent) {
        api.facesOnScreen().then((f) => { setFaces(f); setV((x) => x + 1); }).catch(() => {});
      }
    };
    loadPeople(); poll();
    timer.current = window.setInterval(poll, 3000);
    knownPeopleChanged.addEventListener("changed", loadPeople);
    return () => { clearInterval(timer.current); knownPeopleChanged.removeEventListener("changed", loadPeople); };
  }, []);

  return (
    <div id="faces-on-screen" className="card">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="flex items-center gap-2 font-semibold">
          <ScanFace size={16} className="text-accent" /> Faces on screen
          <span className="text-sm font-normal text-muted">({faces.length})</span>
        </h2>
      </div>
      <datalist id="known-names">{people.map((p) => <option key={p.id} value={p.name} />)}</datalist>
      {done && (
        <div className="flex items-start gap-2 border-b border-border bg-good/10 px-4 py-2 text-xs text-good">
          <Check size={14} className="mt-0.5 shrink-0" /> <span>{done}</span>
        </div>
      )}
      {faces.length > 0 ? (
        <ul className="max-h-[340px] divide-y divide-border overflow-auto">
          {faces.map((f) => (
            <NameFace key={f.track} face={f} people={people} v={v}
              onDone={(text) => { setDone(text); setFaces((all) => all.filter((x) => x.track !== f.track)); }} />
          ))}
        </ul>
      ) : (
        <p className="px-4 py-4 text-sm text-muted">
          Everyone in view is named, or no face is clear and close enough yet. Unknown people appear
          here when they face the camera.
        </p>
      )}
      <p className="flex gap-2 border-t border-border px-4 py-2 text-[11px] text-muted">
        <ShieldCheck size={13} className="mt-px shrink-0" />
        Name people only with their consent. Faces of people not named are never stored.
      </p>
    </div>
  );
}
