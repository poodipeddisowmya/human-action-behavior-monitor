import { useEffect, useRef, useState } from "react";
import { ImagePlus, ShieldCheck, Trash2, UserPlus, X } from "lucide-react";
import { api, KnownPerson } from "../services/api";

// The people face recognition may name. Only enrolled people are ever named;
// everyone else stays "Unknown". A face is biometric data: enrol only with
// the person's consent, and delete on request.

export const knownPeopleChanged = new EventTarget();

export default function KnownPeople() {
  const [people, setPeople] = useState<KnownPerson[]>([]);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const addTo = useRef<number | null>(null);
  const filePicker = useRef<HTMLInputElement>(null);

  const load = () => api.knownPeople().then(setPeople).catch(() => {});
  useEffect(() => {
    load();
    const on = () => load();
    knownPeopleChanged.addEventListener("changed", on);
    return () => knownPeopleChanged.removeEventListener("changed", on);
  }, []);

  const report = (added: number, refused: { file: string; reason: string }[]) => {
    const parts = [];
    if (added) parts.push(`${added} photo${added > 1 ? "s" : ""} enrolled`);
    refused.forEach((r) => parts.push(`${r.file}: ${r.reason}`));
    setMsg({ ok: added > 0 && refused.length === 0, text: parts.join(" · ") || "Nothing added" });
  };

  const create = async () => {
    if (!name.trim() || files.length === 0) return;
    setBusy(true); setMsg(null);
    try {
      const p = await api.addKnownPerson(name.trim());
      const r = await api.addFacePhotos(p.id, files);
      if (r.added.length === 0) await api.deleteKnownPerson(p.id);   // no usable face: undo
      report(r.added.length, r.refused);
      if (r.added.length) { setAdding(false); setName(""); setFiles([]); }
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    }
    setBusy(false);
    load();
  };

  const addPhotos = async (list: FileList | null) => {
    const id = addTo.current;
    if (!list || id == null) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.addFacePhotos(id, Array.from(list));
      report(r.added.length, r.refused);
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    }
    setBusy(false);
    load();
  };

  const remove = async (p: KnownPerson) => {
    if (!confirm(`Delete ${p.name} and all their face photos? Past events keep the name they were logged with.`)) return;
    await api.deleteKnownPerson(p.id).catch(() => {});
    load();
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="font-semibold">Known people ({people.length})</h2>
        <button onClick={() => { setAdding(!adding); setMsg(null); }}
          className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent hover:text-accent">
          {adding ? <X size={13} /> : <UserPlus size={13} />} {adding ? "Cancel" : "Add person"}
        </button>
      </div>

      {adding && (
        <div className="space-y-2 border-b border-border p-3 text-sm">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name"
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 outline-none focus:border-accent" />
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-border px-3 py-2 text-muted hover:border-accent">
            <ImagePlus size={14} />
            {files.length ? `${files.length} photo${files.length > 1 ? "s" : ""} selected` : "Choose 2-5 clear face photos"}
            <input type="file" accept="image/*" multiple className="hidden"
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
          </label>
          <button onClick={create} disabled={busy || !name.trim() || files.length === 0}
            className="w-full rounded-lg bg-accent/90 px-3 py-2 font-medium text-white disabled:opacity-40">
            {busy ? "Enrolling…" : "Enrol"}
          </button>
          <p className="flex items-start gap-1.5 text-[11px] text-muted">
            <ShieldCheck size={12} className="mt-0.5 shrink-0" />
            Enrol only with the person's consent. Faces stay on this device and are deleted with the person.
          </p>
        </div>
      )}
      {msg && (
        <div className={"border-b border-border px-4 py-2 text-xs " + (msg.ok ? "text-good" : "text-warn")}>{msg.text}</div>
      )}

      <ul className="max-h-[260px] divide-y divide-border overflow-auto">
        {people.map((p) => (
          <li key={p.id} className="flex items-center gap-3 px-4 py-2.5">
            <div className="flex -space-x-2">
              {p.samples.slice(0, 3).map((s) => (
                <img key={s.id} src={api.facePhotoUrl(p.id, s.id)} alt=""
                  className="h-8 w-8 rounded-full border-2 border-elevated object-cover" />
              ))}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{p.name}</div>
              <div className="text-[11px] text-muted">{p.samples.length} face photo{p.samples.length !== 1 ? "s" : ""}</div>
            </div>
            <button title="Add photos" aria-label={`Add photos of ${p.name}`}
              onClick={() => { addTo.current = p.id; filePicker.current?.click(); }}
              className="rounded p-1.5 text-muted hover:text-accent"><ImagePlus size={15} /></button>
            <button title="Delete" aria-label={`Delete ${p.name}`} onClick={() => remove(p)}
              className="rounded p-1.5 text-muted hover:text-danger"><Trash2 size={15} /></button>
          </li>
        ))}
        {people.length === 0 && !adding && (
          <li className="px-4 py-5 text-center text-sm text-muted">
            No one enrolled: everyone shows as Unknown. Add a person, or enrol a face from a proof image in the events table.
          </li>
        )}
      </ul>
      <input ref={filePicker} type="file" accept="image/*" multiple className="hidden"
        onChange={(e) => { addPhotos(e.target.files); e.target.value = ""; }} />
    </div>
  );
}
