import { useEffect, useMemo, useState } from "react";
import { format } from "date-fns";
import clsx from "clsx";
import {
  ArrowRight, Camera, Cctv, Cpu, Database, Images, PersonStanding, Play, ScanFace, ShieldCheck, Sparkles, UserRound,
} from "lucide-react";
import { Activities, api, Camera as Cam, EventMessage, KnownPerson, Recordings, Showcase, ShowcaseTotals } from "../services/api";
import { NEW_ACTIVITIES, NEW_LABELS } from "../services/clips";
import EvidenceViewer from "./EvidenceViewer";
import { knownPeopleChanged } from "./KnownPeople";

// The demo's front page: the four things the system does, each with real
// results from this machine and a button that shows it working.

export type Go = (tab: "overview" | "live" | "database" | "compare", then?: "log" | "people") => void;

// Body postures from the skeleton: logged too, but not what a client came to see.
const POSTURES = ["standing", "sitting", "walking", "running", "lying", "kneeling", "bending", "crawling", "falling", "push-up"];

// The middle proof photo: the activity under way, not its first frame.
const mid = (e: EventMessage) => Math.min(1, (e.evidence?.length ?? 1) - 1);

const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);

function Section({ n, icon, title, lead, children, actions }: {
  n: number; icon: React.ReactNode; title: string; lead: string;
  children: React.ReactNode; actions: React.ReactNode;
}) {
  return (
    <section className="card flex flex-col overflow-hidden">
      <div className="flex items-start gap-3 border-b border-border px-5 py-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-accent/40 bg-accent/15 text-accent">
          {icon}
        </div>
        <div className="min-w-0">
          <h2 className="font-semibold">
            <span className="mr-2 text-muted tabular-nums">{n}</span>{title}
          </h2>
          <p className="mt-0.5 text-sm text-muted">{lead}</p>
        </div>
      </div>
      <div className="flex-1 px-5 py-4">{children}</div>
      <div className="flex flex-wrap gap-2 border-t border-border px-5 py-3">{actions}</div>
    </section>
  );
}

function Action({ onClick, primary, children, disabled, title }: {
  onClick: () => void; primary?: boolean; children: React.ReactNode; disabled?: boolean; title?: string;
}) {
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-40",
        primary ? "bg-accent text-white hover:bg-accent/85"
                : "border border-border bg-surface text-slate-200 hover:border-accent hover:text-accent",
      )}>
      {children}
    </button>
  );
}

function Stat({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-2xl font-semibold tabular-nums">{value}</div>
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-muted">{sub}</div>
    </div>
  );
}

function DeviceRow({ name, totals, of, best }: {
  name: string; totals: ShowcaseTotals; of: number; best: boolean;
}) {
  const share = totals.clear / of;
  return (
    <div className="grid grid-cols-[8.5rem_1fr_auto] items-center gap-3 text-sm">
      <span className={clsx("truncate", best ? "font-semibold" : "text-muted")}>{name}</span>
      <div className="h-2.5 overflow-hidden rounded-full bg-surface">
        <div className={clsx("h-full rounded-full", share >= 0.6 ? "bg-good" : share >= 0.3 ? "bg-warn" : "bg-danger")}
          style={{ width: `${Math.max(2, share * 100)}%` }} />
      </div>
      <span className="w-40 text-right tabular-nums">
        <span className="font-semibold">{totals.clear}/{of}</span>
        <span className="text-muted"> · {pct(totals.mean_share)} · {Math.round(totals.median_fps ?? 0)} fps</span>
      </span>
    </div>
  );
}

function Shot({ src, alt }: { src?: string; alt: string }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return <div className="flex h-full w-full items-center justify-center bg-surface text-muted"><Camera size={18} /></div>;
  }
  return <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} className="h-full w-full object-cover" />;
}

export default function Overview({ go }: { go: Go }) {
  const [show, setShow] = useState<Showcase | null>(null);
  const [people, setPeople] = useState<KnownPerson[]>([]);
  const [events, setEvents] = useState<EventMessage[]>([]);
  const [clips, setClips] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);
  const [open, setOpen] = useState<{ event: EventMessage; index: number } | null>(null);
  const [recs, setRecs] = useState<Recordings | null>(null);
  const [rec, setRec] = useState(0);
  const [cctv, setCctv] = useState<Cam[]>([]);
  const [acts, setActs] = useState<Activities | null>(null);

  useEffect(() => {
    api.showcase().then(setShow).catch(() => {});
    api.recordings().then(setRecs).catch(() => {});
    api.activities().then(setActs).catch(() => {});
    api.listSources().then((r) => {
      setClips(new Set(r.clips.map((c) => c.filename)));
      setCctv((r.cameras ?? []).filter((c) => c.kind === "cctv"));
    }).catch(() => {});
    const loadPeople = () => api.knownPeople().then(setPeople).catch(() => {});
    const loadEvents = () => api.recentEvents(400).then(setEvents).catch(() => {});
    loadPeople(); loadEvents();
    const t = setInterval(loadEvents, 10_000);
    knownPeopleChanged.addEventListener("changed", loadPeople);
    return () => { clearInterval(t); knownPeopleChanged.removeEventListener("changed", loadPeople); };
  }, []);

  const withProof = useMemo(() => events.filter((e) => (e.evidence?.length ?? 0) > 0), [events]);
  const latestOf = (label: string) => withProof.find((e) => e.label === label);
  const countOf = (label: string) => events.filter((e) => e.label === label).length;
  const namedIn = (name: string) => events.filter((e) => e.identity === name).length;
  // Gallery: the newest photo of each activity, the new activities first;
  // plain postures (standing, walking...) only if there is room left.
  const gallery = useMemo(() => {
    const rank = (e: EventMessage) => (NEW_LABELS.includes(e.label) ? 0 : POSTURES.includes(e.label) ? 2 : 1);
    const seen = new Set<string>(); const out: EventMessage[] = [];
    for (const e of [...withProof].sort((a, b) => rank(a) - rank(b))) {
      if (!seen.has(e.label)) { seen.add(e.label); out.push(e); }
      if (out.length === 6) break;
    }
    return out;
  }, [withProof]);
  // Proof photos that carry a name, newest first, one per activity.
  const named = useMemo(() => {
    const seen = new Set<string>();
    return withProof.filter((e) => e.identity && !POSTURES.includes(e.label) && !seen.has(e.label) && seen.add(e.label)).slice(0, 3);
  }, [withProof]);

  const play = async (filename: string) => {
    setBusy(filename);
    try { await api.switchSource(filename); go("live"); } catch { /* clip missing */ }
    setBusy(null);
  };

  const devices = show?.available ? [
    { name: show.left.sub, totals: show.totals.left },
    { name: show.right.sub, totals: show.totals.right },
    ...(show.extra ?? []).map((x) => ({ name: x.short, totals: x.totals })),
  ] : [];
  const bestClear = Math.max(0, ...devices.map((d) => d.totals.clear));
  const jetson = show?.extra?.find((x) => x.key === "jetson");
  const tabletsClip = NEW_ACTIVITIES[1].clips.find((c) => clips.has(c));
  const enrolledSamples = people.reduce((n, p) => n + p.samples.length, 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-3xl">
          <h2 className="text-2xl font-semibold">Who is doing what, with proof</h2>
          <p className="mt-1 text-muted">
            Ordinary camera video in; every person tracked, their activity recognised, enrolled people
            named, and photo evidence kept for each activity. Running here on an NVIDIA Jetson, on site:
            no video leaves the building.
          </p>
        </div>
        <Action primary onClick={() => go("live")}><Play size={14} /> Open the live monitor</Action>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat value={jetson ? `${jetson.totals.clear}/${show!.labelled_clips}` : "–"} label="Clips recognised on the Jetson"
          sub={show?.available ? `laptop CPU: ${show.totals.left.clear}/${show.labelled_clips}` : "comparison not installed"} />
        <Stat value={String(NEW_ACTIVITIES.length)} label="Activities added without training"
          sub="making tea · taking tablets · falling down" />
        <Stat value={String(people.length)} label={people.length === 1 ? "Person enrolled" : "People enrolled"}
          sub={`${enrolledSamples} face photo${enrolledSamples === 1 ? "" : "s"} · everyone else stays Unknown`} />
        <Stat value={String(withProof.length)} label="Logged activities with proof"
          sub="2–3 photos each: start, middle, end" />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section n={1} icon={<Cpu size={18} />} title="Same video, three machines"
          lead={show?.available
            ? `The same ${show.labelled_clips} clips through the same software, ${show.seconds_per_clip} s each, recorded and frame-synced.`
            : "Recorded comparison of a laptop CPU, a cloud GPU and the Jetson."}
          actions={<Action primary onClick={() => go("compare")}>Open the comparison <ArrowRight size={14} /></Action>}>
          {devices.length > 0 ? (
            <div className="space-y-3">
              <div className="grid grid-cols-[8.5rem_1fr_auto] gap-3 text-[11px] uppercase tracking-wide text-muted">
                <span>Machine</span><span>Clips clearly recognised</span>
                <span className="w-40 text-right">clear · correct · speed</span>
              </div>
              {devices.map((d) => (
                <DeviceRow key={d.name} name={d.name} totals={d.totals} of={show!.labelled_clips}
                  best={d.totals.clear >= bestClear - 1} />
              ))}
              {show!.new_activities && (
                <>
                  <div className="pt-2 text-[11px] uppercase tracking-wide text-muted">
                    New activities · {show!.new_activities.clips} clips · not run on the CPU
                  </div>
                  {Object.entries(show!.new_activities.totals).map(([k, t]) => (
                    <DeviceRow key={k} of={show!.new_activities!.clips} totals={t} best
                      name={k === "right" ? show!.right.sub : show!.extra?.find((x) => x.key === k)?.short ?? k} />
                  ))}
                </>
              )}
              <p className="pt-1 text-xs text-muted">
                A GPU is what makes it accurate, not just fast: the activity model gets several looks a second
                instead of one every 1.5 s. The Jetson matches the cloud Tesla T4.
              </p>
            </div>
          ) : <p className="text-sm text-muted">No comparison recording installed on this server.</p>}
        </Section>

        <Section n={2} icon={<Sparkles size={18} />} title="New activities, no retraining"
          lead="Activities the base model was never trained on, added in a line of configuration each."
          actions={<>
            {NEW_ACTIVITIES.map((a) => {
              const f = a.clips.find((c) => clips.has(c));
              return (
                <Action key={a.label} onClick={() => f && play(f)} disabled={!f || busy !== null}
                  title={f ? `Play a ${a.title.toLowerCase()} clip in the live monitor` : "No demo clip on this server"}>
                  <Play size={13} /> {busy === f ? "Starting…" : a.title}
                </Action>
              );
            })}
          </>}>
          <div className="grid grid-cols-3 gap-3">
            {NEW_ACTIVITIES.map((a) => {
              const e = latestOf(a.label);
              const n = countOf(a.label);
              return (
                <button key={a.label} disabled={!e} onClick={() => e && setOpen({ event: e, index: mid(e) })}
                  className="group overflow-hidden rounded-lg border border-border bg-surface text-left enabled:hover:border-accent">
                  <div className="aspect-[4/3] overflow-hidden">
                    <Shot src={e?.evidence?.[Math.min(1, e.evidence.length - 1)]} alt={a.title} />
                  </div>
                  <div className="px-2.5 py-2">
                    <div className={clsx("text-sm font-semibold", a.label === "falling down" && "text-danger")}>{a.title}</div>
                    <div className="text-[11px] leading-snug text-muted">{a.how}</div>
                    <div className="mt-1 text-[11px] tabular-nums text-slate-300">
                      {n ? `${n} in the log` : "not seen yet"}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-muted">
            On the 29 other comparison clips, neither GPU ever showed a new activity by mistake.
          </p>
          {acts?.trained && acts.trained.activities.length > 0 && (
            <div className="mt-3 rounded-md border border-border bg-surface px-3 py-2 text-xs">
              <div className="font-medium text-slate-200">Trained on example clips (docs/TRAINING.md)</div>
              {acts.trained.activities.map((a) => {
                const r = acts.trained!.held_out.activities?.[a];
                return (
                  <div key={a} className="mt-0.5 text-muted">
                    <span className="capitalize text-slate-200">{a}</span>
                    {r && ` · recognised in ${r.recognised} of ${r.clips} clips it was not trained on`}
                    {acts.trained!.held_out.other_clips != null &&
                      ` · false alarm in ${acts.trained!.held_out.false_alarm_clips} of ${acts.trained!.held_out.other_clips} ordinary clips`}
                    {` · ${countOf(a)} in the log`}
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        <Section n={3} icon={<ScanFace size={18} />} title="Face recognition: who is doing it"
          lead="Enrolled people are named on the video, in the subtitle, in the log and on the proof photos."
          actions={<>
            <Action primary onClick={() => tabletsClip && play(tabletsClip)} disabled={!tabletsClip || busy !== null}>
              <Play size={13} /> See it live
            </Action>
            <Action onClick={() => go("live", "people")}><UserRound size={13} /> Name people on screen</Action>
          </>}>
          {people.length > 0 ? (
            <ul className="space-y-2">
              {people.map((p) => (
                <li key={p.id} className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2">
                  <div className="flex -space-x-2">
                    {p.samples.slice(0, 3).map((s) => (
                      <img key={s.id} src={api.facePhotoUrl(p.id, s.id)} alt=""
                        className="h-10 w-10 rounded-full border-2 border-elevated object-cover" />
                    ))}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold">{p.name}</div>
                    <div className="text-xs text-muted">
                      {p.samples.length} face photo{p.samples.length === 1 ? "" : "s"} · named in {namedIn(p.name)} logged activities
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted">Nobody enrolled yet: add a person with 2–5 photos, or from any proof image.</p>
          )}
          {named.length > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-3">
              {named.map((e) => (
                <button key={e.id} onClick={() => setOpen({ event: e, index: mid(e) })}
                  className="overflow-hidden rounded-lg border border-border bg-surface text-left hover:border-accent">
                  <div className="aspect-[4/3] overflow-hidden"><Shot src={e.evidence![mid(e)]} alt="" /></div>
                  <div className="truncate px-2.5 py-1.5 text-xs">
                    <span className="font-medium">{e.identity}</span> <span className="text-muted">is {e.label}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
          <ul className="mt-3 space-y-1 text-xs text-muted">
            <li className="flex gap-2"><PersonStanding size={14} className="shrink-0" />
              Anyone else stays <span className="text-slate-300">Unknown</span> until named: type a name under
              their face in the live monitor's <span className="text-slate-300">Faces on screen</span>, and they are
              recognised from then on.</li>
            <li className="flex gap-2"><ShieldCheck size={14} className="shrink-0" />
              Enrol only with consent. Face data stays on this device; deleting a person removes all of it.</li>
          </ul>
        </Section>

        <Section n={4} icon={<Images size={18} />} title="Every activity logged, with proof photos"
          lead="Each logged activity keeps photos of the person at its start, middle and end. Click one to open it."
          actions={<Action primary onClick={() => go("live", "log")}>Open the activity log <ArrowRight size={14} /></Action>}>
          {gallery.length > 0 ? (
            <div className="grid grid-cols-3 gap-3">
              {gallery.map((e) => (
                <button key={e.id ?? `${e.person_id}${e.start_time}`} onClick={() => setOpen({ event: e, index: mid(e) })}
                  className="overflow-hidden rounded-lg border border-border bg-surface text-left hover:border-accent">
                  <div className="aspect-[4/3] overflow-hidden"><Shot src={e.evidence![mid(e)]} alt={e.label} /></div>
                  <div className="px-2.5 py-1.5 text-xs">
                    <div className={clsx("truncate font-medium capitalize", e.label === "falling down" && "text-danger")}>{e.label}</div>
                    <div className="truncate text-muted">
                      {e.identity ?? `Unknown ${e.person_id}`} · {format(new Date(e.start_time), "HH:mm")}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : <p className="text-sm text-muted">No activity logged yet: play a clip in the live monitor.</p>}
        </Section>

        <div className="xl:col-span-2">
          <Section n={5} icon={<Cctv size={18} />} title="An Indian office, through a CCTV camera input"
            lead="Office footage played into the monitor as a network camera, the way a CCTV camera connects: recognised live on the Jetson, and every activity written to the database as it ends."
            actions={<>
              <Action primary onClick={() => go("database")}><Database size={14} /> Watch the database fill</Action>
              {cctv[0] && <Action onClick={() => play(cctv[0].filename)}><Cctv size={14} /> Live: {cctv[0].name}</Action>}
              {clips.has("office-floor.mp4") && <Action onClick={() => play("office-floor.mp4")}><Play size={14} /> Play the office clips</Action>}
            </>}>
            {recs && recs.videos.length > 0 ? (
              <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
                <div>
                  <video key={recs.videos[rec].url} src={recs.videos[rec].url} controls autoPlay muted loop playsInline
                    className="w-full rounded-lg border border-border bg-black aspect-video" />
                  <p className="mt-2 text-sm"><span className="font-medium">{recs.videos[rec].title}:</span>{" "}
                    <span className="text-muted">{recs.videos[rec].shows}. Shown: {recs.videos[rec].labels.join(", ")}.</span></p>
                  {recs.how && <p className="mt-1 text-xs text-muted">{recs.how}{recs.recorded ? ` Recorded ${recs.recorded}` : ""}{recs.device ? ` on ${recs.device}.` : "."}</p>}
                </div>
                <ul className="grid grid-cols-2 gap-2 self-start lg:grid-cols-1">
                  {recs.videos.map((v, i) => (
                    <li key={v.file}>
                      <button onClick={() => setRec(i)}
                        className={clsx("w-full rounded-md border px-3 py-2 text-left text-sm transition",
                          i === rec ? "border-accent bg-accent/10 text-accent" : "border-border hover:border-accent")}>
                        <div className="font-medium">{v.title}</div>
                        <div className="truncate text-xs text-muted capitalize">{v.labels.join(" · ")}</div>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : <p className="text-sm text-muted">No recordings on this machine (they are stored with Git LFS: <span className="font-mono">git lfs pull</span>).</p>}
          </Section>
        </div>
      </div>

      {open && <EvidenceViewer event={open.event} start={open.index} onClose={() => setOpen(null)} />}
    </div>
  );
}
