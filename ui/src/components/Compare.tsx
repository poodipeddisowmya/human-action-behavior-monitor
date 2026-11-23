import { useEffect, useMemo, useRef, useState } from "react";
import { Cpu, Pause, Play, RotateCcw, Zap } from "lucide-react";
import { api, Showcase, ShowcaseClip, ShowcaseSide, ShowcaseTotals } from "../services/api";

// The recorded CPU-vs-GPU comparison: the same clips, the same software,
// one side on a laptop CPU and one on a GPU, plus any further devices
// recorded on the same clips (a Jetson). All videos play in lock-step so any
// frame can be compared directly.

type Kind = "cpu" | "gpu" | "edge";
interface Column {
  key: string;                 // "left" | "right" | an extra side's key (also its video path)
  kind: Kind;
  tag: string;
  sub: string;
  short: string;               // name in the totals when there are more than two sides
  totals: ShowcaseTotals;
  of: (c: ShowcaseClip) => ShowcaseSide | null | undefined;
}

const pct = (x: number | null | undefined) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const secs = (x: number | null | undefined) => (x == null ? "never" : `${x.toFixed(1)} s`);
// New-activity clips are named after their source (mixkit-46792): show the
// activity instead.
const clipTitle = (c: ShowcaseClip) =>
  c.new_activity && c.expected ? c.expected[0].toUpperCase() + c.expected.slice(1) : title(c.name);
const title = (name: string) =>
  name.replace(/^demo-|^hd-/, "").replace(/-\d+$/, "").replace(/-/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase());

function Delta({ label, values, names }: { label: string; values: string[]; names: string[] }) {
  // Two sides: "before → after". More: the GPU sides each named.
  const [base, ...rest] = values;
  return (
    <div className="card px-4 py-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-1 tabular-nums">
        <span className="text-muted line-through decoration-muted/60 whitespace-nowrap">{base}</span>
        <span className="text-muted">→</span>
        {rest.map((v, i) => (
          <span key={i} className="whitespace-nowrap">
            <span className="text-xl font-semibold text-good">{v}</span>
            {rest.length > 1 && <span className="ml-1 text-[11px] text-muted">{names[i + 1]}</span>}
            {i < rest.length - 1 && <span className="ml-2 text-muted">·</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

function SidePanel({ col, s, expected, video }: {
  col: Column; s: ShowcaseSide | null | undefined;
  expected: string | null; video: (el: HTMLVideoElement | null) => void;
}) {
  const gpu = col.kind !== "cpu";
  const badge = { cpu: "bg-danger/80", gpu: "bg-good/90", edge: "bg-accent/90" }[col.kind];
  return (
    <div className="card overflow-hidden min-w-0">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
        <span className={"flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-bold tracking-wider text-white " + badge}>
          {gpu ? <Zap size={12} /> : <Cpu size={12} />} {col.tag}
        </span>
        <span className="text-sm text-muted truncate">{col.sub}</span>
      </div>
      {!s ? (
        <div className="flex aspect-video items-center justify-center bg-black text-sm text-muted">
          Not recorded on this device
        </div>
      ) : (<>
      <video ref={video} src={api.showcaseVideo(col.key, s.video)} muted playsInline loop
        preload="auto" className="block w-full aspect-video bg-black object-contain" />
      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-border text-sm">
        {expected ? (
          <>
            <Stat k={`"${expected}" on screen`} v={pct(s.share)} strong={gpu} />
            <Stat k="first shown" v={secs(s.first_s)} strong={gpu} />
            <Stat k="playback" v={s.fps ? `${Math.round(s.fps)} fps` : "—"} />
            <Stat k="people tracked" v={s.people?.toFixed(1) ?? "—"} />
          </>
        ) : (
          <>
            <Stat k="people tracked per frame" v={s.people?.toFixed(1) ?? "—"} strong={gpu} wide />
            <Stat k="playback" v={s.fps ? `${Math.round(s.fps)} fps` : "—"} wide />
          </>
        )}
      </dl>
      </>)}
    </div>
  );
}

function Stat({ k, v, strong, wide }: { k: string; v: string; strong?: boolean; wide?: boolean }) {
  return (
    <div className={"bg-elevated px-3 py-2 " + (wide ? "col-span-2" : "")}>
      <dt className="text-[11px] text-muted truncate">{k}</dt>
      <dd className={"tabular-nums " + (strong ? "text-base font-semibold text-slate-100" : "text-slate-300")}>{v}</dd>
    </div>
  );
}

export default function Compare() {
  const [data, setData] = useState<Showcase | null>(null);
  const [error, setError] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  // One <video> per column, by column key. "right" (the GPU) is the clock.
  const videos = useRef<Record<string, HTMLVideoElement | null>>({});
  const all = () => Object.values(videos.current).filter((v): v is HTMLVideoElement => !!v);
  // The clock: the T4 video, or the first one there when the T4 lacks the clip.
  const clockVideo = () => videos.current.right ?? all()[0] ?? null;

  useEffect(() => {
    api.showcase().then((d) => {
      setData(d);
      const first = d.clips?.find((c) => c.featured) ?? d.clips?.[0];
      if (first) setCurrent(first.name);
    }).catch(() => setError(true));
  }, []);

  const cols: Column[] = useMemo(() => {
    if (!data?.available) return [];
    return [
      { key: "left", kind: "cpu", tag: data.left.tag, sub: data.left.sub, short: "CPU",
        totals: data.totals.left, of: (c) => c.left ?? null },
      { key: "right", kind: "gpu", tag: data.right.tag, sub: data.right.sub, short: data.right.sub,
        totals: data.totals.right, of: (c) => c.right },
      ...(data.extra ?? []).map((x): Column => ({
        key: x.key, kind: "edge", tag: x.tag, sub: x.sub, short: x.short,
        totals: x.totals, of: (c) => c.extra?.[x.key] })),
    ];
  }, [data]);

  const clips = useMemo(() => {
    if (!data?.clips) return [];
    const list = showAll ? data.clips : data.clips.filter((c) => c.featured);
    return [...list].sort((a, b) => Number(b.featured) - Number(a.featured));
  }, [data, showAll]);
  const clip: ShowcaseClip | undefined = data?.clips?.find((c) => c.name === current);

  // Lock-step playback: the GPU video is the clock; the others follow.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const clock = clockVideo();
      if (clock && !clock.paused) {
        for (const v of all()) {
          if (v !== clock && Math.abs(v.currentTime - clock.currentTime) > 0.08) v.currentTime = clock.currentTime;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [current]);

  // The button reflects what the video is really doing: a browser may refuse
  // autoplay, and then "Pause" over a stopped video reads as broken.
  useEffect(() => {
    const v = clockVideo();
    if (!v) return;
    const on = () => setPlaying(true), off = () => setPlaying(false);
    v.addEventListener("play", on);
    v.addEventListener("pause", off);
    return () => { v.removeEventListener("play", on); v.removeEventListener("pause", off); };
  }, [current]);

  useEffect(() => {                       // new clip: start all from 0
    all().forEach((v) => {
      v.currentTime = 0;
      // A refused autoplay fires no "pause" event; reflect it here.
      if (playing) v.play().catch(() => setPlaying(false));
    });
  }, [current]);

  const toggle = () => {
    const next = !playing;
    setPlaying(next);
    all().forEach((v) => (next ? v.play().catch(() => {}) : v.pause()));
  };
  const restart = () => all().forEach((v) => {
    v.currentTime = 0;
    if (playing) v.play().catch(() => {});
  });

  if (error) return <p className="text-danger">Could not load the comparison.</p>;
  if (!data) return <p className="text-muted">Loading the comparison…</p>;
  if (!data.available) {
    return (
      <div className="card p-6 max-w-2xl">
        <h2 className="font-semibold mb-2">No comparison recorded yet</h2>
        <p className="text-sm text-muted">
          Record one with <code className="font-mono">scripts/kaggle_auto.py showcase</code> and
          install it with <code className="font-mono">scripts/install_showcase.py</code>.
        </p>
      </div>
    );
  }

  const n = data.labelled_clips;
  const names = cols.map((c) => c.short);
  const many = cols.length > 2;
  const fps = (x: number | null) => `${Math.round(x ?? 0)} fps`;
  const where = many
    ? cols.map((c, i) => `${["left", "middle", "right"][i] ?? `then`} on ${c.sub}`).join(", ")
    : `left on ${data.left.sub}, right on ${data.right.sub}`;
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-xl font-semibold">
          {many ? cols.map((c) => c.sub.replace(/^laptop CPU$/, "Laptop CPU")).join(" vs ") : "Without a GPU vs with a GPU"}
        </h2>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          The same {n} clips through the same software and models: {where}.
          Each clip played {data.seconds_per_clip} s; every figure is what the screen showed. Recorded {data.recorded}
          {(data.extra ?? []).map((x) => x.recorded ? ` (${x.short}: ${x.recorded})` : "").join("")}.
        </p>
      </section>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Delta label="Activities clearly recognised" names={names} values={cols.map((c) => `${c.totals.clear}/${n}`)} />
        <Delta label="Correct label on screen (average)" names={names} values={cols.map((c) => pct(c.totals.mean_share))} />
        <Delta label="Correct label appeared at all" names={names} values={cols.map((c) => `${c.totals.appeared}/${n}`)} />
        <Delta label="Playback speed (median)" names={names} values={cols.map((c) => fps(c.totals.median_fps))} />
      </section>

      {data.new_activities && (
        <section className="card px-4 py-3 text-sm">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="rounded bg-accent/20 px-1.5 text-[10px] font-semibold text-accent">NEW</span>
            <span className="font-medium capitalize">{data.new_activities.labels.join(" · ")}</span>
            <span className="text-muted">
              — {data.new_activities.clips} clips, added after the CPU recording
            </span>
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-6 gap-y-1">
            {cols.map((c) => {
              const t = data.new_activities!.totals[c.key];
              return (
                <span key={c.key} className="whitespace-nowrap">
                  <span className="text-muted">{c.short}: </span>
                  {t ? (
                    <>
                      <b className="text-good">{t.clear}/{t.recorded}</b> clearly recognised,{" "}
                      <b className="text-good">{t.appeared}/{t.recorded}</b> shown at all,{" "}
                      {pct(t.mean_share)} on screen
                    </>
                  ) : (
                    <span className="text-muted">not recorded</span>
                  )}
                </span>
              );
            })}
          </div>
        </section>
      )}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          {clips.map((c) => {
            const vals = cols.map((col) => col.of(c));
            const [base, ...rest] = vals.map((v) => c.expected ? pct(v?.share) : (v?.people?.toFixed(1) ?? "—"));
            return (
              <button key={c.name} onClick={() => setCurrent(c.name)}
                className={"rounded-lg border px-3 py-1.5 text-left text-xs transition " +
                  (c.name === current ? "border-accent bg-accent/15" : "border-border bg-elevated hover:border-accent/60")}>
                <div className="font-medium text-slate-100">
                  {clipTitle(c)}
                  {c.name.startsWith("hd-") ? " (HD)" : ""}
                  {c.new_activity && <span className="ml-1.5 rounded bg-accent/20 px-1 text-[10px] font-semibold text-accent">NEW</span>}
                </div>
                <div className="text-muted tabular-nums">
                  {base} → {rest.join(" · ")}{c.expected ? "" : " people"}
                </div>
              </button>
            );
          })}
          <button onClick={() => setShowAll(!showAll)}
            className="text-xs text-accent hover:underline px-2">
            {showAll ? "Show highlights" : `Show all ${data.clips.length}`}
          </button>
        </div>

        {clip && (
          <>
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm">
                <span className="font-semibold">{clipTitle(clip)}</span>
                <span className="text-muted"> · {clip.expected ? <>expected activity: <b className="text-slate-200">{clip.expected}</b></> : "crowded scene: people found and tracked"}</span>
              </div>
              <div className="flex gap-2">
                <button onClick={toggle} aria-label={playing ? "Pause all" : "Play all"}
                  className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent hover:text-accent">
                  {playing ? <Pause size={13} /> : <Play size={13} />} {playing ? "Pause" : "Play"}
                </button>
                <button onClick={restart} aria-label="Restart all"
                  className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent hover:text-accent">
                  <RotateCcw size={13} /> Restart
                </button>
              </div>
            </div>
            <div className={"grid grid-cols-1 md:grid-cols-2 gap-4 " + (many ? "xl:grid-cols-3" : "")}>
              {cols.map((col) => (
                <SidePanel key={col.key} col={col} s={col.of(clip)} expected={clip.expected}
                  video={(el) => { videos.current[col.key] = el; }} />
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
