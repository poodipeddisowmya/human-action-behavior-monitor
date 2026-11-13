import { useEffect, useMemo, useRef, useState } from "react";
import { Cpu, Pause, Play, RotateCcw, Zap } from "lucide-react";
import { api, Showcase, ShowcaseClip, ShowcaseSide } from "../services/api";

// The recorded CPU-vs-GPU comparison: the same clips, the same software,
// one side on a laptop CPU and one on a GPU. Both videos play in lock-step
// so any frame can be compared directly.

const pct = (x: number | null | undefined) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const secs = (x: number | null | undefined) => (x == null ? "never" : `${x.toFixed(1)} s`);
const title = (name: string) =>
  name.replace(/^demo-|^hd-/, "").replace(/-\d+$/, "").replace(/-/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase());

function Delta({ label, left, right }: { label: string; left: string; right: string }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 flex items-baseline gap-2 tabular-nums">
        <span className="text-muted line-through decoration-muted/60 whitespace-nowrap">{left}</span>
        <span className="text-muted">→</span>
        <span className="text-xl font-semibold text-good whitespace-nowrap">{right}</span>
      </div>
    </div>
  );
}

function SidePanel({ side, s, tag, sub, expected, video }: {
  side: "left" | "right"; s: ShowcaseSide; tag: string; sub: string;
  expected: string | null; video: (el: HTMLVideoElement | null) => void;
}) {
  const gpu = side === "right";
  return (
    <div className="card overflow-hidden min-w-0">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
        <span className={"flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-bold tracking-wider text-white " +
          (gpu ? "bg-good/90" : "bg-danger/80")}>
          {gpu ? <Zap size={12} /> : <Cpu size={12} />} {tag}
        </span>
        <span className="text-sm text-muted truncate">{sub}</span>
      </div>
      <video ref={video} src={api.showcaseVideo(side, s.video)} muted playsInline loop
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
  const left = useRef<HTMLVideoElement | null>(null);
  const right = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    api.showcase().then((d) => {
      setData(d);
      const first = d.clips?.find((c) => c.featured) ?? d.clips?.[0];
      if (first) setCurrent(first.name);
    }).catch(() => setError(true));
  }, []);

  const clips = useMemo(() => {
    if (!data?.clips) return [];
    const list = showAll ? data.clips : data.clips.filter((c) => c.featured);
    return [...list].sort((a, b) => Number(b.featured) - Number(a.featured));
  }, [data, showAll]);
  const clip: ShowcaseClip | undefined = data?.clips?.find((c) => c.name === current);

  // Lock-step playback: the right-hand video is the clock; the left follows.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const a = left.current, b = right.current;
      if (a && b && !b.paused && Math.abs(a.currentTime - b.currentTime) > 0.08) {
        a.currentTime = b.currentTime;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [current]);

  // The button reflects what the video is really doing: a browser may refuse
  // autoplay, and then "Pause" over a stopped video reads as broken.
  useEffect(() => {
    const v = right.current;
    if (!v) return;
    const on = () => setPlaying(true), off = () => setPlaying(false);
    v.addEventListener("play", on);
    v.addEventListener("pause", off);
    return () => { v.removeEventListener("play", on); v.removeEventListener("pause", off); };
  }, [current]);

  useEffect(() => {                       // new clip: start both from 0
    [left.current, right.current].forEach((v) => {
      if (!v) return;
      v.currentTime = 0;
      // A refused autoplay fires no "pause" event; reflect it here.
      if (playing) v.play().catch(() => setPlaying(false));
    });
  }, [current]);

  const toggle = () => {
    const next = !playing;
    setPlaying(next);
    [left.current, right.current].forEach((v) => v && (next ? v.play().catch(() => {}) : v.pause()));
  };
  const restart = () => [left.current, right.current].forEach((v) => {
    if (!v) return;
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

  const t = data.totals;
  const n = data.labelled_clips;
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-xl font-semibold">Without a GPU vs with a GPU</h2>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          The same {n} clips through the same software and models: left on {data.left.sub}, right on {data.right.sub}.
          Each clip played {data.seconds_per_clip} s; every figure is what the screen showed. Recorded {data.recorded}.
        </p>
      </section>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Delta label="Activities clearly recognised" left={`${t.left.clear}/${n}`} right={`${t.right.clear}/${n}`} />
        <Delta label="Correct label on screen (average)" left={pct(t.left.mean_share)} right={pct(t.right.mean_share)} />
        <Delta label="Correct label appeared at all" left={`${t.left.appeared}/${n}`} right={`${t.right.appeared}/${n}`} />
        <Delta label="Playback speed (median)" left={`${Math.round(t.left.median_fps ?? 0)} fps`} right={`${Math.round(t.right.median_fps ?? 0)} fps`} />
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          {clips.map((c) => (
            <button key={c.name} onClick={() => setCurrent(c.name)}
              className={"rounded-lg border px-3 py-1.5 text-left text-xs transition " +
                (c.name === current ? "border-accent bg-accent/15" : "border-border bg-elevated hover:border-accent/60")}>
              <div className="font-medium text-slate-100">{title(c.name)}{c.name.startsWith("hd-") ? " (HD)" : ""}</div>
              <div className="text-muted tabular-nums">
                {c.expected ? `${pct(c.left.share)} → ${pct(c.right.share)}`
                  : `${c.left.people?.toFixed(1)} → ${c.right.people?.toFixed(1)} people`}
              </div>
            </button>
          ))}
          <button onClick={() => setShowAll(!showAll)}
            className="text-xs text-accent hover:underline px-2">
            {showAll ? "Show highlights" : `Show all ${data.clips.length}`}
          </button>
        </div>

        {clip && (
          <>
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm">
                <span className="font-semibold">{title(clip.name)}</span>
                <span className="text-muted"> · {clip.expected ? <>expected activity: <b className="text-slate-200">{clip.expected}</b></> : "crowded scene: people found and tracked"}</span>
              </div>
              <div className="flex gap-2">
                <button onClick={toggle} aria-label={playing ? "Pause both" : "Play both"}
                  className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent hover:text-accent">
                  {playing ? <Pause size={13} /> : <Play size={13} />} {playing ? "Pause" : "Play"}
                </button>
                <button onClick={restart} aria-label="Restart both"
                  className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent hover:text-accent">
                  <RotateCcw size={13} /> Restart
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SidePanel side="left" s={clip.left} tag={data.left.tag} sub={data.left.sub}
                expected={clip.expected} video={(el) => (left.current = el)} />
              <SidePanel side="right" s={clip.right} tag={data.right.tag} sub={data.right.sub}
                expected={clip.expected} video={(el) => (right.current = el)} />
            </div>
          </>
        )}
      </section>
    </div>
  );
}
