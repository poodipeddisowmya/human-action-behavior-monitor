import { useEffect, useRef, useState } from "react";
import { Maximize2, MapPin, Users, Video, VideoOff, Zap } from "lucide-react";
import { api, BROWSER_CAMERA, isLive } from "../services/api";
import { useBrowserCamera } from "../hooks/useBrowserCamera";

export interface Caption {
  text: string | null;
  scene: string | null;
}

// "in a kitchen" -> "Kitchen", "indoors" -> "Indoors"
function sceneName(phrase: string): string {
  const s = phrase.replace(/^(in|on|at) (a|an|the) /, "");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export default function VideoStream({
  annotations,
  caption,
}: {
  annotations: any[];
  caption: Caption | null;
}) {
  const people = annotations.length;
  const stage = useRef<HTMLDivElement | null>(null);
  const [current, setCurrent] = useState<string | null>(null);
  // A camera attached to the server itself (the RTX laptop on site). When
  // there is none — Kaggle, a cloud GPU — "Go live" uses this browser's
  // webcam instead and streams it to the server.
  const [serverCamera, setServerCamera] = useState<string | null>(null);
  const lastClip = useRef<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const cam = useBrowserCamera();
  const camStartedAt = useRef(0);
  const live = isLive(current);

  const check = () =>
    api.listSources().then((r) => {
      setCurrent(r.current);
      setServerCamera(r.cameras?.[0]?.filename ?? null);
      if (r.current && !isLive(r.current)) lastClip.current = r.current;
      else if (!lastClip.current && r.clips[0]) lastClip.current = r.clips[0].filename;
    }).catch(() => {});
  useEffect(() => {
    check();
    const t = setInterval(check, 3000);
    return () => clearInterval(t);
  }, []);

  // The server moved off this browser's camera (the Source list, another
  // viewer, or it gave up after the feed went quiet): release the camera.
  useEffect(() => {
    if (cam.state === "on" && current !== BROWSER_CAMERA &&
        Date.now() - camStartedAt.current > 4000) cam.stop();
  }, [current, cam.state]);

  useEffect(() => {
    if (!cam.error) return;
    setNotice(cam.error);
    const t = setTimeout(() => setNotice(null), 7000);
    return () => clearTimeout(t);
  }, [cam.error]);

  // One click to go live from the video window itself, and one to go back.
  const toggleLive = async () => {
    if (busy) return;
    setBusy(true);
    try {
      if (live) {
        cam.stop();
        if (lastClip.current) {
          await api.switchSource(lastClip.current);
          setCurrent(lastClip.current);
        }
      } else if (serverCamera) {
        await api.switchSource(serverCamera);
        setCurrent(serverCamera);
      } else if (await cam.start()) {
        camStartedAt.current = Date.now();
        await api.switchSource(BROWSER_CAMERA);
        setCurrent(BROWSER_CAMERA);
      }
    } catch {
      setNotice("Could not switch the source. Try again.");
    } finally {
      setBusy(false);
    }
  };
  // Full-screen viewing window for monitoring and demos. Browsers allow this
  // only from a click, and may refuse it; then the panel simply stays as is.
  const fullscreen = () => {
    const el = stage.current;
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else el.requestFullscreen?.().catch(() => {});
  };
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="font-semibold flex items-center gap-2">
          {current === BROWSER_CAMERA ? "Your webcam" : live ? "Live camera" : "Video feed"}
          {live && (
            <span className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wider bg-danger text-white">
              <span className="live-dot" /> LIVE
            </span>
          )}
        </h2>
        <div className="flex items-center gap-4 text-xs text-muted">
          {caption?.scene && (
            <span className="flex items-center gap-1.5 text-slate-200">
              <MapPin size={14} /> {sceneName(caption.scene)}
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <Users size={14} /> {people} tracked
          </span>
          <span className="flex items-center gap-1.5 text-good">
            <Zap size={14} /> streaming
          </span>
          <button
            onClick={toggleLive}
            disabled={busy || cam.state === "starting"}
            title={live ? "Stop the live camera"
              : serverCamera ? "Start live detection from the server's camera"
              : "Start live detection from this device's webcam"}
            className={
              "flex items-center gap-1.5 px-2.5 py-1 rounded-md border font-semibold transition disabled:opacity-40 " +
              (live
                ? "border-danger text-danger bg-danger/10 hover:bg-danger/20"
                : "border-accent text-accent bg-accent/10 hover:bg-accent/20")
            }
          >
            {live ? <VideoOff size={13} /> : <Video size={13} />}
            {live ? "Stop live" : cam.state === "starting" ? "Starting…" : "Go live"}
          </button>
          <button
            onClick={fullscreen}
            title="Full screen"
            aria-label="Full screen"
            className="flex items-center gap-1.5 px-2 py-1 rounded-md border border-border hover:border-accent hover:text-accent focus:outline-none focus-visible:border-accent transition"
          >
            <Maximize2 size={13} /> Full screen
          </button>
        </div>
      </div>
      <div ref={stage} className="relative bg-black aspect-video flex items-center justify-center">
        <img
          src={api.videoUrl}
          alt="video stream"
          className="w-full h-full object-contain"
          onError={(e) => {
            // A dropped stream (server restart, tunnel blip) retries on its
            // own instead of leaving a dead black panel until a reload.
            const img = e.currentTarget;
            setTimeout(() => { img.src = `${api.videoUrl}?t=${Date.now()}`; }, 2000);
          }}
        />
        {notice && (
          <div role="alert" className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-md bg-danger/90 px-3 py-1.5 text-xs font-medium text-white shadow">
            {notice}
          </div>
        )}
        {/* The caption itself is burned into the video frames by the
            server (draw_overlays), so it is not drawn again here. This copy
            is for screen readers only. */}
        {caption?.text && (
          <p aria-live="polite" className="sr-only">{caption.text}</p>
        )}
      </div>
    </div>
  );
}
