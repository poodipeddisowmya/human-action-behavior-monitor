import { useEffect, useRef, useState } from "react";
import { Film, Upload, Video, VideoOff } from "lucide-react";
import { api, BROWSER_CAMERA, Camera, Clip, isLive } from "../services/api";

export default function SourceSelector() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // The clip that was playing before going live, so "Stop live" returns to it.
  const lastClip = useRef<string | null>(null);

  const refresh = () =>
    api.listSources().then((r) => {
      setClips(r.clips);
      setCameras(r.cameras ?? []);
      setCurrent(r.current);
    }).catch(() => {});

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, []);

  const change = async (filename: string) => {
    if (busy || filename === current) return;
    if (current && !isLive(current)) lastClip.current = current;
    setBusy(true);
    try {
      await api.switchSource(filename);
      setCurrent(filename);
    } finally {
      setBusy(false);
    }
  };

  const serverLive = !!current && current.startsWith("camera:");
  const toggleLive = () => {
    if (serverLive) change(lastClip.current ?? clips[0]?.filename ?? "");
    else if (cameras[0]) change(cameras[0].filename);
  };

  const upload = async (file: File) => {
    setBusy(true);
    setProgress(0);
    try {
      const res = await api.uploadSource(file, (p) => setProgress(p));
      await refresh();
      if (res?.filename) setCurrent(res.filename);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
      setProgress(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="flex items-center gap-3">
      <label className="flex items-center gap-2 text-xs text-muted">
        <Film size={14} className="text-accent" />
        <span>Source</span>
        <select
          value={current ?? ""}
          disabled={busy}
          onChange={(e) => change(e.target.value)}
          className="bg-surface border border-border rounded-md px-2 py-1 text-sm text-slate-100 focus:outline-none focus:border-accent disabled:opacity-50"
        >
          {(cameras.length > 0 || current === BROWSER_CAMERA) && (
            <optgroup label="Live cameras">
              {cameras.map((c) => (
                <option key={c.filename} value={c.filename}>
                  ● {c.name}
                </option>
              ))}
              {/* Started from "Go live" on the video panel; shown here so
                  the list reflects what is playing. */}
              {current === BROWSER_CAMERA && (
                <option value={BROWSER_CAMERA} disabled>● Your webcam</option>
              )}
            </optgroup>
          )}
          <optgroup label="Video clips">
            {clips.map((c) => (
              <option key={c.filename} value={c.filename} title={c.description}>
                {c.name}
              </option>
            ))}
          </optgroup>
        </select>
      </label>

      {/* Server-attached cameras only. With none (a cloud GPU), the
          video panel's "Go live" uses the viewer's own webcam instead. */}
      {cameras.length > 0 && (
        <button
          onClick={toggleLive}
          disabled={busy}
          title={serverLive ? "Stop the live camera and go back to the clip"
                 : `Start live detection from ${cameras[0].name}`}
          className={
            "flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md border transition disabled:opacity-50 " +
            (serverLive
              ? "border-danger text-danger bg-danger/10 hover:bg-danger/20"
              : "border-border bg-surface hover:border-accent hover:text-accent")
          }
        >
          {serverLive ? <VideoOff size={13} /> : <Video size={13} />}
          {serverLive ? "Stop live" : "Live camera"}
        </button>
      )}

      <button
        onClick={() => fileRef.current?.click()}
        disabled={busy}
        title="Upload a video (mp4/avi/mov/mkv/webm, up to 500 MB)"
        className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md border border-border bg-surface hover:border-accent hover:text-accent transition disabled:opacity-50"
      >
        <Upload size={13} />
        {progress === null
          ? "Upload"
          : `${Math.round(progress * 100)}%`}
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm"
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        className="hidden"
      />
    </div>
  );
}
