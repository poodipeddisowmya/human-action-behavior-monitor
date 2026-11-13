import { useCallback, useEffect, useRef, useState } from "react";

// Streams this device's webcam to the server (/ws/camera) as JPEG frames.
//
// Used when the server has no camera of its own — a Kaggle notebook or a
// cloud GPU. The annotated result comes back through the normal video
// stream, so nothing else on the page changes.
//
// Flow control: the next frame is sent only after the server acknowledges
// the previous one. On a slow uplink the frame rate drops instead of a
// backlog building up, so what is shown is always close to "now".

const WIDTH = 960;          // the detector works at 640 px; more is wasted upload
const MAX_FPS = 15;
const JPEG_QUALITY = 0.72;

export type CameraState = "off" | "starting" | "on" | "error";

export function useBrowserCamera() {
  const [state, setState] = useState<CameraState>("off");
  const [error, setError] = useState<string | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const running = useRef(false);

  const stop = useCallback(() => {
    running.current = false;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    ws.current?.close();
    ws.current = null;
    setState("off");
  }, []);

  const start = useCallback(async (): Promise<boolean> => {
    if (running.current) return true;
    setError(null);
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setError("The browser only allows the camera on https:// or localhost.");
      setState("error");
      return false;
    }
    setState("starting");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: WIDTH }, height: { ideal: 540 }, frameRate: { ideal: MAX_FPS } },
        audio: false,
      });
    } catch (e: any) {
      setError(e?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in the address bar and try again."
        : "No camera could be opened on this device.");
      setState("error");
      return false;
    }

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream.current;
    await video.play().catch(() => {});

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws/camera`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    running.current = true;

    let lastSent = 0;
    const send = () => {
      if (!running.current || sock.readyState !== WebSocket.OPEN) return;
      const wait = 1000 / MAX_FPS - (performance.now() - lastSent);
      if (wait > 0) { setTimeout(send, wait); return; }
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) { setTimeout(send, 100); return; }
      const scale = Math.min(1, WIDTH / vw);
      canvas.width = Math.round(vw * scale);
      canvas.height = Math.round(vh * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {
        if (!blob || !running.current || sock.readyState !== WebSocket.OPEN) return;
        lastSent = performance.now();
        sock.send(await blob.arrayBuffer());
      }, "image/jpeg", JPEG_QUALITY);
    };

    return new Promise<boolean>((resolve) => {
      sock.onopen = () => { setState("on"); send(); resolve(true); };
      sock.onmessage = () => send();             // ack -> next frame
      sock.onerror = () => {
        stop();                                  // release the camera too
        setError("Could not connect the camera to the server.");
        setState("error");
        resolve(false);
      };
      sock.onclose = () => { if (running.current) stop(); };
    });
  }, [stop]);

  useEffect(() => stop, [stop]);                 // release the camera on unmount

  return { state, error, start, stop };
}
