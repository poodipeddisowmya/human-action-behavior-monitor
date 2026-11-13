# Client demo on Kaggle's free GPU

Runs the whole system on a Kaggle **Tesla T4** and gives you a public **HTTPS**
link to the live dashboard. The client opens it in any browser — laptop or
phone — and can watch demo clips, upload their own footage, or press
**Go live** to use *their own webcam*, processed on the GPU.

## Measured on the Kaggle T4 (2026-11-04)

| | Result |
|---|---|
| Live accuracy, 15 demo clips | **15/15**, correct label on screen in 0.5–2.3 s |
| Detection per frame | 14–17 ms (video plays at the clip's own frame rate) |
| Through the public link | ~20 fps video; webcam round trip 7 ms |
| Stability | 0 restarts, 0 errors; 0.7 GB of 15 GB GPU memory |

## Once per account

1. **Verify your phone** at kaggle.com → Settings → Phone verification.
   Without it Kaggle silently gives sessions *no GPU and no internet*.
2. Sign this PC in to Kaggle: save an API token from kaggle.com → Settings →
   API as `%USERPROFILE%\.kaggle\access_token` (or run
   `.venv\Scripts\kaggle.exe auth login`).

## After any code change — one click

Double-click **`kaggle_gpu.bat`**. It:

1. packages the project and uploads it as a **private** Kaggle dataset,
2. runs a full end-to-end test on a T4 (accuracy on GPU, real-time FPS,
   live server, public link, browser-webcam path, batch report),
3. downloads the report to `dist\kaggle-output\gpu_report.md`,
4. saves the **HAM GPU Demo** notebook in your account (not run, no quota used).

## The demo — one cell

Open https://www.kaggle.com/code/sowmyasrmaha/ham-gpu-demo → **Edit** →
**Run All**. In ~4 minutes (≈1 minute on later runs in the same session) it
shows a green box with the **client link**. Send that whole link — the access
token is part of it; without it the page asks for one. Leave the cell running
during the demo: it keeps the session alive and restarts the server if needed.

## During the demo

| Show | How |
|---|---|
| It runs on a GPU, in real time | Header badge: **GPU · Tesla T4 · fps · ms** |
| Activity recognition | Source → `demo-cooking`, `demo-using-computer`, `demo-exercising-pushup` |
| Multi-person tracking | Source → `office-people`, `store-aisle`, `street-surveillance` |
| Their own footage | **Upload** (keep under ~100 MB over the public link) |
| Their own camera | **Go live** → allow camera. Good light, 1–3 m away |
| Clean view | **Full screen** |

If the link ever fails, the notebook's **offline report** cell processes any
clip with the same pipeline and plays the annotated result inside Kaggle.

## Limits

- 30 GPU hours a week; a session ends after 12 h. Run the **stop** cell after the demo.
- The `trycloudflare.com` address changes each start. For a fixed address add
  an `NGROK_AUTHTOKEN` secret (Add-ons → Secrets); `ACCESS_TOKEN` fixes the token.
- Kaggle is for demos. Production deployment: [DEPLOYMENT.md](DEPLOYMENT.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Report says `no GPU` and `internet: false` | Phone verification (step 1 above). |
| `Add the 'human-action-monitor' dataset` | Right panel → Add Input → your dataset. |
| `server exited during start-up` | The log tail is printed; also `!python scripts/kaggle_launch.py logs 100`. |
| Link opens, video black | Wait 10 s; it reconnects on its own. |
| **Go live** says camera not allowed | Allow the camera in the browser's address bar. |
