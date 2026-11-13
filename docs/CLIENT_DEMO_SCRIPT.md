# Client demo script — Human Action & Behavior Monitor

About 15 minutes. Words in *italics* are what to say; **bold** is what to click.

---

## Before the client joins (5 minutes)

1. Double-click **`run.bat`**. Edge opens at `http://localhost:8000`.
2. Click **CPU vs GPU** once, check the videos play, then go back to **Live monitor**.
3. Have one short video of your own ready to upload (a phone clip of someone eating,
   typing, exercising — under 100 MB).
4. Close other heavy programs (the laptop CPU does the live part).
5. Good light on your face if you will use the webcam.

---

## 1. Opening — the problem and the answer (1 minute)

*"Cameras record everything, but nobody has time to watch them. This system watches for
you. It finds every person, follows them with a stable ID, recognises what they are doing,
and writes a plain-English description of the scene — in real time."*

*"I'll show you three things: the live system, the difference a GPU makes measured on the
same clips, and how we add new activities for your site."*

---

## 2. Live monitor (4 minutes) — the product working

On **Live monitor**:

1. **Source** list (top right) → **demo-exercising-pushup**.
   *"Each person gets a box and an ID — P001. The label is the recognised activity with its
   confidence."*
2. Point to the subtitle at the bottom of the video.
   *"This sentence is written by the system, from what it sees."*
3. **Source** → **demo-using-computer**, then **demo-writing**.
   *"Different activity, recognised within a couple of seconds. Everything it recognises is
   also logged with start time and duration"* — scroll to the **events table**.
4. **Source** → **office-people** or **store-aisle**.
   *"Several people at once, each tracked separately."*
5. **Upload** → your own clip.
   *"Any video you give us — it starts processing immediately."*
6. Optional: **Go live** → webcam. Do something clear (type on a keyboard, drink from a cup).
   *"And it works on a live camera."*

*"This is running on my laptop's processor. Now let me show you what changes on a GPU."*

---

## 3. CPU vs GPU — the proof (5 minutes)

Click **CPU vs GPU**.

*"Same 29 video clips, same software, the same AI models. The only difference is the
hardware: left is a laptop processor, right is an NVIDIA Tesla T4 graphics processor."*

Point at the four figures at the top:

| | Without GPU | With GPU |
|---|---|---|
| Activities clearly recognised | 1 of 29 | **23 of 29** |
| Correct label on screen (average) | 4% | **70%** |
| Correct label appeared at all | 2 of 29 | **26 of 29** |
| Playback speed | 18 fps | **24 fps** |

*"On a laptop, models this strong simply cannot keep up — it almost never gets to the
answer. On a GPU it recognises 23 of 29 activities, and shows the right label about 1.2
seconds after the activity starts."*

Then click through three or four pairs (both videos play in sync):

- **Dancing (HD)** — 0% → 93%. *"Left: still analysing. Right: dancing, from the first second."*
- **Eating (HD)** — 0% → 95%. Point to the GPU subtitle: *"The person is eating a sandwich
  with avocado and eggs, holding a fork."* *"That description is generated live from the video."*
- **Cleaning mopping** — 0% → 84%. *"Older, low-resolution footage — still recognised."*
- **Street surveillance** — 0.0 → 2.2 people. *"In a crowd, the GPU finds and tracks people the
  laptop misses completely."*

*"Every figure here is what the screen showed — nothing is hand-picked. 'Show all 30' lists every
clip, including the ones it still gets wrong."*

---

## 4. Honest limits and the next step (2 minutes)

Say this — it builds trust:

*"Today the system uses general AI models trained on hundreds of thousands of public videos.
It does well on clear, everyday activities. On footage it has never seen — unusual camera
angles, crowds, people far away — it is right roughly 55 to 70 percent of the time."*

*"The step from here to production accuracy is training on your own cameras: a few hundred
short labelled clips per activity. We train on a GPU, test on footage the model has never
seen, and agree a target with you before go-live — for example 90% per activity."*

---

## 5. "Can we add our own activities?" (2 minutes)

*"Yes, in two ways."*

**Way 1 — activities the AI already knows (minutes, no training).**
The model knows 400 everyday actions — for example washing dishes, folding clothes, sweeping,
yoga, playing guitar, hugging, shaking hands, clapping, jogging. Adding one is a single line.
*"That is exactly how we added swimming and dancing this week, and measured them on real
footage."*

How to do it (if they want to try):
1. Open `src/action_recognition/target_vocabulary.py`.
2. Add a line, e.g. `"housekeeping": ["washing dishes", "folding clothes", "sweeping floor"],`
3. Stop the server (Ctrl+C) and double-click `run.bat` again.
4. **Upload** a clip of that activity — the new label appears.

**Way 2 — activities specific to your site (training).**
For example "wearing a safety helmet", "operating machine X", "picking battery cells".
*"You give us footage, we label it — or you do, with our tool — and we train the model on a
GPU. Typically: one to two weeks to collect and label, one to three hours of GPU training,
then a test on unseen footage with the accuracy figures, like the comparison you just saw."*

---

## 6. Close (30 seconds)

*"So: it works today on clear everyday activities, a GPU makes it real-time and accurate, and
we have a clear path to train it on your site. The next step is a short pilot — you share
footage from two or three cameras and the activities that matter to you, and we come back
with measured accuracy on your own video."*

---

## What has been achieved (for your own reference)

- **Detection and tracking**: every person found and followed with a stable ID.
- **15 activities**: brushing teeth, cleaning, cooking, cycling, **dancing**, drinking, driving,
  eating, exercising, reading, smoking, **swimming**, using computer, using phone, writing —
  plus posture and motion (standing, sitting, walking, running, lying, falling…).
- **Detailed subtitles on GPU**: a vision-language model describes the scene in a sentence,
  grounded in what was detected; gender is never guessed; switched off on low-resolution
  video where it would invent details.
- **Stable labels**: brief posture wobbles no longer flicker on screen.
- **GPU vs CPU proof**: same models, 29 clips, 1/29 → 23/29 clearly recognised.
- **Production features**: access-token protection for remote use, health and metrics
  endpoints, self-recovery if a video or camera fails, Docker GPU deployment, one-click Kaggle
  GPU testing, 81 automated tests.
- **Where it runs**: laptop CPU (lighter models), NVIDIA GPU server or cloud (tested on a Tesla
  T4), and NVIDIA Jetson edge devices (install steps ready, not yet tested on a device).

---

## Questions the client may ask

**"How accurate will it be on our cameras?"**
*"On clear footage of the activities shown, very good on a GPU. On your specific cameras we
measure it in the pilot — we don't promise a number before seeing your footage."*

**"What hardware do we need?"**
*"One NVIDIA GPU handles a live stream comfortably; we tested on a Tesla T4. Edge devices like
an NVIDIA Jetson are an option for on-site processing."*

**"Is video stored or sent anywhere?"**
*"It runs on your own hardware. Nothing leaves your network unless you choose a cloud setup.
Remote access is protected by an access token."*

**"Can it handle several cameras?"**
*"Each GPU runs one or more streams depending on resolution; we size this in the pilot."*

**"What if it is wrong?"**
*"Every label carries a confidence, and events are logged with time and duration so they can be
reviewed. Training on your footage is what reduces errors for your site."*

**"Why did some clips fail?"**
*"Those are cases the general model has never been trained for — for example a swimmer mostly
underwater, or people dancing very close together. That is precisely what training on real
site footage fixes."*

---

## If something goes wrong during the demo

| Problem | Fix |
|---|---|
| Page doesn't load | Wait 15 seconds (models loading), refresh. |
| Videos in CPU vs GPU don't play | Click **Play**; some browsers block autoplay. |
| Live video is choppy | Expected on a laptop CPU — use it as the bridge to the GPU comparison. |
| Webcam doesn't start | Allow the camera in the address bar, or skip it. |
| Anything else | Use the **CPU vs GPU** tab — it is recorded and always works. |
