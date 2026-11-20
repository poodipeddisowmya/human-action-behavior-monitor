# Face recognition: who is doing what

Tracked people are named when they have been **enrolled**: "Ramesh is taking
tablets in a living room." Everyone else stays **Unknown**, with their
tracking number (P001...). Names appear on the video, in the subtitle, in
the events log and on its proof images.

## Models

| | Model | Licence | On the Orin Nano's CPU |
|---|---|---|---|
| Find faces | YuNet (OpenCV Zoo, 2023) | MIT | 25–80 ms a frame |
| Recognise | SFace (OpenCV Zoo, 2021) | Apache-2.0 | ~25 ms a face |

Both run through OpenCV, already installed: no new dependency, no GPU
memory. InsightFace's models, the other common choice, are licensed for
non-commercial research only and are not used. The model files are in
`models/face/` (Git LFS).

## How a person gets a name

A background thread looks at each tracked person's head a few times a
second until they are named, then every 4 s to confirm.

- Only clear faces count: detection score ≥ 0.85 and at least 40 px wide.
  Blurred, tiny or turned-away faces are skipped, not guessed.
- The face must be at this person's own head (their nose and eye
  keypoints). In a crowd the head crop can also hold someone else's larger
  face; taking the largest face named a man after the woman in front of him
  (classroom clip, 2026-11-20). Without head keypoints a face counts only
  if it is the single face in the upper half of the person's box.
- A face matches an enrolled person when its cosine similarity to their best
  photo is at least 0.363 (SFace's published bar) **and** 0.05 ahead of the
  next enrolled person.
- A person is named when 2 of their last 3 clear looks agree, and keeps the
  name; switching to another name takes 3 agreeing looks in a row.

Settings: `face_id:` in `config/model_config.yaml`; `FACE_ID=0` turns it off.

### Measured (Jetson, 2026-11-19)

| | Result |
|---|---|
| Same person across a 14 s clip, clear faces | similarity 0.45–0.96 |
| Different people, any pair | at most 0.27 |
| Two people enrolled from one photo each, their own clips | named within 1.5 s, held 27 of 28 looks |
| Four other people (74 looks) | all Unknown |
| Live dashboard | "Ramesh is taking tablets indoors." |

## Enrolling people

### From the video: Faces on screen

The live monitor's **Faces on screen** panel shows everyone in view whose
face is clear (at least 60 px wide) and who has no name yet, with their face.
Type a name (or pick someone already enrolled) and press **Name**:

- their 3 best faces from the video are enrolled, with the same checks as
  photos (below);
- the box, subtitle and log name them at once, and the activities they
  were logged doing since they came into view get their name too;
- from then on they are recognised in any video, with a new tracking number.

Tested on a classroom clip: named from the panel in one step, 8 earlier
activities named; after a server restart she was recognised again (similarity 0.89)
and the two others in view stayed Unknown.

The faces of people not named are held in memory only, and only while they
are in view: they are dropped when the person leaves or the video changes.
Nothing about them is written to disk or the database.

### Photos (the Known people panel)

- **Add person:** a name and 2–5 clear photos of their face.
- **From the events log:** open any proof image, choose "This face is …",
  pick the person (or a new name), **Enrol face**.
- **Add photos** to someone already enrolled: different lighting and angles
  make recognition steadier.
- **Delete:** removes the person, every face photo and every embedding.
  Past events keep the name they were logged with.

Every photo is checked before it is stored, with the reason shown when it
is refused:

| Refused when | Why |
|---|---|
| no face, a blurred face, a face under 60 px | it would match badly |
| several faces of similar size | it would be a guess which one is meant |
| it looks like a different enrolled person | prevents duplicate or wrong enrolments |
| it does not look like this person's other photos | a wrong photo would make the system name the wrong face |

## Privacy and the law

A face is **biometric data**. Under India's Digital Personal Data
Protection Act 2023 (and the GDPR in Europe) the person must agree to be
enrolled, know what it is for, and be able to have it deleted.

- Enrol people only with their consent; the panel says so.
- Face photos and embeddings stay on the device: `data/faces/` and the
  local database, both git-ignored. Nothing is sent anywhere.
- Deleting a person removes all of their face data at once.
- The API that serves them is behind `ACCESS_TOKEN`: set it on any server
  others can reach.
- People who are not enrolled are never identified; the system does not
  build a gallery of strangers.

## Limits

- Faces must be visible: a back-of-head view, a mask or a distant camera
  (faces under 40 px) leaves the person Unknown.
- Lighting and angle very different from every enrolled photo can leave
  someone Unknown for a while; enrol 3–5 varied photos.
- Twins and close look-alikes can be confused; the 0.05 margin makes it say
  Unknown rather than guess when two enrolled people are both close.
