// Readable names for the clips in data/samples/, shared by the Source menu
// and the Overview page. File names stay what they are on disk.

import { Clip } from "./api";

// The activities added on top of the Kinetics model (docs/NEW_ACTIVITIES.md),
// with the local demo clips that show them (data/samples/new-*, git-ignored:
// the stock footage may not be re-shared, so other machines have none).
export const NEW_ACTIVITIES = [
  {
    label: "making tea",
    title: "Making tea",
    how: "Described in words to a video transformer (X-CLIP). No training clips.",
    clips: ["new-making-tea-1.mp4", "new-making-tea-2.mp4", "new-making-tea-3.mp4"],
  },
  {
    label: "taking tablets",
    title: "Taking tablets",
    how: "Described in words to a video transformer (X-CLIP). No training clips.",
    clips: ["new-taking-tablets-1.mp4", "new-taking-tablets-2.mp4"],
  },
  {
    label: "falling down",
    title: "Falling down",
    how: "From the skeleton over time: upright, then on the ground within a second. Shown as an alert.",
    clips: ["new-falling-down.mp4"],
  },
] as const;

export const NEW_LABELS: string[] = NEW_ACTIVITIES.map((a) => a.label);

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// Indian corporate office (data/samples/office-*, fetched by
// scripts/fetch_new_activity_clips.py; Pexels, RDNE Stock project).
// office-people.mp4 is the older corridor tracking scene.
const isOffice = (f: string) => f.startsWith("office-") && f !== "office-people.mp4";

export function clipGroup(filename: string): "Office" | "New activities" | "Activities" | "Scenes" {
  if (isOffice(filename)) return "Office";
  if (filename.startsWith("new-")) return "New activities";
  if (filename.startsWith("demo-") || filename.startsWith("hd-")) return "Activities";
  return "Scenes";
}

/** "new-making-tea-2.mp4" -> "Making tea 2"; "hd-dancing-3.mp4" -> "Dancing 3 (HD)". */
export function clipTitle(filename: string): string {
  const stem = filename.replace(/\.[^.]+$/, "");
  const hd = stem.startsWith("hd-");
  const BASIC: Record<string, string> = { run: "running", sit: "sitting", walk: "walking", stand: "standing" };
  let t = stem.replace(isOffice(filename) ? /^office-/ : /^(new|demo|hd)-/, "").replace(/^basic-(\w+)/, (_, w) => BASIC[w] ?? w);
  t = t.replace(/^exercising-(\w+)/, "exercising ($1)").replace(/-(\d+)$/, " $1").replace(/-/g, " ");
  return cap(t) + (hd ? " (HD)" : "");
}

export function groupClips(clips: Clip[]): [string, Clip[]][] {
  const order = ["Office", "New activities", "Activities", "Scenes"];
  const groups = new Map<string, Clip[]>(order.map((g) => [g, []]));
  clips.forEach((c) => groups.get(clipGroup(c.filename))!.push(c));
  groups.forEach((list) => list.sort((a, b) => clipTitle(a.filename).localeCompare(clipTitle(b.filename))));
  return [...groups].filter(([, list]) => list.length > 0);
}
