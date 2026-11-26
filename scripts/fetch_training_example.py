"""Download the example training set: people asleep at desks, and ordinary
office activity, so the whole training workflow (docs/TRAINING.md) can be
tried before collecting clips from your own cameras.

    python scripts/fetch_training_example.py
    python scripts/train_activities.py

Stock footage from Pexels (Pexels License: free to use, not to re-share as
is), so it is downloaded rather than kept in the repository:
  data/training/sleeping at desk/   34 clips, various people and authors
  data/training/other/              18 clips of an Indian corporate office
                                    (RDNE Stock project): desk work, phone
                                    calls, meetings, a pantry
Each clip's page is https://www.pexels.com/video/<file name>/.

Clips from your own cameras teach the monitor far better than these: use
them as a start, then add or replace with your own.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "training"
_UA = "human-action-behavior-monitor/1.0 (demo clip fetcher)"

SLEEPING = [
    ("man-sleeping-on-a-desk-7581176", "https://videos.pexels.com/video-files/7581176/7581176-hd_1280_720_30fps.mp4"),
    ("man-sleeping-on-table-7581344", "https://videos.pexels.com/video-files/7581344/7581344-hd_1280_720_30fps.mp4"),
    ("man-sleeping-on-a-table-7581346", "https://videos.pexels.com/video-files/7581346/7581346-hd_1280_720_30fps.mp4"),
    ("woman-sleeping-on-her-desk-7882762", "https://videos.pexels.com/video-files/7882762/7882762-hd_1280_720_60fps.mp4"),
    ("a-man-sleeping-on-his-office-desk-then-waking-up-6615062", "https://videos.pexels.com/video-files/6615062/6615062-hd_1280_720_25fps.mp4"),
    ("a-student-sleeping-on-a-desk-9489688", "https://videos.pexels.com/video-files/9489688/9489688-hd_1280_720_25fps.mp4"),
    ("a-woman-sleeping-in-front-of-her-laptop-4363839", "https://videos.pexels.com/video-files/4363839/4363839-hd_1280_720_50fps.mp4"),
    ("woman-sleeping-at-work-7882753", "https://videos.pexels.com/video-files/7882753/7882753-hd_1280_720_60fps.mp4"),
    ("woman-sleeping-at-work-7882755", "https://videos.pexels.com/video-files/7882755/7882755-hd_1280_720_60fps.mp4"),
    ("a-woman-sleeping-on-her-desk-7844957", "https://videos.pexels.com/video-files/7844957/7844957-hd_1280_720_30fps.mp4"),
    ("a-woman-sleeping-beside-her-laptop-while-sitting-down-4115789", "https://videos.pexels.com/video-files/4115789/4115789-hd_1280_720_50fps.mp4"),
    ("a-tired-woman-sleeping-on-a-table-7882756", "https://videos.pexels.com/video-files/7882756/7882756-hd_1280_720_60fps.mp4"),
    ("a-woman-sleeping-on-the-desk-beside-her-laptop-4115788", "https://videos.pexels.com/video-files/4115788/4115788-hd_1280_720_50fps.mp4"),
    ("young-girl-napping-over-the-drawing-table-6476164", "https://videos.pexels.com/video-files/6476164/6476164-hd_1280_720_25fps.mp4"),
    ("student-sleeping-in-classroom-6672265", "https://videos.pexels.com/video-files/6672265/6672265-hd_1280_720_24fps.mp4"),
    ("boy-sleeping-on-library-8499776", "https://videos.pexels.com/video-files/8499776/8499776-hd_1280_720_30fps.mp4"),
    ("a-man-fell-asleep-while-working-6614775", "https://videos.pexels.com/video-files/6614775/6614775-hd_720_1280_25fps.mp4"),
    ("a-man-fell-asleep-while-studying-8478111", "https://videos.pexels.com/video-files/8478111/8478111-hd_1280_720_25fps.mp4"),
    ("young-man-falling-asleep-in-office-6614770", "https://videos.pexels.com/video-files/6614770/6614770-hd_1280_720_25fps.mp4"),
    ("person-sleeping-on-table-8395244", "https://videos.pexels.com/video-files/8395244/8395244-hd_720_1280_24fps.mp4"),
    ("woman-sleeping-over-her-studies-8086497", "https://videos.pexels.com/video-files/8086497/8086497-hd_1366_720_25fps.mp4"),
    ("muslim-girl-napping-on-a-stack-of-books-6245553", "https://videos.pexels.com/video-files/6245553/6245553-hd_1366_720_30fps.mp4"),
    ("woman-sleeping-on-her-desk-7605069", "https://videos.pexels.com/video-files/7605069/7605069-hd_1366_720_25fps.mp4"),
    ("elderly-man-sleeping-at-work-8719401", "https://videos.pexels.com/video-files/8719401/8719401-hd_1366_720_25fps.mp4"),
    ("tired-woman-sleeping-on-a-desk-7605075", "https://videos.pexels.com/video-files/7605075/7605075-hd_720_1366_25fps.mp4"),
    ("a-woman-sleeping-on-her-work-desk-9063078", "https://videos.pexels.com/video-files/9063078/9063078-hd_1366_720_25fps.mp4"),
    ("tired-woman-sleeping-on-the-table-7983323", "https://videos.pexels.com/video-files/7983323/7983323-hd_1366_720_25fps.mp4"),
    ("a-tired-woman-fell-asleep-while-working-8086500", "https://videos.pexels.com/video-files/8086500/8086500-hd_1366_720_25fps.mp4"),
    ("an-elderly-man-sleeping-on-the-job-8719396", "https://videos.pexels.com/video-files/8719396/8719396-hd_1366_720_25fps.mp4"),
    ("woman-sleeping-over-her-studies-8086504", "https://videos.pexels.com/video-files/8086504/8086504-hd_1366_720_25fps.mp4"),
    ("a-man-sleeping-on-the-job-8719390", "https://videos.pexels.com/video-files/8719390/8719390-hd_720_1366_25fps.mp4"),
    ("woman-fell-asleep-on-the-table-while-working-8086498", "https://videos.pexels.com/video-files/8086498/8086498-hd_1366_720_25fps.mp4"),
    ("woman-sleeping-in-desk-7605081", "https://videos.pexels.com/video-files/7605081/7605081-hd_720_1366_25fps.mp4"),
    ("a-woman-sleeping-on-her-work-desk-9063074", "https://videos.pexels.com/video-files/9063074/9063074-hd_720_1366_25fps.mp4"),
]

OTHER = [
    ("boss-talking-on-the-phone-7581264", "https://videos.pexels.com/video-files/7581264/7581264-hd_1280_720_30fps.mp4"),
    ("man-analyzing-documents-7581201", "https://videos.pexels.com/video-files/7581201/7581201-hd_1280_720_30fps.mp4"),
    ("man-busy-with-paperworks-7581303", "https://videos.pexels.com/video-files/7581303/7581303-hd_1280_720_30fps.mp4"),
    ("man-checking-the-documents-7581169", "https://videos.pexels.com/video-files/7581169/7581169-hd_1280_720_30fps.mp4"),
    ("man-checking-the-documents-7581380", "https://videos.pexels.com/video-files/7581380/7581380-hd_1280_720_30fps.mp4"),
    ("man-holding-paper-while-thinking-7581238", "https://videos.pexels.com/video-files/7581238/7581238-hd_1280_720_30fps.mp4"),
    ("man-moving-out-from-his-office-7581306", "https://videos.pexels.com/video-files/7581306/7581306-hd_1280_720_30fps.mp4"),
    ("man-talking-on-the-telephone-7581273", "https://videos.pexels.com/video-files/7581273/7581273-hd_1280_720_30fps.mp4"),
    ("man-working-in-an-office-7581337", "https://videos.pexels.com/video-files/7581337/7581337-hd_1280_720_30fps.mp4"),
    ("man-working-on-his-laptop-while-talking-on-the-phone-7581373", "https://videos.pexels.com/video-files/7581373/7581373-hd_1280_720_30fps.mp4"),
    ("men-having-conversation-while-on-break-7581299", "https://videos.pexels.com/video-files/7581299/7581299-hd_1280_720_30fps.mp4"),
    ("people-at-the-office-having-a-meeting-7581170", "https://videos.pexels.com/video-files/7581170/7581170-hd_1280_720_30fps.mp4"),
    ("people-holding-mugs-having-a-conversation-7688038", "https://videos.pexels.com/video-files/7688038/7688038-hd_1280_720_30fps.mp4"),
    ("people-working-in-an-office-7581212", "https://videos.pexels.com/video-files/7581212/7581212-hd_1280_720_30fps.mp4"),
    ("people-working-on-office-using-computer-7581202", "https://videos.pexels.com/video-files/7581202/7581202-hd_1280_720_30fps.mp4"),
    ("video-of-a-man-working-while-thinking-7581227", "https://videos.pexels.com/video-files/7581227/7581227-hd_1280_720_30fps.mp4"),
    ("woman-holding-a-mug-while-talking-on-her-phone-7581290", "https://videos.pexels.com/video-files/7581290/7581290-hd_1280_720_30fps.mp4"),
    ("woman-talking-to-her-officemates-7581162", "https://videos.pexels.com/video-files/7581162/7581162-hd_1280_720_30fps.mp4"),
]


def main() -> int:
    failed = 0
    for folder, clips in (("sleeping at desk", SLEEPING), ("other", OTHER)):
        d = DEST / folder
        d.mkdir(parents=True, exist_ok=True)
        for name, url in clips:
            out = d / f"{name}.mp4"
            if out.exists() and out.stat().st_size > 0:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(req, timeout=120) as r, open(out.with_suffix(".part"), "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                out.with_suffix(".part").replace(out)
                print(f"  {folder}/{out.name}", flush=True)
            except Exception as e:
                out.with_suffix(".part").unlink(missing_ok=True)
                failed += 1
                print(f"  FAILED {name}: {e}", flush=True)
    phrase = DEST / "sleeping at desk" / "phrase.txt"
    if not phrase.exists():
        phrase.write_text("sleeping at their desk\n", encoding="utf-8")
    print(f"\n{len(SLEEPING) + len(OTHER) - failed} of {len(SLEEPING) + len(OTHER)} clips in {DEST}."
          + ("  Run it again for the others." if failed else
             "  Next: python scripts/train_activities.py"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
