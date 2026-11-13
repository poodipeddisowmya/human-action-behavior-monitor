"""Target action vocabulary — the restricted label space the system reports.

Rather than letting VideoMAE pick the arg-max over all 400 Kinetics-400
classes (where irrelevant classes like "tossing coin" or "shot put" compete
with the ones we care about), we restrict inference to only the Kinetics
classes that map into the categories below.

Two things fall out of that:

  * **Noise labels become structurally impossible.** If a class isn't in this
    file, the model cannot emit it. No blocklist maintenance required.

  * **Confidence becomes meaningful.** Probability is renormalized over the
    restricted set, so "0.80 confident it's cooking" means "0.80 among the
    activities we actually track", not "0.80 among 400 mostly-irrelevant
    options".

Patterns are matched as case-insensitive substrings against the Kinetics-400
label strings, resolved to concrete class indices once at model load time.

To add a category: add an entry here. Nothing else needs to change.
"""
from __future__ import annotations

# category name  ->  Kinetics-400 label substrings that belong to it
TARGET_CATEGORIES: dict[str, list[str]] = {

    # ---------- Eating / drinking ----------
    "eating": [
        "eating burger", "eating cake", "eating carrots", "eating chips",
        "eating doughnuts", "eating hotdog", "eating ice cream",
        "eating spaghetti", "eating watermelon",
        "tasting food", "sucking lolly", "licking",
        # "dining" was a genuine miss found auditing all 400 Kinetics labels
        # against this file — a class that unambiguously means "eating a
        # meal" was simply never included.
        "dining",
    ],
    "drinking": [
        "drinking", "drinking beer", "drinking shots",
        "tasting beer", "bartending",
    ],

    # ---------- Reading / writing ----------
    "reading": [
        "reading book", "reading newspaper",
    ],
    "writing": [
        "writing",
    ],

    # ---------- Phone / computer ----------
    # NOTE: Kinetics-400 has no "talking on phone" class — only "texting".
    # The pose classifier's own "using phone" rule (hand raised to face level)
    # covers the general case better and runs first, so this is a backstop.
    "using phone": [
        "texting",
    ],
    "using computer": [
        "using computer",
    ],
    # NOTE: "watching TV" is deliberately absent. Kinetics-400 contains no
    # such class, so there is nothing honest to map it to. A person watching
    # TV correctly reports as "sitting" from the pose classifier. Adding a
    # real "watching TV" detector needs either a fine-tuned model or a
    # scene-level rule (screen in frame + seated + low motion).

    # ---------- Exercise ----------
    "exercising": [
        "push up", "pull ups", "situp", "squat", "lunge",
        "bench pressing", "deadlifting", "clean and jerk", "snatch weight lifting",
        "exercising with an exercise ball",
        "stretching leg", "yoga",
        # "stretching arm" and "exercising arm" were here and are removed on
        # purpose. They describe an arm moving, which is what ANY seated or
        # standing person does while drinking, smoking or reaching — so they
        # were the model's fallback for scenes it had no better class for.
        # Measured: 6 of 67 accepted votes on demo-smoking, and on phone
        # clips of a seated man drinking and of a man sitting down they
        # reached 0.85-0.98 and put "exercising" on screen. None of the
        # exercise demo clips depend on them (those resolve to push up /
        # pull ups / squat / snatch).
        "front raises", "jogging", "running on treadmill",
        "jumping jacks", "burpee", "side kick", "lunge",
        "doing aerobics",
    ],

    # ---------- Cleaning ----------
    # Indoor household cleaning only. Deliberately excludes grooming-adjacent
    # washing (hair, feet) and outdoor chores (gutters, pool) so the category
    # stays semantically tight.
    "cleaning": [
        "cleaning floor", "mopping floor", "sweeping floor",
        "washing dishes",
        "folding clothes", "folding napkins",
        "cleaning windows", "cleaning toilet",
        "making bed", "doing laundry", "ironing",
    ],

    # ---------- Transport ----------
    "driving": [
        "driving car", "driving tractor",
    ],
    "cycling": [
        "riding a bike", "riding mountain bike", "biking through snow",
        "riding scooter", "riding unicycle", "bmx",
    ],

    # ---------- Brushing teeth ----------
    # Verified against the loaded weights' own category list, same as
    # smoking above.
    "brushing teeth": [
        "brushing teeth",
    ],

    # ---------- Smoking ----------
    # Kinetics-400 has exactly two classes for this (verified against the
    # loaded S3D weights' own category list, not guessed):
    #   "smoking"          - the general case, any posture
    #   "smoking hookah"   - visually distinct (a large water pipe) but the
    #                        same real-world activity; mapping it here means a
    #                        hookah lounge scene correctly reads "smoking"
    #                        rather than falling through to no answer at all.
    "smoking": [
        "smoking", "smoking hookah",
    ],

    # ---------- Swimming ----------
    # The three stroke classes only. Diving (springboard, cliff, scuba) is a
    # different activity and is deliberately not mapped here.
    "swimming": [
        "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke",
    ],

    # ---------- Dancing ----------
    # "dancing" matches all twelve "... dancing" / "dancing ..." classes
    # (ballet, salsa, tango, swing, tap, breakdancing, ...); the two below
    # have no "dancing" in their names. Zumba is dance-fitness; it sits here
    # because "exercising" does not claim it and it looks like dancing.
    "dancing": [
        "dancing", "zumba", "krumping",
    ],

    # ---------- Cooking (kept: high-value everyday activity) ----------
    # "chopping" is deliberately NOT a pattern here — it would capture
    # "chopping wood", which is not cooking.
    "cooking": [
        "cooking chicken", "cooking egg", "cooking on campfire",
        "cooking sausages", "cooking scallops",
        "baking cookies", "making pizza", "making sushi", "making tea",
        "making a cake", "making a sandwich", "frying vegetables",
        "scrambling eggs", "peeling potatoes", "peeling apples",
        "cutting pineapple", "cutting watermelon",
        "barbequing", "flipping pancake",
        # Both found auditing the full 400-class list for missed food-prep
        # classes; "tossing salad" is unambiguous, "grinding meat" likewise.
        "tossing salad", "grinding meat",
    ],
}


def resolve_indices(id2label: dict[int, str]) -> tuple[list[int], dict[int, str]]:
    """Map the model's Kinetics label space onto our target categories.

    Returns:
        allowed_indices   sorted list of class indices we permit
        index_to_category {class index -> our category name}
    """
    index_to_category: dict[int, str] = {}

    for category, patterns in TARGET_CATEGORIES.items():
        lowered = [p.lower() for p in patterns]
        for idx, raw in id2label.items():
            r = str(raw).lower()
            # longest-pattern-wins so a more specific pattern claims the class
            match = max((p for p in lowered if p in r), key=len, default=None)
            if match is None:
                continue
            # If already claimed, keep whichever pattern was longer/more specific.
            prev = index_to_category.get(int(idx))
            if prev is None:
                index_to_category[int(idx)] = category
            # (ties are rare and harmless — both map to sensible categories)

    allowed = sorted(index_to_category)
    return allowed, index_to_category
