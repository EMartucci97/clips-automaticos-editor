"""Ad hoc review tool: split a highlight window into camera shots and save
one annotated frame per shot (faces boxed + labeled by track id), so a
human can pick which track is the speaker per shot before regenerating.

Usage: python _review_clip.py <source> <start> <end> <name>
Writes output/_review/<name>/shot_XX.png and prints a summary table.
"""
import os
import sys

from shorts_generator.local.active_speaker import list_shots_for_review, render_shot_image
from shorts_generator.local.clipper import _cut_subclip

source, start, end, name = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]

review_dir = os.path.join("output", "_review", name)
os.makedirs(review_dir, exist_ok=True)
cut_path = os.path.join(review_dir, "_cut.mp4")

print(f"cutting [{start},{end}] from {source} ...", flush=True)
_cut_subclip(source, start, end, cut_path)

print("analyzing shots ...", flush=True)
shots = list_shots_for_review(cut_path)

print(f"\n{len(shots)} shots found:\n")
for shot in shots:
    idx = shot["shot_index"]
    tids = [f["track_id"] for f in shot["faces"]]
    img_path = os.path.join(review_dir, f"shot_{idx:02d}.png")
    if shot["frame_index"] is not None and shot["faces"]:
        render_shot_image(cut_path, shot["frame_index"], shot["faces"], img_path)
        note = img_path
    else:
        note = "(no face detected in this shot)"
    print(f"  shot {idx:2d}  [{shot['start_time']:.1f}s - {shot['end_time']:.1f}s]  tracks={tids}  {note}")

print(f"\nimages saved under {review_dir}/")
