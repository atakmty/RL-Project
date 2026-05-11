"""Check TensorBoard log contents from WSL paths."""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = './tensorboard_logs/PPO_2'
event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

if not event_files:
    print("No event files found!")
    exit(1)

print(f"Event file: {os.path.basename(event_files[0])}")
print(f"File size: {os.path.getsize(event_files[0])/1024:.1f} KB")

ea = EventAccumulator(log_dir)
ea.Reload()
tags = ea.Tags()
scalar_tags = tags.get('scalars', [])
print(f"\nTotal scalar tags: {len(scalar_tags)}")
print("=" * 80)

for tag in sorted(scalar_tags):
    events = ea.Scalars(tag)
    vals = [e.value for e in events]
    if vals:
        print(f"  {tag:<40s} n={len(vals):>4d}  min={min(vals):>9.4f}  max={max(vals):>9.4f}  last={vals[-1]:>9.4f}")

# Check custom episode metrics
print("\n" + "=" * 80)
print("Custom per-objective metrics:")
print("=" * 80)
expected = ["episode/r_struct", "episode/r_gc", "episode/p_homo",
            "episode/r_mfe", "episode/is_success"]
for tag in expected:
    if tag in scalar_tags:
        events = ea.Scalars(tag)
        vals = [e.value for e in events]
        print(f"  OK  {tag:<30s} ({len(vals)} data points)")
        if len(vals) >= 5:
            print(f"       First 3: {[round(v,4) for v in vals[:3]]}")
            print(f"       Last  3: {[round(v,4) for v in vals[-3:]]}")
        elif vals:
            print(f"       Values: {[round(v,4) for v in vals]}")
    else:
        print(f"  XX  {tag:<30s} MISSING!")
