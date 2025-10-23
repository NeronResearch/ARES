import json
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_voxels(data):
    """Extract voxel information as a dict: key=(x,y,z), value=intensity."""
    voxels = {}
    for v in data.get("voxels", []):
        coords = tuple(v["coordinates"])
        voxels[coords] = v.get("intensity", 0)
    return voxels

def compare_voxels(voxels_a, voxels_b, intensity_tolerance=1e-5):
    """Compare two voxel dictionaries and return counts of changes."""
    coords_a = set(voxels_a.keys())
    coords_b = set(voxels_b.keys())

    added = coords_b - coords_a
    removed = coords_a - coords_b
    common = coords_a & coords_b

    intensity_changed = 0
    for c in common:
        if abs(voxels_a[c] - voxels_b[c]) > intensity_tolerance:
            intensity_changed += 1

    total_diff = len(added) + len(removed) + intensity_changed
    return {
        "added": len(added),
        "removed": len(removed),
        "intensity_changed": intensity_changed,
        "total_diff": total_diff
    }

def main(directory):
    files = sorted([f for f in os.listdir(directory) if f.endswith(".json")])
    batch_size = 3

    # Group files into batches of 3
    batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]

    for i in range(len(batches) - 1):
        batch_a = batches[i]
        batch_b = batches[i + 1]

        print(f"\nComparing batch {i+1}: {batch_a} vs {batch_b}")
        for j in range(min(len(batch_a), len(batch_b))):
            file_a = os.path.join(directory, batch_a[j])
            file_b = os.path.join(directory, batch_b[j])

            data_a = load_json(file_a)
            data_b = load_json(file_b)

            voxels_a = extract_voxels(data_a)
            voxels_b = extract_voxels(data_b)

            result = compare_voxels(voxels_a, voxels_b)
            print(f"  {batch_a[j]} vs {batch_b[j]} -> {result['total_diff']} changed "
                  f"({result['added']} added, {result['removed']} removed, "
                  f"{result['intensity_changed']} intensity changed)")

if __name__ == "__main__":
    main("E:\\Code\\Neron\\ARES\\Frames\\FernBellPark")  # Replace with your actual directory path
