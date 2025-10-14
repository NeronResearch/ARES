import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
exe_path = r".\bin\main.exe"
scenario_path = r"E:\Code\Neron\ARES\Scenarios\FernBellPark\scenario.json"
output_dir = r"E:\Code\Neron\ARES\Frames\FernBellPark"

frame_step = 15
start_frame = 1
end_frame = 2280
max_threads = 6  # Adjust based on CPU and I/O capability

# --- Setup ---
os.makedirs(output_dir, exist_ok=True)

def run_simulation(run_id, frame1, frame2):
    """Run a single .exe call for the given frame range."""
    output_path = os.path.join(output_dir, f"{run_id:04d}.json")
    command = [
        exe_path,
        scenario_path,
        str(frame1),
        str(frame2),
        output_path
    ]
    print(f"[Run {run_id:04d}] Executing: {' '.join(command)}")
    subprocess.run(command, shell=True)
    print(f"[Run {run_id:04d}] Completed.")
    return run_id

def main():
    # Generate all frame ranges
    tasks = []
    frame1 = start_frame
    frame2 = frame1 + frame_step
    run_id = 1

    while frame2 <= end_frame:
        tasks.append((run_id, frame1, frame2))
        frame1 += frame_step
        frame2 += frame_step
        run_id += 1

    # Run threaded
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(run_simulation, rid, f1, f2): rid for rid, f1, f2 in tasks}
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Run {run_id:04d}] Error: {e}")

if __name__ == "__main__":
    main()
