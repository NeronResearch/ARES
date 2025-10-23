# hybrid_infer_stress_test.py
import zmq
import json
import time
import statistics

def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect("tcp://192.168.0.193:5555")
    print("Connected to inference service at tcp://192.168.0.193:5555")

    payload = {
        "frames_dir": "/home/connor/code/ARES/Frames/FernBellPark/",
        "frame_num": 125
    }

    requests_per_second = 28
    interval = 1.0 / requests_per_second
    total_requests = 280  # ~10 seconds of load; adjust as desired

    latencies = []
    start_time = time.time()

    for i in range(total_requests):
        send_time = time.perf_counter()
        socket.send_string(json.dumps(payload))

        # Wait for reply
        reply = socket.recv_json()
        recv_time = time.perf_counter()

        latency = (recv_time - send_time) * 1000.0  # ms
        latencies.append(latency)

        if (i + 1) % requests_per_second == 0:
            print(f"Sent {i + 1} requests...")

        # Maintain ~28Hz request rate
        elapsed = time.perf_counter() - send_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

    total_time = time.time() - start_time
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    throughput = total_requests / total_time

    print("\n=== Stress Test Results ===")
    print(f"Total requests:     {total_requests}")
    print(f"Test duration:      {total_time:.2f} s")
    print(f"Target rate:        {requests_per_second} req/s")
    print(f"Actual throughput:  {throughput:.2f} req/s")
    print(f"Average latency:    {avg_latency:.2f} ms")
    print(f"Min latency:        {min_latency:.2f} ms")
    print(f"Max latency:        {max_latency:.2f} ms")
    print(f"Std deviation:      {stdev_latency:.2f} ms")

if __name__ == "__main__":
    main()
