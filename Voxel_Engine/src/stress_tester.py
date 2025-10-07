import subprocess
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class StressTester:
    def __init__(self, command, duration=20, max_threads=None):
        self.command = command
        self.duration = duration
        self.max_threads = max_threads or os.cpu_count()
        self.results = []
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
        
    def run_single_command(self, thread_id):
        """Run a single instance of the command and measure execution time"""
        start = time.time()
        try:
            # Run the command and capture output
            result = subprocess.run(
                self.command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30  # 30 second timeout per command
            )
            end = time.time()
            
            execution_time = end - start
            success = result.returncode == 0
            
            with self.lock:
                self.results.append({
                    'thread_id': thread_id,
                    'execution_time': execution_time,
                    'success': success,
                    'start_time': start,
                    'end_time': end,
                    'return_code': result.returncode,
                    'stdout_length': len(result.stdout) if result.stdout else 0,
                    'stderr_length': len(result.stderr) if result.stderr else 0
                })
            
            return execution_time, success
            
        except subprocess.TimeoutExpired:
            end = time.time()
            with self.lock:
                self.results.append({
                    'thread_id': thread_id,
                    'execution_time': end - start,
                    'success': False,
                    'start_time': start,
                    'end_time': end,
                    'return_code': -1,
                    'stdout_length': 0,
                    'stderr_length': 0,
                    'error': 'Timeout'
                })
            return end - start, False
            
        except Exception as e:
            end = time.time()
            with self.lock:
                self.results.append({
                    'thread_id': thread_id,
                    'execution_time': end - start,
                    'success': False,
                    'start_time': start,
                    'end_time': end,
                    'return_code': -1,
                    'stdout_length': 0,
                    'stderr_length': 0,
                    'error': str(e)
                })
            return end - start, False

    def worker_thread(self, thread_id):
        """Worker function for each thread"""
        thread_results = []
        
        while time.time() - self.start_time < self.duration:
            exec_time, success = self.run_single_command(thread_id)
            thread_results.append((exec_time, success))
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
        return thread_results

    def run_stress_test(self):
        """Run the stress test with multiple threads"""
        print(f"Starting stress test...")
        print(f"Command: {self.command}")
        print(f"Duration: {self.duration} seconds")
        print(f"Max threads: {self.max_threads}")
        print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        self.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit worker threads
            futures = [executor.submit(self.worker_thread, i) for i in range(self.max_threads)]
            
            # Monitor progress
            start_monitoring = time.time()
            while time.time() - start_monitoring < self.duration:
                elapsed = time.time() - start_monitoring
                completed_runs = len(self.results)
                rate = completed_runs / elapsed if elapsed > 0 else 0
                
                print(f"\rProgress: {elapsed:.1f}s | Completed runs: {completed_runs} | Rate: {rate:.2f} runs/sec", end="", flush=True)
                time.sleep(1)
            
            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()
        
        self.end_time = time.time()
        print(f"\nStress test completed!")
        
    def analyze_results(self):
        """Analyze and display the results"""
        if not self.results:
            print("No results to analyze!")
            return
            
        total_duration = self.end_time - self.start_time
        total_runs = len(self.results)
        successful_runs = sum(1 for r in self.results if r['success'])
        failed_runs = total_runs - successful_runs
        
        # Calculate timing statistics
        execution_times = [r['execution_time'] for r in self.results if r['success']]
        
        print("\n" + "="*60)
        print("STRESS TEST RESULTS")
        print("="*60)
        print(f"Test Duration: {total_duration:.2f} seconds")
        print(f"Total Runs: {total_runs}")
        print(f"Successful Runs: {successful_runs}")
        print(f"Failed Runs: {failed_runs}")
        print(f"Success Rate: {(successful_runs/total_runs)*100:.1f}%")
        print(f"Average Rate: {total_runs/total_duration:.2f} runs/second")
        print(f"Successful Rate: {successful_runs/total_duration:.2f} successful runs/second")
        
        if execution_times:
            print(f"\nExecution Time Statistics (successful runs only):")
            print(f"  Mean: {statistics.mean(execution_times):.3f}s")
            print(f"  Median: {statistics.median(execution_times):.3f}s")
            print(f"  Min: {min(execution_times):.3f}s")
            print(f"  Max: {max(execution_times):.3f}s")
            print(f"  Std Dev: {statistics.stdev(execution_times):.3f}s")
        
        # Thread performance
        thread_counts = {}
        for result in self.results:
            tid = result['thread_id']
            if tid not in thread_counts:
                thread_counts[tid] = {'total': 0, 'success': 0}
            thread_counts[tid]['total'] += 1
            if result['success']:
                thread_counts[tid]['success'] += 1
        
        print(f"\nPer-Thread Performance:")
        for tid in sorted(thread_counts.keys()):
            stats = thread_counts[tid]
            print(f"  Thread {tid}: {stats['total']} runs, {stats['success']} successful ({(stats['success']/stats['total'])*100:.1f}%)")
        
        # Error analysis
        if failed_runs > 0:
            print(f"\nError Analysis:")
            error_counts = {}
            return_codes = {}
            
            for result in self.results:
                if not result['success']:
                    error = result.get('error', f"Return code {result['return_code']}")
                    error_counts[error] = error_counts.get(error, 0) + 1
                    
                    code = result['return_code']
                    return_codes[code] = return_codes.get(code, 0) + 1
            
            for error, count in error_counts.items():
                print(f"  {error}: {count} occurrences")

def main():
    # The command to stress test
    command = r".\main.exe E:\Code\Neron\ARES\Scenarios\Scenario2\scenario.json"
    
    # Create and run the stress tester
    tester = StressTester(command, duration=20)
    
    try:
        tester.run_stress_test()
        tester.analyze_results()
    except KeyboardInterrupt:
        print("\nStress test interrupted by user!")
        if tester.results:
            tester.end_time = time.time()
            tester.analyze_results()

if __name__ == "__main__":
    main()