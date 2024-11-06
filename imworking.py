import time
import random
from colorama import init, Fore, Style
import datetime
import os

init()

class MLSimulator:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.dataset_size = 150000
        self.batch_size = 32
        self.steps_per_epoch = self.dataset_size // self.batch_size
        self.learning_rate = 0.001
        self.metrics = {'loss': 5.0, 'accuracy': 0.2}

        self.processing_patterns = [
            ["Loading batch {}", "Decompressing data", "Transferring to GPU"],
            ["Forward pass batch {}", "Calculating activations", "Applying dropout"],
            ["Backward pass batch {}", "Computing gradients", "Applying optimizer"],
            ["Evaluating batch {}", "Calculating metrics (Precision, Recall, F1)", "Updating confusion matrix"],
            ["Saving checkpoint", "Validating model", "Calculating AUC"],
            ["Shuffling dataset", "Augmenting data", "Caching data"]
        ]

        self.gpu_temp = 65.0
        self.gpu_util = 85.0
        self.memory_used = 10.5

        self.last_progress_time = time.time()
        self.progress = 0

    def update_gpu_stats(self):
        self.gpu_temp = max(45, min(95, self.gpu_temp + random.uniform(-0.5, 0.8)))
        self.gpu_util = max(30, min(99, self.gpu_util + random.uniform(-2, 2)))
        self.memory_used = max(8, min(15.8, self.memory_used + random.uniform(-0.1, 0.15)))

    def format_batch_range(self):
        start = self.batch_idx
        end = min(self.batch_idx + self.batch_size, self.dataset_size)
        return f"{start}-{end}"

    def generate_process_message(self):
        pattern = random.choice(self.processing_patterns)
        messages = []
        delays = []
        for msg in pattern:
            messages.append(msg.format(self.format_batch_range() if "{}" in msg else ""))
            delays.append(random.uniform(0.1, 0.3))
        return messages, delays

    def get_error_message(self):
        errors = [
            (Fore.RED, "× python setup.py egg_info did not run successfully."),
            (Fore.RED, "× Encountered error while generating package metadata."),
            (Fore.RED, "× CUDA out of memory detected. Attempting recovery"),
            (Fore.RED, "× ImportError: cannot import name 'Protocol' from 'typing'"),
            (Fore.RED, "× RuntimeError: CUDA error: device-side assert triggered"),
            (Fore.BLUE, "Warning: High GPU memory fragmentation detected"),
            (Fore.BLUE, "Warning: Gradient norm larger than threshold: {:.2f}"),
            (Fore.BLUE, "Warning: Learning rate dropped below 1e-6")
        ]
        error = random.choice(errors)
        if '{}' in error[1]:
            return (error[0], error[1].format(random.uniform(10, 100)))
        return error

    def generate_fancy_progress_bar(self, message="Processing", length=20, error=False):
        self.progress = 0
        for i in range(101):
            progress = i / 100
            bar = '█' * int(length * progress)
            spaces = ' ' * (length - len(bar))
            percentage = f"{i:3d}%"

            if progress < 0.3:
                color = Fore.YELLOW
            elif progress < 0.7:
                color = Fore.CYAN
            elif progress < 0.95:
                color = Fore.GREEN
            else:
                color = Fore.MAGENTA

            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            if error:
                print(f"{current_time} - {Fore.RED}{message}{Style.RESET_ALL} {color}┃{bar}{spaces}┃{Style.RESET_ALL} {percentage}", end='\r', flush=True)
            else:
                print(f"{current_time} - {message} {color}┃{bar}{spaces}┃{Style.RESET_ALL} {percentage}", end='\r', flush=True)
            time.sleep(random.uniform(0.01, 0.03))

        final_message = f"{message} {Fore.GREEN if not error else Fore.RED}┃{'█' * length}┃{Style.RESET_ALL} 100%"
        if error:
            final_message = f"× {final_message}"
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {final_message}")

    def generate_log(self):
        last_gpu_log = time.time()
        last_metric_log = time.time()

        while True:
            current_time = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

            if random.random() < 0.05:
                color, message = self.get_error_message()
                print(f"\n{current_time} - {color}{message}{Style.RESET_ALL}")

                error_messages = [
                    "Attempting to recover from CUDA error",
                    "Restarting Python kernel",
                    "Re-initializing model weights",
                    "Debugging error"
                ]
                self.generate_fancy_progress_bar(random.choice(error_messages), error=True)

                if "did not run successfully" in message:
                    print(f"{current_time} - │ exit code: 1")
                    print(f"{current_time} - ╰─> [30 lines of output]")
                    print(f"{current_time} -      c:\\users\\usrname\\venv\\tf1\\lib\\site-packages\\setuptools\\pep425tags.py:89: RuntimeWarning: Config variable 'Py_DEBUG' is unset, Python ABI tag may be incorrect")
                    print(f"{current_time} -      c:\\users\\usrname\\venv\\tf1\\lib\\site-packages\\setuptools\\pep425tags.py:93: RuntimeWarning: Config variable 'WITH_PYMALLOC' is unset, Python ABI tag may be incorrect")
                    print(f"{current_time} -      ImportError: cannot import name 'Protocol' from 'typing'")
                time.sleep(random.uniform(1.5, 2.5))
                continue

            if time.time() - last_metric_log > random.uniform(0.1, 1): 
                    self.metrics['loss'] *= random.uniform(0.98, 0.995)
                    self.metrics['accuracy'] = min(1.0, self.metrics['accuracy'] * random.uniform(1.001, 1.004))

                    precision = random.uniform(0.5, 0.95)
                    recall = random.uniform(0.5, 0.95)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    auc = random.uniform(0.6, 0.98)

                    metrics_str = f"loss: {self.metrics['loss']:.4f}, acc: {self.metrics['accuracy']:.4f}, "
                    metrics_str += f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, "
                    metrics_str += f"lr: {self.learning_rate:.6f}"
                    print(f"{current_time} - {metrics_str}")

                    last_metric_log = time.time()

            if time.time() - last_gpu_log > random.uniform(8, 12):
                if self.gpu_temp > 80:
                    color = Fore.RED
                else:
                    color = Style.RESET_ALL
                print(f"{current_time} - +---------------------------------------------------------------------------------------+")
                print(f"{current_time} - | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |")
                print(f"{current_time} - |-----------------------------------------+----------------------+----------------------+|")
                print(f"{current_time} - | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |")
                print(f"{current_time} - | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |")
                print(f"{current_time} - |                                         |                      |               MIG M. |")
                print(f"{current_time} - |=========================================+======================+======================|")
                print(f"{current_time} - |   0  NVIDIA A100-SXM4-40GB          Off | 00000000:00:04.0 Off |                    0 |")
                print(f"{current_time} - | N/A   {self.gpu_temp}C    P0              46W / 400W |      2MiB / 40960MiB |      {self.gpu_util}%      Default |")
                print(f"{current_time} - |                                         |                      |             Disabled |")
                print(f"{current_time} - +-----------------------------------------+----------------------+----------------------+|")
                last_gpu_log = time.time()


            if time.time() - last_metric_log > random.uniform(4, 6):
                self.metrics['loss'] *= random.uniform(0.98, 0.995)
                self.metrics['accuracy'] = min(1.0, self.metrics['accuracy'] * random.uniform(1.001, 1.004))
                print(f"{current_time} - loss: {self.metrics['loss']:.4f}, acc: {self.metrics['accuracy']:.4f}, "
                      f"lr: {self.learning_rate:.6f}")
                last_metric_log = time.time()

            messages, delays = self.generate_process_message()
            if isinstance(messages, list):
                for msg, delay in zip(messages, delays):
                    print(f"{current_time} - {msg}")
                    time.sleep(delay)
            else:
                print(f"{current_time} - {messages}")
                time.sleep(delays)


            if random.random() < 0.1 and time.time() - self.last_progress_time > 5:
                messages = [
                    "Optimizing model weights",
                    "Compiling model for inference",
                    "Quantizing model",
                    "Generating model summary"
                ]
                message = random.choice(messages)
                self.generate_fancy_progress_bar(message)
                self.last_progress_time = time.time()


            self.batch_idx += self.batch_size

            if self.batch_idx >= self.dataset_size:
                self.epoch += 1
                self.batch_idx = 0
                print(f"{current_time} - Completed epoch {self.epoch}")
                self.learning_rate *= 0.95
                time.sleep(random.uniform(0.2, 0.5))


            if random.random() < 0.001:
                time.sleep(random.uniform(1.5, 2.5))
            else:
                time.sleep(random.uniform(0.02, 0.05))




if __name__ == "__main__":
    try:
        rows, columns = os.popen('stty size', 'r').read().split()
        columns = int(columns)
    except ValueError:
        columns = 80

    print(f"Initializing training environment\n")
    time.sleep(0.5)
    print(f"Loading model weights and optimizer states\n")
    time.sleep(0.3)
    simulator = MLSimulator()
    simulator.generate_log()