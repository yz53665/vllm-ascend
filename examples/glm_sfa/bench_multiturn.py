import argparse
import asyncio
import json
import queue
import random
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import requests
import aiohttp
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional
import os

from vllm.tokenizers import get_tokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=256,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=128,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--disable-random-sample",
        action="store_true",
        help="If set, disable random sampling of requests from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--enable-round-barrier",
        action="store_true",
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Length of the sub question input for each request, if set 0 use request_length",
    )
    parser.add_argument(
        "--ready-queue-policy",
        type=str,
        default="random",
        help="Policy for popping requests from the ready queue (random or fifo)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag of a certain run in the log file",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help="Min rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Max rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=1.0,
        help="Length variation ratio for prompts and outputs (1.0 = no variation, 0.5 = 50%% variation)",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="String of LoRA path. Currently we only support benchmarking on a single LoRA adaptor.",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="openai",
        choices=["sglang", "openai"],
        help="API format to use: 'sglang' for native /generate endpoint, "
        "'openai' for OpenAI-compatible /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--round-gap",
        type=float,
        default=0.0,
        help="Seconds to sleep before releasing the next round when "
        "--enable-round-barrier is set. Useful for observing KV lease "
        "expiry / cache eviction between rounds.",
    )
    return parser.parse_args()


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl", tag=""):
    """Append the data with a timestamp and tag to the specified JSONL file."""
    timestamped_data = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    try:
        with open(file_path, "a") as file:
            file.write(
                json.dumps(timestamped_data) + "\n"
            )  # Write as a single line in JSONL format
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    prompt_len: int = 0
    output_len: int = 0
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""

@dataclass
class DatasetRow:
    prompt: Any
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    routing_key: Optional[str] = None
    extra_request_body: Optional[Dict[str, Any]] = None  # Per-request API parameters

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0
        if self.extra_request_body is None:
            self.extra_request_body = {}


def gen_payload_openai(messages, output_len, model):
    return {
        "model": model,
        "messages": messages,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text

def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> List[DatasetRow]:
    input_lens = [input_len] * num_prompts
    output_lens = [output_len] * num_prompts

    if return_text:
        # Need to truncate input_len as server encode will add special token.
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        for i in range(num_prompts):
            input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[DatasetRow] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            input_content = input_ids
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            # Use int() to convert numpy.int64 to native Python int for JSON serialization
            input_content = [
                int((offsets[i] + i + j) % tokenizer.vocab_size)
                for j in range(input_lens[i])
            ]
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests

async def async_request_openai_chat_completions(
    payload,
    url,
    pbar=None,
):
    """Send a streaming request to an OpenAI-compatible /v1/chat/completions endpoint.

    Returns a RequestFuncOutput with the same dynamic attributes as
    async_request_sglang_generate (except output_ids, which is unavailable).
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        generated_text = ""
        ttft = 0.0
        latency = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    completion_tokens = 0

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # Streaming token chunks
                            if data.get("choices"):
                                raw_delta = data["choices"][0].get("delta") or {}
                                # Reasoning models (e.g. GLM with a reasoning
                                # parser) stream thinking tokens into
                                # `reasoning_content` while `content` stays
                                # empty, so count both when measuring TTFT/ITL.
                                text = raw_delta.get("content") or raw_delta.get(
                                    "reasoning_content"
                                ) or ""
                                if text:
                                    generated_text += text
                                    timestamp = time.perf_counter()

                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp

                            # Final chunk with usage stats
                            usage = data.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                details = usage.get("prompt_tokens_details", {}) or {}
                                cached_tokens = details.get("cached_tokens", 0)

                    output.generated_text = generated_text
                    output.output_ids = []  # Not available from OpenAI endpoint
                    output.success = True
                    # Assigned once here: an empty stream never reaches the
                    # per-chunk branches above, which used to leave the
                    # dataclass default in place and break the printed metrics.
                    output.ttft = ttft if ttft > 0.0 else latency
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.generated_len = (
                        completion_tokens if completion_tokens else len(output.itl) + 1
                    )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output

class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        self.api_format = args.api_format
        self.model_path = args.model_path

        # Construct the base URL and select request/payload functions
        if self.api_format == "openai":
            self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
            self.request_func = async_request_openai_chat_completions

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None
        self.lora_path = args.lora_path

        self.sent_requests = 0
        self.completed_requests = 0

        # Resolve per-client round counts
        min_rounds = args.min_rounds
        max_rounds = args.max_rounds
        if min_rounds == 0 and max_rounds == 0:
            # Backward compat: all clients use --num-rounds
            min_rounds = args.num_rounds
            max_rounds = args.num_rounds
        elif min_rounds == 0:
            min_rounds = max_rounds
        elif max_rounds == 0:
            max_rounds = min_rounds
        if min_rounds < 1:
            raise ValueError(f"--min-rounds must be >= 1, got {min_rounds}")
        if min_rounds > max_rounds:
            raise ValueError(
                f"--min-rounds ({min_rounds}) must be <= --max-rounds ({max_rounds})"
            )

        self.min_rounds = min_rounds
        self.max_rounds = max_rounds

        if min_rounds == max_rounds:
            # All clients have the same round count; skip randint to preserve random state
            self.client_total_rounds = [min_rounds] * args.num_clients
        else:
            self.client_total_rounds = [
                random.randint(min_rounds, max_rounds) for _ in range(args.num_clients)
            ]

        # clients_per_round[r] = number of clients participating in round r
        self.clients_per_round = [
            sum(1 for t in self.client_total_rounds if t > r) for r in range(max_rounds)
        ]
        self.total_requests = sum(self.client_total_rounds)

        range_ratio = args.range_ratio

        # Use return_text=False to get token ids instead of text
        first_round_samples = sample_random_requests(
            input_len=args.request_length,
            output_len=args.output_length,
            num_prompts=args.num_clients,
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )
        # Store per-sample output_len for first round
        first_round_output_lens = [row.output_len for row in first_round_samples]
        # r.prompt is now List[int] when return_text=False
        self.candidate_inputs = [list(i.prompt) for i in first_round_samples]

        if args.sub_question_input_length != 0:
            sub_question_input_length = args.sub_question_input_length
        else:
            sub_question_input_length = args.request_length

        num_sub_questions = sum(max(t - 1, 0) for t in self.client_total_rounds)

        self.sub_question_inputs = sample_random_requests(
            input_len=sub_question_input_length,
            output_len=args.output_length,
            num_prompts=max(num_sub_questions, 1),
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )

        if self.api_format == "openai":
            # OpenAI mode: history is a messages list for /v1/chat/completions
            initial_messages = {
                i: [
                    {
                        "role": "user",
                        "content": self.tokenizer.decode(self.candidate_inputs[i]),
                    }
                ]
                for i in range(args.num_clients)
            }
            init_requests = [
                (
                    i,
                    gen_payload_openai(
                        initial_messages[i],
                        first_round_output_lens[i],
                        self.model_path,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": initial_messages[i],
                    "total_rounds": self.client_total_rounds[i],
                }
                for i in range(args.num_clients)
            }
        self.ready_queue = ReadyQueue(
            init_requests=init_requests, policy=args.ready_queue_policy
        )
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests)
        self.performance_metrics = {
            "ttft": [],
            "itl": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }
        self.enable_round_barrier = args.enable_round_barrier
        if self.enable_round_barrier:
            # Add round-specific metrics while preserving the original structure
            for i in range(self.max_rounds):
                self.performance_metrics[f"round_{i}"] = {
                    "ttft": [],
                    "latency": [],
                    "prompt_len": [],
                    "cached_tokens": [],
                    "generated_len": [],
                }
        self.num_clients = args.num_clients

        self.num_rounds = self.max_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length
        self.round_gap = args.round_gap

    async def handle_request(self, item):
        client_id, payload = item
        try:
            response = await self.request_func(payload, self.url, self.pbar)
            if self.pbar.n == self.pbar.total:
                self.finished_time = time.perf_counter()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed for client {client_id}: {e}")
            failed_response = RequestFuncOutput()
            failed_response.success = False
            failed_response.error = str(e)
            self.response_queue.put((client_id, failed_response))

    def request_sender(self):
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if self.pbar.n == self.pbar.total:
                    break

                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        next_round_reqs = []
        current_barrier_round = 0
        barrier_round_completed = 0
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    print(f"Request failed for client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue
                # Extend history with response
                if self.api_format == "openai":
                    if response.generated_text:
                        self.client_records[client_id]["history"].append(
                            {"role": "assistant", "content": response.generated_text}
                        )
                else:
                    self.client_records[client_id]["history"].extend(
                        response.output_ids
                    )
                current_round = self.client_records[client_id]["round"]
                self.client_records[client_id]["round"] += 1
                print(
                    f"[client {client_id}] round {current_round} "
                    f"prompt_len={response.prompt_len} "
                    f"cached_tokens={response.cached_tokens} "
                    f"generated_len={response.generated_len} "
                    f"latency={response.latency:.3f}s ttft={response.ttft:.3f}s"
                )
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["generated_len"].append(response.generated_len)
                if self.enable_round_barrier:
                    self.performance_metrics[f"round_{current_round}"]["ttft"].append(
                        response.ttft
                    )
                    self.performance_metrics[f"round_{current_round}"][
                        "latency"
                    ].append(response.latency)
                    self.performance_metrics[f"round_{current_round}"][
                        "prompt_len"

                    ].append(response.prompt_len)
                    self.performance_metrics[f"round_{current_round}"][
                        "cached_tokens"
                    ].append(response.cached_tokens)
                    self.performance_metrics[f"round_{current_round}"][
                        "generated_len"
                    ].append(response.generated_len)
                self.completed_requests += 1

                client_total = self.client_records[client_id]["total_rounds"]
                if self.client_records[client_id]["round"] < client_total:
                    sub_q = self.sub_question_inputs.pop()
                    if self.api_format == "openai":
                        # Append sub-question as a new user message
                        sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                        self.client_records[client_id]["history"].append(
                            {"role": "user", "content": sub_q_text}
                        )
                        new_req = (
                            client_id,
                            gen_payload_openai(
                                self.client_records[client_id]["history"],
                                sub_q.output_len,
                                self.model_path,
                            ),
                        )
                    if self.enable_round_barrier:
                        next_round_reqs.append(new_req)
                    else:
                        self.ready_queue.append(new_req)

                # Barrier logic: release next round when all clients for
                # current barrier round have completed
                if (
                    self.enable_round_barrier
                    and current_barrier_round < self.max_rounds
                ):
                    barrier_round_completed += 1
                    expected = self.clients_per_round[current_barrier_round]
                    if barrier_round_completed == expected:
                        print(
                            f"\n  Barrier: round {current_barrier_round} complete "
                            f"({expected} clients), releasing {len(next_round_reqs)} "
                            f"requests for round {current_barrier_round + 1}"
                        )
                        # self._send_heartbeat(input_len=10, output_len=1)
                        # time.sleep(0.005)
                        if self.round_gap > 0:
                            print(
                                f"  Round gap: sleeping {self.round_gap}s before "
                                f"releasing round {current_barrier_round + 1}"
                            )
                            time.sleep(self.round_gap)
                        for req in next_round_reqs:
                            self.ready_queue.append(req)
                        next_round_reqs = []
                        current_barrier_round += 1
                        barrier_round_completed = 0
            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break
            except ValueError as e:
                print(f"Error processing response for client {client_id}: {e}")
                continue

    def _send_heartbeat(self, input_len=100, output_len=20):
        """Send a small heartbeat request to the server."""
        heartbeat_input = [1] * input_len
        heartbeat_text = self.tokenizer.decode(heartbeat_input)
        messages = [{"role": "user", "content": heartbeat_text}]
        payload = gen_payload_openai(messages, output_len, self.model_path)
        try:
            requests.post(self.url, json=payload, timeout=30)
        except Exception as e:
            print(f"Heartbeat request failed: {e}")

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        # response = requests.post('http://localhost:8100/start_profile')
        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()
        self.pbar.close()

        duration = self.finished_time - self.start_time
        sorted_ttft = sorted(self.performance_metrics["ttft"])
        sorted_latency = sorted(self.performance_metrics["latency"])
        sorted_itl = sorted(self.performance_metrics["itl"])
        sorted_prompt_len = sorted(self.performance_metrics["prompt_len"])
        sorted_output_len = sorted(self.performance_metrics["generated_len"])

        def percentile(sorted_vals, q):
            if not sorted_vals:
                return 0.0
            idx = int(q * len(sorted_vals))
            if idx >= len(sorted_vals):
                idx = len(sorted_vals) - 1
            return sorted_vals[idx]

        def max_or_zero(sorted_vals):
            return sorted_vals[-1] if sorted_vals else 0.0

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "request_rate": self.request_rate,
                "average_prompt_len": (
                    sum(self.performance_metrics["prompt_len"])
                    / len(self.performance_metrics["prompt_len"])
                    if self.performance_metrics["prompt_len"]
                    else 0.0
                ),
                "average_output_len": (
                    sum(self.performance_metrics["generated_len"])
                    / len(self.performance_metrics["generated_len"])
                    if self.performance_metrics["generated_len"]
                    else 0.0
                ),
                "p90_prompt_len": percentile(sorted_prompt_len, 0.9),
                "p99_prompt_len": percentile(sorted_prompt_len, 0.99),
                "p90_output_len": percentile(sorted_output_len, 0.9),
                "p99_output_len": percentile(sorted_output_len, 0.99),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": percentile(sorted_ttft, 0.9),
                "p99_ttft": percentile(sorted_ttft, 0.99),
                "median_ttft": percentile(sorted_ttft, 0.5),
                "max_ttft": max_or_zero(sorted_ttft),
                "average_itl": (
                    sum(self.performance_metrics["itl"])
                    / len(self.performance_metrics["itl"])
                    if self.performance_metrics["itl"]
                    else 0.0
                ),
                "p90_itl": percentile(sorted_itl, 0.9),
                "p99_itl": percentile(sorted_itl, 0.99),
                "median_itl": percentile(sorted_itl, 0.5),
                "max_itl": max_or_zero(sorted_itl),
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": percentile(sorted_latency, 0.9),
                "p99_latency": percentile(sorted_latency, 0.99),
                "median_latency": percentile(sorted_latency, 0.5),
                "max_latency": max_or_zero(sorted_latency),
                "input_token_throughput": sum(self.performance_metrics["prompt_len"])
                / duration,
                "output_token_throughput": sum(
                    self.performance_metrics["generated_len"]
                )
                / duration,
                "throughput": self.pbar.total / duration,
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        if self.enable_round_barrier:
            performance_data["round"] = {}
            for round_num in range(self.num_rounds):
                round_key = f"round_{round_num}"
                round_metrics = self.performance_metrics[round_key]
                performance_data["round"][round_key] = {
                    "average_ttft": (
                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])
                        if round_metrics["ttft"]
                        else 0
                    ),
                    "cache_hit_rate": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "request_count": len(round_metrics["ttft"]),
                }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']} at {performance_data['summary']['request_rate']} requests per second"
        )
        print(
            f"  Average Prompt Length: {performance_data['summary']['average_prompt_len']:.2f} tokens"
        )
        print(
            f"  Average Output Length: {performance_data['summary']['average_output_len']:.2f} tokens"
        )
        print(
            f"  P90 Prompt Length: {performance_data['summary']['p90_prompt_len']:.0f} tokens"
        )
        print(
            f"  P99 Prompt Length: {performance_data['summary']['p99_prompt_len']:.0f} tokens"
        )
        print(
            f"  P90 Output Length: {performance_data['summary']['p90_output_len']:.0f} tokens"
        )
        print(
            f"  P99 Output Length: {performance_data['summary']['p99_output_len']:.0f} tokens"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  P99 TTFT: {performance_data['summary']['p99_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(f"  Max TTFT: {performance_data['summary']['max_ttft']:.2f}")
        print(f"  Average ITL: {performance_data['summary']['average_itl']:.4f}")
        print(f"  P90 ITL: {performance_data['summary']['p90_itl']:.4f}")
        print(f"  P99 ITL: {performance_data['summary']['p99_itl']:.4f}")
        print(f"  Median ITL: {performance_data['summary']['median_itl']:.4f}")
        print(f"  Max ITL: {performance_data['summary']['max_itl']:.4f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  P99 latency: {performance_data['summary']['p99_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(f"  Max latency: {performance_data['summary']['max_latency']:.2f}")
        print(
            f"  Input token throughput: {performance_data['summary']['input_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Output token throughput: {performance_data['summary']['output_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Request Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        if self.enable_round_barrier:
            # Print round-basedsummary
            print("Per-round metrics:")
            if "round" in performance_data:
                for round_num in range(self.num_rounds):
                    round_key = f"round_{round_num}"
                    if round_key in performance_data["round"]:
                        round_data = performance_data["round"][round_key]
                        avg_ttft = round_data["average_ttft"]
                        cache_hit_rate = round_data["cache_hit_rate"]
                        request_count = round_data["request_count"]
                        clients_in_round = self.clients_per_round[round_num]
                        print(
                            f"  Round {round_num}: Average TTFT = {avg_ttft:.2f}s, "
                            f"Cache Hit Rate = {cache_hit_rate:.6f} "
                            f"({request_count} requests, "
                            f"{clients_in_round} clients)"
                        )
                    else:
                        print(f"  Round {round_num}: No requests completed")
        # response = requests.post('http://localhost:8100/stop_profile')
        return performance_data


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
    else:
        print("Auto-running with different request rates...")
        request_rates = [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    for rate in request_rates:
        args.request_rate = rate
        performance_data = WorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, tag=args.tag)