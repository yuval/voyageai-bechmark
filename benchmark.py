#!/usr/bin/env python3
"""
VoyageAI TPM (Tokens Per Minute) Rate Limit Tester
Tests the maximum throughput before hitting rate limits.
"""

import asyncio
import gzip
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import argparse
from collections import deque
import statistics

import voyageai

@dataclass
class TPMTracker:
    """Tracks tokens per minute and latency metrics"""
    start_time: float = 0
    total_tokens: int = 0
    request_count: int = 0
    token_history: deque = field(default_factory=deque)  # (timestamp, tokens) pairs
    latencies: list[float] = field(default_factory=list)  # Request latencies in seconds
    peak_tpm: float = 0.0
    window_seconds: int = 60
    
    def __init__(self, window_seconds: int = 60):
        self.start_time = 0.0
        self.total_tokens = 0
        self.request_count = 0
        self.token_history = deque()
        self.window_seconds = window_seconds
        self.latencies = []
        self.peak_tpm = 0.0
    
    def add_request(self, tokens: int, latency: float):
        """Add a completed request to the tracker"""
        now = time.time()
        if self.start_time == 0:
            self.start_time = now
        
        self.total_tokens += tokens
        self.request_count += 1
        self.latencies.append(latency)
        self.token_history.append((now, tokens))
        
        # Clean old entries outside the window
        cutoff = now - self.window_seconds
        while self.token_history and self.token_history[0][0] < cutoff:
            self.token_history.popleft()

        # Track peak TPM
        current_tpm = self.get_current_tpm()
        if current_tpm > self.peak_tpm:
            self.peak_tpm = current_tpm
    
    def get_current_tpm(self) -> float:
        """Tokens per minute over a strict 60s rolling window."""
        if not self.token_history:
            return 0.0
        now = time.time()
        cutoff = now - self.window_seconds
        # token_history may already be pruned by add_request; prune again defensively
        while self.token_history and self.token_history[0][0] < cutoff:
            self.token_history.popleft()
        window_tokens = sum(tokens for _, tokens in self.token_history)
        return (window_tokens / self.window_seconds) * 60.0
    
    
    def get_latency_stats(self) -> dict:
        """Get latency statistics"""
        if not self.latencies:
            return {
                'mean': 0,
                'median': 0,
                'p95': 0,
                'p99': 0,
                'min': 0,
                'max': 0
            }

        def _pct(sorted_vals, p):
            n = len(sorted_vals)
            idx = max(0, min(n-1, math.ceil(p*n) - 1))
            return sorted_vals[idx]

        sorted_latencies = sorted(self.latencies)

        return {
            'mean': statistics.mean(self.latencies),
            'median': statistics.median(self.latencies),
            'p95': _pct(sorted_latencies, 0.95),
            'p99': _pct(sorted_latencies, 0.99),
            'min': sorted_latencies[0],
            'max': sorted_latencies[-1]
        }


def load_chunks(file_path: str) -> list[str]:
    """Load chunks from compressed JSONL file"""
    chunks = []
    
    # Determine if file is compressed based on extension
    if file_path.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open
    
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Assume chunk is in a field like 'text', 'content', or 'chunk'
                if isinstance(data, str):
                    chunks.append(data)
                elif 'text' in data:
                    chunks.append(data['text'])
                elif 'content' in data:
                    chunks.append(data['content'])
                elif 'chunk' in data:
                    chunks.append(data['chunk'])
                else:
                    # If it's a dict without known keys, use the first string value
                    for value in data.values():
                        if isinstance(value, str):
                            chunks.append(value)
                            break
    return chunks


def estimate_tokens(text: str) -> int:
    """Quick token estimation (4 chars ≈ 1 token)"""
    return len(text) // 4


def create_batches(
    chunks: list[str],
    batch_size: int,
    max_tokens_per_batch: Optional[int] = None,
    use_tokenizer: bool = False,
    model: str = "voyage-3-large"
) -> list[list[str]]:
    """Create batches respecting size and token limits"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Initialize tokenizer if needed
    tokenizer = None
    if use_tokenizer and max_tokens_per_batch:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                "This command needs the transformers package. "
            ) from e
        
        tokenizer = AutoTokenizer.from_pretrained(f'voyageai/{model}')
            
    for chunk in chunks:
        # Estimate tokens for this chunk
        if tokenizer and max_tokens_per_batch:
            chunk_tokens = len(tokenizer.encode(chunk))
        elif max_tokens_per_batch:
            chunk_tokens = estimate_tokens(chunk)
        else:
            chunk_tokens = 0
        
        # Check if adding this chunk would exceed limits
        should_start_new_batch = (
            len(current_batch) >= batch_size or
            (max_tokens_per_batch and current_tokens + chunk_tokens > max_tokens_per_batch)
        )
        
        if should_start_new_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        current_batch.append(chunk)
        current_tokens += chunk_tokens
    
    # Add remaining batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


async def embed_batch(
    client: voyageai.AsyncClient,
    batch: list[str],
    model: str,
    semaphore: asyncio.Semaphore,
    tracker: TPMTracker,
    batch_id: int
) -> None:
    """Embed a single batch and update tracker"""
    async with semaphore:
        start_time = time.time()
        
        try:
            result = await client.embed(
                texts=batch,
                model=model,
                input_type="document"
            )
            
            latency = time.time() - start_time
            
            # Track request
            tracker.add_request(result.total_tokens, latency)
            
            # Print progress
            print(f"Batch {batch_id:4d}: {result.total_tokens:6d} tokens | "
                  f"Latency: {latency*1000:6.1f}ms | "
                  f"Current TPM: {tracker.get_current_tpm():,.0f}")
            
        except Exception as e:
            tracker.add_request(0, time.time() - start_time) 
            print(f"Batch {batch_id} failed: {e}")
            return


async def run_test(
    file_path: str,
    batch_size: int,
    max_tokens_per_batch: Optional[int],
    concurrent_requests: int,
    use_tokenizer: bool,
    model: str
):
    """Main test runner"""
    print(f"Loading chunks from {file_path}...")
    chunks = load_chunks(file_path)
    print(f"Loaded {len(chunks)} chunks")
    
    print(f"\nCreating batches (size={batch_size}, max_tokens={max_tokens_per_batch})...")
    batches = create_batches(chunks, batch_size, max_tokens_per_batch, use_tokenizer, model)
    print(f"Created {len(batches)} batches")
    
    # Initialize client and tracker
    client = voyageai.AsyncClient()
    tracker = TPMTracker()
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    print(f"\nStarting embedding with {concurrent_requests} concurrent requests...")
    print("=" * 80)
    
    # Create all tasks
    tasks = [
        embed_batch(client, batch, model, semaphore, tracker, i)
        for i, batch in enumerate(batches)
    ]
    
    # Run all tasks
    start_time = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Calculate true average TPM using wall-clock time
    true_avg_tpm = (tracker.total_tokens / elapsed) * 60 if elapsed > 0 else 0.0

    # Get latency statistics
    latency_stats = tracker.get_latency_stats()

    # Final statistics
    print("=" * 80)
    print(f"\nTEST COMPLETE")
    print(f"\n* THROUGHPUT METRICS:")
    print(f"  Total time:           {elapsed:.2f} seconds")
    print(f"  Total tokens:         {tracker.total_tokens:,}")
    print(f"  Total requests:       {tracker.request_count}")
    print(f"  Average TPM:          {true_avg_tpm:,.0f}")
    print(f"  Peak TPM (60s):       {tracker.peak_tpm:,.0f}")
    print(f"  Requests/second:      {tracker.request_count/elapsed:.2f}")
    
    print(f"\n* LATENCY METRICS (ms):")
    print(f"  Mean:                 {latency_stats['mean']*1000:.1f}")
    print(f"  Median:               {latency_stats['median']*1000:.1f}")
    print(f"  P95:                  {latency_stats['p95']*1000:.1f}")
    print(f"  P99:                  {latency_stats['p99']*1000:.1f}")
    print(f"  Min:                  {latency_stats['min']*1000:.1f}")
    print(f"  Max:                  {latency_stats['max']*1000:.1f}")
    
    print(f"\n* CONFIGURATION:")
    print(f"  Concurrent requests:  {concurrent_requests}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Max tokens/batch:     {max_tokens_per_batch or 'None'}")
    print(f"  Tokenizer:            {'Actual' if use_tokenizer else 'Estimated'}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VoyageAI TPM limits and measure latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with default settings
  %(prog)s chunks.jsonl.gz
  
  # High throughput test
  %(prog)s chunks.jsonl.gz --concurrent 50 --batch-size 20
  
  # Token-limited batches with accurate counting
  %(prog)s chunks.jsonl.gz --max-tokens 5000 --use-tokenizer
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to compressed JSONL file containing chunks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of chunks per batch (default: 1000)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128000,
        help="Maximum tokens per batch (optional)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="Use actual tokenizer instead of estimation (slower but more accurate)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voyage-3-large",
        help="Model name for embedding and tokenizer (default: voyage-3-large)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_test(
        args.file_path,
        args.batch_size,
        args.max_tokens,
        args.concurrent,
        args.use_tokenizer,
        args.model
    ))


if __name__ == "__main__":
    main()