"""
VoyageAI TPM (Tokens Per Minute) Rate Limit Tester
Tests the maximum throughput before hitting rate limits.
"""

import asyncio
import gzip
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional
import argparse
from collections import deque
import statistics

try:
    import voyageai
except ImportError:
    print("Warning: voyageai package not installed. Install with: pip install voyageai")
    voyageai = None

try:
    import transformers
except ImportError:
    print("Warning: transformers package not installed. Install with: pip install transformers")
    transformers = None


@dataclass
class RequestResult:
    """Result of a single API request."""
    timestamp: float
    success: bool
    tokens: int = 0
    latency: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    """Statistics for the benchmark run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    rate_limited_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def tokens_per_minute(self) -> float:
        if self.total_time == 0:
            return 0.0
        return (self.total_tokens / self.total_time) * 60
    
    @property
    def requests_per_minute(self) -> float:
        if self.total_time == 0:
            return 0.0
        return (self.successful_requests / self.total_time) * 60
    
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.quantiles(self.latencies, n=20)[18]  # 95th percentile


class VoyageAIBenchmark:
    """VoyageAI API benchmark runner."""
    
    def __init__(self, api_key: str, model: str = "voyage-large-2"):
        """Initialize the benchmark runner.
        
        Args:
            api_key: VoyageAI API key
            model: Model to use for embeddings
        """
        if voyageai is None:
            raise ImportError("voyageai package is required. Install with: pip install voyageai")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        
        # Initialize tokenizer if transformers is available
        if transformers is not None:
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained("voyage-ai/voyage-large-2")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                print("Using fallback token counting method.")
                self.tokenizer = None
        else:
            self.tokenizer = None
            
        self.results: List[RequestResult] = []
        self.stats = BenchmarkStats()
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception:
                pass
        
        # Fallback to approximate token count
        return int(len(text.split()) * 1.3)  # rough approximation
    
    async def make_request(self, texts: List[str]) -> RequestResult:
        """Make a single API request."""
        start_time = time.time()
        total_tokens = sum(self.count_tokens(text) for text in texts)
        
        try:
            result = await self.client.embed(texts, model=self.model)
            end_time = time.time()
            
            return RequestResult(
                timestamp=start_time,
                success=True,
                tokens=total_tokens,
                latency=end_time - start_time
            )
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            is_rate_limited = "rate limit" in error_msg.lower() or "429" in error_msg
            
            return RequestResult(
                timestamp=start_time,
                success=False,
                tokens=total_tokens,
                latency=end_time - start_time,
                error=error_msg
            )
    
    async def run_concurrent_requests(self, 
                                    texts: List[str], 
                                    concurrency: int = 5,
                                    duration: int = 60) -> BenchmarkStats:
        """Run concurrent requests for a specified duration.
        
        Args:
            texts: List of texts to embed
            concurrency: Number of concurrent requests
            duration: Duration to run in seconds
            
        Returns:
            BenchmarkStats with results
        """
        start_time = time.time()
        end_time = start_time + duration
        
        async def worker():
            """Worker coroutine that makes requests until time is up."""
            text_idx = 0
            while time.time() < end_time:
                # Select texts for this batch (cycling through available texts)
                batch_size = min(8, len(texts))  # Reasonable batch size
                batch_texts = []
                for _ in range(batch_size):
                    batch_texts.append(texts[text_idx % len(texts)])
                    text_idx += 1
                
                result = await self.make_request(batch_texts)
                self.results.append(result)
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
        
        # Start concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        
        # Wait for all workers to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        self.stats.total_time = time.time() - start_time
        self.stats.total_requests = len(self.results)
        
        for result in self.results:
            if result.success:
                self.stats.successful_requests += 1
                self.stats.total_tokens += result.tokens
                self.stats.latencies.append(result.latency)
            else:
                self.stats.failed_requests += 1
                if result.error and ("rate limit" in result.error.lower() or "429" in result.error):
                    self.stats.rate_limited_requests += 1
        
        return self.stats
    
    def print_stats(self):
        """Print benchmark statistics."""
        print("\n" + "="*60)
        print("VOYAGEAI BENCHMARK RESULTS")
        print("="*60)
        print(f"Duration: {self.stats.total_time:.2f} seconds")
        print(f"Total Requests: {self.stats.total_requests}")
        print(f"Successful Requests: {self.stats.successful_requests}")
        print(f"Failed Requests: {self.stats.failed_requests}")
        print(f"Rate Limited Requests: {self.stats.rate_limited_requests}")
        print(f"Success Rate: {self.stats.success_rate:.2%}")
        print(f"Total Tokens Processed: {self.stats.total_tokens:,}")
        print(f"Tokens Per Minute: {self.stats.tokens_per_minute:.0f}")
        print(f"Requests Per Minute: {self.stats.requests_per_minute:.1f}")
        
        if self.stats.latencies:
            print(f"Average Latency: {self.stats.avg_latency:.3f}s")
            print(f"95th Percentile Latency: {self.stats.p95_latency:.3f}s")
            print(f"Min Latency: {min(self.stats.latencies):.3f}s")
            print(f"Max Latency: {max(self.stats.latencies):.3f}s")
        
        print("="*60)


def load_sample_texts(file_path: Optional[str] = None) -> List[str]:
    """Load sample texts for benchmarking.
    
    Args:
        file_path: Optional path to text file (JSON or plain text)
        
    Returns:
        List of sample texts
    """
    if file_path:
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [str(item) for item in data]
                else:
                    return [str(data)]
            except json.JSONDecodeError:
                # Treat as plain text, split by lines
                return [line.strip() for line in content.split('\n') if line.strip()]
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            print("Using default sample texts...")
    
    # Default sample texts
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we work and live.",
        "Machine learning models require large amounts of data to train effectively.",
        "Natural language processing has made significant advances in recent years.",
        "Deep learning architectures like transformers have revolutionized NLP.",
        "Embeddings capture semantic meaning in high-dimensional vector spaces.",
        "Vector databases are essential for similarity search applications.",
        "Large language models can understand and generate human-like text.",
        "Fine-tuning allows models to adapt to specific domains and tasks.",
        "Retrieval-augmented generation combines search with text generation."
    ] * 10  # Repeat to have more variety


async def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="VoyageAI TPM Rate Limit Tester")
    parser.add_argument("--api-key", required=True, help="VoyageAI API key")
    parser.add_argument("--model", default="voyage-large-2", help="Model to use")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run in seconds")
    parser.add_argument("--texts-file", help="Optional file with sample texts (JSON or plain text)")
    
    args = parser.parse_args()
    
    print(f"Starting VoyageAI benchmark with model: {args.model}")
    print(f"Concurrency: {args.concurrency}, Duration: {args.duration}s")
    
    # Load sample texts
    texts = load_sample_texts(args.texts_file)
    print(f"Loaded {len(texts)} sample texts")
    
    # Initialize benchmark
    benchmark = VoyageAIBenchmark(args.api_key, args.model)
    
    # Run benchmark
    print("\nStarting benchmark...")
    stats = await benchmark.run_concurrent_requests(
        texts=texts,
        concurrency=args.concurrency,
        duration=args.duration
    )
    
    # Print results
    benchmark.print_stats()


if __name__ == "__main__":
    asyncio.run(main())