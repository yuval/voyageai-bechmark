# VoyageAI Benchmark

VoyageAI TPM (Tokens Per Minute) Rate Limit Tester - Tests the maximum throughput before hitting rate limits.

## Installation

This project uses Poetry for dependency management. Install the dependencies:

```bash
# Install Poetry if you haven't already
pip install poetry

# Install project dependencies
poetry install
```

Alternatively, install dependencies directly with pip:

```bash
pip install voyageai transformers torch
```

## Usage

### Basic Usage

```bash
# Using Poetry
poetry run python voyageai_benchmark.py --api-key YOUR_API_KEY

# Or directly with Python
python voyageai_benchmark.py --api-key YOUR_API_KEY
```

### Advanced Options

```bash
python voyageai_benchmark.py \
  --api-key YOUR_API_KEY \
  --model voyage-large-2 \
  --concurrency 10 \
  --duration 120 \
  --texts-file sample_texts.json
```

### Parameters

- `--api-key`: **Required** - Your VoyageAI API key
- `--model`: Model to use for embeddings (default: `voyage-large-2`)
- `--concurrency`: Number of concurrent requests (default: `5`)
- `--duration`: Duration to run the benchmark in seconds (default: `60`)
- `--texts-file`: Optional file with sample texts (JSON array or plain text, one per line)

### Sample Text File Format

**JSON format:**
```json
[
  "Your first text to embed",
  "Your second text to embed",
  "More texts..."
]
```

**Plain text format:**
```
Your first text to embed
Your second text to embed
More texts...
```

## Output

The benchmark will output statistics including:
- Total requests made
- Success/failure rates
- Tokens per minute (TPM)
- Requests per minute
- Latency statistics (average, 95th percentile, min/max)
- Rate limiting information

## Development

```bash
# Install with dev dependencies
poetry install

# Run linting
poetry run black voyageai_benchmark.py
poetry run isort voyageai_benchmark.py

# Run tests
poetry run pytest
```

## Requirements

- Python 3.9+
- VoyageAI API key
- Internet connection for API calls

## License

Apache License 2.0