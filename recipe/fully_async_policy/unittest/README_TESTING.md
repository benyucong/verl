# Unit Tests for Message Queue

This directory contains comprehensive unit tests for the message queue implementations.

## Test Files

- **`test_message_queue_new.py`** - Unit tests for the fastmq-based message queue (`message_queue_new.py`)

## Running the Tests


### Run All Tests

From the repository root:

```bash
# Using unittest (built-in)
python -m recipe.fully_async_policy.unittest.test_message_queue_new

# Or using pytest (if installed)
pytest recipe/fully_async_policy/unittest/test_message_queue_new.py -v

# Or from the unittest directory
cd recipe/fully_async_policy/unittest
python test_message_queue_new.py
```

### Run Specific Test Classes

```bash
# Test only serialization helpers
python -m unittest recipe.fully_async_policy.unittest.test_message_queue_new.TestSerializationHelpers

# Test only basic queue operations
python -m unittest recipe.fully_async_policy.unittest.test_message_queue_new.TestMessageQueueBasics

# Test staleness handling
python -m unittest recipe.fully_async_policy.unittest.test_message_queue_new.TestStalenessHandling
```

### Run Specific Test Methods

```bash
python -m unittest recipe.fully_async_policy.unittest.test_message_queue_new.TestMessageQueueBasics.test_put_and_get_sample
```

### Using pytest with verbose output

```bash
pytest recipe/fully_async_policy/unittest/test_message_queue_new.py -v -s
```

Options:
- `-v` : Verbose output
- `-s` : Show print statements
- `-k PATTERN` : Run tests matching pattern
- `--tb=short` : Shorter traceback format

Example:
```bash
pytest recipe/fully_async_policy/unittest/test_message_queue_new.py -v -k "staleness"
```

## Test Coverage

The test suite covers the following aspects:

### 1. Serialization Helpers (`TestSerializationHelpers`)
- Basic type serialization/deserialization
- Bytes passthrough
- ByteArray handling
- Invalid data handling

### 2. Basic Queue Operations (`TestMessageQueueBasics`)
- Queue initialization
- Put and get single sample
- Multiple samples handling
- Queue size tracking

### 3. Staleness Handling (`TestStalenessHandling`)
- Staleness threshold enforcement
- Dropping old samples
- Parameter version updates

### 4. Queue Overflow (`TestQueueOverflow`)
- Overflow behavior
- Oldest sample dropping when full

### 5. Statistics and Memory (`TestStatisticsAndMemory`)
- Statistics tracking
- Memory usage estimation
- Queue clearing

### 6. Shutdown Behavior (`TestShutdown`)
- Graceful shutdown
- Consumer wake-up on shutdown

### 7. Validation Path (`TestValidationPath`)
- Validation data put/get
- Separate validation queue

### 8. Client Operations (`TestMessageQueueClient`)
- Client wrapper functionality
- Direct SPSC path (when available)
- RPC fallback
- Sync/async methods

### 9. Concurrent Operations (`TestConcurrentOperations`)
- Multiple producers
- Producer-consumer balance
- Concurrent stress testing

## Test Structure

Each test class inherits from `unittest.IsolatedAsyncioTestCase` for async support:

```python
class TestMyFeature(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Ray once per test class
        
    async def asyncSetUp(self):
        # Set up test fixtures before each test
        
    async def test_something(self):
        # Your async test code
        
    async def asyncTearDown(self):
        # Clean up after each test
```

## Troubleshooting

### Ray initialization errors

If you get Ray initialization errors:
```bash
# Clean up any stale Ray processes
ray stop
# Or kill all Ray processes
pkill -9 ray
```

### Shared memory errors

If you get POSIX shared memory errors:
```bash
# Clean up shared memory segments
ls /dev/shm/ | grep mq_shard
rm /dev/shm/mq_shard_*
```

### Test timeouts

Some tests may timeout if the system is under heavy load. You can adjust timeouts in the test code if needed.

## Adding New Tests

To add new tests:

1. Create a new test class inheriting from `unittest.IsolatedAsyncioTestCase`
2. Add test methods starting with `test_`
3. Use `async def` for async test methods
4. Add the test class to the `run_tests()` function:

```python
def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(YourNewTestClass))
    # ... existing test classes ...
```

## Continuous Integration

To integrate with CI/CD:

```bash
# Run tests with coverage
pytest recipe/fully_async_policy/unittest/test_message_queue_new.py \
    --cov=recipe.fully_async_policy.message_queue_new \
    --cov-report=html \
    --cov-report=term

# Run with XML output for CI systems
pytest recipe/fully_async_policy/unittest/test_message_queue_new.py \
    --junitxml=test-results.xml
```

## Expected Output

Successful test run should look like:

```
test_clear_queue (test_message_queue_new.TestStatisticsAndMemory) ... ok
test_memory_usage (test_message_queue_new.TestStatisticsAndMemory) ... ok
test_multiple_samples (test_message_queue_new.TestMessageQueueBasics) ... ok
test_put_and_get_sample (test_message_queue_new.TestMessageQueueBasics) ... ok
test_queue_initialization (test_message_queue_new.TestMessageQueueBasics) ... ok
...

----------------------------------------------------------------------
Ran 25 tests in 15.234s

OK
```

## Performance Benchmarks

For performance testing, you can add benchmark tests:

```python
async def test_throughput_benchmark(self):
    """Benchmark message queue throughput"""
    num_messages = 1000
    start_time = time.time()
    
    for i in range(num_messages):
        await self.client.put_sample({"id": i}, param_version=0)
    
    elapsed = time.time() - start_time
    throughput = num_messages / elapsed
    print(f"Throughput: {throughput:.2f} messages/sec")
```

## Notes

- Tests use small queue sizes and shard counts for faster execution
- Tests include sleep delays to allow polling thread to process messages
- Ray is initialized once per test class for efficiency
- Each test has its own queue actor to ensure isolation
