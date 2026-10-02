const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const context = { window: {} };
const source = fs.readFileSync(path.join(__dirname, "../src/perf.js"), "utf8");
vm.runInNewContext(source, context);
const parseThroughput = context.window.LlamaStudioPerf.parseThroughput;

test("parses llama.cpp prompt and decode summaries with long and short units", () => {
  assert.equal(
    parseThroughput("prompt eval time = 1234.56 ms / 1000 tokens ( 200.14 tokens per second)").pp,
    200.14,
  );
  assert.equal(
    parseThroughput("eval time = 800.00 ms / 30 runs ( 37.50 t/s, 26.67 ms per token)").decode,
    37.5,
  );
});

test("parses current llama.cpp prompt-processing and generation log lines", () => {
  assert.equal(
    parseThroughput("0.57.201.055 I slot print_timing: id 0 | task 0 | prompt processing, n_tokens = 2048, progress = 0.47, t = 5.45 s / 376.05 tokens per second").pp,
    376.05,
  );
  assert.equal(
    parseThroughput("I slot print_timing: id 0 | task 0 | generation, n_tokens = 64, t = 1.60 s, speed = 40.00 tokens per second").decode,
    40,
  );
  assert.equal(
    parseThroughput("I slot print_timing: id 0 | task 0 | generation, n_tokens = 48, t = 3.20 s, 15.00 tokens per second").decode,
    15,
  );
});

test("parses llama.cpp rolling generation stats and prefers the smoothed rate", () => {
  assert.equal(
    parseThroughput("6.11.062.614 I slot print_timing: id 0 | task 0 | n_gen = 100, tg = 29.86 t/s, tg_3s = 30.16 t/s").decode,
    30.16,
  );
  assert.equal(
    parseThroughput("6.38.201.658 I slot print_timing: id 0 | task 0 | n_gen = 921, tg = 30.17 t/s").decode,
    30.17,
  );
});

test("parses grouped vLLM and SGLang throughput reports", () => {
  assert.deepEqual(
    { ...parseThroughput("Avg prompt throughput: 198.2 tokens/s, Avg generation throughput: 37.4 tokens/s") },
    { pp: 198.2, decode: 37.4 },
  );
  assert.deepEqual(
    { ...parseThroughput("Prefill throughput: 123.4 tokens/s, Decode throughput: 45.6 tokens/s") },
    { pp: 123.4, decode: 45.6 },
  );
});

test("accepts grouped token counts and ignores unrelated lines", () => {
  assert.equal(
    parseThroughput("prompt eval time = 3,000.0 ms / 1,000 tokens (333.33 tokens/s)").pp,
    333.33,
  );
  assert.equal(parseThroughput("server is loading model tensors"), null);
});
