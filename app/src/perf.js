/* Parse throughput summaries from the supported backend log formats. */
(function (root) {
  function number(value) {
    const parsed = Number(String(value).replace(/,/g, ""));
    return Number.isFinite(parsed) ? parsed : null;
  }

  function parseThroughput(text) {
    const line = String(text || "");
    const pair = line.match(
      /(?:avg\s+)?(?:prompt|prefill) throughput:\s*([\d,.]+)\s*tokens\/s.*?(?:avg\s+)?(?:generation|decode) throughput:\s*([\d,.]+)\s*tokens\/s/i
    );
    if (pair) return { pp: number(pair[1]), decode: number(pair[2]) };

    const llamaDecode = line.match(
      /\bn_gen\s*=\s*[\d,]+.*?\btg\s*=\s*([\d,.]+)\s*t\/s(?:\s*,\s*tg_3s\s*=\s*([\d,.]+)\s*t\/s)?/i
    );
    if (llamaDecode) {
      return { decode: number(llamaDecode[2] || llamaDecode[1]) };
    }

    const currentLlama = line.match(
      /\b(prompt processing|generation|decode processing|output processing|decode)\b.*?([\d,.]+)\s*(?:tokens?\s+per\s+second|tokens?\/s|t\/s)/i
    );
    if (currentLlama) {
      const rate = number(currentLlama[2]);
      return /^prompt processing$/i.test(currentLlama[1])
        ? { pp: rate }
        : { decode: rate };
    }

    const llama = line.match(
      /(prompt eval time|eval time)\s*=\s*[\d,.]+\s*ms\s*\/\s*[\d,]+\s*(?:tokens?|runs?)\s*\(\s*([\d,.]+)\s*(?:tokens?\s+per\s+second|tokens?\/s|t\/s)/i
    );
    if (!llama) return null;

    const rate = number(llama[2]);
    return /^prompt eval time$/i.test(llama[1])
      ? { pp: rate }
      : { decode: rate };
  }

  root.LlamaStudioPerf = { parseThroughput };
})(typeof window === "undefined" ? globalThis : window);
