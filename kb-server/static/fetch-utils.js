/**
 * fetchWithTimeout — 带超时 + 自动重试的 fetch 封装
 * @param {string} url
 * @param {object} options - fetch options
 * @param {number} timeoutMs - 超时毫秒（默认 15000）
 * @param {number} retries - 重试次数（默认 2）
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(url, options, timeoutMs, retries) {
  timeoutMs = timeoutMs || 15000;
  retries = retries || 2;
  var lastErr = null;
  for (var attempt = 0; attempt <= retries; attempt++) {
    try {
      var controller = new AbortController();
      var timer = setTimeout(function() { controller.abort(); }, timeoutMs);
      var fetchOpts = Object.assign({}, options || {}, { signal: controller.signal });
      var resp = await fetch(url, fetchOpts);
      clearTimeout(timer);
      if (!resp.ok && attempt < retries && resp.status >= 500) {
        // 仅服务器错误重试，客户端错误不重试
        await sleep(1000 * (attempt + 1)); // 递增等待
        continue;
      }
      return resp;
    } catch (e) {
      lastErr = e;
      if (e.name === 'AbortError') {
        console.warn('fetch timeout:', url, timeoutMs + 'ms');
      }
      if (attempt < retries) {
        await sleep(1000 * (attempt + 1));
      }
    }
  }
  throw lastErr || new Error('fetch failed after ' + retries + ' retries');
}

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }
