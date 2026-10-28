const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const frontend = path.join(__dirname, '../Frontend');
const config = fs.readFileSync(path.join(frontend, 'api-config.js'), 'utf8');
for (const protocol of ['http:', 'https:', 'file:']) {
    const context = {window: {location: {protocol}}};
    vm.runInNewContext(config, context);
    assert.equal(context.window.MEMORYBOT_API_BASE, protocol === 'file:' ? 'http://127.0.0.1:8000' : '');
}
for (const filename of ['chat.html', 'index.html']) {
    const html = fs.readFileSync(path.join(frontend, filename), 'utf8');
    assert.match(html, /src="api-config.js"/);
    assert.doesNotMatch(html, /http:\/\/127\.0\.0\.1:8000/);
    for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
}
console.log('PASS: same-origin API config, file fallback, no hardcoded API origin in pages');
