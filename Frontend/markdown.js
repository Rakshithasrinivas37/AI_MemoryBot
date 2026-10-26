/* Render Markdown without allowing embedded HTML or executable links. */
(function (root) {
    const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
    const renderer = new marked.Renderer();
    const originalLink = renderer.link;
    renderer.html = token => escapeHTML(token.text);
    renderer.link = function (token) {
        // Decode entities before checking the protocol to catch obfuscated URLs.
        const href = token.href.replace(/&#(x[0-9a-f]+|[0-9]+);?/gi, (_, number) => {
            const code = number[0].toLowerCase() === 'x' ? parseInt(number.slice(1), 16) : parseInt(number, 10);
            return code <= 0x10ffff ? String.fromCodePoint(code) : '';
        }).replace(/&colon;/gi, ':').trim();
        if (!/^https?:\/\//i.test(href) && !/^mailto:/i.test(href)) {
            return this.parser.parseInline(token.tokens);
        }
        return originalLink.call(this, {...token, href});
    };
    // Saved conversations should not load remote tracking images.
    renderer.image = token => escapeHTML(token.text);
    const parser = new marked.Marked({renderer, gfm: true, breaks: true});
    root.renderMarkdown = text => parser.parse(String(text ?? ''));
})(globalThis);
