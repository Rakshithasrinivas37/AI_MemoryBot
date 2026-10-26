// Run with node tests/test_memory_details_ui.cjs. No browser dependencies.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../Frontend/chat.html'), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const elements = new Map();
function element() {
    return {children:[], dataset:{}, value:'', textContent:'', innerHTML:'', open:false,
        classList:{add(){},remove(){},toggle(){}}, setAttribute(){}, addEventListener(){}, focus(){},
        appendChild(child){this.children.push(child)}, append(...children){this.children.push(...children)},
        replaceChildren(){this.children=[]}, showModal(){this.open=true}, close(){this.open=false}};
}
const record = {id:'memory-1', title:'Planning a Singapore Career', summary:'A summary <script>not HTML</script>',
    created_at:'2026-10-26', conversations:Array.from({length:5},(_,i)=>({user:`Question ${i}`,assistant:`Full answer ${i}`}))};
let deleteBody;
const context = vm.createContext({URLSearchParams, AbortController, setTimeout, clearTimeout, setInterval,
    window:{addEventListener(){}},
    document:{hidden:false,activeElement:null,addEventListener(){},querySelectorAll(){return []},createElement:element,
        getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)}},
    fetch:async(url, options)=>{if(options?.method === "DELETE") deleteBody = JSON.parse(options.body);return {ok:true,json:async()=>({memories:[record]})}}
});
for (const filename of ['marked.min.js', 'markdown.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../Frontend', filename), 'utf8'), context);
}
vm.runInContext(script,context);
(async()=>{
    await vm.runInContext('loadMemories("alice")',context);
    const row=elements.get('memory-list').children[0];
    assert.equal(row.children[0].textContent, record.title);
    row.children[0].onclick();
    assert.equal(elements.get('memory-detail').open,true);
    assert.match(elements.get('memory-detail-summary').innerHTML, /&lt;script&gt;/);
    assert.doesNotMatch(elements.get('memory-detail-summary').innerHTML, /<script>/);
    assert.equal(elements.get('memory-detail-conversation').children.length,5);
    assert.match(elements.get('memory-detail-conversation').children[4].children[2].children[0].innerHTML,/Full answer 4/);
    assert.equal(elements.get('memory-detail-conversation').children[0].children[1].className, 'question-box memory-question');
    context.legacy={title:'Legacy',summary:'Old summary'};
    vm.runInContext('openMemory(legacy)',context);
    assert.match(elements.get('memory-detail-conversation').children[0].textContent,/not saved/);
    row.children[2].children[0].onclick();
    await vm.runInContext('confirmDelete()',context);
    assert.equal(deleteBody.memory_id, 'memory-1');
    assert.equal(deleteBody.memory, record.summary);
    console.log('PASS: titled sidebar, click-to-open, five full exchanges, safe text rendering, legacy message, deletion flow');
})().catch(error=>{console.error(error);process.exitCode=1});
