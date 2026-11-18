// DOM/HTTP simulation only: validates keyboard/save logic, not browser rendering.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.split('<script>')[1].split('</script>')[0].replace('__TOKEN__','"test"');
const nodes={},listeners={};let posts=0,labels={};
function element(id='') {return {id,value:'',tagName:'DIV',disabled:false,textContent:'',innerHTML:'',
 classList:{toggle(){}},append(){},removeAttribute(){},replaceWith(next){nodes[this.id]=next;}};}
function node(id){return nodes[id]||(nodes[id]=element(id));}
const candidates=[{id:'a',pool:'identity'},{id:'b',pool:'motion'},{id:'t',pool:'review'}].map(c=>({...c,session_id:'s',start_seconds:0,end_seconds:5.166667,score:0,reasons:[]}));
const context=vm.createContext({console,URLSearchParams,location:{search:'?view=approval'},confirm:()=>false,
 document:{getElementById:node,querySelectorAll:()=>[],createElement:()=>element(),addEventListener:(key,fn)=>listeners[key]=fn},
 fetch:async(url,opts)=>{if(url==='/data')return {ok:true,json:async()=>({candidates,labels,references:[],initial_ids:['a','b'],calibration_ids:['t'],proposals:{}})};
 if(url==='/label'){posts++;const d=JSON.parse(opts.body);if(d.label===null)delete labels[d.id];else labels[d.id]={label:d.label};return{ok:true,json:async()=>({labels:{...labels}})};}throw Error('unexpected fetch');}});
node('session').value='all';
const run=s=>vm.runInContext(s,context);
const key=k=>listeners.keydown({key:k,target:{tagName:'BODY'},preventDefault(){}});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 vm.runInContext(script,context);await tick();assert.equal(run('queue.length'),2);assert.equal(run('pos'),0);
 const oldImage=node('strip');key('ArrowRight');assert.equal(run('pos'),1);assert.equal(posts,0);
 oldImage.onload();assert.equal(run('imageReady'),false);key('ArrowUp');await tick();assert.equal(posts,0);
 node('strip').onload();key('Enter');await tick();assert.equal(posts,1);assert.equal(labels.b.label,'motion');assert.equal(run('queue.length'),2);
 key('ArrowLeft');assert.equal(posts,1);node('strip').onload();key('ArrowDown');await tick();assert.equal(labels.a.label,'reject');
 key('ArrowLeft');node('strip').onload();key('Enter');await tick();assert.equal(labels.a.label,'reject');assert.equal(posts,2);
 node('filter').value='calibration';run('rebuild(false)');assert.equal(run('queue.length'),1);node('strip').onload();
 key('Enter');await tick();assert.equal(posts,2);key('ArrowRight');await tick();assert.equal(posts,2);
 node('strip').onload();key('h');await tick();assert.equal(labels.t.label,'hold');assert.equal(posts,3);
 console.log('keyboard/navigation/approval/load-race simulation passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
