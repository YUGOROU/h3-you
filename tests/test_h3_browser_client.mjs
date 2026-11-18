import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const html=fs.readFileSync(new URL('../research/h3_browser_client.html',import.meta.url),'utf8');
const helpers=html.split('// PURE HELPERS BEGIN')[1].split('// PURE HELPERS END')[0];
const ctx=vm.createContext({TextEncoder,Uint8Array,DataView,AbortController});
vm.runInContext(helpers+'\nglobalThis.api={parseManifest,localPath,codecFromInit,appendSerial,removeSerial,mayEnd,requireMSE};',ctx);
const {parseManifest,localPath,codecFromInit,appendSerial,removeSerial,mayEnd,requireMSE}=ctx.api;
const playlist='#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-PLAYLIST-TYPE:EVENT\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:2.000,\nseg-000.m4s';
test('strict full event history and EOS',()=>{
 const m=parseManifest(playlist);assert.equal(m.segments.length,1);assert.equal(mayEnd(m,1),false);
 const end=parseManifest(playlist+'\n#EXT-X-ENDLIST');assert.equal(mayEnd(end,0),false);assert.equal(mayEnd(end,1),true);
 for(const line of ['https://evil/seg-000.m4s','../seg-000.m4s','seg-001.m4s','/seg-000.m4s','seg-000.m4s?x=1'])assert.throws(()=>parseManifest(playlist.replace('seg-000.m4s',line)));
 for(const tag of ['#EXT-X-KEY:METHOD=AES-128','#EXT-X-DISCONTINUITY','#EXT-X-STREAM-INF:BANDWIDTH=1','#UNKNOWN'])assert.throws(()=>parseManifest(playlist+'\n'+tag));
 assert.throws(()=>parseManifest(playlist.replace('URI="init.mp4"','URI="https://evil/init.mp4"')));
 assert.throws(()=>parseManifest(playlist.replace('2.000','9.000')));
 assert.throws(()=>parseManifest('x'.repeat(65537)));
 for(const p of ['//evil/init.mp4','/../init.mp4','/init.mp4?x','https://evil','/api/other'])assert.throws(()=>localPath(p));
});
function box(type,...payload){const body=Buffer.concat(payload.map(x=>Buffer.from(x)));const head=Buffer.alloc(8);head.writeUInt32BE(body.length+8);head.write(type,4);return Buffer.concat([head,body]);}
function init(level=42){const avcc=box('avcC',[1,100,0,level,255,225,0,4,103,100,0,level,1,0,1,104]);const avc=box('avc1',Buffer.alloc(78),avcc);const stsd=box('stsd',[0,0,0,0,0,0,0,1],avc);return box('moov',box('trak',box('mdia',box('minf',box('stbl',stsd)))));}
test('AVC codec derives SPS profile compatibility and level',()=>{
 for(const level of [31,42,51]){const b=init(level);assert.equal(codecFromInit(b),`video/mp4; codecs="avc1.6400${level.toString(16)}, mp4a.40.2"`);}
 assert.throws(()=>codecFromInit(Buffer.concat([init(),init()])));
 assert.throws(()=>codecFromInit(init().subarray(0,20)));
 const bad=init();bad[bad.indexOf(Buffer.from([103,100,0,42]))+3]=31;assert.throws(()=>codecFromInit(bad));
});
test('codec-less MP4 false is not a valid MSE rejection',()=>{
 let bareCalls=0;
 class MSE {static isTypeSupported(mime){if(mime==='video/mp4'){bareCalls++;return false;}return mime.includes('mp4a.40.2');}}
 assert.equal(requireMSE(MSE),MSE);
 assert.equal(bareCalls,0);
 assert.equal(MSE.isTypeSupported(codecFromInit(init())),true);
 assert.throws(()=>requireMSE(undefined),/API unavailable/);
 assert.throws(()=>requireMSE({}),/API unavailable/);
 assert.doesNotMatch(html,/isTypeSupported\('video\/mp4'\)/);
});
test('append waits for updateend, disallows concurrent append, propagates abort',async()=>{
 class SB extends EventTarget {updating=false; items=[];appendBuffer(x){this.updating=true;this.items.push(x);}finish(){this.updating=false;this.dispatchEvent(new Event('updateend'));}}
 const sb=new SB(), c=new AbortController();let resolved=false;
 const first=appendSerial(sb,1,c.signal).then(()=>{resolved=true;});await Promise.resolve();assert.equal(resolved,false);
 await assert.rejects(appendSerial(sb,2,c.signal));sb.finish();await first;
 const second=appendSerial(sb,2,c.signal);sb.finish();await second;assert.deepEqual(sb.items,[1,2]);
 const third=appendSerial(sb,3,c.signal);c.abort();await assert.rejects(third);
});
test('removal obeys the same serial and abort contract',async()=>{
 class SB extends EventTarget {updating=false;remove(){this.updating=true;}}
 const sb=new SB(), c=new AbortController();const pending=removeSerial(sb,0,4,c.signal);
 c.abort();await assert.rejects(pending);
});
test('pending playlist is not a stream failure and bodies are incrementally bounded',async()=>{
 let cancelled=false,released=false;
 const local=vm.createContext({Uint8Array,AbortController,localPath,
  fetch:async()=>({ok:false,status:404,headers:{get:()=>null}})});
 const source='function read('+html.split('function read(')[1].split('const sleep=')[0];
 vm.runInContext('let controller=new AbortController();let queue=Promise.resolve();'+source+';globalThis.readBody=read;',local);
 assert.equal(await local.readBody('/index.m3u8',64,{},true),null);
 await assert.rejects(local.readBody('/seg-000.m4s',64,{},true),/HTTP 404/);
 let count=0;
 local.fetch=async()=>({ok:true,status:200,headers:{get:()=>null},body:{getReader:()=>({
  read:async()=>count++<2?{done:false,value:new Uint8Array(3)}:{done:true},
  cancel:async()=>{cancelled=true;},releaseLock:()=>{released=true;}})}});
 await assert.rejects(local.readBody('/init.mp4',4),/too large/);
 assert.equal(cancelled,true);assert.equal(released,true);
});
test('GPU warm-up and active playback clocks are separate and never renewed',async()=>{
 const armed=[];let state='waiting';
 const local=vm.createContext({TextDecoder,TextEncoder,Number,LIMITS:{seconds:240},
  read:async()=>new TextEncoder().encode(JSON.stringify({output:{default_is_builtin_speaker:true},stream_state:state,startup_seconds:1200,max_seconds:240})).buffer,
  armPhase:n=>armed.push(n),stop:async()=>{throw Error('unexpected stop');}});
 const source='async function checkOutput('+html.split('async function checkOutput(')[1].split('prepare.onclick=')[0];
 vm.runInContext('let activeArmed=false,startupArmed=false;'+source+';globalThis.check=checkOutput;',local);
 await local.check();await local.check();state='streaming';await local.check();await local.check();
 assert.deepEqual(armed,[1200,240]);
});
test('self-contained explicit playback, one multiplexed buffer, no capture or fallback',()=>{
 assert.equal((html.match(/<video\s/g)||[]).length,1);
 assert.equal((html.match(/<script\b/g)||[]).length,1);
 assert.equal((html.match(/addSourceBuffer\(/g)||[]).length,1);
 assert.doesNotMatch(html,/<script[^>]+src=|<link|autoplay|getUserMedia|getDisplayMedia|RTCPeerConnection|setSinkId|window\.open|https?:\/\//i);
 assert.match(html,/await checkOutput\(\);if\(stopped\)return;approved=true;await video.play\(\)/);
 assert.match(html,/redirect:'error'/);assert.match(html,/total>cap/);assert.match(html,/ahead\(\)\+duration>LIMITS.buffer/);
});
