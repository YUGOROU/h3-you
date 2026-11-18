import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
const code = readFileSync(new URL('../research/obs_portrait_browser.js', import.meta.url), 'utf8');
function fixture({width=1920,height=1080,unsupported=false,failPlay=false,unprepared=false,label='OBS Virtual Camera'}={}) {
 class Track extends EventTarget {
  constructor(kind,label) { super(); Object.assign(this,{kind,label,readyState:'live',stops:0,calls:[],frames:0}); }
  stop() { this.stops++; this.readyState='ended'; }
  async applyConstraints(c) { this.calls.push(c); }
  requestFrame() { this.frames++; }
 }
 class Stream {
  constructor(tracks) { this.tracks=tracks; }
  getTracks() { return this.tracks; }
  getVideoTracks() { return this.tracks.filter(t=>t.kind==='video'); }
 }
 const source=new Track('video',label), audio=new Track('audio','original'), other=new Track('video','other');
 const stream=new Stream([source,audio,other]);
 const replacement=new Track('video','canvas');
 const draws=[], cancellations=[], captures=[];
 let callback, requests=0, oldRestores=0;
 const video=Object.assign(new EventTarget(),{
  readyState:1,videoWidth:width,videoHeight:height,paused:false,
  requestVideoFrameCallback(fn) { callback=fn; return 42; },
  cancelVideoFrameCallback(id) { cancellations.push(id); },
  async play() { if(failPlay) throw Error('private source label/device ID/exception detail'); },
  pause() { this.paused=true; }
 });
 if(unsupported) video.requestVideoFrameCallback=undefined;
 const canvas={getContext:()=>({drawImage:(...args)=>draws.push(args)}),captureStream:(rate)=>{captures.push(rate);return new Stream([replacement]);}};
 const original=async()=>{requests++;return stream;};
 const devices={getUserMedia:original};
 const pixels=new Uint8ClampedArray(16*9*4);
 for(let i=0;i<pixels.length;i+=4){pixels[i]=unprepared?60:0;pixels[i+3]=255;}
 const check={getContext:()=>({drawImage(){},getImageData:()=>({data:pixels})})};
 let canvases=0;
 const context=vm.createContext({navigator:{mediaDevices:devices},document:{createElement:tag=>tag==='video'?video:(canvases++===0?canvas:check)}, MediaStream:Stream,MediaStreamTrack:Track,setTimeout,clearTimeout,h3OBSPortrait:{restore(){oldRestores++;}}});
 vm.runInContext(code,context);
 return {source,audio,other,stream,replacement,draws,captures,video,canvas,devices,original,context,cancellations,requests:()=>requests,oldRestores:()=>oldRestores,tick:()=>callback()};
}
test('complete centered portrait extracted without source constraints or extra capture; audio identity',async()=>{
 const f=fixture(); assert.equal(f.requests(),0); assert.equal(f.oldRestores(),1);
 const result=await f.devices.getUserMedia({video:true,audio:true});
 assert.equal(f.requests(),1);
 assert.deepEqual(Array.from(result.getTracks()),[f.replacement,f.audio,f.other]);
 assert.deepEqual(Array.from(f.video.srcObject.getTracks()),[f.source]);
 assert.equal(f.video.muted,true);
 assert.deepEqual(f.draws[0].slice(1),[656.25,0,607.5,1080,0,0,720,1280]);
 assert.deepEqual(f.captures,[0]); assert.equal(f.source.calls.length,0);
 assert.equal(f.canvas.width/f.canvas.height,9/16);
 f.tick(); assert.equal(f.replacement.frames,2);
 f.context.h3OBSPortrait.restore(); assert.equal(f.devices.getUserMedia,f.original);
 assert.equal(f.source.stops,0); // undo only unhooks
 f.replacement.stop();
});
for(const [options,reason] of [
 [{width:720,height:1280},'geometry_mismatch'],
 [{width:1440,height:1080},'geometry_mismatch'],
 [{unsupported:true},'capability_unavailable'],
 [{failPlay:true},'preparation_failed'],
 [{unprepared:true},'letterbox_unprepared']
]) {
 test(`safe original fallback ${JSON.stringify(options)}`,async()=>{
  const f=fixture(options); assert.equal(await f.devices.getUserMedia({video:true}),f.stream);
  assert.equal(f.source.stops,0);assert.equal(f.audio.stops,0);assert.equal(f.source.calls.length,0);
  assert.equal(f.video.srcObject,null);assert.equal(f.video.paused,true);
  assert.deepEqual(status(f),{installed:true,activeConversionCount:0,fallbackCount:1,
   lastReason:reason,output:{width:720,height:1280}});
  assert.equal(f.requests(),1);
  assert.deepEqual(f.audio.calls,[]);
  assert.deepEqual(Array.from(f.stream.getTracks()),[f.source,f.audio,f.other]);
 });
}
test('non OBS streams unchanged',async()=>{
 const f=fixture({label:'physical'});assert.equal(await f.devices.getUserMedia({video:true}),f.stream);
 assert.equal(f.draws.length,0);assert.equal(f.source.calls.length,0);
 assert.equal(status(f).activeConversionCount,0);assert.equal(status(f).fallbackCount,0);
 assert.equal(status(f).lastReason,null);
});
test('replacement stop closes source and frame pump, never audio or unrelated video',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 f.replacement.stop();f.replacement.stop();f.tick();
 assert.equal(f.source.stops,1);assert.equal(f.replacement.stops,1);
 assert.equal(f.audio.stops,0);assert.equal(f.other.stops,0);
 assert.deepEqual(f.cancellations,[42]);assert.equal(f.video.srcObject,null);
 assert.equal(f.video.paused,true);assert.equal(f.draws.length,1);
 assert.equal(status(f).activeConversionCount,0);
 assert.equal(status(f).lastReason,'conversion_stopped');
 assert.equal(status(f).fallbackCount,0);
});
test('source ended closes replacement',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 f.source.dispatchEvent(new Event('ended'));
 assert.equal(f.replacement.stops,1);assert.equal(f.video.srcObject,null);assert.equal(f.audio.stops,0);
 assert.equal(status(f).activeConversionCount,0);assert.equal(status(f).lastReason,'source_ended');
 f.replacement.stop();assert.equal(status(f).activeConversionCount,0);
});
test('landscape and advanced resolution constraints cannot change portrait geometry',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 await f.replacement.applyConstraints({width:{exact:1920},height:1080,aspectRatio:16/9,resizeMode:'crop-and-scale',frameRate:24,advanced:[{width:640,height:480,frameRate:15}]});
 const applied=JSON.parse(JSON.stringify(f.replacement.calls[0]));
 assert.deepEqual(applied,{frameRate:24,advanced:[{frameRate:15}]});
 assert.equal(f.canvas.width,720);assert.equal(f.canvas.height,1280);assert.equal(f.source.calls.length,0);
 f.replacement.stop();
});

function status(f) {
 return JSON.parse(JSON.stringify(f.context.h3OBSPortrait.status()));
}
test('installation is not active conversion; status is detached metadata only',()=>{
 const f=fixture();
 const expected={installed:true,activeConversionCount:0,fallbackCount:0,lastReason:null,
  output:{width:720,height:1280}};
 assert.deepEqual(status(f),expected);assert.equal(f.requests(),0);
 const snapshot=f.context.h3OBSPortrait.status();
 snapshot.output.width=1;snapshot.activeConversionCount=99;
 assert.deepEqual(status(f),expected);
 f.context.h3OBSPortrait.restore();
 assert.deepEqual(status(f),{...expected,installed:false});
 assert.equal(f.requests(),0);
});
test('restore only unhooks future calls; existing conversion stays counted until stopped',async()=>{
 const f=fixture(), api=f.context.h3OBSPortrait;
 const result=await f.devices.getUserMedia({video:true,audio:true});
 assert.equal(status(f).activeConversionCount,1);
 api.restore();api.restore();
 assert.equal(status(f).installed,false);assert.equal(status(f).activeConversionCount,1);
 assert.equal(f.source.stops,0);assert.equal(f.replacement.stops,0);
 assert.equal(await f.devices.getUserMedia({video:true,audio:true}),f.stream);
 assert.equal(status(f).activeConversionCount,1);assert.equal(f.requests(),2);
 result.getVideoTracks()[0].stop();
 assert.equal(status(f).activeConversionCount,0);
 assert.equal(f.audio.stops,0);assert.deepEqual(f.audio.calls,[]);
});
test('source ending after restore still cleans up and updates retained status',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 f.context.h3OBSPortrait.restore();
 f.source.dispatchEvent(new Event('ended'));
 assert.equal(status(f).installed,false);assert.equal(status(f).activeConversionCount,0);
 assert.equal(status(f).lastReason,'source_ended');
 assert.equal(f.replacement.stops,1);assert.equal(f.audio.stops,0);
});
test('replacement ended event retires conversion once without touching audio',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 f.replacement.dispatchEvent(new Event('ended'));
 f.replacement.stop();
 assert.equal(status(f).activeConversionCount,0);
 assert.equal(status(f).lastReason,'conversion_ended');
 assert.equal(f.source.stops,1);assert.equal(f.audio.stops,0);
});
test('geometry drift retires active conversion with fixed reason, not a fallback claim',async()=>{
 const f=fixture();await f.devices.getUserMedia({video:true});
 f.video.videoWidth=1440;f.tick();
 assert.equal(status(f).activeConversionCount,0);
 assert.equal(status(f).lastReason,'geometry_mismatch');
 assert.equal(status(f).fallbackCount,0);assert.equal(f.audio.stops,0);
});
test('source already ended falls back with sanitized reason',async()=>{
 const f=fixture();f.source.readyState='ended';
 assert.equal(await f.devices.getUserMedia({video:true}),f.stream);
 assert.equal(status(f).lastReason,'source_ended');
 assert.equal(status(f).activeConversionCount,0);assert.equal(status(f).fallbackCount,1);
 assert.equal(f.source.stops,0);assert.equal(f.audio.stops,0);
});
