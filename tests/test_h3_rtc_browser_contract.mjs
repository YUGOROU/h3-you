import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Script contract only. No browser, signaling network, playback or capture.
test('single RTC control channel matches native bridge and presentation gates ACK', async()=>{
 const html=fs.readFileSync(new URL('../research/h3_rtc_client.html',import.meta.url),'utf8');
 assert.ok(!html.includes('getUserMedia'));
 const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
 const sent=[];let channel,peer;const nodes={};
 const video={currentTime:0,paused:false,srcObject:null,muted:true,
  play:async()=>{},pause(){this.paused=true},requestVideoFrameCallback(fn){this.callback=fn}};
 nodes['#media']=video;nodes['#status']={textContent:''};
 for(const k of ['#connect','#unmute','#stop'])nodes[k]={};
 const mb={session:'session',profile_sha256:'a'.repeat(64),epoch:1,fps:24,sample_rate:48000,max_segments:16,max_total_frames:1654};
 const binding={job_id:'b'.repeat(24),session:'session',boot:'c'.repeat(32),expires_at:Date.now()/1000+60,ice_profile:'host-only-v1'};
 class Stream{constructor(){this.tracks=[]}getTracks(){return this.tracks}addTrack(t){this.tracks.push(t)}}
 class PC{constructor(){peer=this;this.iceGatheringState='complete'}addTransceiver(){return {setCodecPreferences(){}}}
  createDataChannel(label,options){assert.equal(label,'control');assert.equal(options.ordered,true);channel={readyState:'open',bufferedAmount:0,send(x){sent.push(x)}};return channel}
  async createOffer(){return {type:'offer',sdp:'not-a-network-offer'}}
  async setLocalDescription(d){this.localDescription=d}
  async setRemoteDescription(){this.ontrack({track:{kind:'audio',stop(){}}});this.ontrack({track:{kind:'video',stop(){}}});channel.onopen()}
  close(){}}
 const context=vm.createContext({console,JSON,Math,Date,Number,TextEncoder,Error,
  document:{querySelector:k=>nodes[k]},location:{hash:'#cap',pathname:'/'},history:{replaceState(){}},window:{addEventListener(){}},
  performance:{now:()=>1000},MediaStream:Stream,RTCPeerConnection:PC,RTCRtpReceiver:{getCapabilities:()=>({codecs:[{mimeType:'audio/opus'},{mimeType:'video/H264'}]})},
  AbortSignal:{timeout:()=>null},setTimeout:()=>1,clearTimeout(){},setInterval:()=>2,clearInterval(){},
  fetch:async(path)=>({ok:true,json:async()=>path==='/config'?{binding,media_binding:mb,iceServers:[]}:path==='/offer'?{session:binding.session,boot:binding.boot,description:{type:'answer',sdp:'answer'}}:{}})});
 vm.runInContext(script,context);
 await nodes['#connect'].onclick();
 assert.deepEqual(sent,['start']);channel.onmessage({data:'started'});
 const identity={schema:'h3-rtc-media-v1',session:mb.session,epoch:1,profile_sha256:mb.profile_sha256};
 channel.onmessage({data:JSON.stringify({...identity,kind:'window',index:0,frame_start:0,frame_stop:1,sample_start:0,sample_stop:2000,fps:24,sample_rate:48000,final:true})});
 assert.equal(sent.length,1,'metadata alone must not acknowledge');
 video.callback(0,{mediaTime:0,presentedFrames:1});
 assert.equal(sent.length,1,'first presentation lacks full audio-clock coverage');
 video.currentTime=1/24+0.001;vm.runInContext('progress()',context);
 const ack=JSON.parse(sent[1]);
 assert.deepEqual(ack,{...identity,kind:'sink-accepted',index:0,video_frame_stop:1,audio_sample_stop:2000});
 channel.onmessage({data:JSON.stringify({...identity,kind:'source_eos',index:0,video_frame_stop:1,audio_sample_stop:2000})});
 assert.equal(sent.length,2,'no forged extra end/ack message');
 assert.equal(video.muted,true,'output remains explicit user opt-in');
 vm.runInContext('stop()',context);
 assert.ok(peer);
});
