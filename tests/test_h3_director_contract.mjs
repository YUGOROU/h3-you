import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDraft,validateVisualIntent,makeTools,BASELINE,MODEL,SYSTEM,VISUAL_SYSTEM,VISUAL_TURN_SCHEMA,VISUAL_INTENT_SCHEMA} from '../research/director/contract.mjs';
const Type={Null:()=>({type:'null'}),Object:(p,o)=>({p,o}),String:o=>({type:'string',...o}),Union:v=>({anyOf:v}),Literal:v=>({const:v}),Integer:()=>({type:'integer'}),Unknown:()=>({})};
const draft={reply:'こんにちは。',expression:'neutral',motion:'still',tone:'calm'};
const intent=instruction=>({schema:VISUAL_INTENT_SCHEMA,instruction});
const visual=instruction=>({schema:VISUAL_TURN_SCHEMA,reply:'少し周りを見せるね。',visual_intent:intent(instruction)});

test('ASR policy favors contextual correction without semantic narrowing or invented facts',()=>{
  assert.match(SYSTEM,/ASR transcript/);
  assert.match(SYSTEM,/silently correct recognition errors instead of asking/);
  assert.match(SYSTEM,/授業員 as 従業員/);
  assert.match(SYSTEM,/do not narrow 従業員 to 正社員/);
  assert.match(SYSTEM,/does not justify inventing facts/);
});

test('strict proposals cannot change model, devices or reference',()=>{
  assert.deepEqual(validateDraft(draft),draft);
  for (const field of ['checkpoint','camera','microphone','reference','command'])assert.throws(()=>validateDraft({...draft,[field]:'override'}));
  assert.throws(()=>validateDraft({...draft,motion:'turn_camera_on'}));
  assert.throws(()=>validateDraft({...draft,reply:'長'.repeat(161)}));
  assert.equal(MODEL,'Qwen/Qwen3.8-27B:cerebras');
  assert.throws(()=>{BASELINE.canvas[0]=999;});
});

test('custom tools only stage once; abort cannot stage',async()=>{
  const state={id:'test',epoch:1,canceled:false,draft:null};const tools=makeTools(Type,state);
  assert.deepEqual(tools.map(t=>t.name),['h3_get_state','h3_submit_turn']);
  const snapshot=await tools[0].execute();assert.equal(snapshot.structuredContent.sensors,false);
  const output=await tools[1].execute('1',draft,new AbortController().signal);
  assert.equal(output.terminate,true);assert.equal(output.structuredContent.effects_executed,false);
  await assert.rejects(tools[1].execute('2',draft));
  const canceled={id:'old',epoch:0,canceled:true,draft:null};
  await assert.rejects(makeTools(Type,canceled)[1].execute('3',draft));
  assert.equal(canceled.draft,null);
});

test('visual contract is explicit, versioned and has no action preset list',()=>{
  const proposed=visual('Turn the handheld view away from the face to show the surroundings.');
  assert.throws(()=>validateDraft(proposed));
  assert.throws(()=>validateDraft(draft,{visualIntents:true}));
  assert.deepEqual(validateDraft(proposed,{visualIntents:true}),proposed);
  assert.deepEqual(validateDraft(visual(null),{visualIntents:true}).visual_intent,intent(null));
  for (const field of ['achieved','clip','expression','device','reference'])
    assert.throws(()=>validateDraft({...proposed,[field]:true},{visualIntents:true}));
  assert.throws(()=>validateDraft(proposed,{visualIntents:1}));
});

test('visual directions are bounded data; null continues and no success is inferred',()=>{
  assert.deepEqual(validateVisualIntent(intent('  Return to self view.  ')),intent('Return to self view.'));
  assert.deepEqual(validateVisualIntent(intent('😀'.repeat(256))),intent('😀'.repeat(256)));
  for (const bad of ['', ' ', 'x'.repeat(257), 'line\ncommand', '\u007f', '\ud800', 1, false, []])
    assert.throws(()=>validateVisualIntent(intent(bad)));
  assert.throws(()=>validateVisualIntent({...intent(null),achieved:true}));
  assert.throws(()=>validateVisualIntent({...intent(null),schema:'future'}));
});

test('same two tools stage visual proposal only and reject canceled callbacks',async()=>{
  const state={id:'visual-turn',epoch:2,canceled:false,draft:null};
  const tools=makeTools(Type,state,{visualIntents:true});
  assert.deepEqual(tools.map(t=>t.name),['h3_get_state','h3_submit_turn']);
  assert.equal(tools[1].parameters.p.schema.const,VISUAL_TURN_SCHEMA);
  const output=await tools[1].execute('visual',visual('Return to the self-facing view.'));
  assert.equal(output.structuredContent.effects_executed,false);
  assert.equal(output.structuredContent.epoch,2);
  assert.equal(output.structuredContent.achieved,undefined);
  await assert.rejects(tools[1].execute('duplicate',visual(null)));
  const old={id:'old',epoch:1,canceled:true,draft:null};
  await assert.rejects(makeTools(Type,old,{visualIntents:true})[1].execute('late',visual('Turn away.')));
  assert.equal(old.draft,null);
});

test('visual system separates handheld direction, continuing intent and observed achievement',()=>{
  assert.match(VISUAL_SYSTEM,/null to continue the current visual action/);
  assert.match(VISUAL_SYSTEM,/neutral natural handheld selfie view/);
  assert.match(VISUAL_SYSTEM,/without importing camera instructions from an earlier session/);
  assert.match(VISUAL_SYSTEM,/ordinary questions or mentions of the background, use null/);
  assert.match(VISUAL_SYSTEM,/Only an explicit request/);
  assert.match(VISUAL_SYSTEM,/small bounded adjustment/);
  assert.match(VISUAL_SYSTEM,/then hold the resulting natural composition/);
  assert.match(VISUAL_SYSTEM,/excessive movement requests/);
  assert.match(VISUAL_SYSTEM,/Null retains that instruction and composition/);
  assert.doesNotMatch(VISUAL_SYSTEM,/propose turning the generated handheld view away/);
  assert.match(VISUAL_SYSTEM,/returning to a self-facing view/);
  assert.match(VISUAL_SYSTEM,/paraphrases and context, not exact keywords/);
  assert.match(VISUAL_SYSTEM,/previous proposal is not visual success evidence/);
  assert.doesNotMatch(VISUAL_SYSTEM,/visual prompt unchanged/);
  assert.match(SYSTEM,/visual prompt unchanged/);
});
