import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDraft,STABLE_SELFIE_SYSTEM,VISUAL_TURN_SCHEMA,VISUAL_INTENT_SCHEMA} from '../research/director/contract.mjs';
const draft=instruction=>({schema:VISUAL_TURN_SCHEMA,reply:'Hello',visual_intent:{schema:VISUAL_INTENT_SCHEMA,instruction}});
test('stable selfie accepts only null, leaving visual legacy unchanged',()=>{
  assert.equal(validateDraft(draft(null),{visualIntents:true,stableSelfie:true}).visual_intent.instruction,null);
  assert.throws(()=>validateDraft(draft('Pan right'),{visualIntents:true,stableSelfie:true}),/forbids/);
  assert.equal(validateDraft(draft('Pan right'),{visualIntents:true}).visual_intent.instruction,'Pan right');
  assert.throws(()=>validateDraft(draft(null),{stableSelfie:true}),/selection/);
  assert.throws(()=>validateDraft(draft(null),{visualIntents:true,stableSelfie:1}),/selection/);
  assert.match(STABLE_SELFIE_SYSTEM,/always set visual_intent.instruction to null/);
});
