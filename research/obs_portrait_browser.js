/* Paste in the Discord page only, before camera OFF→ON.
 * Requires a complete 9:16 portrait centered full-height in a 16:9 OBS frame.
 * No capture requests, audio processing, network, storage, or source constraints.
 * Metadata only: h3OBSPortrait.status(); active counts are not remote acceptance.
 * Undo hook only: h3OBSPortrait.restore(). Existing conversions keep their cleanup.
 * Installation alone implies no active conversion or layout/identity success.
 */
(() => {
  'use strict';
  globalThis.h3OBSPortrait?.restore();
  const devices = navigator.mediaDevices;
  const original = devices.getUserMedia;
  let activeConversionCount = 0, fallbackCount = 0, lastReason = null;
  class PreparationFailure extends Error {
    constructor(reason) { super('Portrait preparation failed'); this.reason = reason; }
  }
  const prepare = async (source) => {
    const video = document.createElement('video');
    let output, replacement, callback, timer;
    let closed = false;
    let active = false;
    let reason = 'preparation_failed';
    let readyCleanup = () => {};
    const cleanup = (stopSource = false, endedReason = null) => {
      if (closed) return;
      closed = true;
      if (active) {
        active = false;
        activeConversionCount--;
        if (endedReason) lastReason = endedReason;
      }
      readyCleanup();
      clearTimeout(timer);
      if (callback !== undefined) video.cancelVideoFrameCallback?.(callback);
      video.pause();
      video.srcObject = null;
      video.removeEventListener('error', onEnded);
      source.removeEventListener('ended', onEnded);
      replacement?.removeEventListener('ended', onReplacementEnded);
      output?.getTracks().forEach(track => nativeStop.call(track));
      if (stopSource) source.stop();
    };
    const onEnded = event => {
      reason = event.type === 'ended' ? 'source_ended' : 'preparation_failed';
      cleanup(false, reason);
    };
    const onReplacementEnded = () => cleanup(true, 'conversion_ended');
    const nativeStop = globalThis.MediaStreamTrack?.prototype.stop;
    try {
      if (!nativeStop || !video.requestVideoFrameCallback || !video.cancelVideoFrameCallback) {
        reason = 'capability_unavailable';
        throw new Error('frame callbacks unavailable');
      }
      video.muted = true;
      video.playsInline = true;
      video.srcObject = new MediaStream([source]);
      await new Promise((resolve, reject) => {
        const loaded = () => resolve();
        const failed = event => {
          if (event?.type === 'ended') reason = 'source_ended';
          reject(new Error('source unavailable'));
        };
        readyCleanup = () => {
          video.removeEventListener('loadedmetadata', loaded);
          video.removeEventListener('error', failed);
          source.removeEventListener('ended', failed);
          clearTimeout(timer);
        };
        video.addEventListener('loadedmetadata', loaded);
        video.addEventListener('error', failed);
        source.addEventListener('ended', failed);
        timer = setTimeout(failed, 3000);
        if (video.readyState >= 1) resolve();
      });
      readyCleanup();
      const validGeometry = () => video.videoWidth > 0 && video.videoHeight > 0 &&
        Math.abs(video.videoWidth / video.videoHeight - 16 / 9) < 0.002;
      if (source.readyState === 'ended') {
        reason = 'source_ended';
        throw new Error('source ended');
      }
      if (!validGeometry()) {
        reason = 'geometry_mismatch';
        throw new Error('not 16:9');
      }
      const canvas = document.createElement('canvas');
      canvas.width = 720;
      canvas.height = 1280;
      const context = canvas.getContext('2d');
      if (!context || !canvas.captureStream) {
        reason = 'capability_unavailable';
        throw new Error('canvas unavailable');
      }
      output = canvas.captureStream(0);
      replacement = output.getVideoTracks()[0];
      if (!replacement?.requestFrame) {
        reason = 'capability_unavailable';
        throw new Error('manual frames unavailable');
      }
      const apply = replacement.applyConstraints.bind(replacement);
      replacement.applyConstraints = async (constraints = {}) => {
        // Resolution belongs to the canvas, never to the physical/OBS source.
        // Ignore requested geometry (including advanced geometry), retain others.
        const filter = value => Object.fromEntries(Object.entries(value).filter(
          ([key]) => !['width', 'height', 'aspectRatio', 'resizeMode', 'advanced'].includes(key)));
        const safe = filter(constraints);
        if (constraints.advanced) safe.advanced = constraints.advanced.map(filter);
        await apply(safe);
      };
      replacement.stop = () => cleanup(true, 'conversion_stopped');
      replacement.addEventListener('ended', onReplacementEnded);
      source.addEventListener('ended', onEnded);
      video.addEventListener('error', onEnded);
      const draw = () => {
        if (closed) return;
        try {
          if (!validGeometry()) {
            reason = 'geometry_mismatch';
            throw new Error('source geometry changed');
          }
          const height = video.videoHeight;
          const width = height * 9 / 16;
          context.drawImage(video, (video.videoWidth - width) / 2, 0,
            width, height, 0, 0, 720, 1280);
          replacement.requestFrame();
          callback = video.requestVideoFrameCallback(draw);
        } catch (_) { cleanup(active, reason); }
      };
      await video.play();
      if (closed) throw new Error('source ended');
      // Refuse an unprepared/standby frame instead of zooming its center.
      const check = document.createElement('canvas');
      check.width = 16; check.height = 9;
      const probe = check.getContext('2d', { willReadFrequently: true });
      if (!probe) {
        reason = 'capability_unavailable';
        throw new Error('letterbox check unavailable');
      }
      probe.drawImage(video, 0, 0, 16, 9);
      const pixels = probe.getImageData(0, 0, 16, 9).data;
      for (const x of [0, 15]) for (const y of [2, 4, 6]) {
        const i = (y * 16 + x) * 4;
        if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 24 || pixels[i + 3] < 250) {
          reason = 'letterbox_unprepared';
          throw new Error('prepared black sidebars absent');
        }
      }
      draw();
      if (closed) throw new Error('drawing failed');
      active = true;
      activeConversionCount++;
      console.info('H3 full-frame portrait ready: 720x1280; centered letterbox extracted, no source-track crop.');
      return { replacement, rollback: () => cleanup() };
    } catch (_) {
      cleanup();
      throw new PreparationFailure(reason);
    }
  };
  const hook = async function (...args) {
    const stream = await original.apply(devices, args);
    const prepared = [];
    try {
      const tracks = [];
      for (const track of stream.getTracks()) {
        if (track.kind !== 'video' || track.label !== 'OBS Virtual Camera') {
          tracks.push(track);
        } else {
          const item = await prepare(track);
          prepared.push(item);
          tracks.push(item.replacement);
        }
      }
      return prepared.length ? new MediaStream(tracks) : stream;
    } catch (error) {
      fallbackCount++;
      lastReason = error instanceof PreparationFailure ? error.reason : 'preparation_failed';
      console.warn('H3 portrait transport unavailable; original OBS stream retained. No portrait success.');
      prepared.forEach(item => item.rollback());
      return stream;
    }
  };
  devices.getUserMedia = hook;
  console.info('H3 portrait hook installed only; no active conversion implied. Check h3OBSPortrait.status().');
  globalThis.h3OBSPortrait = {
    // Fresh metadata snapshots only: no stream, track, label, ID or media references.
    status() {
      return {
        installed: devices.getUserMedia === hook,
        activeConversionCount,
        fallbackCount,
        lastReason,
        output: { width: 720, height: 1280 }
      };
    },
    restore() {
      if (devices.getUserMedia === hook) devices.getUserMedia = original;
    }
  };
})();
