import { useEffect, useRef } from 'react';

/**
 * Simple hook to play looping ambient audio with graceful failure handling.
 * Errors from decoding are caught to avoid "Unable to decode audio data".
 */
export function useAmbientAudio(url: string, volume = 0.5) {
  const audioCtxRef = useRef<AudioContext>();
  const sourceRef = useRef<AudioBufferSourceNode>();

  useEffect(() => {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    audioCtxRef.current = ctx;

    fetch(url)
      .then(resp => resp.arrayBuffer())
      .then(data => ctx.decodeAudioData(data))
      .then(buffer => {
        const src = ctx.createBufferSource();
        src.buffer = buffer;
        src.loop = true;
        const gain = ctx.createGain();
        gain.gain.value = volume;
        src.connect(gain).connect(ctx.destination);
        src.start(0);
        sourceRef.current = src;
      })
      .catch(err => {
        console.error('Failed to start ambient audio', err);
      });

    return () => {
      sourceRef.current?.stop();
      ctx.close();
    };
  }, [url, volume]);
}
