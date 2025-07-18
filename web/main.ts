const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

async function useAmbientAudio(url: string) {
  try {
    const res = await fetch(url);
    const arrayBuffer = await res.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = true;
    source.connect(audioContext.destination);
    source.start();
  } catch (err) {
    console.error('Unable to decode audio data', err);
  }
}

useAmbientAudio('ambient.mp3');

const app = document.getElementById('app');
if (app) {
  app.textContent = 'Ambient audio demo';
}
