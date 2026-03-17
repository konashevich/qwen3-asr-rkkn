const form = document.getElementById('transcribe-form');
const fileInput = document.getElementById('audio-input');
const languageInput = document.getElementById('language-input');
const contextInput = document.getElementById('context-input');
const timestampsInput = document.getElementById('timestamps-input');
const submitButton = document.getElementById('submit-button');
const submitStatus = document.getElementById('submit-status');
const recordingStatus = document.getElementById('recording-status');
const startButton = document.getElementById('record-start');
const stopButton = document.getElementById('record-stop');
const clearButton = document.getElementById('record-clear');
const resultEmpty = document.getElementById('result-empty');
const resultCard = document.getElementById('result-card');
const resultLanguage = document.getElementById('result-language');
const resultElapsed = document.getElementById('result-elapsed');
const resultCount = document.getElementById('result-count');
const resultText = document.getElementById('result-text');
const resultJson = document.getElementById('result-json');

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedBlob = null;

function updateDiagnostics(diagnostics) {
  const loaded = document.getElementById('runtime-loaded');
  const cacheHome = document.getElementById('runtime-cache-home');
  if (loaded && diagnostics?.runtime) {
    loaded.textContent = diagnostics.runtime.loaded ? 'yes' : 'no';
  }
  if (cacheHome && diagnostics?.runtime?.cache_home) {
    cacheHome.textContent = diagnostics.runtime.cache_home;
  }
}

function setBusy(isBusy, message) {
  submitButton.disabled = isBusy;
  submitStatus.textContent = message;
}

function showResult(payload) {
  const items = payload.transcription.raw_output || [];
  const first = items[0] || {};
  updateDiagnostics(payload.diagnostics);
  resultEmpty.classList.add('hidden');
  resultCard.classList.remove('hidden');
  resultLanguage.textContent = first.language || 'Language: auto';
  resultElapsed.textContent = `${payload.transcription.elapsed_seconds.toFixed(2)}s`;
  resultCount.textContent = `${payload.transcription.result_count} result(s)`;
  resultText.textContent = payload.transcription.parsed_text || '[no text returned]';
  resultJson.textContent = JSON.stringify(payload, null, 2);
}

function showError(payload) {
  if (payload.diagnostics) {
    updateDiagnostics(payload.diagnostics);
  }
  resultEmpty.classList.add('hidden');
  resultCard.classList.remove('hidden');
  resultLanguage.textContent = payload.error?.type || 'Error';
  resultElapsed.textContent = 'request failed';
  resultCount.textContent = '';
  resultText.textContent = payload.error?.message || 'Unknown error';
  resultJson.textContent = JSON.stringify(payload, null, 2);
}

async function startRecording() {
  recordedChunks = [];
  recordedBlob = null;
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const preferredMime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : undefined;
  mediaRecorder = new MediaRecorder(mediaStream, preferredMime ? { mimeType: preferredMime } : undefined);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
    recordingStatus.textContent = `Recorded clip ready: ${(recordedBlob.size / 1024).toFixed(1)} KiB.`;
    clearButton.disabled = false;
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
  };
  mediaRecorder.start();
  startButton.disabled = true;
  stopButton.disabled = false;
  clearButton.disabled = true;
  recordingStatus.textContent = 'Recording from browser microphone...';
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  startButton.disabled = false;
  stopButton.disabled = true;
}

function clearRecording() {
  recordedChunks = [];
  recordedBlob = null;
  clearButton.disabled = true;
  recordingStatus.textContent = 'Idle. If you record, the captured clip will be used when no file is selected.';
}

startButton.addEventListener('click', async () => {
  try {
    await startRecording();
  } catch (error) {
    recordingStatus.textContent = `Microphone error: ${error.message}`;
  }
});

stopButton.addEventListener('click', () => stopRecording());
clearButton.addEventListener('click', () => clearRecording());

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData();
  const selectedFile = fileInput.files[0];

  if (selectedFile) {
    formData.append('audio', selectedFile, selectedFile.name);
  } else if (recordedBlob) {
    formData.append('audio', recordedBlob, 'browser_recording.webm');
  } else {
    showError({ error: { type: 'ValidationError', message: 'Choose an audio file or record a clip first.' } });
    return;
  }

  formData.append('language', languageInput.value);
  formData.append('context', contextInput.value);
  formData.append('return_time_stamps', timestampsInput.checked ? 'true' : 'false');

  setBusy(true, 'Transcription running. First request may take several minutes while the model loads.');

  try {
    const response = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== 'ok') {
      showError(payload);
      setBusy(false, 'Request failed. Inspect the error details below.');
      return;
    }
    showResult(payload);
    setBusy(false, 'Transcription completed.');
  } catch (error) {
    showError({ error: { type: 'NetworkError', message: error.message } });
    setBusy(false, 'Request failed before the server returned a result.');
  }
});