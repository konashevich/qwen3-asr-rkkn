const form = document.getElementById('encoder-form');
const featureInput = document.getElementById('input-features');
const selfTestButton = document.getElementById('npu-self-test-button');
const runButton = document.getElementById('encoder-run-button');
const submitStatus = document.getElementById('submit-status');
const resultEmpty = document.getElementById('result-empty');
const resultCard = document.getElementById('result-card');
const resultLabel = document.getElementById('result-label');
const resultElapsed = document.getElementById('result-elapsed');
const resultShape = document.getElementById('result-shape');
const resultText = document.getElementById('result-text');
const resultJson = document.getElementById('result-json');

function updateDiagnostics(diagnostics) {
  const npuStatus = document.getElementById('npu-status');
  const npuMaxDiff = document.getElementById('npu-max-diff');
  const rknnPath = document.getElementById('rknn-path');
  if (npuStatus && diagnostics?.npu) {
    npuStatus.textContent = diagnostics.npu.board_validation_report?.status || 'not run';
  }
  if (npuMaxDiff && diagnostics?.npu) {
    npuMaxDiff.textContent = diagnostics.npu.board_validation_report?.validation?.max_abs_diff ?? 'n/a';
  }
  if (rknnPath && diagnostics?.npu?.encoder_rknn_path) {
    rknnPath.textContent = diagnostics.npu.encoder_rknn_path;
  }
}

function setBusy(isBusy, message) {
  selfTestButton.disabled = isBusy;
  runButton.disabled = isBusy;
  submitStatus.textContent = message;
}

function showReport(label, payload) {
  updateDiagnostics(payload.diagnostics);
  const report = payload.npu_report || {};
  resultEmpty.classList.add('hidden');
  resultCard.classList.remove('hidden');
  resultLabel.textContent = label;
  resultElapsed.textContent = report.validation?.max_abs_diff !== undefined
    ? `max diff ${report.validation.max_abs_diff}`
    : report.status || 'done';
  resultShape.textContent = report.output_shape ? `output ${report.output_shape.join('x')}` : '';
  resultText.textContent = report.status === 'ok'
    ? 'Native RKNN execution completed on the Rockchip NPU.'
    : (payload.error?.message || 'Native RKNN execution failed.');
  resultJson.textContent = JSON.stringify(payload, null, 2);
}

function showError(payload) {
  if (payload.diagnostics) {
    updateDiagnostics(payload.diagnostics);
  }
  resultEmpty.classList.add('hidden');
  resultCard.classList.remove('hidden');
  resultLabel.textContent = payload.error?.type || 'Error';
  resultElapsed.textContent = 'request failed';
  resultShape.textContent = '';
  resultText.textContent = payload.error?.message || 'Unknown error';
  resultJson.textContent = JSON.stringify(payload, null, 2);
}

selfTestButton.addEventListener('click', async () => {
  setBusy(true, 'Running native RKNN self-test on the NPU.');
  try {
    const response = await fetch('/api/npu-self-test', { method: 'POST' });
    const payload = await response.json();
    if (!response.ok || payload.status !== 'ok') {
      showError(payload);
      setBusy(false, 'NPU self-test failed.');
      return;
    }
    showReport('NPU self-test', payload);
    setBusy(false, 'NPU self-test completed.');
  } catch (error) {
    showError({ error: { type: 'NetworkError', message: error.message } });
    setBusy(false, 'NPU self-test request failed.');
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const selectedFile = featureInput.files[0];
  if (!selectedFile) {
    showError({ error: { type: 'ValidationError', message: 'Choose a .npy feature tensor first.' } });
    return;
  }

  const formData = new FormData();
  formData.append('input_features', selectedFile, selectedFile.name);
  setBusy(true, 'Running uploaded tensor on the native RKNN encoder.');

  try {
    const response = await fetch('/api/run-encoder', {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== 'ok') {
      showError(payload);
      setBusy(false, 'Native encoder run failed.');
      return;
    }
    showReport('Uploaded tensor', payload);
    setBusy(false, 'Native encoder run completed.');
  } catch (error) {
    showError({ error: { type: 'NetworkError', message: error.message } });
    setBusy(false, 'Native encoder request failed.');
  }
});
