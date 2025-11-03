(function () {
  const settingsForm = document.getElementById('settings-form');
  const floatBtn = document.getElementById('floatBtn');
  const processedVideo = document.getElementById('processedVideo');
  const startButton = document.getElementById('startButton');
  const stopButton = document.getElementById('stopButton');

  let hideTimer = null;
  let videoRunning = false;

  function showMenu() {
    settingsForm.classList.remove('hidden');
    floatBtn.classList.remove('visible');
  }

  function hideMenu() {
    settingsForm.classList.add('hidden');
  }

  function showFloatBtn() {
    floatBtn.classList.add('visible');
  }

  function hideFloatBtn() {
    floatBtn.classList.remove('visible');
  }

  function clearHideTimer() {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function resetHideTimer() {
    if (!videoRunning) return;
    showMenu();
    clearHideTimer();
    hideTimer = setTimeout(() => {
      hideMenu();
      showFloatBtn();
    }, 5000);
  }

  if (startButton) {
    startButton.addEventListener('click', (e) => {
      videoRunning = true;
      resetHideTimer();
    });
  }

  if (stopButton) {
    stopButton.addEventListener('click', (e) => {
      videoRunning = false;
      clearHideTimer();
      showMenu();
      hideFloatBtn();
    });
  }

  if (processedVideo) {
    processedVideo.addEventListener('playing', () => {
      videoRunning = true;
      resetHideTimer();
    });
    processedVideo.addEventListener('pause', () => {
      videoRunning = false;
      clearHideTimer();
      showMenu();
      hideFloatBtn();
    });
    processedVideo.addEventListener('ended', () => {
      videoRunning = false;
      clearHideTimer();
      showMenu();
      hideFloatBtn();
    });
  }

  settingsForm.addEventListener('mouseenter', () => {
    clearHideTimer();
  });
  settingsForm.addEventListener('mouseleave', () => {
    if (videoRunning) resetHideTimer();
  });

  document.addEventListener('mousemove', (e) => {
    if (!videoRunning) return;

    if (settingsForm.classList.contains('hidden')) {
      showFloatBtn();
    } else {
      resetHideTimer();
    }
  });

  floatBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    showMenu();
    hideFloatBtn();
    resetHideTimer();
  });

  settingsForm.addEventListener('click', (e) => {
    e.stopPropagation();
  });

  document.addEventListener('click', (e) => {
    if (!videoRunning) return;
    if (settingsForm.contains(e.target) || floatBtn.contains(e.target)) return;
    hideMenu();
    showFloatBtn();
    clearHideTimer();
  });

  window.addEventListener('load', () => {
    if (processedVideo && !processedVideo.paused && !processedVideo.ended) {
      videoRunning = true;
      resetHideTimer();
    } else {
      videoRunning = false;
      showMenu();
      hideFloatBtn();
    }
  });
})();
