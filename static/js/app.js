// app.js — shared utilities (minimal, most logic is inline in each HTML page)

// Keyboard shortcut: press Enter in any input → trigger detect (index page only)
if (document.getElementById('detectBtn')) {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && document.activeElement.tagName === 'INPUT') {
      e.preventDefault();
      runDetection();
    }
  });
}

