/* Code-Aware Lossy KV Reuse deck — pager logic.
   External file so it runs even when the viewer blocks inline <script> (CSP).
   The indicator is driven THREE ways so at least one always fires:
     1. IntersectionObserver (root=track)  — authoritative, any scroll method
     2. debounced scroll listener          — recomputes i from scrollLeft
     3. wheel discrete paging               — fixes the main gesture (vertical
        wheel was eaten by .slide overflow-y:auto, freezing the indicator)
   The progress bar is driven by CSS scroll-timeline (see <style>), so it
   advances even with JS fully disabled. */
(function () {
  var track = document.getElementById('track');
  if (!track) return;
  var slides = document.querySelectorAll('.slide');
  var n = slides.length;
  var i = 0;
  var prev = document.getElementById('prev');
  var next = document.getElementById('next');
  var cur = document.getElementById('cur');
  var total = document.getElementById('total');
  var sectitle = document.getElementById('sectitle');
  var dots = document.querySelectorAll('.dot');

  if (total) total.textContent = n;
  slides.forEach(function (s, k) { s.dataset.idx = k; });
  dots.forEach(function (d, k) { d.addEventListener('click', function () { go(k); }); });

  function titleOf(k) {
    var h = slides[k] && slides[k].querySelector('h1');
    return h ? h.textContent.trim() : '';
  }

  function render() {
    if (cur) cur.textContent = i + 1;
    if (sectitle) sectitle.textContent = titleOf(i);
    dots.forEach(function (d, k) { d.classList.toggle('active', k === i); });
    if (prev) prev.disabled = i === 0;
    if (next) next.disabled = i === n - 1;
  }

  function go(k) {
    i = Math.max(0, Math.min(n - 1, k));
    track.scrollTo({ left: i * track.clientWidth, behavior: 'smooth' });
    render();
  }
  function fwd() { if (i < n - 1) { i++; go(i); } }
  function back() { if (i > 0) { i--; go(i); } }

  if (next) next.addEventListener('click', fwd);
  if (prev) prev.addEventListener('click', back);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); fwd(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); back(); }
    else if (e.key === 'Home') { go(0); }
    else if (e.key === 'End') { go(n - 1); }
  });

  // (1) IntersectionObserver — authoritative current-slide source.
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting && e.intersectionRatio > 0.55) {
          var k = Number(e.target.dataset.idx);
          if (k !== i) { i = k; render(); }
        }
      });
    }, { root: track, threshold: [0.55, 0.75] });
    slides.forEach(function (s) { io.observe(s); });
  }

  // (2) debounced scroll fallback — always attached, recomputes i from scrollLeft.
  var ticking = false;
  track.addEventListener('scroll', function () {
    if (ticking) return; ticking = true;
    requestAnimationFrame(function () {
      var w = track.clientWidth || 1;
      var k = Math.max(0, Math.min(n - 1, Math.round(track.scrollLeft / w)));
      if (k !== i) { i = k; render(); }
      ticking = false;
    });
  }, { passive: true });

  // (3) wheel → discrete paging (fixes the frozen-indicator root cause:
  //     vertical wheel was consumed by .slide overflow-y:auto, so the track
  //     never advanced. shift+wheel still scrolls inside a tall slide.)
  var wacc = 0; var WTH = 60;
  track.addEventListener('wheel', function (e) {
    if (e.shiftKey) return;
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return; // horizontal wheel: native
    e.preventDefault();
    wacc += e.deltaY;
    if (wacc > WTH) { wacc = 0; fwd(); }
    else if (wacc < -WTH) { wacc = 0; back(); }
  }, { passive: false });

  // swipe (touch)
  var sx = 0;
  document.addEventListener('touchstart', function (e) { sx = e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', function (e) {
    var dx = e.changedTouches[0].clientX - sx;
    if (dx < -50) fwd(); else if (dx > 50) back();
  }, { passive: true });

  // keep current slide in view on resize
  window.addEventListener('resize', function () { track.scrollTo({ left: i * track.clientWidth }); });

  render();
})();
