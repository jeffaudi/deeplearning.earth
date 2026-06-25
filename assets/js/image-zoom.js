(function () {
  function zoomBackground() {
    return document.body.classList.contains("colorscheme-dark")
      ? "rgba(0, 0, 0, 0.92)"
      : "rgba(255, 255, 255, 0.96)";
  }

  function initImageZoom() {
    if (typeof mediumZoom === "undefined") {
      return;
    }

    var images = document.querySelectorAll(".post-content .post-image-zoom");
    if (!images.length) {
      return;
    }

    mediumZoom(".post-content .post-image-zoom", {
      margin: 24,
      background: zoomBackground(),
      scrollOffset: 0,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initImageZoom);
  } else {
    initImageZoom();
  }
})();
