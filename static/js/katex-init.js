document.addEventListener("DOMContentLoaded", function () {
  function loadScript(src, onload) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    document.head.appendChild(s);
  }

  loadScript(
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
    function () {
      loadScript(
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js",
        function () {
          renderMathInElement(document.body, {
            delimiters: [
              { left: "$$", right: "$$", display: true },
              { left: "$",  right: "$",  display: false }
            ],
            throwOnError: false
          });
        }
      );
    }
  );
});
