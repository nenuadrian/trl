"use strict";

(function () {
  const LATEX_COMMENT_RE = /^\s*#\s*Latex:\s*(.+?)\s*$/;
  const COMMENT_SELECTORS = [
    "pre code span.c",
    "pre code span.c1",
    "pre code span.cm",
    "pre code span.cp",
    "pre code span.cw",
  ];

  function replaceLatexComments(root) {
    const nodes = root.querySelectorAll(COMMENT_SELECTORS.join(","));
    const replaced = [];

    nodes.forEach((node) => {
      const text = (node.textContent || "").trim();
      const match = text.match(LATEX_COMMENT_RE);
      if (!match) {
        return;
      }

      const latex = match[1].trim();
      if (!latex) {
        return;
      }

      const latexNode = document.createElement("span");
      latexNode.className = "latex-code-comment";
      latexNode.setAttribute("data-latex", latex);
      latexNode.textContent = `\\(${latex}\\)`;
      node.replaceWith(latexNode);
      replaced.push(latexNode);
    });

    return replaced;
  }

  function typeset(elements) {
    if (elements.length === 0) {
      return;
    }

    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise(elements).catch((error) => {
        console.error("Failed to typeset # Latex comments:", error);
      });
    }
  }

  function renderLatexInCode(root) {
    const replaced = replaceLatexComments(root);
    typeset(replaced);
  }

  function init() {
    if (window.document$ && typeof window.document$.subscribe === "function") {
      window.document$.subscribe(() => renderLatexInCode(document));
      return;
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => renderLatexInCode(document));
      return;
    }

    renderLatexInCode(document);
  }

  init();
})();
