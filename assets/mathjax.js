MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    options: {
      renderActions: {
        addStyles: [10, () => {
          // 为 MathJax 设置字体样式
          const styles = document.createElement('style');
          styles.innerHTML = `
            .mjx-math {
              font-family: 'TeX Gyre Pagella Math', serif !important;
            }
            .mjx-container {
              font-family: 'TeX Gyre Pagella Math', serif !important;
            }
          `;
          document.head.appendChild(styles);
        }]
      }
    },
    svg: {
      fontCache: 'global',
      font: 'TeX Gyre Pagella Math', // 使用 MathJax 的默认字体
      scale: 1 // 调整公式整体大小，默认值为1
    },
    startup: {
      ready: () => {
        MathJax.startup.defaultReady();
      }
    }
  };
  