/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface:   "#0b1220",
        elevated:  "#111a2e",
        border:    "#1f2a44",
        accent:    "#3b82f6",
        good:      "#22c55e",
        warn:      "#eab308",
        danger:    "#ef4444",
        muted:     "#94a3b8",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
