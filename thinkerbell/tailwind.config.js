/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'display': ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Thinkerbell brand colors
        'tb-pink': {
          50: '#fef2f4',
          100: '#fde6e8',
          200: '#fbcfd5',
          300: '#f7aab4',
          400: '#f27a8a',
          500: '#ea4c62',
          600: '#d63851',
          700: '#b42a42',
          800: '#96253e',
          900: '#80243b',
          950: '#470f1c',
        },
        'tb-magenta': '#FF1493',
        'tb-green': {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        'tb-neon-green': '#00FF00',
        success: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
        },
        warning: {
          50: '#fffbeb',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
        },
        error: {
          50: '#fef2f2',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
        },
      },
      animation: {
        'pulse-tb': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-tb': 'bounce 1s infinite',
      }
    },
  },
  plugins: [],
}
